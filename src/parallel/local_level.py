"""Rank-local esoteric level — production promotion of the gate-proven code.

`extract_level` copies everything a rank needs from a BUILT production level
(each rank runs the full deterministic SimulationSetup build, then keeps only
its wrap-sliced slab and frees the rest — bit-identical per-rank state
without a distributed initializer). `LocalLevel.advance` replicates
Simulation._advance_esoteric exactly: optional dyn_smag pre-pass (macro LOAD
-> nu_t) then the fused esoteric cumulant kernel, with an optional ALM body
force. Verified bit-for-bit against MultiLevelGrid.advance() in gates
G-M1/M2a/M2b/M3 (patch 17).
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from src.kernels.esoteric_cumulant_d3q27 import EsotericCumulantKernelD3Q27
from src.kernels.esoteric_d3q27 import (
    EsotericMacroKernelD3Q27, esoteric_scatter_std)
from src.kernels.dyn_smag_d3q27 import DynSmagKernelD3Q27


def extract_level(lev) -> dict:
    """Reference (NOT copy) everything the driver needs from a built level.

    Views only — at D40 the per-level f copy alone is up to 2.9GB and stacked
    copies OOM a 24GB card during runner construction (M5 field finding).
    LocalLevel wrap-slices its slab out of these views; the caller then
    releases the source arrays (see DistributedMLGRunner) so the transient
    peak stays at (t=0 build state + one slab) and shrinks level by level.
    """
    shape = lev.domain_shape
    feq27 = None
    if lev.f is None:                                  # dist-init: uniform IC
        rho0, u0 = lev._dist_init_ic
        xp0 = lev.xp
        rho_t = xp0.full((1, 1, 1), rho0, dtype=xp0.float32)
        u_t = xp0.zeros((3, 1, 1, 1), dtype=xp0.float32)
        for d in range(min(3, len(u0))):
            u_t[d] = u0[d]
        # same per-cell math as the full elementwise init -> bit-equal
        feq27 = lev.collision.compute_equilibrium(rho_t, u_t)
    _oh = getattr(lev, '_eso_omega_high', 1.0 / lev.tau)
    return dict(
        shape=shape,
        omega=1.0 / lev.tau,
        omega_bulk=getattr(lev, '_eso_omega_bulk', 1.0 / lev.tau),
        omega_high=_oh,
        omega_345=getattr(lev, '_eso_omega_345', (_oh, _oh, _oh)),
        lam=getattr(lev, '_eso_lambda', 0.0),
        bgk=(type(lev.collision).__name__ == 'BGKCollision'),
        sgs=dict(lev._sgs_cfg),
        f0=lev.physical_f,                             # standard phys, t=0
        feq27=feq27,                                   # dist-init constant
        node_type=lev._eso_node_type.reshape(shape),
        bc_rho=lev._eso_bc_rho.reshape(shape),
        bc_ux=lev._eso_bc_ux.reshape(shape),
        bc_uy=lev._eso_bc_uy.reshape(shape),
        bc_uz=lev._eso_bc_uz.reshape(shape),
    )


def wrap_slice(arr, part, spatial_offset: int = 0):
    """Wrap-slice along the partition axis (works for (...,X,Y,Z) arrays).

    Accepts host numpy (dist-init metadata) or device arrays; always
    returns a device array."""
    ax = spatial_offset + part.axis
    lo = part.own_start - part.ghost
    idx = (np.arange(lo, lo + part.local_shape[part.axis]) % arr.shape[ax])
    if isinstance(arr, np.ndarray):
        return cp.asarray(np.take(arr, idx, axis=ax))
    return cp.take(arr, cp.asarray(idx), axis=ax)


class LocalLevel:
    """Rank-local esoteric level replicating Simulation._advance_esoteric."""

    def __init__(self, ld: dict, part, t0: int = 0) -> None:
        self.part = part
        self.dims = tuple(part.local_shape)
        n = int(np.prod(self.dims))
        # t0: restart support — the esoteric parity must CONTINUE from the
        # restored step (checkpoints store parity-free std f; scattering at
        # the restored parity reproduces the uninterrupted memory state).
        #
        # CHUNKED construction: the whole-slab scatter held THREE slab-sized
        # transients (wrap copy + eso convert + roll) — native-24GB OOM at
        # NR=1 D40 (WSL2 oversubscription masked it locally) and it would
        # break the 450M dist-init slabs the same way. Chunk regions scatter
        # additively (each pair-slot write is independent and sourced from
        # its own chunk) -> union over chunks == full scatter, bit-exact.
        from src.kernels.esoteric_d3q27 import esoteric_scatter_std_region
        cp.get_default_memory_pool().free_all_blocks()   # defragment first
        self.mem = cp.empty((27,) + self.dims, cp.float32)
        cax = 0 if part.axis != 0 else 1          # chunk along a non-split axis
        rest = 1
        for d in range(3):
            if d != cax:
                rest *= self.dims[d]
        ch = min(self.dims[cax], max(1, int(4_000_000 // max(rest, 1))))
        # ONE reused chunk buffer (cp.take(out=...)): per-chunk allocations
        # fragmented the pool under a hard 24GB limit — the level-mem
        # allocs then found no contiguous block despite GBs nominally free
        cshape = list(self.dims)
        cshape[cax] = ch
        cbuf = cp.empty((27,) + tuple(cshape), cp.float32)
        wrap_idx = None
        if ld["f0"] is not None:
            lo = part.own_start - part.ghost
            wrap_idx = cp.asarray(
                (np.arange(lo, lo + part.local_shape[part.axis])
                 % ld["f0"].shape[1 + part.axis]))
        else:
            cp.copyto(cbuf, cp.asarray(ld["feq27"],
                                       dtype=cp.float32).reshape(27, 1, 1, 1))
        for c0 in range(0, self.dims[cax], ch):
            n_c = min(ch, self.dims[cax] - c0)
            sl = slice(c0, c0 + n_c)
            reg = [slice(None)] * 3
            reg[cax] = sl
            view_ix = [slice(None)] * 4
            view_ix[1 + cax] = slice(0, n_c)
            view = cbuf[tuple(view_ix)]
            if ld["f0"] is not None:
                # non-split axes: local == global index, so the chunk slice
                # applies to the source directly; wrap only on the axis
                src = [slice(None)] * 4
                src[1 + cax] = sl
                cp.take(ld["f0"][tuple(src)], wrap_idx,
                        axis=1 + part.axis, out=view)
            vals = view if view.flags.c_contiguous \
                else cp.ascontiguousarray(view)
            esoteric_scatter_std_region(cp, self.mem, vals, t0, tuple(reg))
        del cbuf
        self.t = t0
        self.omega = ld["omega"]
        self.ob, self.oh = ld["omega_bulk"], ld["omega_high"]
        self.w345 = ld.get("omega_345", (self.oh, self.oh, self.oh))
        self.lam = ld.get("lam", 0.0)
        self.nt = wrap_slice(ld["node_type"], part).ravel().copy()
        if t0 == 0:
            # fresh IC only: seed the implicit-HWBB bounce slots on the slab
            # (restart t0>0 scatters the checkpointed deposits bit-exactly —
            # seeding would overwrite them; see eso_seed_solid_bounce_ic).
            # Ghost-edge seeds touched by the LOCAL wrap of the roll are
            # overwritten by the first halo sync (every level syncs before
            # its first advance), so only owned-region correctness matters.
            from src.kernels.esoteric_d3q27 import eso_seed_solid_bounce_ic
            if ld["f0"] is not None:
                get_dir = lambda q: wrap_slice(ld["f0"][q], part)
            else:
                get_dir = lambda q: float(ld["feq27"][q].ravel()[0])
            eso_seed_solid_bounce_ic(
                cp, self.mem, self.nt.reshape(self.dims), get_dir, t0)
        self.b_r = wrap_slice(ld["bc_rho"], part).ravel().copy()
        self.b_x = wrap_slice(ld["bc_ux"], part).ravel().copy()
        self.b_y = wrap_slice(ld["bc_uy"], part).ravel().copy()
        self.b_z = wrap_slice(ld["bc_uz"], part).ravel().copy()
        self.rho = cp.empty(self.dims, cp.float32)
        self.u = cp.empty((3,) + self.dims, cp.float32)
        self.sgs = ld["sgs"]
        dyn = self.sgs["model"] == "dyn_smag"
        self.is_bgk = ld.get("bgk", False)
        if self.is_bgk:
            if self.sgs["model"] not in ("off", "none", "dyn_smag"):
                raise ValueError(
                    "esoteric BGK runner: SGS must be off or dyn_smag "
                    f"(got '{self.sgs['model']}')")
            from src.kernels.esoteric_d3q27 import EsotericBGKKernelD3Q27
            self.ker = EsotericBGKKernelD3Q27()
        else:
            self.ker = EsotericCumulantKernelD3Q27(
                sgs_model=("wale" if dyn else "off"))
        self.mk = EsotericMacroKernelD3Q27()           # ALM macro pre-pass
        if dyn:
            self.dk = DynSmagKernelD3Q27()
            self.rho_b = cp.empty(n, cp.float32)
            self.u_b = cp.empty((3, n), cp.float32)
            self.nut_in = cp.empty(n, cp.float32)
            self.nut = cp.zeros(n, cp.float32)

    def macro_pre_pass(self) -> None:
        """LOAD-only rho/u into self.rho/self.u (ALM sampling input)."""
        nx, ny, nz = self.dims
        self.mk.launch(self.mem, self.rho.ravel(), self.u.reshape(3, -1),
                       nx, ny, nz, self.t)

    def advance(self, force=None) -> None:
        nx, ny, nz = self.dims
        kw = {}
        if self.sgs["model"] == "dyn_smag":
            self.mk.launch(self.mem, self.rho_b, self.u_b, nx, ny, nz, self.t)
            self.dk.launch(self.u_b[0], self.u_b[1], self.u_b[2],
                           self.nut_in, nx, ny, nz, dx=1.0,
                           Cs_max=float(self.sgs["Cs_max"]),
                           alpha_sq=float(self.sgs["alpha_sq"]))
            kw = dict(nu_t_in=self.nut_in, nu_t_out=self.nut)
        if self.is_bgk:
            self.ker.launch(self.mem, self.rho, self.u, self.nt, self.b_r,
                            self.b_x, self.b_y, self.b_z,
                            self.omega, nx, ny, nz,
                            t_step=self.t, force=force, **kw)
        else:
            self.ker.launch(self.mem, self.rho, self.u, self.nt, self.b_r,
                            self.b_x, self.b_y, self.b_z,
                            self.omega, self.ob, self.oh, nx, ny, nz,
                            t_step=self.t, force=force,
                            Cs=float(self.sgs.get("Cs", 0.0)),
                            omega_3=self.w345[0], omega_4=self.w345[1],
                            omega_5=self.w345[2], lambda_lim=self.lam, **kw)
        self.t += 1
