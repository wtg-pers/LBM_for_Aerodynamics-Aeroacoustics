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


def extract_level(lev, part=None) -> dict:
    """Reference (NOT copy) everything the driver needs from a built level.

    Views only — at D40 the per-level f copy alone is up to 2.9GB and stacked
    copies OOM a 24GB card during runner construction (M5 field finding).
    LocalLevel wrap-slices its slab out of these views; the caller then
    releases the source arrays (see DistributedMLGRunner) so the transient
    peak stays at (t=0 build state + one slab) and shrinks level by level.

    `f0` is the ONE exception to "views only" — it is a standard-ordered
    copy. Given `part` it is SLAB-scoped (slab_std_f): only this rank's
    own+ghost rows are materialised, which is all it ever uses. The whole-
    level form is kept for a caller with no partition.

    The esoteric original is dead once that copy exists, so it is released
    HERE rather than by the caller after LocalLevel is built — that keeps the
    next level's peak from carrying this one's f.

    Why it matters: the full copy used to be the standing limit on
    replicated-build size. At octo8 v1 the largest level is 39,635,241 cells
    = 4.28 GB per copy, and gathering it OOM'd a 24 GB card that already held
    21.2 GB (2026-08-10 cluster run). Slab-scoped, that transient is ~1/n_ranks
    of it.
    """
    shape = lev.domain_shape
    # dist-init RESTART: the checkpoint's full field sits on the HOST; take
    # this rank's wrapped slab off it and never upload the rest.
    restart_host = getattr(lev, '_dist_restart_f', None)
    feq27 = None
    if lev.f is None and restart_host is None:         # dist-init: uniform IC
        rho0, u0 = lev._dist_init_ic
        xp0 = lev.xp
        rho_t = xp0.full((1, 1, 1), rho0, dtype=xp0.float32)
        u_t = xp0.zeros((3, 1, 1, 1), dtype=xp0.float32)
        for d in range(min(3, len(u0))):
            u_t[d] = u0[d]
        # same per-cell math as the full elementwise init -> bit-equal
        feq27 = lev.collision.compute_equilibrium(rho_t, u_t)
    _oh = getattr(lev, '_eso_omega_high', 1.0 / lev.tau)

    # Slab-scoped when the caller knows this rank's partition (the whole
    # point — see slab_std_f); full-level copy otherwise, which is the old
    # behaviour and what a 1-rank/no-partition caller still gets.
    f0_is_slab = False
    if restart_host is not None and part is not None:
        f0 = wrap_slice(restart_host, part, 1)   # host -> device, slab only
        f0_is_slab = True
        lev._dist_restart_f = None               # release the host reference
    elif lev.f is None:
        f0 = None
    elif part is not None:
        f0 = slab_std_f(lev, part)
        f0_is_slab = True
    else:
        f0 = lev.physical_f
    if f0 is not None:
        try:
            lev.f = None                    # esoteric original is dead now
        except AttributeError:
            pass                            # read-only property: caller frees
        else:
            cp.get_default_memory_pool().free_all_blocks()

    return dict(
        shape=shape,
        omega=1.0 / lev.tau,
        omega_bulk=getattr(lev, '_eso_omega_bulk', 1.0 / lev.tau),
        omega_high=_oh,
        omega_345=getattr(lev, '_eso_omega_345', (_oh, _oh, _oh)),
        lam=getattr(lev, '_eso_lambda', 0.0),
        bgk=(type(lev.collision).__name__ == 'BGKCollision'),
        sgs=dict(lev._sgs_cfg),
        f0=f0,                                         # standard phys, t=0
        f0_is_slab=f0_is_slab,                         # local coords already
        feq27=feq27,                                   # dist-init constant
        node_type=lev._eso_node_type.reshape(shape),
        # Wall-aware coupling skip (mlg.wall_coupling.mode='exclude'), or
        # None under the strict default. Computed on the FULL solid mask at
        # setup and sliced here exactly like node_type, so the flags are a
        # global geometric property — identical for any rank count.
        coupling_skip=(None if getattr(lev, '_coupling_skip_nt', None) is None
                       else lev._coupling_skip_nt.reshape(shape)),
        coupling_skip_dirs=tuple(getattr(lev, '_coupling_skip_dirs', ())),
        bc_rho=lev._eso_bc_rho.reshape(shape),
        bc_ux=lev._eso_bc_ux.reshape(shape),
        bc_uy=lev._eso_bc_uy.reshape(shape),
        bc_uz=lev._eso_bc_uz.reshape(shape),
    )


def slab_std_f(lev, part):
    """Standard-ordered f on THIS RANK's slab only — (27, *local_shape).

    `physical_f` materialises the WHOLE level even though a rank only ever
    uses its own slab of it. At octo8 v1 that is 4.28 GB for L2 alone, and it
    OOM'd a 24 GB card during runner construction with 21.2 GB already held
    (2026-08-10). This gathers 1/n_ranks of it (plus ghosts) instead.

    Wrapping is preserved exactly as wrap_slice does it: the local range
    [own_start - ghost, + local_shape) is taken modulo the global extent, so
    a rank at either end reads across the seam. That range is covered in at
    most two CONTIGUOUS pieces, both of which stay on the region kernel's
    fast path (it rejects out-of-bounds slices and would otherwise fall back
    to the python index path).
    """
    from src.kernels.esoteric_d3q27 import esoteric_gather_std_region

    xp = lev.xp
    ax = part.axis
    n_glob = int(lev.domain_shape[ax])
    n_loc = int(part.local_shape[ax])
    lo = part.own_start - part.ghost
    out, pos = None, 0
    while pos < n_loc:
        start = (lo + pos) % n_glob
        cnt = min(n_loc - pos, n_glob - start)
        reg = [slice(None)] * 3
        reg[ax] = slice(start, start + cnt)
        piece = esoteric_gather_std_region(
            xp, lev.f, lev._esoteric_step, tuple(reg))
        if out is None:
            out = xp.empty((piece.shape[0],) + tuple(part.local_shape),
                           dtype=piece.dtype)
        dst = [slice(None)] * 4
        dst[1 + ax] = slice(pos, pos + cnt)
        out[tuple(dst)] = piece
        del piece
        pos += cnt
    return out


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
        slab_src = bool(ld.get("f0_is_slab"))
        wrap_idx = None
        if ld["f0"] is None:                  # dist-init: uniform IC, no field
            cp.copyto(cbuf, cp.asarray(ld["feq27"],
                                       dtype=cp.float32).reshape(27, 1, 1, 1))
        elif not slab_src:                    # full field: wrap on the axis
            lo = part.own_start - part.ghost
            wrap_idx = cp.asarray(
                (np.arange(lo, lo + part.local_shape[part.axis])
                 % ld["f0"].shape[1 + part.axis]))
        # slab_src: already wrapped into local coords by slab_std_f
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
                if slab_src:
                    # already wrapped into local coords by slab_std_f
                    cp.copyto(view, ld["f0"][tuple(src)])
                else:
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
        # Flags the COUPLING scatters skip, per direction. Each defaults to
        # self.nt — the patch-12 solid-only skip — so a strict build, and
        # any direction the policy does not cover, is unchanged.
        _cs = ld.get("coupling_skip")
        _dirs = ld.get("coupling_skip_dirs") or ()
        _wall = (None if _cs is None
                 else wrap_slice(_cs, part).ravel().copy())
        self.nt_c2f = _wall if (_wall is not None and "c2f" in _dirs) else self.nt
        self.nt_f2c = _wall if (_wall is not None and "f2c" in _dirs) else self.nt
        if t0 == 0:
            # fresh IC only: seed the implicit-HWBB bounce slots on the slab
            # (restart t0>0 scatters the checkpointed deposits bit-exactly —
            # seeding would overwrite them; see eso_seed_solid_bounce_ic).
            # Ghost-edge seeds touched by the LOCAL wrap of the roll are
            # overwritten by the first halo sync (every level syncs before
            # its first advance), so only owned-region correctness matters.
            from src.kernels.esoteric_d3q27 import eso_seed_solid_bounce_ic
            if ld["f0"] is not None:
                get_dir = ((lambda q: ld["f0"][q]) if slab_src
                           else (lambda q: wrap_slice(ld["f0"][q], part)))
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
        self.mk.launch(self.mem, self.nt, self.rho.ravel(),
                       self.u.reshape(3, -1), nx, ny, nz, self.t)

    def _sgs_mask_solid_u(self) -> None:
        """No-slip the SGS input u at SOLID cells (patch 12 F1e — same
        rationale as Simulation._sgs_mask_solid_u: the gradient/test-filter
        stencils read u at solid neighbours; u=0 is the physical value)."""
        idx = getattr(self, '_sgs_solid_idx', None)
        if idx is None:
            import cupy as cp
            idx = cp.where(self.nt == 1)[0]
            self._sgs_solid_idx = idx
        if idx.size:
            self.u_b[:, idx] = 0.0

    def _sgs_wall_damp(self) -> None:
        """Zero nu_t_in within sgs.wall_damp_cells of the body (patch 12
        follow-up; same rationale as Simulation._sgs_wall_damp — the
        staircase wall jump is not resolved strain). Slab-local dilation is
        correct in the owned interior: ghosts (>=5) exceed the damp radius."""
        n = int(self.sgs.get("wall_damp_cells", 0))
        if n <= 0:
            return
        idx = getattr(self, '_sgs_damp_idx', None)
        if idx is None:
            import cupy as cp
            from cupyx.scipy import ndimage
            solid3 = (self.nt == 1).reshape(self.dims)
            shell = ndimage.binary_dilation(
                solid3, iterations=n, brute_force=True)
            idx = cp.where(shell.ravel())[0]
            self._sgs_damp_idx = idx
        if idx.size:
            self.nut_in[idx] = 0.0

    def advance(self, force=None) -> None:
        nx, ny, nz = self.dims
        kw = {}
        if self.sgs["model"] == "dyn_smag":
            self.mk.launch(self.mem, self.nt, self.rho_b, self.u_b,
                           nx, ny, nz, self.t)
            self._sgs_mask_solid_u()
            self.dk.launch(self.u_b[0], self.u_b[1], self.u_b[2],
                           self.nut_in, nx, ny, nz, dx=1.0,
                           Cs_max=float(self.sgs["Cs_max"]),
                           alpha_sq=float(self.sgs["alpha_sq"]))
            self._sgs_wall_damp()
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
