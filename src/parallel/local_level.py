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
    return dict(
        shape=shape,
        omega=1.0 / lev.tau,
        omega_bulk=lev._eso_omega_bulk,
        omega_high=lev._eso_omega_high,
        sgs=dict(lev._sgs_cfg),
        f0=lev.physical_f,                             # standard phys, t=0
        node_type=lev._eso_node_type.reshape(shape),
        bc_rho=lev._eso_bc_rho.reshape(shape),
        bc_ux=lev._eso_bc_ux.reshape(shape),
        bc_uy=lev._eso_bc_uy.reshape(shape),
        bc_uz=lev._eso_bc_uz.reshape(shape),
    )


def wrap_slice(arr, part, spatial_offset: int = 0):
    """Wrap-slice along the partition axis (works for (...,X,Y,Z) arrays)."""
    ax = spatial_offset + part.axis
    lo = part.own_start - part.ghost
    idx = (np.arange(lo, lo + part.local_shape[part.axis]) % arr.shape[ax])
    return cp.take(arr, cp.asarray(idx), axis=ax)


class LocalLevel:
    """Rank-local esoteric level replicating Simulation._advance_esoteric."""

    def __init__(self, ld: dict, part, t0: int = 0) -> None:
        self.part = part
        self.dims = tuple(part.local_shape)
        n = int(np.prod(self.dims))
        # t0: restart support — the esoteric parity must CONTINUE from the
        # restored step (checkpoints store parity-free std f; scattering at
        # the restored parity reproduces the uninterrupted memory state)
        self.mem = esoteric_scatter_std(cp, wrap_slice(ld["f0"], part, 1), t0)
        self.t = t0
        self.omega = ld["omega"]
        self.ob, self.oh = ld["omega_bulk"], ld["omega_high"]
        self.nt = wrap_slice(ld["node_type"], part).ravel().copy()
        self.b_r = wrap_slice(ld["bc_rho"], part).ravel().copy()
        self.b_x = wrap_slice(ld["bc_ux"], part).ravel().copy()
        self.b_y = wrap_slice(ld["bc_uy"], part).ravel().copy()
        self.b_z = wrap_slice(ld["bc_uz"], part).ravel().copy()
        self.rho = cp.empty(self.dims, cp.float32)
        self.u = cp.empty((3,) + self.dims, cp.float32)
        self.sgs = ld["sgs"]
        dyn = self.sgs["model"] == "dyn_smag"
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
        self.ker.launch(self.mem, self.rho, self.u, self.nt, self.b_r,
                        self.b_x, self.b_y, self.b_z,
                        self.omega, self.ob, self.oh, nx, ny, nz,
                        t_step=self.t, force=force,
                        Cs=float(self.sgs.get("Cs", 0.0)), **kw)
        self.t += 1
