"""G-M2b — REAL bench5 5-level MLG, 2-rank decomposed vs production 1-rank.

Reference = the actual production path: bench5_pure_lbm built by
SimulationSetup with LBM_ESOTERIC=1 (esoteric cumulant + dyn_smag + eq/sponge
BCs), advanced by MultiLevelGrid.advance(). The decomposed run drives the
SAME kernels per rank on wrap-sliced local slabs (f, node_type, bc arrays),
with the production advance replicated (dyn_smag pre-pass -> WALE-branch
cumulant), per-level halo exchange, and RankLocalCouplingV1 for all 4 level
pairs (fine cuts derived down the chain).

ghost = 3 here (NOT 2): dyn_smag needs u one cell beyond the collided cell,
so the redundant-ghost-compute argument needs one extra synced layer
(u corrupt at ghost-3 -> nu_t valid at ghost-1 -> owned protected).

Run from repo root:  python patch_notes/hpc_upgrade/gates/mgpu_m2b_gate.py
"""
import os
os.environ["LBM_ESOTERIC"] = "1"
for _k in ("MLG_CUDA_GRAPH", "MLG_PROFILE", "MLG_NVTX"):
    os.environ.pop(_k, None)
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp
from src.kernels.esoteric_cumulant_d3q27 import EsotericCumulantKernelD3Q27
from src.kernels.esoteric_d3q27 import (
    EsotericMacroKernelD3Q27, esoteric_gather_std, esoteric_scatter_std,
    esoteric_gather_std_region)
from src.kernels.dyn_smag_d3q27 import DynSmagKernelD3Q27
from src.parallel import Partition1D, HaloBandExchangerV1, LoopbackTransport
from src.parallel.mlg_coupling import (
    RankLocalCouplingV1, fine_range_from_coarse)

NSTEPS, NR, GHOST = 2, 2, 3


def build_reference():
    sys.argv = ["m2b", "--config", "configs/hpc_bench/bench5_pure_lbm.py",
                "--gpu", "0", "--no-vtk", "--no-force"]
    from src.io.args_parser import parse_args
    from src.solver.setup import SimulationSetup
    from src.solver.initializer import SolverInitializer
    a = parse_args()
    s = SimulationSetup(a)
    s.start_log_capture()
    mlg = s.build_simulation()
    SolverInitializer(s).initialize(mlg, a)
    s.stop_log_capture()
    mlg._graph_enabled = False
    return mlg


def extract_level(lev):
    """Copy everything the distributed driver needs from a built level."""
    shape = lev.domain_shape
    d = dict(
        shape=shape,
        omega=1.0 / lev.tau,
        omega_bulk=lev._eso_omega_bulk,
        omega_high=lev._eso_omega_high,
        sgs=dict(lev._sgs_cfg),
        f0=lev.physical_f.copy(),                      # standard phys, t=0
        node_type=lev._eso_node_type.reshape(shape).copy(),
        bc_rho=lev._eso_bc_rho.reshape(shape).copy(),
        bc_ux=lev._eso_bc_ux.reshape(shape).copy(),
        bc_uy=lev._eso_bc_uy.reshape(shape).copy(),
        bc_uz=lev._eso_bc_uz.reshape(shape).copy(),
    )
    return d


def wrap_slice(arr, part, spatial_offset=0):
    """Wrap-slice along the partition axis (works for (…,X,Y,Z) arrays)."""
    ax = spatial_offset + part.axis
    lo = part.own_start - part.ghost
    idx = (np.arange(lo, lo + part.local_shape[part.axis]) % arr.shape[ax])
    return cp.take(arr, cp.asarray(idx), axis=ax)


class LocalLevel:
    """Rank-local esoteric level replicating Simulation._advance_esoteric."""

    def __init__(self, ld, part):
        self.part = part
        self.dims = tuple(part.local_shape)
        n = int(np.prod(self.dims))
        self.mem = esoteric_scatter_std(cp, wrap_slice(ld["f0"], part, 1), 0)
        self.t = 0
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
        if dyn:
            self.mk = EsotericMacroKernelD3Q27()
            self.dk = DynSmagKernelD3Q27()
            self.rho_b = cp.empty(n, cp.float32)
            self.u_b = cp.empty((3, n), cp.float32)
            self.nut_in = cp.empty(n, cp.float32)
            self.nut = cp.zeros(n, cp.float32)

    def advance(self):
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
                        t_step=self.t, force=None,
                        Cs=float(self.sgs.get("Cs", 0.0)), **kw)
        self.t += 1


def run_distributed(level_data, couplings, axis):
    NL = len(level_data)
    parts = []                                     # parts[k][r]
    p0 = [Partition1D(level_data[0]["shape"], NR, r, axis, ghost=GHOST)
          for r in range(NR)]
    parts.append(p0)
    for k in range(1, NL):
        gc = couplings[k - 1]
        fdc = gc._region.fine_domain_coarse
        lo = (fdc.x_start, fdc.y_start, fdc.z_start)[axis]
        hi = (fdc.x_end, fdc.y_end, fdc.z_end)[axis]
        nf = gc._region.fine_shape[axis]
        row = []
        for r in range(NR):
            f0_, fc_ = fine_range_from_coarse(parts[k - 1][r], lo, hi, nf)
            assert fc_ >= GHOST, f"L{k} rank{r} own={fc_} < ghost={GHOST}"
            row.append(Partition1D.from_range(
                level_data[k]["shape"], NR, r, axis, f0_, fc_, ghost=GHOST))
        parts.append(row)

    lb = LoopbackTransport()
    lv = [[LocalLevel(level_data[k], parts[k][r]) for r in range(NR)]
          for k in range(NL)]
    ex = [[HaloBandExchangerV1(parts[k][r], lb, cp) for r in range(NR)]
          for k in range(NL)]
    rlc = [[RankLocalCouplingV1(couplings[k], parts[k][r], parts[k + 1][r], cp)
            for r in range(NR)] for k in range(NL - 1)]
    fprev = [[None] * NR for _ in range(NL - 1)]

    def sync(k):
        for r in range(NR):
            ex[k][r].post(lv[k][r].mem, lv[k][r].t)
        for r in range(NR):
            ex[k][r].complete(lv[k][r].mem, lv[k][r].t)

    def save_fprev(k):                              # pair k, coarse level k
        for r in range(NR):
            reg = rlc[k][r].coarse_block_region_local()
            fprev[k][r] = (esoteric_gather_std_region(
                cp, lv[k][r].mem, lv[k][r].t, reg) if reg else None)

    def advance_fine(k):
        has_finer = k + 1 < NL
        if has_finer:
            sync(k); save_fprev(k)
        sync(k)
        for r in range(NR):
            lv[k][r].advance()
        sync(k - 1)
        for r in range(NR):
            rlc[k - 1][r].c2f(lv[k - 1][r].mem, lv[k][r].mem,
                              lv[k - 1][r].t, lv[k][r].t,
                              is_half_step=True,
                              f_prev_sub_loc=fprev[k - 1][r])
        if has_finer:
            advance_fine(k + 1)
        if has_finer:
            sync(k); save_fprev(k)
        sync(k)
        for r in range(NR):
            lv[k][r].advance()
        for r in range(NR):
            rlc[k - 1][r].c2f(lv[k - 1][r].mem, lv[k][r].mem,
                              lv[k - 1][r].t, lv[k][r].t, is_half_step=False)
        if has_finer:
            advance_fine(k + 1)
        sync(k)
        for r in range(NR):
            rlc[k - 1][r].f2c(lv[k][r].mem, lv[k - 1][r].mem,
                              lv[k][r].t, lv[k - 1][r].t)

    for _ in range(NSTEPS):
        sync(0); save_fprev(0)
        for r in range(NR):
            lv[0][r].advance()
        advance_fine(1)

    out = []
    for k in range(NL):
        pieces = [esoteric_gather_std(cp, lv[k][r].mem, lv[k][r].t)
                  [(slice(None),) + parts[k][r].owned_local()]
                  for r in range(NR) if parts[k][r].own_count > 0]
        out.append(cp.concatenate(pieces, axis=1 + axis))
    return out


def main():
    mlg = build_reference()
    NL = mlg.num_levels
    level_data = [extract_level(mlg.get_level(k)) for k in range(NL)]
    couplings = mlg._couplings
    shapes = [level_data[k]["shape"] for k in range(NL)]
    print("  level shapes:", shapes)

    # auto-axis must REJECT x: the bench5 nested boxes collapse the x chain
    # (pair2 box x[18,62] falls inside rank0's derived L2 range -> L3/L4
    # would land entirely on one rank). Chain-aware choose_axis catches it.
    from src.parallel import choose_axis
    boxes = []
    for gc in couplings:
        b = gc._region.fine_domain_coarse
        boxes.append(((b.x_start, b.x_end), (b.y_start, b.y_end),
                      (b.z_start, b.z_end)))
    ax_auto = choose_axis(shapes, boxes, n_ranks=NR)
    print(f"  auto axis = {ax_auto} (x correctly rejected: {ax_auto != 0})")
    assert ax_auto != 0

    # distributed FIRST (from pristine t=0 copies), then reference advances.
    dist = {}
    for axis, nm in ((1, "y"), (2, "z")):
        dist[axis] = run_distributed(level_data, couplings, axis)
        cp.get_default_memory_pool().free_all_blocks()

    for _ in range(NSTEPS):
        mlg.advance()
    ref = [mlg.get_level(k).physical_f for k in range(NL)]

    ok = True
    for axis, nm in ((1, "y"), (2, "z")):
        bits, md = [], 0.0
        for k in range(NL):
            b = bool(cp.all(dist[axis][k] == ref[k]))
            md = max(md, float(cp.max(cp.abs(dist[axis][k] - ref[k]))))
            bits.append(b)
        print(f"  axis={nm}: per-level bit={bits}  max|d|={md:.2e}")
        ok = ok and all(bits)
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
