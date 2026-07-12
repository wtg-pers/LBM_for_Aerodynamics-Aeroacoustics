"""G-M3 — distributed ALM (partial-sum allreduce) vs production 1-rank.

Two bench5 configs, 2 ranks, split axis y:
  bench5_purealm_m3 : pure ALM (no corrections)      — sampling/spreading core
  bench5_archb_m3   : radial trunc + kleine STRAIGHT — production cases 3/4/6 path

Per rank a REAL ActuatorLineModel (from an extra identical build) runs with
the M3 hooks: _grid_offset, local _F_grid, _velocity_sampler = barrier-
allreduce partial sums (ThreadAllreduce mirrors MPI.Allreduce control flow;
each rank's step() runs in its own thread). BEM/kleine solve replicate
identically on every rank from the shared u_markers.

Bit-parity is NOT expected (straddling markers change fp summation order);
gate: per-level field max|df| < 1e-4 and F_grid owned-assembled relative
error < 1e-5 after NSTEPS coarse steps.

Run from repo root:  python patch_notes/hpc_upgrade/gates/mgpu_m3_gate.py
"""
import os, sys, threading, types
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp

# load the M2b gate as a library (main() guarded out)
_g = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mgpu_m2b_gate.py")
_src = open(_g).read().replace('if __name__ == "__main__":\n    main()', '')
m2b = types.ModuleType("m2b"); m2b.__dict__['__file__'] = _g
exec(compile(_src, _g, "exec"), m2b.__dict__)

from src.parallel import Partition1D, HaloBandExchangerV1, LoopbackTransport
from src.parallel.mlg_coupling import RankLocalCouplingV1, fine_range_from_coarse
from src.parallel.alm_dist import ThreadAllreduce, make_distributed_sampler
from src.kernels.esoteric_d3q27 import esoteric_gather_std, esoteric_gather_std_region

NSTEPS, NR, GHOST, AXIS = 2, 2, 3, 1


def build(config):
    sys.argv = ["m3", "--config", config, "--gpu", "0", "--no-vtk", "--no-force"]
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
    return mlg, s


def run_case(config, name):
    # ── reference build (also the level-data source at t=0) ──
    mlg, s0 = build(config)
    NL = mlg.num_levels
    level_data = [m2b.extract_level(mlg.get_level(k)) for k in range(NL)]
    couplings = mlg._couplings
    alm_level = NL - 1
    assert mlg.get_level(alm_level).al_model is not None

    # ── per-rank ALM models from extra identical builds ──
    models = []
    for r in range(NR):
        mr, sr = build(config)
        models.append(mr.get_level(alm_level).al_model)
        del mr, sr
    import gc as _gc
    _gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── partitions / halo / coupling (as in M2b) ──
    parts = [[Partition1D(level_data[0]["shape"], NR, r, AXIS, ghost=GHOST)
              for r in range(NR)]]
    for k in range(1, NL):
        fdc = couplings[k - 1]._region.fine_domain_coarse
        lo = (fdc.x_start, fdc.y_start, fdc.z_start)[AXIS]
        hi = (fdc.x_end, fdc.y_end, fdc.z_end)[AXIS]
        nf = couplings[k - 1]._region.fine_shape[AXIS]
        parts.append([Partition1D.from_range(
            level_data[k]["shape"], NR, r, AXIS,
            *fine_range_from_coarse(parts[k - 1][r], lo, hi, nf), ghost=GHOST)
            for r in range(NR)])
    lb = LoopbackTransport()
    lv = [[m2b.LocalLevel(level_data[k], parts[k][r]) for r in range(NR)]
          for k in range(NL)]
    ex = [[HaloBandExchangerV1(parts[k][r], lb, cp) for r in range(NR)]
          for k in range(NL)]
    rlc = [[RankLocalCouplingV1(couplings[k], parts[k][r], parts[k + 1][r], cp)
            for r in range(NR)] for k in range(NL - 1)]
    fprev = [[None] * NR for _ in range(NL - 1)]

    # ── configure the per-rank ALM models (production M3 hooks) ──
    allred = ThreadAllreduce(NR)
    Lp = parts[alm_level]
    for r, m in enumerate(models):
        off = np.zeros(3)
        off[AXIS] = Lp[r].own_start - Lp[r].ghost
        m._global_domain_shape = tuple(level_data[alm_level]["shape"])
        m.domain_shape = tuple(Lp[r].local_shape)
        m._F_grid = cp.zeros((3,) + tuple(Lp[r].local_shape), dtype=cp.float64)
        m._grid_offset = off
        m._velocity_sampler = make_distributed_sampler(allred, r, Lp[r], cp)

    def sync(k):
        for r in range(NR):
            ex[k][r].post(lv[k][r].mem, lv[k][r].t)
        for r in range(NR):
            ex[k][r].complete(lv[k][r].mem, lv[k][r].t)

    def save_fprev(k):
        for r in range(NR):
            reg = rlc[k][r].coarse_block_region_local()
            fprev[k][r] = (esoteric_gather_std_region(
                cp, lv[k][r].mem, lv[k][r].t, reg) if reg else None)

    def alm_kernel_step(k):
        """Production _advance_esoteric_with_alm, distributed."""
        # pass 1: macro pre-pass into lvl.rho/u (flat views), per rank
        for r in range(NR):
            L = lv[k][r]
            nx, ny, nz = L.dims
            L.mk.launch(L.mem, L.rho.ravel(), L.u.reshape(3, -1),
                        nx, ny, nz, L.t) if hasattr(L, 'mk') else None
        # models step in THREADS (sampler barrier-allreduces inside)
        forces = [None] * NR
        errs = []

        def worker(r):
            try:
                F = models[r].step(lv[k][r].u, dt=1.0)
                forces[r] = F.astype(cp.float32) if F.dtype != cp.float32 else F
            except Exception as e:                     # pragma: no cover
                errs.append((r, e))
        th = [threading.Thread(target=worker, args=(r,)) for r in range(NR)]
        for t in th: t.start()
        for t in th: t.join(180)
        if errs:
            raise RuntimeError(f"rank ALM step failed: {errs}")
        for r in range(NR):
            lv[k][r].advance(force=forces[r])

    def advance_level(k):
        if k == alm_level:
            alm_kernel_step(k)
        else:
            for r in range(NR):
                lv[k][r].advance()

    def advance_fine(k):
        has_finer = k + 1 < NL
        if has_finer:
            sync(k); save_fprev(k)
        sync(k)
        advance_level(k)
        sync(k - 1)
        for r in range(NR):
            rlc[k - 1][r].c2f(lv[k - 1][r].mem, lv[k][r].mem,
                              lv[k - 1][r].t, lv[k][r].t,
                              is_half_step=True, f_prev_sub_loc=fprev[k - 1][r])
        if has_finer:
            advance_fine(k + 1)
        if has_finer:
            sync(k); save_fprev(k)
        sync(k)
        advance_level(k)
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
        advance_level(0)
        advance_fine(1)

    # ── reference: production advance on the pristine build ──
    for _ in range(NSTEPS):
        mlg.advance()
    ref_f = [mlg.get_level(k).physical_f for k in range(NL)]
    ref_F = mlg.get_level(alm_level).al_model._F_grid

    # ── compare owned assemblies ──
    ok, worst = True, 0.0
    for k in range(NL):
        pieces = [esoteric_gather_std(cp, lv[k][r].mem, lv[k][r].t)
                  [(slice(None),) + parts[k][r].owned_local()]
                  for r in range(NR)]
        df = float(cp.max(cp.abs(cp.concatenate(pieces, axis=1 + AXIS)
                                 - ref_f[k])))
        worst = max(worst, df)
    Fd = cp.concatenate(
        [models[r]._F_grid[(slice(None),) + Lp[r].owned_local()]
         for r in range(NR)], axis=1 + AXIS)
    Frel = float(cp.abs(Fd - ref_F).sum() / (cp.abs(ref_F).sum() + 1e-300))
    print(f"  [{name}] field max|df|={worst:.3e} (gate<1e-4)   "
          f"F_grid rel={Frel:.3e} (gate<1e-5)")
    del mlg, s0, models, lv
    _gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    return worst < 1e-4 and Frel < 1e-5


def main():
    ok = run_case("configs/hpc_bench/bench5_purealm_m3.py", "pure-ALM")
    ok &= run_case("configs/hpc_bench/bench5_archb_m3.py", "archB-straight")
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
