"""Phase 1d smoke — bench5_baseline (HVAB-mini, ALM ON) standard vs esoteric.

The real thing at D16: 5-level MLG + cumulant + dyn_smag + eq/sponge BC +
rotating ALM on the finest level. Runs N coarse steps in both modes and
compares the spread rotor force (sum of the ALM F_grid, i.e. total thrust
vector in lu) per coarse step, plus stability and GPU memory (used + pool).

Free-wake/BEM fp sensitivity means bit-parity is NOT expected (repo-established
gate for ALM fp perturbations = CV-band +-3%, see project memory: GPU freewake
5.6e-16 per-call -> chaotic amplification, integrated C_T ~2%). Chaos-aware
gate: median(tail rel diff) < 1e-3 (catches a SYSTEMATIC esoteric bias) AND
max(tail) < 3e-2 (allows bounded chaotic spikes), no NaN.

Run from repo root:  python patch_notes/hpc_upgrade/gates/eso_bench5_alm_smoke.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

NSTEPS = int(os.environ.get("SMOKE_STEPS", "20"))


def run(esoteric: bool):
    import cupy as cp
    os.environ["LBM_ESOTERIC"] = "1" if esoteric else "0"
    for k in ("MLG_CUDA_GRAPH", "MLG_PROFILE", "MLG_NVTX"):
        os.environ.pop(k, None)
    sys.argv = ["eso_bench5_smoke", "--config",
                "configs/hpc_bench/bench5_baseline.py",
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

    # locate the ALM level
    alm_lev = None
    for k in range(mlg.num_levels):
        if getattr(mlg.get_level(k), "al_model", None) is not None:
            alm_lev = mlg.get_level(k)
    assert alm_lev is not None, "no ALM level found in bench5_baseline"
    eso_on = bool(getattr(alm_lev, "_use_esoteric", False))

    F_hist = []
    nan = False
    for _ in range(NSTEPS):
        mlg.advance()
        Fg = alm_lev.al_model._F_grid          # (3, Nx, Ny, Nz), lu
        F = [float(cp.sum(Fg[d])) for d in range(3)]
        F_hist.append(F)
        nan = nan or any(not np.isfinite(v) for v in F)
    cp.cuda.runtime.deviceSynchronize()
    nan = nan or bool(cp.any(~cp.isfinite(alm_lev.rho)))

    pool = cp.get_default_memory_pool()
    used, total = pool.used_bytes(), pool.total_bytes()
    del mlg, s, alm_lev, Fg
    import gc
    gc.collect()                     # break ref cycles before the next build
    pool.free_all_blocks()
    return eso_on, np.array(F_hist), nan, used, total


def main():
    print(f"[eso_bench5_alm_smoke] bench5_baseline (HVAB-mini D16, ALM), {NSTEPS} coarse steps")
    eso0, F_s, nan_s, used_s, tot_s = run(esoteric=False)
    eso1, F_e, nan_e, used_e, tot_e = run(esoteric=True)

    print(f"  std: eso={eso0} NaN={nan_s}  mem used={used_s/1e9:.3f}GB pool={tot_s/1e9:.3f}GB")
    print(f"  eso: eso={eso1} NaN={nan_e}  mem used={used_e/1e9:.3f}GB pool={tot_e/1e9:.3f}GB")

    # total force magnitude trace (thrust builds under the ramp)
    mag_s = np.linalg.norm(F_s, axis=1)
    mag_e = np.linalg.norm(F_e, axis=1)
    print("  step:  |F|_std      |F|_eso      rel_diff")
    rels = []
    for k in range(NSTEPS):
        denom = max(abs(mag_s[k]), 1e-30)
        rel = abs(mag_e[k] - mag_s[k]) / denom
        rels.append(rel)
        if k % 4 == 0 or k == NSTEPS - 1:
            print(f"  {k:4d}:  {mag_s[k]:.6e}  {mag_e[k]:.6e}  {rel:.3e}")
    # settle: ignore the first quarter (ramp start, tiny denominators)
    tail = rels[NSTEPS // 4:]
    rel_max = max(tail) if tail else 1.0
    rel_med = float(np.median(tail)) if tail else 1.0
    print(f"  tail rel diff: median={rel_med:.3e} (gate < 1e-3, systematic bias)  "
          f"max={rel_max:.3e} (gate < 3e-2, chaotic spikes / repo CV-band)")

    ok = ((not eso0) and eso1 and (not nan_s) and (not nan_e)
          and rel_med < 1e-3 and rel_max < 3e-2)
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
