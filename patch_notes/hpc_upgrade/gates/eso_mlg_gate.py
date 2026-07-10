"""Phase 1c gate — MLG machinery with esoteric levels (bench5 topology).

Builds bench5_pure_lbm (HVAB-mini: 5-level MLG, cumulant, dyn_smag, eq+sponge
BCs, ALM off, fluid at rest) twice: standard vs LBM_ESOTERIC=1. Runs N coarse
steps and compares every level's (rho, u) plus mass conservation.

At rest the fields are direction-dependent constants (f_q = w_q rho), so any
slot/parity/BC-mapping error corrupts them immediately; spatial-offset errors
are covered by eso_gather_scatter_gate (dynamic reference). Full dynamics with
ALM forcing is exercised in Phase (d)'s bench5_baseline smoke.

Run from repo root:  python patch_notes/hpc_upgrade/gates/eso_mlg_gate.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

NSTEPS = 3


def build_and_run(esoteric: bool):
    import cupy as cp
    os.environ["LBM_ESOTERIC"] = "1" if esoteric else "0"
    for k in ("MLG_CUDA_GRAPH", "MLG_PROFILE", "MLG_NVTX"):
        os.environ.pop(k, None)
    sys.argv = ["eso_mlg_gate", "--config", "configs/hpc_bench/bench5_pure_lbm.py",
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

    eso_flags = [bool(getattr(mlg.get_level(k), "_use_esoteric", False))
                 for k in range(mlg.num_levels)]
    m0 = sum(float(cp.sum(mlg.get_level(k).f)) for k in range(mlg.num_levels))
    for _ in range(NSTEPS):
        mlg.advance()
    cp.cuda.runtime.deviceSynchronize()
    m1 = sum(float(cp.sum(mlg.get_level(k).f)) for k in range(mlg.num_levels))

    fields = []
    nan = False
    for k in range(mlg.num_levels):
        lev = mlg.get_level(k)
        nan = nan or bool(cp.any(~cp.isfinite(lev.rho))) or bool(cp.any(~cp.isfinite(lev.u)))
        fields.append((cp.asnumpy(lev.rho), cp.asnumpy(lev.u)))
    n_levels = mlg.num_levels

    del mlg, s
    cp.get_default_memory_pool().free_all_blocks()
    return eso_flags, (m1 - m0) / m0, nan, fields, n_levels


def main():
    print(f"[eso_mlg_gate] bench5_pure_lbm 5-level MLG, {NSTEPS} coarse steps")
    flags_s, drift_s, nan_s, fields_s, nl = build_and_run(esoteric=False)
    flags_e, drift_e, nan_e, fields_e, _ = build_and_run(esoteric=True)

    print(f"  std: eso_levels={flags_s} mass_drift={drift_s:+.3e} NaN={nan_s}")
    print(f"  eso: eso_levels={flags_e} mass_drift={drift_e:+.3e} NaN={nan_e}")

    ok = (not any(flags_s)) and all(flags_e) and (not nan_s) and (not nan_e)
    dmax = 0.0
    for k in range(nl):
        dr = float(np.max(np.abs(fields_s[k][0] - fields_e[k][0])))
        du = float(np.max(np.abs(fields_s[k][1] - fields_e[k][1])))
        dmax = max(dmax, dr, du)
        print(f"  L{k}: max|drho|={dr:.3e}  max|du|={du:.3e}")
    ok = ok and dmax < 1e-5 and abs(drift_e) < 1e-5
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
