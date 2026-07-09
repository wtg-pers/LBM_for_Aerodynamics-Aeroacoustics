"""Phase 1c S1 gate — MLG whole-step CUDA graph replay == eager (bit-identical).

Builds the REAL pure-LBM MLG (configs/hpc_bench/bench5_pure_lbm.py) via the
production setup+initialiser path, then from ONE identical initial state runs
K coarse steps twice:
  (A) eager   (MLG_CUDA_GRAPH off)
  (B) graph   (MLG_CUDA_GRAPH on) — 3 eager warmups, capture, then replay
and asserts every level's f is bit-identical (max|Δ| == 0) and that a graph was
actually captured (not the eager fallback). Also prints a rough wall ratio.

Snapshot/restore of each level's f (not two independent builds) makes the two
runs start from provably identical state, independent of init determinism.

Run from repo root on a CUDA GPU:
    python patch_notes/hpc_upgrade/gates/p1c_s1_graph_gate.py
"""
import os
import sys
import time

# repo root (gates/ -> hpc_upgrade/ -> patch_notes/ -> root) on sys.path so
# `from src.* import ...` resolves regardless of CWD.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

sys.argv = ["p1c_s1_graph_gate", "--config",
            "configs/hpc_bench/bench5_pure_lbm.py",
            "--gpu", "0", "--no-vtk", "--no-force"]
# Ensure the graph env starts OFF; run (B) flips the MLG attribute directly.
os.environ.pop("MLG_CUDA_GRAPH", None)
os.environ.pop("MLG_PROFILE", None)
os.environ.pop("MLG_NVTX", None)

import cupy as cp  # noqa: E402
from src.io.args_parser import parse_args  # noqa: E402
from src.solver.setup import SimulationSetup  # noqa: E402
from src.solver.initializer import SolverInitializer  # noqa: E402

K = 8  # >= WARMUP(3)+capture(1)+>=1 replay to exercise the replay path


def build_mlg():
    args = parse_args()
    setup = SimulationSetup(args)
    setup.start_log_capture()
    mlg = setup.build_simulation()
    SolverInitializer(setup).initialize(mlg, args)
    setup.stop_log_capture()
    return mlg


def snapshot_f(mlg):
    return [lev.f.copy() for lev in mlg._levels]


def restore_f(mlg, snap):
    for lev, f0 in zip(mlg._levels, snap):
        lev.f[...] = f0
        lev.step_count = 0
    mlg._step_count = 0
    mlg._f_prev_initialized = False          # rebuild f_prev from restored f
    # reset graph state machine
    mlg._graph = None
    mlg._graph_stream = None
    mlg._graph_warmup = 0
    mlg._graph_failed = False
    mlg._graph_ok = None


def run_k(mlg, k):
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(k):
        mlg.advance()
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter() - t0


def main():
    print(f"[gate] building pure-LBM MLG (bench5_pure_lbm), K={K} coarse steps")
    mlg = build_mlg()
    nlev = mlg.num_levels
    print(f"[gate] num_levels={nlev}  shapes="
          f"{[tuple(l.domain_shape) for l in mlg._levels]}")

    init = snapshot_f(mlg)

    # (A) eager reference
    mlg._graph_enabled = False
    t_eager = run_k(mlg, K)
    ref = [lev.f.copy() for lev in mlg._levels]

    # (B) graph: identical initial state, capture+replay
    restore_f(mlg, init)
    mlg._graph_enabled = True
    t_graph = run_k(mlg, K)

    # --- bit-identical check across every level ---
    print("\n[gate] per-level max|Δ| (graph replay vs eager):")
    worst = 0.0
    for k in range(nlev):
        d = float(cp.abs(mlg._levels[k].f - ref[k]).max())
        worst = max(worst, d)
        print(f"    L{k}: max|Δ| = {d:.3e}")
    captured = (mlg._graph is not None) and (not mlg._graph_failed)

    print(f"\n[gate] graph captured        : {captured}")
    print(f"[gate] coarse-step counter   : {mlg.step_count} (expect {K})")
    print(f"[gate] wall  eager={t_eager*1e3:8.1f} ms  graph={t_graph*1e3:8.1f} ms"
          f"  ratio={t_eager/t_graph:.2f}x  (tiny grid — indicative only)")

    ok = (worst == 0.0) and captured and (mlg.step_count == K)
    print("\n[gate] VERDICT:", "PASS" if ok else "FAIL")
    if not ok:
        if not captured:
            print("  -> graph was NOT captured (eager fallback); see capture "
                  "FAILED message above for the host-sync/alloc culprit.")
        if worst != 0.0:
            print(f"  -> replay diverged from eager (max|Δ|={worst:.3e}); a "
                  "captured buffer pointer is unstable across steps.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
