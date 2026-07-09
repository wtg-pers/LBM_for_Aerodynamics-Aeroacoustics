"""Profiling driver — build bench5_pure_lbm MLG, warm up, profile N coarse steps.

Wraps the profiled region in cudaProfilerStart/Stop (via cupy) so both
  nsys  --capture-range=cudaProfilerApi
  ncu   --profile-from-start off
capture ONLY the steady-state coarse steps and exclude one-time setup kernels.
Pure-LBM (ALM off) isolates the LBM bottleneck (C2F coupling per 09/10).

Run from repo root. Local WSL2 blocks nsys CUPTI + ncu HW counters
(ERR_NVGPUCTRPERM) — run on the cluster (native Linux 4090). See note 10 for
the full protocol.

    PROF_STEPS=10 nsys profile -t cuda --capture-range=cudaProfilerApi \
        --capture-range-end=stop -o bench5_purelbm_prof --force-overwrite=true \
        python patch_notes/hpc_upgrade/gates/nsys_purelbm_driver.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.argv = ["nsys_driver", "--config", "configs/hpc_bench/bench5_pure_lbm.py",
            "--gpu", "0", "--no-vtk", "--no-force"]
for _k in ("MLG_CUDA_GRAPH", "MLG_PROFILE", "MLG_NVTX"):
    os.environ.pop(_k, None)

import cupy as cp
from cupy.cuda import profiler
from src.io.args_parser import parse_args
from src.solver.setup import SimulationSetup
from src.solver.initializer import SolverInitializer

N = int(os.environ.get("PROF_STEPS", "10"))
a = parse_args()
s = SimulationSetup(a); s.start_log_capture()
mlg = s.build_simulation(); SolverInitializer(s).initialize(mlg, a); s.stop_log_capture()
mlg._graph_enabled = False

for _ in range(3):            # warmup: JIT kernels, allocate lazy buffers
    mlg.advance()
cp.cuda.runtime.deviceSynchronize()

profiler.start()
for _ in range(N):            # profiled region (setup excluded)
    mlg.advance()
cp.cuda.runtime.deviceSynchronize()
profiler.stop()
print(f"[nsys_driver] profiled {N} coarse steps")
