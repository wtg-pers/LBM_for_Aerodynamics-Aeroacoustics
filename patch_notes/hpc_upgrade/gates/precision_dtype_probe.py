"""Phase 0 audit probe (patch_notes/hpc_upgrade/13) — measure the ACTUAL runtime
dtype of every per-level PDF/macroscopic buffer, plus real bytes/node.

Uses bench5_pure_lbm (HVAB base config -> precision='float32', D=16, pure-LBM)
as a proxy for the HVAB family dtype flow. Runs a couple of advances to trigger
the lazy allocation of f_post / f_prev, then inspects.

Resolves patch13 §1 static ambiguity: is the HVAB-family run actually f32 or f64?

Run from repo root:  python patch_notes/hpc_upgrade/gates/precision_dtype_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp
from src.io.args_parser import parse_args
from src.solver.setup import SimulationSetup
from src.solver.initializer import SolverInitializer


def build(precision_override=None):
    sys.argv = ["probe", "--config", "configs/hpc_bench/bench5_pure_lbm.py",
                "--gpu", "0", "--no-vtk", "--no-force"]
    a = parse_args()
    s = SimulationSetup(a)
    if precision_override is not None:
        s.sim_params["precision"] = precision_override
    s.start_log_capture()
    mlg = s.build_simulation()
    SolverInitializer(s).initialize(mlg, a)
    s.stop_log_capture()
    mlg._graph_enabled = False
    for _ in range(2):        # trigger lazy f_post / f_prev allocation
        mlg.advance()
    cp.cuda.runtime.deviceSynchronize()
    return s, mlg


def dt(x):
    return None if x is None else str(getattr(x, "dtype", "?"))


def report(tag, s, mlg):
    print(f"\n================ {tag}  (config precision -> compute_dtype={np.dtype(s.compute_dtype).name}) ================")
    levels = mlg._levels
    fprev = getattr(mlg, "_f_prev", [None] * len(levels))
    total_nodes = 0
    for k, lev in enumerate(levels):
        n = 1
        for d in lev.domain_shape:
            n *= d
        total_nodes += n
        print(f"  L{k}: nodes={n:>12,}  f={dt(lev.f):8}  f_post={dt(lev.f_post):8}  "
              f"rho={dt(lev.rho):8}  u={dt(lev.u):8}  f_prev={dt(fprev[k] if k < len(fprev) else None)}")
    used = cp.get_default_memory_pool().used_bytes()
    print(f"  --- total nodes={total_nodes:,}   GPU mempool used={used/1e9:.3f} GB   "
          f"=> {used/max(total_nodes,1):.1f} B/node (all buffers+overhead)")
    return total_nodes, used


print("### Phase 0 dtype probe — bench5_pure_lbm (HVAB base, D=16, pure-LBM) ###")

s1, mlg1 = build(precision_override=None)          # native (HVAB base = float32)
n1, u1 = report("NATIVE (as HVAB base ships)", s1, mlg1)
del mlg1, s1
cp.get_default_memory_pool().free_all_blocks()

s2, mlg2 = build(precision_override="float64")     # forced DP
n2, u2 = report("FORCED precision=float64", s2, mlg2)

print("\n### Verdict ###")
print(f"  native B/node = {u1/max(n1,1):.1f} ; float64 B/node = {u2/max(n2,1):.1f} ; ratio = {u2/max(u1,1):.2f}x")
print("  If native f==float32 -> HVAB already stores f32; precision is NOT a further")
print("  memory lever (DPSP/SP == same storage). If native==float64 -> precision halving applies.")
