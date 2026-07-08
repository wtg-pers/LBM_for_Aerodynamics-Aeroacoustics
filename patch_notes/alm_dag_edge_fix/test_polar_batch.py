"""Verify the batched polar lookup (_lookup_cl_cd fast path) against the
original per-marker loop — production HVAB nasa_overflow C81 multi-airfoil
setup, Mach-pass active.

Checks:
  1. Bit-identical CL/CD vs the pre-patch per-marker manager dispatch
     (including inactive markers, u_rel<1e-10 markers, clamped α/Mach).
  2. Group/queryfunc caches hit on repeat calls (same objects, same result).
  3. Speed: ≥10× on the 4-blade 192-marker HVAB lookup.

Run:  python patch_notes/alm_dag_edge_fix/test_polar_batch.py
"""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.actuator.actuator_line import ActuatorLineModel          # noqa: E402
from src.actuator.airfoil_data import MultiAirfoilPolarManager    # noqa: E402

DECKS = "/home/wtg/01_python_project/test_0124/data/airfoils/hvab_nasa_overflow"
NAMES = ["RC4-12", "RC4-10", "RC6-08", "RC6-08T"]

mgr = MultiAirfoilPolarManager()
for nm in NAMES:
    mgr.add_airfoil_from_c81(nm, os.path.join(DECKS, f"{nm}_OVERFLOW_SA.c81"),
                             set_default=(nm == "RC4-12"))
unified = mgr.to_unified_query()

# ---- synthetic HVAB-like blade: 48 markers, spanwise airfoil zones ----------
N = 48
rng = np.random.default_rng(42)
marker_airfoil = (["RC4-12"] * 10 + ["RC4-10"] * 14 + ["RC6-08"] * 14
                  + ["RC6-08T"] * 10)
active = np.ones(N, dtype=bool)
active[:2] = False                                   # root cutout
blade = SimpleNamespace(marker_airfoil=marker_airfoil, marker_active=active)

alpha = rng.uniform(-30.0, 30.0, N)                  # includes beyond-stall
alpha[5] = 200.0                                     # α clamp path
Re = rng.uniform(2e5, 2e6, N)
Mach = rng.uniform(0.0, 1.2, N)                      # includes Mach clamp
u_rel = rng.uniform(0.02, 0.12, N)
u_rel[7] = 1e-12                                     # below-threshold marker


def reference_loop(alpha_deg, Re, u_rel, Mach, blade, n):
    """Pre-patch semantics: per-marker unified-query dispatch."""
    CL = np.zeros(n); CD = np.zeros(n)
    for j in range(n):
        if not blade.marker_active[j] or u_rel[j] < 1e-10:
            continue
        CL[j], CD[j] = unified(float(alpha_deg[j]), float(Re[j]),
                               blade.marker_airfoil[j], mach=float(Mach[j]))
    return CL, CD


slf = SimpleNamespace(_multi_airfoil=True, polar_query=unified,
                      _polar_groups={}, _polar_qf_cache={})

CL_ref, CD_ref = reference_loop(alpha, Re, u_rel, Mach, blade, N)
CL_new, CD_new = ActuatorLineModel._lookup_cl_cd(slf, alpha, Re, u_rel, Mach,
                                                 blade, N)
assert np.array_equal(CL_ref, CL_new) and np.array_equal(CD_ref, CD_new), \
    (np.abs(CL_ref - CL_new).max(), np.abs(CD_ref - CD_new).max())
print(f"1. batch == loop bit-identical (N={N}, incl. inactive/clamped/threshold)")

# no-Mach path (Re-only signature)
CL_r2, CD_r2 = ActuatorLineModel._lookup_cl_cd(slf, alpha, Re, u_rel, None,
                                               blade, N)
CLl = np.zeros(N); CDl = np.zeros(N)
for j in range(N):
    if active[j] and u_rel[j] >= 1e-10:
        CLl[j], CDl[j] = unified(float(alpha[j]), float(Re[j]),
                                 blade.marker_airfoil[j])
assert np.array_equal(CLl, CL_r2) and np.array_equal(CDl, CD_r2)
print("1b. no-Mach path bit-identical")

# 2. cache hits (skipped when the fast path is disabled via ALM_POLAR_BATCH=0)
if os.environ.get('ALM_POLAR_BATCH', '1') == '0':
    print("2/3. skipped (ALM_POLAR_BATCH=0 → legacy path; identity checks above "
          "already passed)")
    print("\nALL CHECKS PASSED")
    sys.exit(0)
g1 = slf._polar_groups[id(blade)]
ActuatorLineModel._lookup_cl_cd(slf, alpha, Re, u_rel, Mach, blade, N)
assert slf._polar_groups[id(blade)] is g1, "group cache rebuilt on repeat call"
print(f"2. caches: {len(slf._polar_qf_cache)} query funcs, group cache stable")

# 3. speed on the real per-sub-step workload: 4 blades × (1 + corrected re-query)
REP = 200
t0 = time.perf_counter()
for _ in range(REP):
    for _b in range(4):
        reference_loop(alpha, Re, u_rel, Mach, blade, N)
t_loop = (time.perf_counter() - t0) / REP * 1e3
t0 = time.perf_counter()
for _ in range(REP):
    for _b in range(4):
        ActuatorLineModel._lookup_cl_cd(slf, alpha, Re, u_rel, Mach, blade, N)
t_batch = (time.perf_counter() - t0) / REP * 1e3
print(f"3. 4-blade lookup: loop {t_loop:.2f} ms → batch {t_batch:.2f} ms "
      f"({t_loop / t_batch:.1f}×)")
assert t_loop / t_batch > 10, "expected ≥10× speedup"
print("\nALL CHECKS PASSED")
