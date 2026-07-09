"""Numerical-equivalence smoke: vectorized dcl_eval vs scalar reference.

Compares the OLD per-marker `lift_curve_slope_batch` (scalar polar_query +
np.polyfit PER marker) against the NEW batched formulation (4 batched C81
lookups + single multi-RHS np.polyfit) on a REAL NASA C81 deck. They must
agree to ~1e-12 (same cubic; batched C81 == scalar per the loader docstring).
"""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.abspath("src"))

from actuator.c81_loader import C81PolarSet, make_c81_polar_query_mach
from actuator.polar_slope import lift_curve_slope_batch

deck = sorted(glob.glob("data/airfoils/hvab_nasa_overflow/*RC4-10*.c81"))[0]
polar = C81PolarSet.from_file(deck)
pq = make_c81_polar_query_mach(polar)
print(f"deck={deck}  supports_batch={getattr(pq,'supports_batch',False)} "
      f"wants_mach={getattr(pq,'wants_mach',False)}")

rng = np.random.default_rng(0)
n = 48
alpha = rng.uniform(-4.0, 12.0, n)          # deg, representative rotor range
Re    = rng.uniform(2e5, 2e6, n)
Mach  = rng.uniform(0.1, 0.65, n)
active = np.ones(n, dtype=bool)
active[:3] = False                          # root cutout (must return 0)

# --- OLD reference: per-marker loop + per-marker polyfit ---
old = lift_curve_slope_batch(pq, alpha, Re, active,
                             multi_airfoil=False, marker_airfoil=None,
                             mach=Mach, delta_deg=1.0)

# --- NEW: batched lookups + single multi-RHS polyfit (mirrors dcl_eval) ---
offs = np.array([-1.5, -0.5, 0.5, 1.5])                       # × δ=1.0 deg
cls = np.stack([pq(alpha + o, Re, mach=Mach)[0] for o in offs])   # (4, n)
xs = offs * (np.pi / 180.0)
coeffs = np.polyfit(xs, cls, 3)
new = np.where(active, coeffs[2], 0.0)

d = np.abs(old - new)
rel = d / (np.abs(old) + 1e-30)
print(f"active markers: {active.sum()}/{n}")
print(f"slope range [1/rad]: old {old[active].min():.3f}..{old[active].max():.3f}")
print(f"max abs diff = {d.max():.3e}   max rel diff = {rel[active].max():.3e}")
print(f"inactive slopes zero: old={np.all(old[~active]==0)} new={np.all(new[~active]==0)}")
ok = d.max() < 1e-10
print("RESULT:", "PASS (numerically equivalent)" if ok else "FAIL")
sys.exit(0 if ok else 1)
