"""Stage 1 gate: GPU C81 LUT + closed-form dcl vs CPU reference.

(1) _C81Table GPU bilinear (cupy) vs CPU RegularGridInterpolator  → <1e-12
(2) _C81Table numpy bilinear vs RegularGridInterpolator            → <1e-12
(3) cubic_slope_4pt (closed-form) vs np.polyfit                    → <1e-12
On the real NASA RC4-10 deck. Small arrays → GPU-safe (no large grid).
"""
import sys, os, glob
import numpy as np
import cupy as cp
sys.path.insert(0, os.path.abspath("src"))

from actuator.c81_loader import C81PolarSet, make_c81_polar_query_mach
from actuator.polar_slope import cubic_slope_4pt, lift_curve_slope_batch

deck = sorted(glob.glob("data/airfoils/hvab_nasa_overflow/*RC4-10*.c81"))[0]
polar = C81PolarSet.from_file(deck)
tab = polar._cl                     # a _C81Table
print(f"deck={os.path.basename(deck)}  aoa[{tab.aoa.shape[0]}] mach[{tab.mach.shape[0]}]")

rng = np.random.default_rng(1)
N = 4 * 48                          # all blades × markers
alpha = rng.uniform(-8, 16, N)
mach  = rng.uniform(0.05, 0.70, N)

# (0) CPU reference (RegularGridInterpolator)
ref = tab(alpha, mach)                                  # numpy → _interp path

# (1) GPU bilinear
gpu = tab(cp.asarray(alpha), cp.asarray(mach))          # cupy → _bilinear
assert isinstance(gpu, cp.ndarray), "GPU path did not return cupy array"
gpu_h = cp.asnumpy(gpu)
d1 = np.abs(ref - gpu_h).max()

# (2) numpy bilinear (xp=np, bypass _interp) — isolates the bilinear math
np_bil = tab._bilinear(np.clip(alpha, tab.aoa[0], tab.aoa[-1]),
                       np.clip(mach, tab.mach[0], tab.mach[-1]), np)
d2 = np.abs(ref - np_bil).max()

# (3) closed-form dcl vs np.polyfit (on real deck C_l samples)
pq = make_c81_polar_query_mach(polar)
Re = rng.uniform(2e5, 2e6, N)
offs = np.array([-1.5, -0.5, 0.5, 1.5])
cls = np.stack([pq(alpha + o, Re, mach=mach)[0] for o in offs])   # (4, N)
xs = offs * (np.pi / 180.0)
slope_polyfit = np.polyfit(xs, cls, 3)[2]
slope_closed  = cubic_slope_4pt(cls, delta_deg=1.0, xp=np)
d3 = np.abs(slope_polyfit - slope_closed).max()
# closed-form on GPU too
slope_gpu = cubic_slope_4pt(cp.asarray(cls), delta_deg=1.0, xp=cp)
d3g = np.abs(slope_polyfit - cp.asnumpy(slope_gpu)).max()

print(f"(1) GPU bilinear   vs RegularGridInterp : max|Δ| = {d1:.3e}")
print(f"(2) numpy bilinear vs RegularGridInterp : max|Δ| = {d2:.3e}")
print(f"(3) closed-form dcl vs np.polyfit (CPU) : max|Δ| = {d3:.3e}")
print(f"(3) closed-form dcl vs np.polyfit (GPU) : max|Δ| = {d3g:.3e}")
ok = max(d1, d2, d3, d3g) < 1e-12
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
