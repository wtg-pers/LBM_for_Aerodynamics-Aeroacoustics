"""S2 Part 1 gate: freewake_influence xp-agnostic (cupy) vs numpy vs reference.

Synthetic helical rotor wake (nc=48, ne=49, L=50 → ns≈2450, matches steady
bench5 ns_kept≈2501). Checks:
  (A) vec-numpy (no prune) vs reference triple-loop  → exact algorithm cross-check
  (B) vec-cupy  (prune)    vs vec-numpy (prune)      → GPU == CPU (the S2 claim)
  (C) vec-numpy (prune)    vs vec-numpy (no prune)   → prune numerically identical
Plus CPU vs GPU wall for the Biot-Savart build.
"""
import sys, os, time
import numpy as np
import cupy as cp
sys.path.insert(0, os.path.abspath("src"))

from actuator.smearing_correction import (
    freewake_influence, _freewake_influence_loop)

# --- synthetic helical wake -------------------------------------------------
ne, nc, L = 49, 48, 50
r_edge = np.linspace(0.1, 8.0, ne)              # trailed-edge radii [lu]
r_ctrl = 0.5 * (r_edge[:-1] + r_edge[1:])       # nc=48 control (marker) radii
dpsi, dz = 0.20, 0.15                           # per-step rotation [rad], descent [lu]

def ring(a):
    c, s = np.cos(-a * dpsi), np.sin(-a * dpsi)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = np.stack([r_edge, np.zeros(ne), np.zeros(ne)], axis=1)  # (ne,3) along x
    pts = pts @ Rz.T
    pts[:, 2] -= a * dz
    return pts

rings_np = [ring(a) for a in range(L)]          # newest-first
eps_np = np.full(ne, 2.0)
ctrl_np = np.stack([r_ctrl, np.zeros(nc), np.full(nc, 0.05)], axis=1)  # blade, tiny z-offset
axis = np.array([0.0, 0.0, 1.0])

# (A) reference (no prune) vs vec-numpy (no prune)
B_ref = _freewake_influence_loop(ctrl_np, rings_np, eps_np, axis)
B_np_full = freewake_influence(ctrl_np, rings_np, eps_np, axis, prune_factor=None)
dA = np.abs(B_ref - B_np_full).max()

# (C) prune vs no-prune (numpy)
B_np_prune = freewake_influence(ctrl_np, rings_np, eps_np, axis, prune_factor=6.0)
dC = np.abs(B_np_prune - B_np_full).max()

# (B) vec-cupy (prune) vs vec-numpy (prune)
rings_cp = [cp.asarray(r) for r in rings_np]
B_cp = freewake_influence(cp.asarray(ctrl_np), rings_cp, cp.asarray(eps_np),
                          cp.asarray(axis), prune_factor=6.0)
assert isinstance(B_cp, cp.ndarray), "cupy path did not return cupy array"
dB = np.abs(B_np_prune - cp.asnumpy(B_cp)).max()

print(f"wake: nc={nc} ne={ne} L={L}  |B|~{np.abs(B_ref).max():.3e}")
print(f"(A) vec-numpy(no-prune) vs reference loop : max|Δ| = {dA:.3e}")
print(f"(C) numpy prune vs no-prune               : max|Δ| = {dC:.3e}")
print(f"(B) vec-CUPY vs vec-numpy (both prune)    : max|Δ| = {dB:.3e}  ★GPU==CPU")

# --- timing (steady-state-ish size) ---
for _ in range(3):
    freewake_influence(ctrl_np, rings_np, eps_np, axis)              # warm CPU
    cp.cuda.runtime.deviceSynchronize()
t0 = time.perf_counter()
for _ in range(20): freewake_influence(ctrl_np, rings_np, eps_np, axis)
t_np = (time.perf_counter() - t0) / 20 * 1e3
for _ in range(3):
    freewake_influence(cp.asarray(ctrl_np), rings_cp, cp.asarray(eps_np), cp.asarray(axis))
cp.cuda.runtime.deviceSynchronize()
t0 = time.perf_counter()
for _ in range(20):
    freewake_influence(cp.asarray(ctrl_np), rings_cp, cp.asarray(eps_np), cp.asarray(axis))
cp.cuda.runtime.deviceSynchronize()
t_cp = (time.perf_counter() - t0) / 20 * 1e3
print(f"freewake_influence wall: CPU {t_np:.2f} ms   GPU {t_cp:.2f} ms   speedup {t_np/t_cp:.1f}x")

ok = max(dA, dB, dC) < 1e-11
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
