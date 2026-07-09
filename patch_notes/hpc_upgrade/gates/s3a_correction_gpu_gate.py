"""S3a gate: correct_noniterative + _triangle xp-agnostic, GPU vs CPU.

Real straight influence_matrix A + xp.linalg.solve. Synthetic analytic polar
(same on both backends) so any Δ is purely the GPU linalg/triangle path.
Must agree <1e-11 (linalg.solve reduction order differs → not 1e-16, but tiny).
"""
import sys, os
import numpy as np
import cupy as cp
sys.path.insert(0, os.path.abspath("src"))
from actuator.smearing_correction import correct_noniterative, influence_matrix

n = 48
rng = np.random.default_rng(3)
r = np.sort(rng.uniform(0.1, 8.0, n))
chord = np.full(n, 0.5)
dr = float(np.mean(np.diff(r)))
eps = np.full(n, 2.0)
u_n = rng.uniform(0.5, 2.0, n)
u_tan = rng.uniform(5.0, 40.0, n)
twist = rng.uniform(2.0, 20.0, n)
Gprev = rng.uniform(-1.0, 1.0, n)
active = np.ones(n, bool); active[:2] = False
A_np = influence_matrix(r, eps, dr)             # real straight A (CPU, numpy)

def make_polar(xp):
    def cl_eval(a_deg):    return 0.11 * a_deg                    # linear C_l(α)
    def dcl_eval(a_deg):   return xp.full_like(a_deg, 0.11 * 180.0 / np.pi)  # [1/rad]
    return cl_eval, dcl_eval

# CPU
cl_c, dcl_c = make_polar(np)
un_c, al_c, G_c, w_c = correct_noniterative(
    r, chord, dr, eps, u_n, u_tan, twist, cl_c, dcl_c, Gprev, A=A_np, active=active)

# GPU (everything cupy)
g = lambda x: cp.asarray(x)
cl_g, dcl_g = make_polar(cp)
un_g, al_g, G_g, w_g = correct_noniterative(
    g(r), g(chord), dr, g(eps), g(u_n), g(u_tan), g(twist),
    cl_g, dcl_g, g(Gprev), A=g(A_np), active=g(active))
assert isinstance(G_g, cp.ndarray)

d = {"u_n_c": np.abs(un_c - cp.asnumpy(un_g)).max(),
     "alpha_c": np.abs(al_c - cp.asnumpy(al_g)).max(),
     "Gamma_new": np.abs(G_c - cp.asnumpy(G_g)).max(),
     "w_corr": np.abs(w_c - cp.asnumpy(w_g)).max()}
for k, v in d.items():
    print(f"  {k:10s} max|Δ| = {v:.3e}")
print(f"|Gamma|~{np.abs(G_c).max():.3e}  |w_corr|~{np.abs(w_c).max():.3e}")
ok = max(d.values()) < 1e-11
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
