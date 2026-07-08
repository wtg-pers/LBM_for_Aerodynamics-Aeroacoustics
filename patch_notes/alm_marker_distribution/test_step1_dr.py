"""Step 1 verification: per-marker dr in influence_matrix + _viscous_core kernel.
Checks: (a) scalar path byte-identical to original formula, (b) array-dr matches a
brute-force reference (axis correctness), (c) uniform array ~= scalar."""
import sys
import numpy as np
sys.path.insert(0, "/home/wtg/01_python_project/test_0124/src")
from actuator.smearing_correction import influence_matrix, _gradient_matrix, _INV4PI

rng = np.random.default_rng(0)
n = 20
r = np.cumsum(0.5 + rng.random(n))          # non-uniform, monotonic radii
eps = 0.8 + 0.3 * rng.random(n)
dr_scalar = 0.37
dr_arr_uniform = np.full(n, dr_scalar)
dr_arr_var = 0.2 + 0.6 * rng.random(n)      # non-uniform per-marker width

def _kmat(r, eps):
    d = r[:, None] - r[None, :]
    far = np.abs(d) > 1e-9
    safe = np.where(far, d, 1.0)
    return np.where(far, np.exp(-(d / eps[None, :]) ** 2) / safe, 0.0)

# ---- reference: A_ik = -(1/4π) Σ_m Kmat_im · Δr_m · G_mk  (explicit loop) ----
def A_reference(r, eps, dr):
    K = _kmat(r, eps); G = _gradient_matrix(r); n = len(r)
    drv = np.full(n, dr) if np.ndim(dr) == 0 else np.asarray(dr)
    A = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            A[i, k] = -_INV4PI * np.sum(K[i, :] * drv * G[:, k])
    return A

# (a) scalar path == original one-liner (byte-identical)
orig_scalar = (-_INV4PI * dr_scalar) * (_kmat(r, eps) @ _gradient_matrix(r))
A_s = influence_matrix(r, eps, dr_scalar)
print("(a) scalar byte-identical to original:", np.array_equal(A_s, orig_scalar))

# (b) array-dr axis correctness vs brute-force reference
A_v = influence_matrix(r, eps, dr_arr_var)
ref_v = A_reference(r, eps, dr_arr_var)
print("(b) array-dr matches brute-force ref: maxrelerr=%.2e" %
      (np.max(np.abs(A_v - ref_v)) / (np.max(np.abs(ref_v)) + 1e-30)))

# (c) uniform array ~= scalar (consistency; differs only by fp order)
A_u = influence_matrix(r, eps, dr_arr_uniform)
print("(c) uniform array vs scalar: maxrelerr=%.2e" %
      (np.max(np.abs(A_u - A_s)) / (np.max(np.abs(A_s)) + 1e-30)))

# (d) scalar reference cross-check (sanity that ref==scalar too)
print("(d) ref(scalar)==influence(scalar): maxrelerr=%.2e" %
      (np.max(np.abs(A_reference(r, eps, dr_scalar) - A_s)) /
       (np.max(np.abs(A_s)) + 1e-30)))

# ---- _viscous_core kernel (replicate the in-method math, inviscid target) ----
def viscous_core(r, eps, Gamma, dr, use_opt=False, eps_opt=None):
    N = len(r); gradG = np.gradient(Gamma, r); inv4pi = 1.0 / (4 * np.pi)
    dr_scal = (np.ndim(dr) == 0); w = np.zeros(N)
    for i in range(N):
        d = r[i] - r; far = np.abs(d) > 1e-9; safe = np.where(far, d, 1.0)
        K = np.exp(-(d / eps) ** 2)
        kernel = np.where(far, K / safe, 0.0)
        w[i] = (-inv4pi * np.sum(gradG * kernel) * dr) if dr_scal \
            else (-inv4pi * np.sum(gradG * kernel * dr))
    return w

# Invariant (memory): influence_matrix @ Γ == viscous_core(inviscid)  — both dr modes
Gamma = np.sin(r / r[-1] * np.pi) * (0.5 + rng.random(n))   # arbitrary Γ(r)
for tag, dr in (("scalar", dr_scalar), ("array-var", dr_arr_var)):
    lhs = influence_matrix(r, eps, dr) @ Gamma
    rhs = viscous_core(r, eps, Gamma, dr)
    print(f"(e/{tag}) influence@Γ == viscous_core: maxabs=%.2e" %
          np.max(np.abs(lhs - rhs)))
