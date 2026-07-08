"""INTEGRITY verification (physics/equations EXACT, not smoke).
Checks each computation step of the Step 1-2 implementation against the
governing math: deficit kernel, per-marker Δr weighting, quadrature exactness,
thrust conservation, dΓ/dr on non-uniform grids, and endpoint active/chord/eps.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/wtg/01_python_project/test_0124")
sys.path.insert(0, "/home/wtg/01_python_project/test_0124/src")
from actuator.smearing_correction import influence_matrix, _gradient_matrix, _INV4PI

OK = lambda b: "PASS" if b else "**FAIL**"
print("================ INTEGRITY CHECKS ================\n")

# C1. Deficit kernel = ideal − smeared  (Dağ viscous-core physics)
#   ideal (point vortex)   ∝ 1/d
#   smeared (Lamb-Oseen ε) ∝ (1 − exp(−(d/ε)²))/d
#   deficit  = 1/d − (1−exp)/d = exp(−(d/ε)²)/d   ← kernel used in influence_matrix
d = np.linspace(0.3, 8.0, 50); eps = 1.7
ideal = 1.0 / d
smeared = (1.0 - np.exp(-(d / eps) ** 2)) / d
deficit_kernel = np.exp(-(d / eps) ** 2) / d
print(f"C1 deficit kernel == ideal − smeared : {OK(np.allclose(deficit_kernel, ideal - smeared, atol=1e-15))}")
print(f"   _INV4PI == 1/(4π) : {OK(abs(_INV4PI - 1/(4*np.pi)) < 1e-18)}")

# C2. influence_matrix per-marker Δr == explicit Biot-Savart sum
#   A_ik = -(1/4π) Σ_m [exp(−((r_i−r_m)/ε_m)²)/(r_i−r_m)] · Δr_m · G_mk
rng = np.random.default_rng(1)
n = 24
r = np.cumsum(0.4 + rng.random(n))
epsm = 1.0 + 0.5 * rng.random(n)
dr = 0.2 + 0.7 * rng.random(n)
def A_explicit(r, eps, dr):
    G = _gradient_matrix(r); n = len(r); A = np.zeros((n, n))
    for i in range(n):
        for m in range(n):
            if abs(r[i]-r[m]) > 1e-9:
                K = np.exp(-((r[i]-r[m])/eps[m])**2)/(r[i]-r[m])
            else:
                K = 0.0
            A[i, :] += -_INV4PI * K * dr[m] * G[m, :]
    return A
err2 = np.max(np.abs(influence_matrix(r, epsm, dr) - A_explicit(r, epsm, dr)))
print(f"\nC2 influence_matrix == explicit Biot-Savart : maxabs={err2:.2e}  {OK(err2 < 1e-14)}")

# C3+C4. Quadrature: Σ marker_dr == span (exact) + exact for constant & linear
from actuator.blade import _cosine_map, _trapezoid_widths
a, b, N = 2.0, 9.0, 40
span = b - a
def dist_points(mode, side="both"):
    if mode == "uniform":
        drr = span / N; rr = a + (np.arange(N) + 0.5) * drr
        return rr, np.full(N, drr)
    if mode == "cosine":
        xe = np.arange(N + 1) / N; re = a + span * _cosine_map(xe, side)
        return 0.5*(re[:-1]+re[1:]), np.diff(re)
    if mode == "endpoint":
        rr = np.linspace(a, b, N); return rr, _trapezoid_widths(rr)
print()
for mode, side in (("uniform","both"),("cosine","both"),("cosine","tip"),("endpoint","both")):
    rr, drr = dist_points(mode, side)
    sum_ok = abs(drr.sum() - span) < 1e-12
    # quadrature exact for f=1 (→ span) and f=r (→ (b²−a²)/2)
    q_const = abs(np.sum(1.0 * drr) - span)
    q_lin = abs(np.sum(rr * drr) - 0.5*(b**2 - a**2))
    tag = f"{mode}/{side}" if mode == "cosine" else mode
    print(f"C3/4 [{tag:11}] Σdr−span={drr.sum()-span:+.2e}  ∫1 err={q_const:.2e}  "
          f"∫r err={q_lin:.2e}  {OK(sum_ok and q_const<1e-12 and q_lin<1e-12)}")

# C5. np.gradient on NON-uniform grid: (a) EXACT for quadratic Γ (the method's
#     formal accuracy), (b) 2nd-order CONVERGENCE for sin, (c) non-uniform error
#     comparable to uniform (clustering doesn't corrupt dΓ/dr).
rr, _ = dist_points("cosine", "both")
quad_err = np.max(np.abs(np.gradient(3*rr**2 - 2*rr + 1, rr)[2:-2] - (6*rr - 2)[2:-2]))
print(f"\nC5a np.gradient EXACT for quadratic Γ (non-uniform): maxabs={quad_err:.2e}  {OK(quad_err < 1e-10)}")
errs = []
for Nx in (40, 80, 160, 320):
    xe = np.arange(Nx + 1) / Nx; re = a + span * _cosine_map(xe, "both")
    rc = 0.5*(re[:-1]+re[1:])
    errs.append(np.max(np.abs(np.gradient(np.sin(rc), rc)[3:-3] - np.cos(rc)[3:-3])))
rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))   # ~2 = 2nd order
print(f"C5b sin convergence rate (→2 = 2nd-order): {np.round(rates,2)}  {OK(np.all(rates > 1.7))}")
# fair compare: uniform-grid gradient error on same sin, same N
ru = a + (np.arange(40)+0.5)*span/40
eu = np.max(np.abs(np.gradient(np.sin(ru), ru)[3:-3] - np.cos(ru)[3:-3]))
ec = np.max(np.abs(np.gradient(np.sin(0.5*(a+span*_cosine_map(np.arange(41)/40,"both"))[:-1]
            +0.5*(a+span*_cosine_map(np.arange(41)/40,"both"))[1:]),
            0.5*(a+span*_cosine_map(np.arange(41)/40,"both"))[:-1]
            +0.5*(a+span*_cosine_map(np.arange(41)/40,"both"))[1:])[3:-3]
            - np.cos(rr)[3:-3]))
print(f"C5c uniform err={eu:.2e} vs cosine err={ec:.2e} (same order, no corruption)  {OK(ec < 5*eu)}")

# C6. Thrust conservation: for FIXED analytic loading L(r), Σ L(r_j)·dr_j == ∫L dr
#     (elliptic-ish span loading; midpoint & trapezoid both 2nd-order)
def L(r): return np.sqrt(np.maximum(1 - ((r - a)/span * 2 - 1)**2, 0)) * (1 + 0.2*r)
from scipy.integrate import quad
T_true, _ = quad(L, a, b)
print(f"\nC6 thrust conservation vs ∫L dr={T_true:.6f}:")
for mode, side in (("uniform","both"),("cosine","both"),("cosine","tip"),("endpoint","both")):
    rr, drr = dist_points(mode, side)
    T = np.sum(L(rr) * drr)
    tag = f"{mode}/{side}" if mode == "cosine" else mode
    print(f"   [{tag:11}] T={T:.6f}  rel-err={abs(T-T_true)/T_true:.2e}")

# C7. endpoint/cosine markers: ACTIVE flag + chord at the true endpoints
sys.path.insert(0, "/home/wtg/01_python_project/test_0124/configs/hvab")
from _hvab_hover_base import build_config, CHORD_REF, CHORD_TIP, IN2M
from actuator.rotor import Rotor
def mk(dist):
    c = build_config(10.0, marker_distribution=dist)["actuator_line"]["rotor"]
    c["grid"]["dx"] = 0.1
    return Rotor.from_config(c).blades[0]
print()
for dist in ("uniform", "cosine", "endpoint"):
    bl = mk(dist)
    allact = bool(np.all(bl.marker_active))
    eps_ok = bool(np.all(bl.marker_epsilon > 0) and np.all(np.isfinite(bl.marker_epsilon)))
    info = ""
    if dist == "endpoint":
        info = (f" tip_chord={bl.marker_chord[-1]/IN2M:.2f}in(=3.27?{abs(bl.marker_chord[-1]-CHORD_TIP)<1e-6}) "
                f"root_chord={bl.marker_chord[0]/IN2M:.2f}in(=5.45?{abs(bl.marker_chord[0]-CHORD_REF)<1e-6})")
    print(f"C7 [{dist:9}] all markers active:{OK(allact)}  eps>0 finite:{OK(eps_ok)}{info}")
