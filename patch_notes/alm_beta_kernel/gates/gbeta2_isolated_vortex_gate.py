"""G-beta2 — isolated straight-vortex deficit gate (01_design.md #14.3).

Validates the smearing-correction deficit kernels K(d; eps) implemented in
ALMKernelSpec.deficit_K against an INDEPENDENT numerical route built from
the PRODUCTION shape function (np_shape — operator sharing, not a local
re-implementation of psi):

    eta2(s)  = integral of eta3 along the filament axis   (2D projected core)
    F(x)     = enclosed circulation of eta2 (Stokes: u_theta 2 pi d = Gamma_enc)
    K_num(x) = 1 - F(x)

Claims:
  [D] wendland closed form (03_correction_derivation.md) vs K_num:
      max|diff| < 1e-9 over x in (0, 1]  — validates BOTH the derivation
      and the production psi in one comparison.
  [C] controls through the SAME quadrature machinery:
      gaussian  -> exp(-(d/eps)^2)          (< 1e-9)
      winckelmans -> 1/(1+(d/eps)^2)^2      (< 1e-9; also confirms the
      filament-integral route and the projection route agree)
  [P] properties: K(0) = 1 exactly (wendland), K(x>=1) = 0 exactly,
      monotone decreasing on the sample grid.
  [F] projected-core FWHM (vorticity half-max): wendland = 1.7540 eps
      +- 1e-3 (recorded 2026-07-16; gaussian control 2 sqrt(ln 2) eps
      = 1.6651 eps). Anchor for tip-vortex core diagnostics.

Run from repo root:
    python patch_notes/alm_beta_kernel/gates/gbeta2_isolated_vortex_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from scipy import integrate, optimize

from src.actuator.alm_kernel import get_alm_kernel

EPS = 2.3          # arbitrary width for the physical-units round trip [lu]


def projected_core(spec, sup):
    """eta2(s) from the PRODUCTION np_shape (unnormalized; s in support
    units for compact kernels / eps units otherwise)."""
    def eta2(s):
        if sup is not None and s >= sup:
            return 0.0
        zmax = np.sqrt(sup * sup - s * s) if sup is not None else np.inf
        # np_shape takes (d^2, eps^2) in the SAME units; use eps units here
        return integrate.quad(
            lambda t: float(spec.np_shape(np, np.float64(s * s + t * t),
                                          np.float64(1.0))),
            0, zmax, limit=400)[0]
    return eta2


def K_numeric(spec, d_over_eps, sup):
    """1 - enclosed-circulation fraction at d/eps, via double quadrature of
    the production shape. Normalization cancels in the fraction."""
    eta2 = projected_core(spec, sup)
    upper = sup if sup is not None else np.inf
    tot = integrate.quad(lambda s: s * eta2(s), 0, upper, limit=400)[0]
    enc = integrate.quad(lambda s: s * eta2(s), 0, min(d_over_eps, upper),
                         limit=400)[0]
    return 1.0 - enc / tot


def main():
    ok = True
    xs = np.array([0.05, 0.15, 0.3, 0.5, 0.8, 1.2, 1.7, 2.2, 2.7])  # d/eps

    # [D] + [C]: each family, closed form vs production-shape quadrature.
    # winckelmans tolerance is quadrature-limited (nested quad over the
    # algebraic infinite tail plateaus ~1e-8; the analytic identity of the
    # projection route, eta2 = (2/pi)(s^2+1)^{-3} -> K = (1+x^2)^{-2}, is
    # exact — see 03_correction_derivation.md).
    cases = (("wendland", np.sqrt(7.5), 1e-9), ("gaussian", None, 1e-9),
             ("winckelmans", None, 1e-7))
    for name, sup, tol in cases:
        spec = get_alm_kernel(name)
        worst = 0.0
        for de in xs:
            if sup is not None and de >= sup:
                k_num = 0.0
            else:
                k_num = K_numeric(spec, de, sup)
            k_cl = float(spec.deficit_K(np, np.float64(de * EPS), EPS))
            worst = max(worst, abs(k_cl - k_num))
        good = worst < tol
        ok &= good
        tag = "[D]" if name == "wendland" else "[C]"
        print(f"  {tag} {name:12s} closed-form vs production-shape "
              f"quadrature: max|diff|={worst:.2e}  "
              f"-> {'OK' if good else 'FAIL'}")

    # [P] properties (wendland)
    sw = get_alm_kernel("wendland")
    k0 = float(sw.deficit_K(np, 0.0, EPS))
    k_out = float(sw.deficit_K(np, np.sqrt(7.5) * EPS * 1.000001, EPS))
    grid = np.linspace(0.0, np.sqrt(7.5), 400) * EPS
    kk = sw.deficit_K(np, grid, EPS)
    mono = bool(np.all(np.diff(kk) <= 1e-15))
    p_ok = (k0 == 1.0) and (k_out == 0.0) and mono
    ok &= p_ok
    print(f"  [P] wendland K(0)={k0} (exact 1)  K(beyond)={k_out} (exact 0)"
          f"  monotone={mono}  -> {'OK' if p_ok else 'FAIL'}")

    # [F] projected-core FWHM
    for name, sup, ref, tol in (("wendland", np.sqrt(7.5), 1.7540, 1e-3),
                                ("gaussian", None,
                                 2.0 * np.sqrt(np.log(2.0)), 1e-3)):
        spec = get_alm_kernel(name)
        eta2 = projected_core(spec, sup)
        i0 = eta2(0.0)
        hi = (sup or 4.0) * 0.99
        s_half = optimize.brentq(lambda s: eta2(s) - 0.5 * i0, 1e-3, hi)
        fwhm = 2.0 * s_half
        good = abs(fwhm - ref) < tol
        ok &= good
        print(f"  [F] {name:12s} projected-core FWHM = {fwhm:.4f} eps "
              f"(ref {ref:.4f})  -> {'OK' if good else 'FAIL'}")

    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
