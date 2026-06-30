"""
Standalone hover BEMT (blade-element momentum theory) — an INDEPENDENT reference
for ALM verification.

Uses the SAME airfoil deck (C81PolarSet) as the ALM but a 1-D momentum inflow
(with Prandtl tip loss) instead of the LBM-resolved wake.  Comparing BEMT vs
ALM+LBM isolates the inflow model:
  - if they agree away from the tip  -> the ALM force/airfoil chain is correct
  - BEMT's proper tip-loss roll-off vs the ALM's tip over-loading pinpoints the
    ALM tip induced-downwash deficit as the source of C_T over-prediction.

Sectional balance (hover, Glauert + Prandtl F):
    4 F λ² x = (σ/2) x² C_n      ->  λ = sqrt(σ x C_n / (8 F))
    φ = atan2(λ, x),  α = θ(x) − φ,  C_n = C_l cosφ − C_d sinφ
    dC_T/dx = (σ/2) x² C_n,   dC_P/dx = (σ/2) x³ (C_l sinφ + C_d cosφ)
with x = r/R, σ = N c /(π R) (disk solidity), local Mach = M_tip·x.

CLI (Caradonna-Tung, untwisted 8 deg):
    python src/utilities/bemt_hover.py --deck data/airfoils/NACA0012_Yee.C81 \\
        --R 1.143 --chord 0.1905 --nblades 2 --twist 8 --rootcut 0.20 --mtip 0.877
"""
import argparse
import math

import numpy as np


def bemt_hover(deck, N, R, chord, twist_deg, mtip, root_cut=0.2, nx=80,
               itmax=200, tol=1e-7):
    """Solve hover BEMT.  chord, twist may be scalars or callables of x=r/R.

    Returns dict: CT, CP, FM, sigma, and spanwise arrays (x, alpha, CL, CD,
    lam, phi_deg, F, dCT_dx, M2CL).
    """
    x = np.linspace(root_cut, 1.0, nx)
    cfun = chord if callable(chord) else (lambda _x: chord)
    tfun = twist_deg if callable(twist_deg) else (lambda _x: twist_deg)

    alpha = np.zeros(nx); CL = np.zeros(nx); CD = np.zeros(nx)
    lam = np.zeros(nx); phid = np.zeros(nx); Ff = np.zeros(nx)
    dCT = np.zeros(nx); dCP = np.zeros(nx)

    for i, xi in enumerate(x):
        c = cfun(xi); th = tfun(xi); M = mtip * xi
        sigma = N * c / (math.pi * R)
        lm = 0.05
        for _ in range(itmax):
            phi = math.atan2(lm, xi)
            a = th - math.degrees(phi)
            cl = float(deck.get_CL(a, M)); cd = float(deck.get_CD(a, M))
            Cn = cl * math.cos(phi) - cd * math.sin(phi)
            f = 0.5 * N * (1.0 - xi) / max(xi * math.sin(phi), 1e-6)
            F = max((2.0 / math.pi) * math.acos(min(1.0, math.exp(-f))), 1e-2)
            # Cn<=0 -> section unloaded; otherwise momentum balance for lambda.
            lm_new = math.sqrt(sigma * xi * Cn / (8.0 * F)) if Cn > 0 else 0.0
            lm_new = min(lm_new, 0.6)             # cap (avoid F->0 tip blow-up)
            if abs(lm_new - lm) < tol:
                lm = lm_new
                break
            lm = 0.7 * lm + 0.3 * lm_new          # heavy damping for stability
        phi = math.atan2(lm, xi)
        a = th - math.degrees(phi)
        cl = float(deck.get_CL(a, M)); cd = float(deck.get_CD(a, M))
        Cn = cl * math.cos(phi) - cd * math.sin(phi)
        Ct = cl * math.sin(phi) + cd * math.cos(phi)
        alpha[i], CL[i], CD[i] = a, cl, cd
        lam[i], phid[i], Ff[i] = lm, math.degrees(phi), F
        dCT[i] = 0.5 * sigma * xi ** 2 * Cn
        dCP[i] = 0.5 * sigma * xi ** 3 * Ct

    CT = float(np.trapezoid(dCT, x))
    CP = float(np.trapezoid(dCP, x))
    FM = CT ** 1.5 / (math.sqrt(2.0) * CP) if CP > 0 else float("nan")
    sigma = N * cfun(0.75) / (math.pi * R)
    return {"CT": CT, "CP": CP, "FM": FM, "sigma": sigma, "x": x,
            "alpha": alpha, "CL": CL, "CD": CD, "lam": lam, "phi_deg": phid,
            "F": Ff, "dCT_dx": dCT, "M2CL": (mtip * x) ** 2 * CL}


def main():
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..", "..")))
    from src.actuator.c81_loader import C81PolarSet
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--R", type=float, required=True)
    ap.add_argument("--chord", type=float, required=True)
    ap.add_argument("--nblades", type=int, required=True)
    ap.add_argument("--twist", type=float, default=8.0, help="collective [deg] (untwisted)")
    ap.add_argument("--rootcut", type=float, default=0.2)
    ap.add_argument("--mtip", type=float, required=True)
    ap.add_argument("--exp-ct", type=float, default=None)
    a = ap.parse_args()
    deck = C81PolarSet.from_file(a.deck)
    r = bemt_hover(deck, a.nblades, a.R, a.chord, a.twist, a.mtip, a.rootcut)
    print("  BEMT hover (deck=%s, M_tip=%.3f, theta=%.1f deg)"
          % (os.path.basename(a.deck), a.mtip, a.twist))
    print("    sigma = %.4f" % r["sigma"])
    print("    C_T = %.5f   C_P = %.6f   FM = %.3f   C_T/sigma = %.4f"
          % (r["CT"], r["CP"], r["FM"], r["CT"] / r["sigma"]))
    if a.exp_ct:
        print("    vs experiment C_T = %.5f  -> %+.1f%%"
              % (a.exp_ct, 100 * (r["CT"] - a.exp_ct) / a.exp_ct))
    print("    tip(x>0.9) alpha=%.2f deg, F(tip)=%.3f, peak dCT/dx at x=%.2f"
          % (r["alpha"][r["x"] > 0.9].mean(), r["F"][-1],
             r["x"][int(np.argmax(r["dCT_dx"]))]))


if __name__ == "__main__":
    main()
