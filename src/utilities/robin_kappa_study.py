"""robin/11 — kappa_n*h wall-pressure channel criterion, a-priori study (K0-K3).

Pure post-processing of the five existing rotor-off runs (robin/03-10): per
orifice, the two wall-Cp channels already on disk
    h0.5  = surface p_state (robin_<run>_cp.csv, kernel sample at 0.5 cell)
    h1.5  = offline volume read (robin_<run>_cp_vol_h15.csv, robin/08)
and the question is whether the per-orifice channel winner is predicted by a
local, body-agnostic dimensionless number kappa*h (facet normal curvature x
readout height), replacing the ROBIN-specific station rule (08 section 6:
st1-4 -> h1.5, st12-14 -> p_state).

Pre-registered (session file, before results):
  K0  orifice curvature tool + analytic (genROBIN superellipse) check <=10%
  K1  sign of D = |err_h15| - |err_h05| monotone in kappa*h (5 runs pooled;
      rank correlation + logistic fit)
  K2  threshold kh* invariant in r (32->40: h_phys x0.8) and alpha (-10..+5)
  K3  the alpha=-10 lower-nose 2-point excess (10 section 2) explained by
      kappa*h or dynamic-pressure weighted (u_t/U)^2 kappa*h
Primary indicator |k|max, secondary mean curvature H (pre-registered);
kappa_n(flow) = normal curvature along the projected freestream direction is
carried as an EXPLORATORY indicator (labelled post-hoc: motivated by the K0
analytic table showing |k|max does not separate nose from tailboom, while
patch 08's kappa_n figures 5.3 -> 1.5 -> 0 are the meridional values).

Usage (main dir):
    python -m src.utilities.robin_kappa_study            # K0-K3 report + figs
Outputs to temp_results/robin/: robin_kappa_stations.png (2x7 station plate),
robin_kappa_scatter.png, robin_kappa_orifices.csv (per-orifice curvature).
Cluster is Python 3.9: no PEP 604 annotations.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utilities.robin_anchor import (_REPO, load_orifices, mirror_pairs,
                                        orifice_curvature_quadric,
                                        kappa_n_direction,
                                        read_sim_csv as load_cp_csv)

STL = os.path.join(_REPO, "input_files", "geom", "robin_mod_v1c.stl")
RES = os.path.join(_REPO, "temp_results", "robin")

#: run -> (csv_h05, csv_h15, alpha_deg, fine cell in R)
RUNS = {
    "r0":   ("robin_r0_cp.csv",   "robin_r0_cp_vol_h15.csv",   0.0,  1.0 / 256.0),
    "r40":  ("robin_r40_cp.csv",  "robin_r40_cp_vol_h15.csv",  0.0,  1.0 / 320.0),
    "am10": ("robin_am10_cp.csv", "robin_am10_cp_vol_h15.csv", -10.0, 1.0 / 256.0),
    "am5":  ("robin_am5_cp.csv",  "robin_am5_cp_vol_h15.csv",  -5.0, 1.0 / 256.0),
    "ap5":  ("robin_ap5_cp.csv",  "robin_ap5_cp_vol_h15.csv",  +5.0, 1.0 / 256.0),
}
H_CELLS = 1.5          # readout height of the outer channel [fine cells]
CREASE_ORF = (52, 57)  # pylon-fuselage crease pairs (52/140, 57/145)


# ──────────────────────────────────────────────────────────────────────
# Analytic genROBIN surface (K0 reference) — coefficients from
# input_files/geom/robin_src/genROBIN.py (rows 0-3 fuselage, 4-5 pylon;
# includes the Applied-Scientific-Research fixes). All 176 orifices lie on
# the fuselage (verified here: nearest-surface assignment, pylon count 0).
# ──────────────────────────────────────────────────────────────────────
_HC = np.array([[1.0, -1.0, -0.4, -0.4, 1.8, 0.0, 0.25, 1.8],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 1.0],
                [1.0, -1.0, -0.8, 1.1, 1.5, 0.05, 0.2, 0.6],
                [1.0, -1.0, -1.9, 0.1, 2.0, 0.0, 0.05, 2.0]])
_WC = np.array([[1.0, -1.0, -0.4, -0.4, 2.0, 0.0, 0.25, 2.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 1.0],
                [1.0, -1.0, -0.8, 1.1, 1.5, 0.05, 0.2, 0.6],
                [1.0, -1.0, -1.9, 0.1, 2.0, 0.0, 0.05, 2.0]])
_ZC = np.array([[1.0, -1.0, -0.4, -0.4, 1.8, -0.08, 0.08, 1.8],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, -1.0, -0.8, 1.1, 1.5, 0.04, -0.04, 0.6],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.04, 0.0, 1.0]])
_NC = np.array([[2.0, 3.0, 0.0, 0.4, 1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0],
                [5.0, -3.0, -0.8, 1.1, 1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0]])


def _sec(x: float) -> int:
    if x < 0.4:
        return 0
    if x < 0.8:
        return 1
    if x < 1.9:
        return 2
    return 3


def _sval(x: float, c: np.ndarray) -> float:
    cv = (x + c[2]) / c[3]
    t = c[0] + c[1] * np.sign(cv) * abs(cv) ** c[4]
    return float(c[5] + c[6] * max(0.0, t) ** (1.0 / c[7]))


def _surf(x: float, th: float) -> np.ndarray:
    k = _sec(x)
    H, W = _sval(x, _HC[k]), _sval(x, _WC[k])
    Z0, N = _sval(x, _ZC[k]), _sval(x, _NC[k])
    den = (0.5 * H * abs(math.sin(th))) ** N + (0.5 * W * abs(math.cos(th))) ** N
    r = 0.25 * H * W / den ** (1.0 / N) if den > 0 else 0.0
    return np.array([x, r * math.sin(th), r * math.cos(th) + Z0])


def analytic_curvature(xyz_R: np.ndarray, dx: float = 2e-4, dt: float = 2e-3
                       ) -> Dict[str, np.ndarray]:
    """Principal curvatures (convex positive) + directions + normal of the
    analytic fuselage at each orifice, from the fundamental forms with
    central differences on the exact surface."""
    m = len(xyz_R)
    out = {"k1": np.empty(m), "k2": np.empty(m),
           "dir1": np.empty((m, 3)), "dir2": np.empty((m, 3)),
           "normal": np.empty((m, 3))}
    for i, p in enumerate(xyz_R):
        x = float(np.clip(p[0], 1e-4, 1.9999))
        Z0 = _sval(x, _ZC[_sec(x)])
        th = math.atan2(p[1], p[2] - Z0)
        Su = (_surf(x + dx, th) - _surf(x - dx, th)) / (2 * dx)
        Sv = (_surf(x, th + dt) - _surf(x, th - dt)) / (2 * dt)
        Suu = (_surf(x + dx, th) - 2 * _surf(x, th) + _surf(x - dx, th)) / dx ** 2
        Svv = (_surf(x, th + dt) - 2 * _surf(x, th) + _surf(x, th - dt)) / dt ** 2
        Suv = (_surf(x + dx, th + dt) - _surf(x + dx, th - dt)
               - _surf(x - dx, th + dt) + _surf(x - dx, th - dt)) / (4 * dx * dt)
        n = np.cross(Su, Sv)
        n /= np.linalg.norm(n)
        if np.dot(n, [0.0, math.sin(th), math.cos(th)]) < 0.0:
            n = -n
        E, F, G = Su @ Su, Su @ Sv, Sv @ Sv
        L, M, N2 = Suu @ n, Suv @ n, Svv @ n
        A = np.linalg.solve(np.array([[E, F], [F, G]]),
                            np.array([[L, M], [M, N2]]))
        w, V = np.linalg.eig(A)
        k = -w.real                      # convex positive
        d3 = [Su * V[0, j].real + Sv * V[1, j].real for j in range(2)]
        d3 = [d / np.linalg.norm(d) for d in d3]
        s = np.argsort(k)[::-1]
        out["k1"][i], out["k2"][i] = k[s]
        out["dir1"][i], out["dir2"][i] = d3[s[0]], d3[s[1]]
        out["normal"][i] = n
    return out


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
def flow_dir_body(alpha_deg: float, sigma: float) -> np.ndarray:
    """Freestream direction in the body/STL frame. sigma fixes the pitch
    sign convention EMPIRICALLY (set from the pt88 windward side, see
    main): u = (cos a, 0, sigma*sin a)."""
    a = math.radians(alpha_deg)
    return np.array([math.cos(a), 0.0, sigma * math.sin(a)])


# ──────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────
def spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.correlation), float(r.pvalue)


def logistic_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit P(y=1) = sigmoid(a + b x) by ML. Returns (a, b, x* = -a/b)."""
    from scipy.optimize import minimize

    def nll(p):
        z = p[0] + p[1] * x
        # log(1+exp(z)) stable
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    best = None
    for b0 in (-1.0, 1.0):
        r = minimize(nll, np.array([0.0, b0]), method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    a, b = best.x
    xs = -a / b if b != 0 else float("nan")
    return float(a), float(b), float(xs)


def binned_frac(x: np.ndarray, y: np.ndarray, nbin: int = 8
                ) -> List[Tuple[float, float, int]]:
    """Quantile bins of x -> (median x, mean y, n)."""
    q = np.quantile(x, np.linspace(0, 1, nbin + 1))
    out = []
    for i in range(nbin):
        m = (x >= q[i]) & (x <= q[i + 1] if i == nbin - 1 else x < q[i + 1])
        if m.sum():
            out.append((float(np.median(x[m])), float(np.mean(y[m])), int(m.sum())))
    return out


def main() -> None:
    orf = load_orifices()
    xyz = orf["xyz"]
    st = orf["station"]
    phi = orf["phi_deg"]

    # ── K0: curvature tool + analytic gate ────────────────────────────
    ana = analytic_curvature(xyz)
    dis = orifice_curvature_quadric(STL, xyz, radius=0.012)
    absmax_a = np.maximum(np.abs(ana["k1"]), np.abs(ana["k2"]))
    absmax_d = np.maximum(np.abs(dis["k1"]), np.abs(dis["k2"]))
    crease = np.isin(orf["orifice"], CREASE_ORF + tuple(o + 88 for o in CREASE_ORF))
    smooth = np.isin(st, [1, 2, 13, 14]) & ~crease
    rel = (absmax_d[smooth] - absmax_a[smooth]) / absmax_a[smooth]
    print("== K0 curvature gate (quadric vs analytic genROBIN, smooth st1/2/13/14)")
    print(f"   |k|max rel err: median {np.median(np.abs(rel)):.3f} "
          f"p90 {np.percentile(np.abs(rel), 90):.3f} max {np.max(np.abs(rel)):.3f} "
          f"(n={smooth.sum()}; gate <=0.10 on station means below)")
    for s in (1, 2, 13, 14):
        m = (st == s) & ~crease
        ra = absmax_d[m].mean() / absmax_a[m].mean() - 1.0
        print(f"   st{s:2d} station-mean rel {ra:+.3f}")

    # ── flow-direction sign: fix sigma from pt88 exp (windward = lower) ──
    exp88, _ = load_cp_csv(os.path.join(RES, RUNS["am10"][0]))
    _, _, e88 = mirror_pairs(exp88)
    st1 = st[:88] == 1
    top = st1 & (np.abs(phi[:88]) < 115)
    bot = st1 & (np.abs(phi[:88]) > 115)
    windward_low = np.nanmean(e88[bot]) > np.nanmean(e88[top])
    sigma = -1.0 if windward_low else 1.0
    print(f"\n== flow sign: pt88 st1 exp Cp mean top {np.nanmean(e88[top]):+.3f} "
          f"bottom {np.nanmean(e88[bot]):+.3f} -> windward "
          f"{'lower' if windward_low else 'upper'}, sigma {sigma:+.0f}")

    # ── indicators per orifice (mirror-mean over the pair) ────────────
    def pairmean(v):
        return 0.5 * (v[:88] + v[88:])

    kmax_p = pairmean(absmax_d)
    kmean_p = pairmean(0.5 * (dis["k1"] + dis["k2"]))
    kmax_pa = pairmean(absmax_a)

    # ── pooled table: one row per (run, mirror pair) ──────────────────
    rows = []          # dicts: run, pair(1..88), station, phi, D, err05, err15
    for run, (f05, f15, alpha, cell) in RUNS.items():
        exp05, sim05 = load_cp_csv(os.path.join(RES, f05))
        exp15, sim15 = load_cp_csv(os.path.join(RES, f15))
        assert np.allclose(np.nan_to_num(exp05), np.nan_to_num(exp15))
        _, _, e = mirror_pairs(exp05)
        _, _, s05 = mirror_pairs(sim05)
        _, _, s15 = mirror_pairs(sim15)
        err05 = s05 - e
        err15 = s15 - e
        h_R = H_CELLS * cell
        kn = kappa_n_direction(dis, flow_dir_body(alpha, sigma))
        kn_p = pairmean(kn)
        kna = kappa_n_direction(ana, flow_dir_body(alpha, sigma))
        kna_p = pairmean(kna)
        qw = np.maximum(1.0 - e, 0.0)          # (u_t/U)^2 proxy from exp Cp
        for i in range(88):
            if not (np.isfinite(err05[i]) and np.isfinite(err15[i])):
                continue
            rows.append({
                "run": run, "pair": i + 1, "station": int(st[i]),
                "phi": float(phi[i]), "alpha": alpha,
                "err05": float(err05[i]), "err15": float(err15[i]),
                "D": float(abs(err15[i]) - abs(err05[i])),
                "kh_max": float(kmax_p[i] * h_R),
                "kh_mean": float(abs(kmean_p[i]) * h_R),
                "kh_flow": float(kn_p[i] * h_R),
                "kh_flow_ana": float(kna_p[i] * h_R),
                "kh_flow_q": float(qw[i] * max(kn_p[i], 0.0) * h_R),
                "crease": bool(crease[i]),
            })
    D = np.array([r["D"] for r in rows])
    y = (D < 0).astype(float)              # 1 = h1.5 wins
    runs_v = np.array([r["run"] for r in rows])
    st_v = np.array([r["station"] for r in rows])
    cr_v = np.array([r["crease"] for r in rows])
    print(f"\n== pooled: {len(rows)} (run,pair) points, h1.5 wins "
          f"{100 * y.mean():.1f}% overall")

    # ── K1: monotonicity, all indicators ──────────────────────────────
    print("\n== K1 — sign(D) vs indicator (pooled; Spearman on D, logistic on win)")
    for key, label in (("kh_max", "|k|max*h   [K0 primary]"),
                       ("kh_mean", "|H|*h      [K0 secondary]"),
                       ("kh_flow", "kn(flow)*h [exploratory, STL]"),
                       ("kh_flow_ana", "kn(flow)*h [exploratory, analytic]"),
                       ("kh_flow_q", "(1-Cp)kn+h [K3 secondary]")):
        x = np.array([r[key] for r in rows])
        ok = np.isfinite(x)
        rho, pv = spearman(x[ok], D[ok])
        a, b, xs = logistic_fit(x[ok], y[ok])
        print(f"   {label:34s} Spearman {rho:+.3f} (p {pv:.1e})  "
              f"logistic slope {b:+.1f}  P=0.5 at {xs:.4f}")
        for xm, ym, n in binned_frac(x[ok], y[ok]):
            print(f"      bin med {xm:+.4f}: h1.5 wins {100 * ym:4.0f}%  (n={n})")

    # ── K2: threshold invariance across r and alpha ───────────────────
    print("\n== K2 — kn(flow)*h logistic threshold per run (invariance)")
    for run in RUNS:
        m = runs_v == run
        x = np.array([r["kh_flow"] for r in rows])[m]
        a, b, xs = logistic_fit(x, y[m])
        rho, pv = spearman(x, D[m])
        print(f"   {run:5s} (a={RUNS[run][2]:+.0f}, cell 1/{round(1/RUNS[run][3])}): "
              f"Spearman {rho:+.3f}  slope {b:+.1f}  kh* {xs:.4f}")
    # kappa alone (no h): r32 pooled vs r40 threshold ratio
    for key, lab in (("kh_flow", "kappa*h"),):
        x32 = np.array([r[key] / (H_CELLS * RUNS[r["run"]][3]) for r in rows])
        m40 = runs_v == "r40"
        m32 = ~m40
        _, _, k32 = logistic_fit(x32[m32], y[m32])
        _, _, k40 = logistic_fit(x32[m40], y[m40])
        print(f"   kappa-only threshold: r32-pool {k32:.3f}/R vs r40 {k40:.3f}/R "
              f"(ratio {k40 / k32:.2f}; kappa*h predicts 1.25, kappa alone 1.00)")

    # ── K3: pt88 lower-nose 2-point excess ────────────────────────────
    print("\n== K3 — am10 st1 lower points (phi 132/150, robin/10 s2)")
    for r in rows:
        if r["run"] == "am10" and r["station"] == 1:
            print(f"   phi {r['phi']:+6.1f}: err15 {r['err15']:+.3f} err05 {r['err05']:+.3f} "
                  f"kh_flow {r['kh_flow']:+.4f} (alpha0 would be "
                  f"{[q['kh_flow'] for q in rows if q['run'] == 'r0' and q['pair'] == r['pair']][0]:+.4f}) "
                  f"kh_q {r['kh_flow_q']:+.4f}")

    # ── outputs: per-orifice curvature CSV + figures ──────────────────
    out_csv = os.path.join(RES, "robin_kappa_orifices.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orifice", "station", "phi_deg", "k1_stl", "k2_stl",
                    "k1_ana", "k2_ana", "kn_flow_a0_stl", "crease"])
        kn0 = kappa_n_direction(dis, flow_dir_body(0.0, sigma))
        for i in range(176):
            w.writerow([int(orf["orifice"][i]), int(st[i]), f"{phi[i]:.2f}",
                        f"{dis['k1'][i]:.4f}", f"{dis['k2'][i]:.4f}",
                        f"{ana['k1'][i]:.4f}", f"{ana['k2'][i]:.4f}",
                        f"{kn0[i]:.4f}", int(crease[i])])
    print(f"\n   csv -> {out_csv}")
    plot_figures(rows, st_v, sigma)


def plot_figures(rows: List[dict], st_v: np.ndarray, sigma: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"r0": "k", "r40": "tab:gray", "am10": "tab:blue",
              "am5": "tab:cyan", "ap5": "tab:red"}
    # station plate 2x7: D vs phi, all runs, kh_flow as background stem
    fig, axes = plt.subplots(2, 7, figsize=(22, 7), sharey=True)
    for s, ax in zip(range(1, 15), axes.ravel()):
        sel = [r for r in rows if r["station"] == s]
        for run, c in colors.items():
            rr = sorted([r for r in sel if r["run"] == run], key=lambda r: r["phi"])
            ax.plot([r["phi"] for r in rr], [r["D"] for r in rr], "-o", ms=2.5,
                    lw=0.8, color=c, label=run)
        ax2 = ax.twinx()
        rr = sorted([r for r in sel if r["run"] == "r0"], key=lambda r: r["phi"])
        ax2.plot([r["phi"] for r in rr], [r["kh_flow"] for r in rr], ":",
                 color="tab:green", lw=1.2)
        ax2.set_ylim(-0.01, 0.05)
        ax2.tick_params(axis="y", labelsize=6, colors="tab:green")
        if s != 7 and s != 14:
            ax2.set_yticklabels([])
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_title(f"st {s}", fontsize=9)
        ax.set_xlim(-185, 185)
        ax.set_ylim(-0.15, 0.15)
        ax.grid(alpha=0.3)
        ax.set_xlabel("phi [deg]", fontsize=7)
    axes[0, 0].set_ylabel("D = |err_h1.5| - |err_h0.5|  (neg = h1.5 wins)")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("robin/11 K1 — channel error difference vs phi by station "
                 "(green dots: kn(flow)*h, r0)")
    fig.tight_layout()
    p = os.path.join(RES, "robin_kappa_stations.png")
    fig.savefig(p, dpi=130)
    print(f"   plot -> {p}")

    # pooled scatter: D vs indicators
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, key, lab in zip(axs, ("kh_max", "kh_mean", "kh_flow"),
                            ("|k|max*h (K0 primary)", "|H|*h (K0 secondary)",
                             "kn(flow)*h (exploratory)")):
        for run, c in colors.items():
            rr = [r for r in rows if r["run"] == run]
            ax.scatter([r[key] for r in rr], [r["D"] for r in rr], s=8,
                       color=c, alpha=0.55, label=run)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("D = |err_h1.5| - |err_h0.5|")
    axs[0].legend(fontsize=7)
    fig.suptitle("robin/11 K1 — pooled 5-run channel error difference")
    fig.tight_layout()
    p = os.path.join(RES, "robin_kappa_scatter.png")
    fig.savefig(p, dpi=130)
    print(f"   plot -> {p}")


if __name__ == "__main__":
    main()
