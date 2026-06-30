"""
Hover Figure-of-Merit post-processing for ALM+LBM collective sweeps.

Reads each run's `csv/rotor_performance.csv`, averages thrust / torque / power
over the converged tail (last `--avg-revs` revolutions), and computes the
rotorcraft hover coefficients + Figure of Merit:

    C_T   = thrust_lu / (rho_ref * area_lu * tip_speed_lu^2)
    C_P   = power_lu  / (rho_ref * area_lu * tip_speed_lu^3)
    C_T/s = C_T / sigma           C_P/s = C_P / sigma
    FM    = C_T^1.5 / (sqrt(2) * C_P)          (ideal/actual power ratio)

(All quantities are dimensionless ratios of lattice values, so no unit
conversion is needed — the normalisation constants are stored per row.)

Outputs (default --outdir aeromechanics_workshop/temp_results):
    hart2_fm_summary.csv        full table incl. convergence scatter (CV%)
    hart2_task3_paste.csv       columns CT, CT/S, CP, CP/S, FM  (AW1 Task3 order)
    hart2_fm_vs_ctsigma.png     FM-CT/s curve + CT/s(collective) + convergence

Usage:
    python src/utilities/hover_fm_post.py \\
        --results "./result_hart2_hover_c*_mlg4" --avg-revs 3 --sigma 0.077

For the workshop submission map these to the *Free wake* (FM3) rows of the
AW1 Task3 sheet — the LBM resolves the rotor wake directly.
"""
import argparse
import csv
import glob
import math
import os
import re

import numpy as np

PERF_COLS = ("step", "time_lt", "time_phys", "revolutions", "thrust_lu",
             "torque_lu", "power_lu", "rho_ref", "area_lu", "tip_speed_lu",
             "omega_lu", "R_lu", "D_lu", "n_lu")


# =============================================================================
# §1. Per-run reduction
# =============================================================================
def _read_perf(path):
    """Read rotor_performance.csv -> dict of float ndarrays (keyed by column)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = {}
    for k in PERF_COLS:
        try:
            out[k] = np.array([float(r[k]) for r in rows], dtype=float)
        except (KeyError, ValueError):
            return None
    return out


def reduce_run(perf, sigma, avg_revs, avg_steps=None):
    """Average the converged tail and compute hover coefficients + FM.

    Robust to restart counter resets (uses the final run-segment only; see
    _tail_select.tail_mask). `avg_steps` overrides the rev-based window with a
    step-based one. Returns a result dict, or None if there is not enough data.
    """
    try:
        from _tail_select import tail_mask          # run as a script
    except ImportError:
        from ._tail_select import tail_mask         # imported as a package
    rev = perf["revolutions"]
    if rev.size < 2:
        return None
    rev_max = float(rev[-1])
    mask, tinfo = tail_mask(perf["step"], rev, avg_revs, avg_steps)
    if mask.sum() < 1:
        mask = np.ones_like(rev, dtype=bool)

    def _avg(key):
        return float(np.mean(perf[key][mask]))

    def _cv(key):                       # coefficient of variation [%] (steadiness)
        v = perf[key][mask]
        m = float(np.mean(v))
        return float(100.0 * np.std(v) / abs(m)) if m != 0 else float("nan")

    rho   = _avg("rho_ref")
    area  = _avg("area_lu")
    tip   = _avg("tip_speed_lu")
    omega = _avg("omega_lu")
    T     = _avg("thrust_lu")
    Q     = _avg("torque_lu")
    P     = _avg("power_lu")
    # Prefer logged power; fall back to Q*omega if power column is zero/missing.
    if not np.isfinite(P) or P == 0.0:
        P = Q * omega

    CT = T / (rho * area * tip ** 2)
    CP = P / (rho * area * tip ** 3)
    CT_s = CT / sigma
    CP_s = CP / sigma
    FM = (CT ** 1.5) / (math.sqrt(2.0) * CP) if (CP > 0 and CT > 0) else float("nan")

    return {
        "rev_last": rev_max, "n_avg": int(mask.sum()),
        "restarted": tinfo["restarted"], "short_tail": tinfo["short_tail"],
        "tail_rev_span": tinfo["rev_span"],
        "tail_step_lo": tinfo["step_lo"], "tail_step_hi": tinfo["step_hi"],
        "thrust_lu": T, "torque_lu": Q, "power_lu": P,
        "CT": CT, "CP": CP, "CT_sigma": CT_s, "CP_sigma": CP_s, "FM": FM,
        "thrust_cv_pct": _cv("thrust_lu"), "torque_cv_pct": _cv("torque_lu"),
        "_history": (perf["revolutions"], perf["thrust_lu"], perf["power_lu"]),
    }


def _collective_from_dir(d):
    """Parse collective [deg] from a 'result_hart2_hover_cNN.N_...' dir name."""
    m = re.search(r"_c(\d+(?:\.\d+)?)", os.path.basename(os.path.normpath(d)))
    return float(m.group(1)) if m else float("nan")


# =============================================================================
# §2. Driver
# =============================================================================
def collect(results_glob, sigma, avg_revs, avg_steps=None):
    rows = []
    for d in sorted(glob.glob(results_glob)):
        perf_path = os.path.join(d, "csv", "rotor_performance.csv")
        if not os.path.exists(perf_path):
            print("  [skip] no rotor_performance.csv in %s" % d)
            continue
        perf = _read_perf(perf_path)
        if perf is None:
            print("  [skip] no data rows yet in %s" % perf_path)
            continue
        res = reduce_run(perf, sigma, avg_revs, avg_steps)
        if res is None:
            print("  [skip] insufficient revolutions in %s" % d)
            continue
        if res.get("restarted") or res.get("short_tail"):
            print("  [warn] %s: restarted=%s short_tail=%s "
                  "(tail %.2f rev, steps %.0f-%.0f, n=%d) -- counter reset; "
                  "tail taken from final segment%s"
                  % (os.path.basename(os.path.normpath(d)), res["restarted"],
                     res["short_tail"], res["tail_rev_span"], res["tail_step_lo"],
                     res["tail_step_hi"], res["n_avg"],
                     "; span < avg_revs" if res["short_tail"] else ""))
        res["collective"] = _collective_from_dir(d)
        res["dir"] = d
        rows.append(res)
    rows.sort(key=lambda r: r["collective"])
    return rows


def write_csvs(rows, outdir, prefix="hart2"):
    os.makedirs(outdir, exist_ok=True)
    summ = os.path.join(outdir, "%s_fm_summary.csv" % prefix)
    with open(summ, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["collective_deg", "rev_last", "n_avg", "thrust_lu",
                    "power_lu", "CT", "CT_sigma", "CP", "CP_sigma", "FM",
                    "thrust_CV_pct"])
        for r in rows:
            w.writerow(["%.1f" % r["collective"], "%.2f" % r["rev_last"],
                        r["n_avg"], "%.6e" % r["thrust_lu"],
                        "%.6e" % r["power_lu"], "%.6f" % r["CT"],
                        "%.6f" % r["CT_sigma"], "%.6f" % r["CP"],
                        "%.6f" % r["CP_sigma"], "%.4f" % r["FM"],
                        "%.2f" % r["thrust_cv_pct"]])
    # AW1 Task3 paste order: CT, CT/S, CP, CP/S, FM
    paste = os.path.join(outdir, "%s_task3_paste.csv" % prefix)
    with open(paste, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["CT", "CT/S", "CP", "CP/S", "FM"])
        for r in rows:
            w.writerow(["%.6f" % r["CT"], "%.6f" % r["CT_sigma"],
                        "%.6f" % r["CP"], "%.6f" % r["CP_sigma"],
                        "%.4f" % r["FM"]])
    return summ, paste


def make_plot(rows, outdir, sigma, prefix="hart2", exp_curves=None):
    """exp_curves: list of (label, DataFrame) experimental references with
    columns collective_deg, CT_sigma, CP_sigma, FM — overlaid as dashed lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cts = np.array([r["CT_sigma"] for r in rows])
    fm = np.array([r["FM"] for r in rows])
    col = np.array([r["collective"] for r in rows])

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("%s hover - Figure of Merit (ALM+LBM)" % prefix.upper(),
                 fontsize=13, fontweight="bold")

    ax[0, 0].plot(cts, fm, "-o", color="tab:blue", label="ALM (ours)")
    for r in rows:
        ax[0, 0].annotate("%.0f°" % r["collective"],
                          (r["CT_sigma"], r["FM"]), fontsize=7,
                          textcoords="offset points", xytext=(3, 3))
    for lbl, e in (exp_curves or []):
        ax[0, 0].plot(e["CT_sigma"], e["FM"], "--", lw=1.3, label="exp %s" % lbl)
    ax[0, 0].set(xlabel=r"$C_T/\sigma$", ylabel="FM",
                 title="Figure of Merit vs $C_T/\\sigma$  (deliverable)")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(col, cts, "-s", color="tab:green", label="ALM (ours)")
    for lbl, e in (exp_curves or []):
        ax[0, 1].plot(e["collective_deg"], e["CT_sigma"], "--", lw=1.3,
                      label="exp %s" % lbl)
    ax[0, 1].set(xlabel="collective @ 0.75R [deg]", ylabel=r"$C_T/\sigma$",
                 title="Thrust loading vs collective")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].legend(fontsize=8)

    cps = np.array([r["CP_sigma"] for r in rows])
    ax[1, 0].plot(cts, cps, "-^", color="tab:red", label="ALM (ours)")
    for lbl, e in (exp_curves or []):
        ax[1, 0].plot(e["CT_sigma"], e["CP_sigma"], "--", lw=1.3,
                      label="exp %s" % lbl)
    ax[1, 0].set(xlabel=r"$C_T/\sigma$", ylabel=r"$C_P/\sigma$",
                 title="Power polar")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(fontsize=8)

    for r in rows:
        rv, th, _ = r["_history"]
        ax[1, 1].plot(rv, th, lw=0.9, label="%.0f°" % r["collective"])
    ax[1, 1].set(xlabel="revolutions", ylabel="thrust [lu]",
                 title="Convergence history (check steady tail)")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(fontsize=7, ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(outdir, "%s_fm_vs_ctsigma.png" % prefix)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="Hover FM post-processing")
    ap.add_argument("--results", default="./result_hart2_hover_c*_mlg*",
                    help="glob of result directories")
    ap.add_argument("--avg-revs", type=float, default=3.0,
                    help="average over the last N revolutions")
    ap.add_argument("--avg-steps", type=float, default=None,
                    help="override --avg-revs with a step-based tail window "
                         "(last N steps of the final run-segment). Use when the "
                         "revolutions counter is unreliable after a restart.")
    ap.add_argument("--sigma", type=float, default=0.0770,
                    help="rotor solidity (HART-II = 0.077)")
    ap.add_argument("--outdir", default="aeromechanics_workshop/temp_results")
    ap.add_argument("--prefix", default="hart2",
                    help="output filename prefix (e.g. 'hvab')")
    ap.add_argument("--exp", action="append", default=[],
                    help="experimental reference to overlay, 'label=path.csv' "
                         "(CSV cols: collective_deg, CT_sigma, CP_sigma, FM). "
                         "Repeatable, e.g. --exp natural=R44.csv --exp forced=R59.csv")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    exp_curves = []
    for spec in args.exp:
        label, _, path = spec.partition("=")
        if not path:
            label, path = os.path.splitext(os.path.basename(spec))[0], spec
        try:
            import pandas as pd
            exp_curves.append((label, pd.read_csv(path)))
        except Exception as e:
            print("  [exp 무시] %s: %s" % (spec, e))

    rows = collect(args.results, args.sigma, args.avg_revs, args.avg_steps)
    if not rows:
        print("No usable runs found for glob: %s" % args.results)
        return

    print("\n  collective   CT/sigma    FM     CT        CP        CV%%(thrust)  revs")
    for r in rows:
        print("    %5.1f      %7.4f  %6.3f  %.5f  %.6f   %6.2f     %.1f"
              % (r["collective"], r["CT_sigma"], r["FM"], r["CT"], r["CP"],
                 r["thrust_cv_pct"], r["rev_last"]))

    summ, paste = write_csvs(rows, args.outdir, prefix=args.prefix)
    print("\n  wrote: %s" % summ)
    print("  wrote: %s  (AW1 Task3 columns: CT, CT/S, CP, CP/S, FM)" % paste)
    if not args.no_plot:
        png = make_plot(rows, args.outdir, args.sigma, prefix=args.prefix,
                        exp_curves=exp_curves)
        print("  wrote: %s" % png)


if __name__ == "__main__":
    main()
