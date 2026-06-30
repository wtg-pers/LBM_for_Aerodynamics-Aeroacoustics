"""
Spanwise blade-loading post-processor for the ALM.

Reads a run's per-marker diagnostics (csv/blade_diagnostics/<j>.csv), averages
each marker over the converged tail (last `--avg-revs` revolutions, across all
blades and steps), and produces the spanwise distributions vs r/R:

    alpha(r/R)      section angle of attack       [deg]
    CL(r/R), CD(r/R)   section coefficients (= sectional C_n for small alpha)
    F_n(r/R)        section normal (thrust-dir) force  [lattice]  -> loading shape
    M(r/R), M^2*CL(r/R)   (if --mtip given)  classic rotor sectional-loading metric

These are the ALM-comparable quantities for hover validation (e.g. Caradonna-
Tung sectional C_n at r/R = 0.50/0.68/0.80/0.89/0.96).  Surface Cp is NOT
produced by an actuator line.

Usage:
    python src/utilities/spanwise_post.py --result ./result_ct_t08.0_M440_mlg4_D32 \\
        --avg-revs 3 --mtip 0.439 [--exp experimental_Cn.csv --exp-col M2CL]
"""
import argparse
import glob
import os

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def load_spanwise(result_dir, avg_revs=3.0):
    """Average each marker's diagnostics over the last `avg_revs` revolutions.

    Returns a DataFrame sorted by r_R (one row per marker), or None.
    """
    bd = os.path.join(result_dir, "csv", "blade_diagnostics")
    files = sorted(f for f in glob.glob(os.path.join(bd, "*.csv"))
                   if os.path.basename(f) != "blade_geometry.csv")
    recs = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty or "revolutions" not in df.columns:
            continue
        rmax = float(df["revolutions"].max())
        sub = df[df["revolutions"] >= rmax - avg_revs]
        if sub.empty:
            sub = df
        recs.append(sub.mean(numeric_only=True))
    if not recs:
        return None
    return pd.DataFrame(recs).sort_values("r_R").reset_index(drop=True)


def compute(result_dir, avg_revs=3.0, mtip=None, umax=0.1):
    """Spanwise table + (optionally) local Mach and M^2*CL."""
    df = load_spanwise(result_dir, avg_revs)
    if df is None:
        return None
    df = df[df.get("r_R", 0) > 0].copy() if "r_R" in df else df
    if mtip is not None and "u_rel" in df.columns:
        # u_rel is lattice; u_rel_lu = umax maps to V_tip = mtip*a  ->  M = u_rel*mtip/umax
        df["mach"] = df["u_rel"] * (mtip / umax)
        df["M2CL"] = df["mach"] ** 2 * df["CL"]
    return df


def make_plot(df, out_path, exp=None, exp_col="CL", title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_m = "M2CL" in df.columns
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Spanwise blade loading" + (" — " + title if title else ""),
                 fontsize=13, fontweight="bold")
    r = df["r_R"].values

    ax[0, 0].plot(r, df["alpha"], "-o", ms=3)
    ax[0, 0].set(xlabel="r/R", ylabel=r"$\alpha$ [deg]", title="Angle of attack")

    ax[0, 1].plot(r, df["CL"], "-o", ms=3, label="ALM $C_L$")
    if exp is not None and exp_col in ("CL",):
        ax[0, 1].plot(exp[:, 0], exp[:, 1], "ks", ms=6, label="experiment")
        ax[0, 1].legend(fontsize=8)
    ax[0, 1].set(xlabel="r/R", ylabel=r"$C_L$ ($\approx C_n$)",
                 title="Section lift coefficient")

    if has_m:
        ax[1, 0].plot(r, df["M2CL"], "-o", ms=3, label=r"ALM $M^2 C_L$")
        if exp is not None and exp_col in ("M2CL", "M2Cn"):
            ax[1, 0].plot(exp[:, 0], exp[:, 1], "ks", ms=6, label="experiment")
            ax[1, 0].legend(fontsize=8)
        ax[1, 0].set(xlabel="r/R", ylabel=r"$M^2 C_L$",
                     title="Normalized sectional loading")
    else:
        ax[1, 0].plot(r, df["CD"], "-o", ms=3, color="tab:red")
        ax[1, 0].set(xlabel="r/R", ylabel=r"$C_D$", title="Section drag")

    ax[1, 1].plot(r, df["F_n"], "-o", ms=3, color="tab:green")
    ax[1, 1].set(xlabel="r/R", ylabel=r"$F_n$ [lattice]",
                 title="Section thrust-direction force (loading shape)")

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    if pd is None:
        raise SystemExit("pandas required for spanwise_post")
    ap = argparse.ArgumentParser(description="Spanwise blade-loading post-processing")
    ap.add_argument("--result", required=True, help="result directory")
    ap.add_argument("--avg-revs", type=float, default=3.0)
    ap.add_argument("--mtip", type=float, default=None,
                    help="tip Mach (enables local Mach + M^2*CL)")
    ap.add_argument("--umax", type=float, default=0.1, help="u_max_lu (scaling)")
    ap.add_argument("--exp", default=None,
                    help="experimental CSV: columns r_R,value (header)")
    ap.add_argument("--exp-col", default="CL", choices=["CL", "M2CL", "M2Cn"])
    ap.add_argument("--outdir", default="aeromechanics_workshop/temp_results")
    args = ap.parse_args()

    df = compute(args.result, args.avg_revs, args.mtip, args.umax)
    if df is None:
        print("No blade diagnostics found in", args.result)
        return
    os.makedirs(args.outdir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.result))
    csv_out = os.path.join(args.outdir, "spanwise_%s.csv" % tag)
    cols = [c for c in ["r_R", "alpha", "phi", "Re", "CL", "CD",
                        "mach", "M2CL", "F_n", "F_theta", "eps_lu"] if c in df.columns]
    df[cols].to_csv(csv_out, index=False)

    exp = None
    if args.exp:
        exp = np.genfromtxt(args.exp, delimiter=",", skip_header=1)
        if exp.ndim == 1:
            exp = exp.reshape(1, -1)

    png = make_plot(df, os.path.join(args.outdir, "spanwise_%s.png" % tag),
                    exp=exp, exp_col=args.exp_col, title=tag)
    print("  markers: %d, r/R %.3f..%.3f" % (len(df), df["r_R"].min(), df["r_R"].max()))
    if "M2CL" in df.columns:
        print("  CL range %.3f..%.3f, peak M^2CL=%.4f"
              % (df["CL"].min(), df["CL"].max(), df["M2CL"].max()))
    print("  wrote:", csv_out)
    print("  wrote:", png)


if __name__ == "__main__":
    main()
