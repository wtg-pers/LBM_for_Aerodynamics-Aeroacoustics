#!/usr/bin/env python3
"""Collect + plot ALM run results — integrated performance & spanwise loading.

Two jobs the per-run diagnostics don't do on their own:
  1. INTEGRATED performance from ``rotor_performance.csv`` (converged tail):
     C_T = T/(ρ A u_tip²),  C_P = P/(ρ A u_tip³),  FM = C_T^{3/2}/(√2 C_P).
  2. SPANWISE loading overlays + a grid-CONVERGENCE view for a D-sweep:
     dC_T,n (grid-invariant, from compare_spanwise) and, when chord is
     available, the standard rotor airload M²c_n = N'/(½ρ a² c).

Reuses :func:`compare_spanwise.load_spanwise` for the (per-marker time-series →
converged spanwise snapshot) reconstruction, so the tricky data-layout logic
lives in one place.  Chord (for M²c_n) is read from ``blade_geometry.csv``.

Usage
-----
Explicit case comparison (e.g. the Kleine wake A/B) ::

    python -m src.utilities.collect_alm_runs \
        --run pure=aeromechanics_workshop/HVAB/260706_5level_test/slab5_pure_alm_csv \
        --run free=aeromechanics_workshop/HVAB/260706_5level_test/slab5_kleine_free_csv \
        --run straight=aeromechanics_workshop/HVAB/0708_straight_kleine/straight_kleine_csv \
        --outdir aeromechanics_workshop/HVAB --prefix slab5_kleineAB --ref-ct 0.00832

Uniform D-sweep (auto-discovers runs, x-axis = rotor diameter in lattice cells) ::

    python -m src.utilities.collect_alm_runs \
        --sweep-glob 'result_hvab_hover_c*_mlg1_D*_unif*' \
        --outdir aeromechanics_workshop/HVAB --prefix unif_sweep --ref-ct 0.00832

``--run label=path`` (repeatable) and ``--sweep-glob PATTERN`` may be combined.
``path`` may point at the run dir, its ``csv/`` dir, or ``blade_diagnostics/``.
A numeric grid size (D_lu) is taken from ``rotor_performance.csv``; the sweep
convergence plot is drawn when ≥3 runs have distinct D.
"""
import argparse
import csv
import glob
import math
import os
import re

import numpy as np

try:                                            # run as script or as module
    from compare_spanwise import load_spanwise
except ImportError:
    from .compare_spanwise import load_spanwise


# ---------------------------------------------------------------------------
# locating the run's files (accepts run dir / csv dir / blade_diagnostics dir)
# ---------------------------------------------------------------------------
def _find_file(path, name):
    """Find ``name`` (e.g. rotor_performance.csv / blade_geometry.csv) for a run."""
    cands = [
        os.path.join(path, name),
        os.path.join(path, "csv", name),
        os.path.join(os.path.dirname(path.rstrip("/")), name),
        os.path.join(path, "blade_diagnostics", name),
        os.path.join(path, "csv", "blade_diagnostics", name),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    hits = glob.glob(os.path.join(path, "**", name), recursive=True)
    return hits[0] if hits else None


def _tail_rows(rows, avg_revs):
    """Last ``avg_revs`` revolutions of the final segment (robust to restarts)."""
    if not rows:
        return rows
    rev = np.array([float(r.get("revolutions", 0) or 0) for r in rows])
    rmax = rev.max()
    keep = rev >= (rmax - avg_revs)
    sel = [r for r, k in zip(rows, keep) if k]
    return sel or rows[-20:]


def load_performance(path, mtip, avg_revs=3.0):
    """Integrated C_T, C_P, FM, thrust, torque, power, D_lu from rotor_performance."""
    f = _find_file(path, "rotor_performance.csv")
    if not f:
        return None
    rows = _tail_rows(list(csv.DictReader(open(f))), avg_revs)
    g = lambda k: float(np.mean([float(r[k]) for r in rows]))
    T, Q, P = g("thrust_lu"), g("torque_lu"), g("power_lu")
    A = float(rows[-1]["area_lu"]); v = float(rows[-1]["tip_speed_lu"])
    rho = float(rows[-1].get("rho_ref", 1.0)); D_lu = float(rows[-1]["D_lu"])
    CT = T / (rho * A * v ** 2)
    CP = P / (rho * A * v ** 3)
    FM = (CT ** 1.5) / (math.sqrt(2.0) * CP) if CP > 0 else float("nan")
    return {"CT": CT, "CP": CP, "FM": FM, "thrust": T, "torque": Q, "power": P,
            "D_lu": D_lu, "R_lu": float(rows[-1]["R_lu"]),
            "rev_last": float(rows[-1].get("revolutions", 0) or 0), "a_lu": v / mtip}


def load_chord(path):
    """{marker: chord_lu} from blade_geometry.csv (for M²c_n)."""
    f = _find_file(path, "blade_geometry.csv")
    if not f:
        return None
    return {int(r["marker"]): float(r["chord_lu"]) for r in csv.DictReader(open(f))}


def sectional_airload(span, chord_by_marker, a_lu):
    """M²c_n = N'/(½ρ a² c) per marker (N' = F_n/dr). None if chord unavailable.
    Also returns dC_T,n (grid-invariant sectional thrust coeff) from load_spanwise."""
    r = span["r_lu"]; Fn = span["F_n"]; mk = span["marker"]
    edges = np.empty(len(r) + 1)
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[0] = r[0] - 0.5 * (r[1] - r[0]); edges[-1] = r[-1] + 0.5 * (r[-1] - r[-2])
    dr = np.diff(edges)
    m2 = None
    if chord_by_marker is not None:
        chord = np.array([chord_by_marker.get(int(m), np.nan) for m in mk])
        m2 = (Fn / dr) / (0.5 * 1.0 * a_lu ** 2 * chord)
    return m2, dr


# ---------------------------------------------------------------------------
# collect one run
# ---------------------------------------------------------------------------
def collect(label, path, mtip, avg_revs, blade=0):
    span = load_spanwise(path, avg_revs=avg_revs, blade=blade)
    perf = load_performance(path, mtip, avg_revs)
    a_lu = perf["a_lu"] if perf else (0.1 / mtip)
    chord = load_chord(path)
    m2, dr = sectional_airload(span, chord, a_lu)
    rR = span["r_R"]
    tip = rR >= 0.8
    dCTn = span.get("dCT_n")
    # tip roll-off failure indicator: is the loading PEAK at (or near) the tip?
    metric = m2 if m2 is not None else (dCTn if dCTn is not None else span["F_n"])
    pk = int(np.nanargmax(metric))
    out = {
        "label": label, "path": path, "span": span, "m2cn": m2, "dCTn": dCTn,
        "rR": rR, "alpha": span.get("alpha"),
        "perf": perf, "a_lu": a_lu,
        "peak_val": float(metric[pk]), "peak_rR": float(rR[pk]),
        "tip_val": float(metric[-1]),                      # outermost marker
        "rolloff_ratio": float(metric[-1] / metric[pk]),   # ~1 → NO roll-off (bad)
    }
    # integrated tip fraction of thrust (grid-invariant via dCT_n if present)
    w = dCTn if dCTn is not None else span["F_n"]
    out["tip_frac"] = float(np.nansum(w[tip]) / np.nansum(w))
    return out


def _guess_D(res, label):
    """Numeric grid size for the sweep x-axis: D_lu from perf, else parse label."""
    if res["perf"] and res["perf"].get("D_lu"):
        return res["perf"]["D_lu"]
    m = re.search(r"[dD](\d+)", label)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_spanwise(results, outdir, prefix, ref_ct=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have_m2 = all(r["m2cn"] is not None for r in results)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for r in results:
        y = r["m2cn"] if have_m2 else r["dCTn"]
        ax[0].plot(r["rR"], y, "-o", ms=3, label=r["label"])
        if r["alpha"] is not None:
            ax[1].plot(r["rR"], r["alpha"], "-o", ms=3, label=r["label"])
    ax[0].set_xlabel("r/R"); ax[0].set_ylabel("M²c_n" if have_m2 else "dC_T,n")
    ax[0].set_title("Spanwise loading" + ("  (M²c_n; KSAS/EXP peak ~0.2 then roll-off)"
                                          if have_m2 else "  (grid-invariant dC_T,n)"))
    ax[0].axvline(0.8, ls=":", c="gray", lw=1); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].set_xlabel("r/R"); ax[1].set_ylabel("alpha [deg]"); ax[1].set_title("Effective AoA")
    ax[1].axvline(0.8, ls=":", c="gray", lw=1); ax[1].grid(alpha=.3); ax[1].legend()
    fig.tight_layout()
    out = os.path.join(outdir, f"{prefix}_spanwise.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_convergence(results, outdir, prefix, ref_ct=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [(_guess_D(r, r["label"]), r) for r in results]
    pts = [(d, r) for d, r in pts if d is not None]
    pts.sort(key=lambda t: t[0])                 # sort by D only (dicts aren't comparable)
    if len({d for d, _ in pts}) < 3:             # need ≥3 DISTINCT grid sizes
        return None
    D = np.array([d for d, _ in pts])
    CT = np.array([r["perf"]["CT"] if r["perf"] else np.nan for _, r in pts])
    FM = np.array([r["perf"]["FM"] if r["perf"] else np.nan for _, r in pts])
    tip = np.array([r["peak_val"] for _, r in pts])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(D, CT, "-o")
    if ref_ct:
        ax[0].axhline(ref_ct, ls="--", c="r", label=f"ref C_T={ref_ct}")
        ax[0].legend()
    ax[0].set_xlabel("rotor D [cells]"); ax[0].set_ylabel("C_T"); ax[0].set_title("C_T vs grid")
    ax[1].plot(D, FM, "-o"); ax[1].set_xlabel("rotor D [cells]")
    ax[1].set_ylabel("Figure of Merit"); ax[1].set_title("FM vs grid")
    ax[2].plot(D, tip, "-o"); ax[2].set_xlabel("rotor D [cells]")
    ax[2].set_ylabel("peak sectional loading"); ax[2].set_title("Tip/peak loading vs grid")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(outdir, f"{prefix}_convergence.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def write_summary(results, outdir, prefix):
    out = os.path.join(outdir, f"{prefix}_summary.csv")
    cols = ["label", "D_lu", "rev", "CT", "CP", "FM", "thrust_lu",
            "peak_load", "peak_rR", "tip_load", "rolloff_ratio", "tip_frac_rR0.8"]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in results:
            p = r["perf"] or {}
            w.writerow([
                r["label"], f"{p.get('D_lu', float('nan')):.0f}" if p else "",
                f"{p.get('rev_last', float('nan')):.1f}" if p else "",
                f"{p.get('CT', float('nan')):.6f}" if p else "",
                f"{p.get('CP', float('nan')):.6f}" if p else "",
                f"{p.get('FM', float('nan')):.4f}" if p else "",
                f"{p.get('thrust', float('nan')):.4f}" if p else "",
                f"{r['peak_val']:.4f}", f"{r['peak_rR']:.3f}",
                f"{r['tip_val']:.4f}", f"{r['rolloff_ratio']:.3f}",
                f"{r['tip_frac']:.3f}"])
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Collect + plot ALM run results")
    ap.add_argument("--run", action="append", default=[],
                    help="label=path (repeatable)")
    ap.add_argument("--sweep-glob", default=None,
                    help="glob of run dirs; label auto = basename")
    ap.add_argument("--mtip", type=float, default=0.65)
    ap.add_argument("--avg-revs", type=float, default=3.0)
    ap.add_argument("--blade", type=int, default=0)
    ap.add_argument("--ref-ct", type=float, default=None,
                    help="reference C_T line (e.g. GT Fluent 0.00832)")
    ap.add_argument("--outdir", default="aeromechanics_workshop/HVAB")
    ap.add_argument("--prefix", default="alm_collect")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit("--run needs label=path: %r" % spec)
        lbl, path = spec.split("=", 1)
        runs.append((lbl, path))
    if args.sweep_glob:
        for path in sorted(glob.glob(args.sweep_glob)):
            if os.path.isdir(path):
                runs.append((os.path.basename(path.rstrip("/")), path))
    if not runs:
        raise SystemExit("no runs (use --run label=path and/or --sweep-glob)")

    os.makedirs(args.outdir, exist_ok=True)
    results = []
    for lbl, path in runs:
        try:
            results.append(collect(lbl, path, args.mtip, args.avg_revs, args.blade))
            p = results[-1]["perf"]
            tag = (f"CT={p['CT']:.5f} FM={p['FM']:.3f} D={p['D_lu']:.0f}"
                   if p else "(no rotor_performance.csv)")
            print(f"  [{lbl:16s}] {tag}  peak={results[-1]['peak_val']:.3f}"
                  f"@r/R={results[-1]['peak_rR']:.3f}  rolloff={results[-1]['rolloff_ratio']:.2f}")
        except Exception as e:
            print(f"  [{lbl:16s}] SKIP: {e}")

    if not results:
        raise SystemExit("no runs loaded")
    s = write_summary(results, args.outdir, args.prefix)
    print(f"\nsummary -> {s}")
    if not args.no_plot:
        sp = plot_spanwise(results, args.outdir, args.prefix, args.ref_ct)
        print(f"spanwise -> {sp}")
        cv = plot_convergence(results, args.outdir, args.prefix, args.ref_ct)
        if cv:
            print(f"convergence -> {cv}")
        else:
            print("convergence plot skipped (<3 runs with distinct D)")


if __name__ == "__main__":
    main()
