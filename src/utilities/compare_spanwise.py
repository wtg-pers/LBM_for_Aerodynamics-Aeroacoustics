"""Spanwise A/B comparison of ALM blade diagnostics (fine-region margin test).

Reconstructs the converged spanwise distribution from a run's per-marker
diagnostics and overlays two (or more) runs so the tip behaviour can be
compared directly.  Built for the fine-region-margin diagnosis:

    narrow (light)      vs   wide (light_wide)
    팁 u_n→0 (다운워시 결손)   팁 u_n 유지 (회복)?

IMPORTANT data layout: `csv/blade_diagnostics/<N>.csv` is the **time series of
marker N** (rows = output steps x blades), NOT one timestep.  A spanwise
snapshot therefore needs the *last converged* rows of *every* marker file.

Usage
-----
    python -m src.utilities.compare_spanwise \\
        --run narrow=result_hvab_hover_c10.0_M650_mlg4_D32_light \\
        --run wide=result_hvab_hover_c10.0_M650_mlg4_D32_light_wide \\
        --mtip 0.65 --avg-revs 3 --outdir aeromechanics_workshop/HVAB

Each --run accepts `label=path` (repeatable).  `path` may point at the run
directory, its `csv/` dir, or the `blade_diagnostics/` dir directly.

The verdict line reports, per run, the tip downwash retention
`u_n(tip) / max u_n(r/R<0.8)` — near 0 = starved tip (grid too small);
recovering toward the inner level = fine region now adequate.  The same
retention metric is reported for the section normal force `F_n` (thrust
loading), together with the integrated thrust `T = sum F_n [lu]`, so the
spanwise loading itself — not just the inflow — can be compared A/B.
"""
import argparse
import csv
import glob
import math
import os

import numpy as np

# columns we average over the converged tail (others ignored if absent)
_NUM_COLS = ("r_R", "r_lu", "eps_lu", "u_n", "u_theta", "u_rel",
             "phi", "alpha", "Re", "CL", "CD", "F_n", "F_theta", "F_L", "F_D")


# =============================================================================
# §1. Per-run spanwise reconstruction
# =============================================================================
def _find_diag_dir(path):
    """Locate the blade_diagnostics directory from a run/csv/diag path."""
    cand = [
        path,
        os.path.join(path, "blade_diagnostics"),
        os.path.join(path, "csv", "blade_diagnostics"),
    ]
    for c in cand:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.csv")):
            # confirm it actually holds marker files (numeric names)
            names = [os.path.basename(f)[:-4]
                     for f in glob.glob(os.path.join(c, "*.csv"))]
            if any(n.isdigit() for n in names):
                return c
    raise FileNotFoundError("blade_diagnostics를 찾을 수 없음: %s" % path)


def load_spanwise(path, avg_revs=3.0, avg_steps=None, blade=None):
    """Reconstruct the converged spanwise profile for one run.

    Returns a dict of np.ndarrays keyed by column (sorted by marker index),
    plus 'marker'.  Each value is averaged over the last `avg_revs` revolutions
    (or `avg_steps` steps) of the final run-segment and over all blades
    (`blade=None`) or over a single blade (`blade=int`, 'blade' column).
    Robust to restart counter resets (see _tail_select.tail_mask).
    """
    try:
        from _tail_select import tail_mask          # run as a script
    except ImportError:
        from ._tail_select import tail_mask         # imported as a package
    diag = _find_diag_dir(path)
    files = [f for f in glob.glob(os.path.join(diag, "*.csv"))
             if os.path.basename(f)[:-4].isdigit()]
    span = {}
    for f in files:
        m = int(os.path.basename(f)[:-4])
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        if blade is not None:
            if "blade" not in rows[0]:
                raise ValueError("blade 컬럼 없음 (per-blade 불가): %s" % f)
            rows = [r for r in rows if int(float(r["blade"])) == blade]
            if not rows:
                continue
        rev = np.array([float(r.get("revolutions", 0) or 0) for r in rows])
        step = np.array([float(r.get("step", 0) or 0) for r in rows])
        rmax = float(rev.max())
        mask, _ = tail_mask(step, rev, avg_revs, avg_steps)
        if mask.sum() == 0:
            mask = np.ones_like(rev, dtype=bool)
        rec = {"marker": m, "rev_last": rmax}
        for k in _NUM_COLS:
            if k in rows[0]:
                vals = np.array([float(r[k]) for r in rows])[mask]
                rec[k] = float(np.mean(vals))
        span[m] = rec
    if not span:
        raise ValueError("마커 데이터 없음: %s" % diag)
    order = sorted(span)
    out = {"marker": np.array(order)}
    for k in _NUM_COLS:
        if all(k in span[m] for m in order):
            out[k] = np.array([span[m][k] for m in order])
    out["rev_last"] = span[order[0]]["rev_last"]

    # ── Grid-invariant sectional loading ────────────────────────────────
    # F_n / F_theta are absolute forces in FINEST-level lattice units; they
    # scale as (resolution)² (chord_lu × dr_lu each ∝ resolution), so raw F_n
    # differs ~4× between a 4-level and 5-level grid for identical physics and
    # must NOT be overlaid across grids. Normalising by ρ·A·u_tip² gives each
    # marker's dimensionless thrust/torque contribution (Σ dCT_n = C_T),
    # which IS grid-invariant and overlays. R_lu is recovered from the span
    # itself (r_lu / r_R, constant); u_tip defaults to umax (0.1 here).
    if "F_n" in out and "r_lu" in out and "r_R" in out:
        with np.errstate(divide="ignore", invalid="ignore"):
            R_lu = np.nanmedian(out["r_lu"] / out["r_R"])
        area_lu = np.pi * R_lu ** 2
        q = 1.0 * area_lu * (0.1 ** 2)            # ρ_ref·A·u_tip²  [lu_force]
        out["_R_lu"] = float(R_lu)
        out["dCT_n"] = out["F_n"] / q             # sectional thrust coeff contrib
        if "F_theta" in out:
            out["dCT_t"] = out["F_theta"] / q     # sectional tangential (∝ torque)
    return out


def list_blades(path):
    """Distinct blade indices present in the diagnostics (sorted)."""
    diag = _find_diag_dir(path)
    f = sorted(glob.glob(os.path.join(diag, "*.csv")))[0]
    ids = set()
    for r in csv.DictReader(open(f)):
        if "blade" not in r:
            return []
        ids.add(int(float(r["blade"])))
    return sorted(ids)


def tip_metrics(s, mtip, umax=0.1, inner_cut=0.8):
    """Tip-starvation metrics + torque split for one run's spanwise dict."""
    rR = s["r_R"]
    un = s.get("u_n")
    inner = rR < inner_cut
    inner_max = float(np.max(un[inner])) if un is not None and inner.any() else float("nan")
    tip_un = float(un[-1]) if un is not None else float("nan")
    retention = tip_un / inner_max if inner_max not in (0.0, float("nan")) else float("nan")
    # collapse radius: outermost r/R where u_n still >= 50% of inner_max
    coll = float("nan")
    if un is not None and np.isfinite(inner_max) and inner_max > 0:
        good = np.where(un >= 0.5 * inner_max)[0]
        coll = float(rR[good[-1]]) if good.size else float("nan")
    m = {
        "rev_last": s.get("rev_last", float("nan")),
        "tip_rR": float(rR[-1]),
        "tip_phi": float(s["phi"][-1]) if "phi" in s else float("nan"),
        "tip_alpha": float(s["alpha"][-1]) if "alpha" in s else float("nan"),
        "tip_un": tip_un, "inner_max_un": inner_max, "retention": retention,
        "collapse_rR": coll,
        "tip_CD": float(s["CD"][-1]) if "CD" in s else float("nan"),
        "tip_M": float(s["u_rel"][-1] * (mtip / umax)) if "u_rel" in s else float("nan"),
    }
    # spanwise loading: integrated thrust, tip force, tip-loading retention
    fn = s.get("F_n")
    if fn is not None:
        inner_fn_max = float(np.max(fn[inner])) if inner.any() else float("nan")
        m["Fn_sum"] = float(np.sum(fn))            # raw [lu] — GRID-SCALE, do not compare across grids
        m["tip_Fn"] = float(fn[-1])
        m["Fn_retention"] = (fn[-1] / inner_fn_max) if inner_fn_max else float("nan")
        m["peak_Fn_rR"] = float(rR[int(np.argmax(fn))])
    dct = s.get("dCT_n")
    if dct is not None:                            # grid-invariant integrated thrust
        m["CT_n"] = float(np.sum(dct))             # Σ sectional dC_T = C_T (cross-grid safe)
    # torque split (induced F_L sinphi vs profile F_D cosphi), arm = r_lu
    if all(k in s for k in ("F_L", "F_D", "phi", "r_lu")):
        phir = np.radians(s["phi"])
        qi = s["r_lu"] * s["F_L"] * np.sin(phir)
        qp = s["r_lu"] * s["F_D"] * np.cos(phir)
        m["Q_ind"], m["Q_prof"] = float(qi.sum()), float(qp.sum())
        tot = m["Q_ind"] + m["Q_prof"]
        m["prof_frac"] = m["Q_prof"] / tot if tot else float("nan")
    return m


# =============================================================================
# §2. Plot
# =============================================================================
def make_plot(runs, outdir, mtip, prefix, umax=0.1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # F_n/F_theta panels use the GRID-INVARIANT normalised loading (dCT_n/dCT_t
    # = F/(ρ·A·u_tip²)) so curves overlay across grids; raw lattice F_n differs
    # ~4× between 4- and 5-level grids and would be misleading. Falls back to raw
    # F_n/F_theta if the normalisation could not be built (e.g. no r_lu column).
    panels = [("u_n", "axial inflow u_n [lu/lt]  (downwash)"),
              ("phi", "inflow angle phi [deg]"),
              ("alpha", "angle of attack alpha [deg]"),
              ("mach", "section Mach"),
              ("dCT_n", "sectional thrust loading  dC_T (norm; grid-invariant)"),
              ("dCT_t", "sectional tangential  dC_T,θ (norm; ∝ torque)"),
              ("CL", "section CL"),
              ("CD", "section CD")]
    nrow, ncol = 2, 4
    fig, ax = plt.subplots(nrow, ncol, figsize=(20, 9))
    fig.suptitle("%s  spanwise A/B  (tip downwash / loading)" % prefix.upper(),
                 fontsize=13, fontweight="bold")
    for j, (key, ylab) in enumerate(panels):
        a = ax[j // ncol, j % ncol]
        for lbl, s in runs:
            rR = s["r_R"]
            if key == "mach":
                y = s["u_rel"] * (mtip / umax) if "u_rel" in s else None
            else:
                y = s.get(key)
            if y is None:
                continue
            a.plot(rR, y, "-o", ms=3, label=lbl)
        a.set(xlabel="r/R", ylabel=ylab)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        if key in ("u_n", "phi", "dCT_n", "dCT_t"):
            a.axhline(0.0, color="k", lw=0.6, ls=":")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(outdir, "%s_spanwise_compare.png" % prefix)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def make_perblade_plot(run_blades, outdir, prefix):
    """One row per run: each blade as a thin line + all-blade mean in black.

    run_blades : list of (label, mean_span, {blade_id: span}).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("u_n", "u_n [lu/lt]"), ("alpha", "alpha [deg]"),
              ("dCT_n", "dC_T (norm)"), ("dCT_t", "dC_T,θ (norm)")]
    nrow = len(run_blades)
    fig, ax = plt.subplots(nrow, len(panels), figsize=(19, 3.4 * nrow),
                           squeeze=False)
    fig.suptitle("%s  per-blade spanwise (thin = blades, black = mean)" % prefix,
                 fontsize=13, fontweight="bold")
    for i, (lbl, mean_s, blades) in enumerate(run_blades):
        for j, (key, ylab) in enumerate(panels):
            a = ax[i][j]
            for b, s in sorted(blades.items()):
                if key in s:
                    a.plot(s["r_R"], s[key], "-", lw=1.0, alpha=0.75,
                           label="b%d" % b)
            if key in mean_s:
                a.plot(mean_s["r_R"], mean_s[key], "k-", lw=2.0, label="mean")
            a.set(xlabel="r/R", ylabel=ylab)
            a.grid(alpha=0.3)
            if j == 0:
                a.set_title(lbl, loc="left", fontsize=10, fontweight="bold")
            if i == 0 and j == len(panels) - 1:
                a.legend(fontsize=7, ncol=2)
            if key in ("u_n", "dCT_n", "dCT_t"):
                a.axhline(0.0, color="k", lw=0.6, ls=":")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(outdir, "%s_spanwise_perblade.png" % prefix)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# =============================================================================
# §3. Driver
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Spanwise A/B (fine-region margin)")
    ap.add_argument("--run", action="append", default=[], required=True,
                    help="label=path (repeatable). path=run dir / csv / blade_diagnostics")
    ap.add_argument("--mtip", type=float, default=0.65)
    ap.add_argument("--avg-revs", type=float, default=3.0)
    ap.add_argument("--avg-steps", type=float, default=None,
                    help="override --avg-revs with a step-based tail window "
                         "(last N steps of the final segment); for restarted runs "
                         "whose revolutions counter reset")
    ap.add_argument("--outdir", default="aeromechanics_workshop/HVAB")
    ap.add_argument("--prefix", default="hvab_marginAB")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--per-blade", action="store_true",
                    help="블레이드별 프로파일 분리: per-blade 지표 표 + "
                         "<prefix>_spanwise_perblade.png (mean은 기존 그대로)")
    args = ap.parse_args()

    runs, paths = [], {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        if not path:
            label, path = os.path.basename(os.path.normpath(spec)), spec
        try:
            runs.append((label, load_spanwise(path, args.avg_revs, args.avg_steps)))
            paths[label] = path
        except Exception as e:
            print("  [무시] %s: %s" % (spec, e))
    if not runs:
        print("사용 가능한 런 없음.")
        return

    print("\n  spanwise A/B (avg %.1f rev, mtip=%.2f)" % (args.avg_revs, args.mtip))
    print("  %-10s rev   tip_un   inner_max  retention  collapse  tip_phi  tip_a  tip_CD  prof/Q" %
          "run")
    metr = {}
    for lbl, s in runs:
        m = tip_metrics(s, args.mtip)
        metr[lbl] = m
        print("  %-10s %4.1f  %+.5f  %.5f    %5.1f%%    r%.3f   %+5.2f  %5.2f  %.4f  %3.0f%%" % (
            lbl, m["rev_last"], m["tip_un"], m["inner_max_un"],
            100 * m["retention"], m["collapse_rR"], m["tip_phi"],
            m["tip_alpha"], m["tip_CD"], 100 * m.get("prof_frac", float("nan"))))

    # spanwise loading: C_T (grid-invariant) + tip retention. Fn_sum[lu] is
    # GRID-SCALE (finest lattice, ∝ resolution²) — shown only for same-grid ref.
    if any("Fn_sum" in metr[lbl] for lbl, _ in runs):
        nan = float("nan")
        print("\n  loading (thrust shape) — C_T/blade=ΣdC_T grid-invariant (×N_b=rotor C_T); sumFn[lu] grid-scale")
        print("  %-11s  CT/blade   Fn_ret   peak@r/R   sumFn[lu](grid-scale)" % "run")
        for lbl, _ in runs:
            m = metr[lbl]
            print("  %-11s  %.5f    %5.1f%%   %.3f      %.4e" % (
                lbl, m.get("CT_n", nan),
                100 * m.get("Fn_retention", nan), m.get("peak_Fn_rR", nan),
                m.get("Fn_sum", nan)))

    # verdict (compares first two runs, ordered as given)
    if len(runs) >= 2:
        a, b = runs[0][0], runs[1][0]
        ra, rb = metr[a]["retention"], metr[b]["retention"]
        print("\n  판정 (%s -> %s):" % (a, b))
        print("    팁 다운워시 retention  %.1f%% -> %.1f%%" % (100 * ra, 100 * rb))
        if np.isfinite(ra) and np.isfinite(rb):
            if rb - ra > 0.15:
                print("    => 팁 u_n 회복: fine-region 과소(격자)가 팁 starvation의 원인 확정.")
            elif abs(rb - ra) <= 0.15:
                print("    => 팁 u_n 거의 불변: 격자 margin이 주범 아님 (inherent ALM/폴라 의심).")
        print("    collapse 반경 r/R  %.3f -> %.3f  (1.0에 가까울수록 건강)" % (
            metr[a]["collapse_rR"], metr[b]["collapse_rR"]))
        fa, fb = metr[a].get("Fn_retention"), metr[b].get("Fn_retention")
        if fa is not None and fb is not None:
            print("    팁 Fn retention  %.1f%% -> %.1f%%  (thrust 결손/회복)" % (
                100 * fa, 100 * fb))
        ta, tb = metr[a].get("CT_n"), metr[b].get("CT_n")
        if ta and tb:
            print("    적분 C_T/blade(ΣdCT, grid-invariant)  %.5f -> %.5f  (%+.1f%%)"
                  % (ta, tb, 100 * (tb - ta) / ta))

    # per-blade breakdown: blade-locked asymmetry vs the all-blade mean
    run_blades = []
    if args.per_blade:
        print("\n  per-blade (tail-avg %.1f rev; 동일 값이어야 축대칭)" % args.avg_revs)
        print("  %-12s bl   T=sumFn[lu]   dT/mean   tip_Fn       tip_a   tip_phi" % "run")
        for lbl, mean_s in runs:
            blades = {}
            for b in list_blades(paths[lbl]):
                blades[b] = load_spanwise(paths[lbl], args.avg_revs,
                                          args.avg_steps, blade=b)
            run_blades.append((lbl, mean_s, blades))
            tsum = {b: float(np.sum(s["F_n"])) for b, s in blades.items()
                    if "F_n" in s}
            tmean = np.mean(list(tsum.values())) if tsum else float("nan")
            for b, s in sorted(blades.items()):
                m = tip_metrics(s, args.mtip)
                print("  %-12s b%d   %.4e   %+5.1f%%   %+.4e   %5.2f   %+5.2f" % (
                    lbl, b, tsum.get(b, float("nan")),
                    100 * (tsum.get(b, float("nan")) - tmean) / tmean,
                    m.get("tip_Fn", float("nan")),
                    m["tip_alpha"], m["tip_phi"]))
            # blade asymmetry: thrust spread + worst spanwise F_n scatter
            if tsum:
                spread = 100 * (max(tsum.values()) - min(tsum.values())) / tmean
                fn = np.array([blades[b]["F_n"] for b in sorted(tsum)])
                rel = np.std(fn, axis=0) / np.maximum(np.abs(np.mean(fn, axis=0)),
                                                      1e-12)
                k = int(np.argmax(rel))
                print("  %-12s      blade thrust spread %.1f%%  |  max F_n scatter "
                      "%.1f%% @ r/R=%.3f" % (
                          lbl, spread, 100 * rel[k],
                          blades[sorted(tsum)[0]]["r_R"][k]))

    if not args.no_plot:
        os.makedirs(args.outdir, exist_ok=True)
        png = make_plot(runs, args.outdir, args.mtip, args.prefix)
        print("\n  wrote: %s" % png)
        if run_blades:
            png2 = make_perblade_plot(run_blades, args.outdir, args.prefix)
            print("  wrote: %s" % png2)


if __name__ == "__main__":
    main()
