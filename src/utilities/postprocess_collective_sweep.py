"""Collective sweep post-processing (6/8/10/12 deg, HVAB D40 farfield40).

Inputs per case (paths configurable below, defaults = cluster testing/ CWD):
  - thrust CSV (main_mpi --csv): step,time_lu,thrust,torque,power,C_T,C_P,FM
  - marker VTPs at the last two rev-locked snapshots (--vtk-every 1257)
Averaging window: LAST 2 REVS (rev = 1257 coarse steps).

Outputs (into --out, default ./sweep_analysis):
  spanwise_M2Cn_collective.png / spanwise_M2Cc_collective.png
  ct_sigma_vs_fom.png, collective_vs_CT_CP_FM.png, sweep_summary.csv

Physics conventions (lattice units, body/finest level):
  M2cn = (F_n/dr) / (0.5 * rho * cs^2 * chord),  cs = 1/sqrt(3), rho = 1
  M2cc = (F_theta/dr) / (0.5 * rho * cs^2 * chord)
  sigma = N_b * sum(c dr) / (pi R^2)  (thrust-weighted-free, geometric)
Cases run at different rank counts mix legitimately: decomposition is
bit-identical to single (final acceptance test), so trajectories match.

    python src/utilities/postprocess_collective_sweep.py [--out DIR]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = 1257
# M2cn normalization sound speed: the PHYSICAL a in lattice units. With
# acoustic_scaling=False the lattice cs=1/sqrt(3) is NOT the physical sound
# speed; a_lu = V_tip_lu / M_tip (farfield40: 0.1/0.65). Using cs made the
# curves ~14x too small vs the EXP sectional data (caught by anchor check).
A_LU = 0.1 / 0.65
# Solidity for CT/sigma: the PUBLISHED nominal-chord convention
# (sigma = Nb*c_ref/(pi*R), HVAB documented value) — NOT the
# marker-integrated blade-area ratio, which excludes the root cutout
# (~0.24R) and includes tip taper and therefore comes out ~25% lower
# (geometric integral prints as a diagnostic: ~0.0784).
SIGMA_REF = 0.1013

# ── case table: (deg, csv, result_dir, last_snapshot_steps) ─────────────
# 10 deg = the existing 25-rev 4-rank archive; 6/8/12 = the 22-rev sweep.
CASES = [
    (6,  "sweep_c6.csv",
     "result_hvab_hover_c06.0_M650_mlg5_D40_farfield40_farfield40_eso_c6_mpi2",
     (26397, 27654)),
    (8,  "sweep_c8.csv",
     "result_hvab_hover_c08.0_M650_mlg5_D40_farfield40_farfield40_eso_c8_mpi2",
     (26397, 27654)),
    # 10 deg: the 25-rev archive has NO marker VTPs (predates M5c) — the
    # sweep script backfills 2 extra revs via checkpoint restart, giving
    # snapshots 32682/33939. Candidates listed newest-last; the newest two
    # that exist are used.
    (10, "mpi4_case1_full.csv",
     "result_hvab_hover_c10.0_M650_mlg5_D40_farfield40_farfield40_eso_mpi4",
     (30168, 31425, 32682, 33939)),
    (12, "sweep_c12.csv",
     "result_hvab_hover_c12.0_M650_mlg5_D40_farfield40_farfield40_eso_c12_mpi2",
     (26397, 27654)),
]


def load_markers(path):
    import vtk
    from vtk.util import numpy_support as ns
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(path); r.Update()
    pd = r.GetOutput().GetPointData()
    out = {}
    for i in range(pd.GetNumberOfArrays()):
        nm = pd.GetArrayName(i)
        out[nm] = ns.vtk_to_numpy(pd.GetArray(i)).copy()
    return out


CHORD_DONOR = None      # set via --chord-donor: NEW-writer VTP of the SAME
                        # rotor/marker layout; legacy VTPs (pre chord_lu)
                        # join chord/r_R by nearest radius (exact stations)


def _donor_join(m):
    d = load_markers(CHORD_DONOR)
    ri = np.argsort(d["radius"][d["blade_id"] == 0])
    dr_ = d["radius"][d["blade_id"] == 0][ri]
    dc_ = d["chord_lu"][d["blade_id"] == 0][ri]
    drr = d["r_R"][d["blade_id"] == 0][ri]
    idx = np.searchsorted(dr_, m["radius"]).clip(0, len(dr_) - 1)
    idx2 = (idx - 1).clip(0)
    pick = np.where(np.abs(dr_[idx] - m["radius"]) <=
                    np.abs(dr_[idx2] - m["radius"]), idx, idx2)
    assert float(np.max(np.abs(dr_[pick] - m["radius"]))) < 0.5, \
        "chord donor stations do not match this rotor"
    m["chord_lu"] = dc_[pick]
    m["r_R"] = drr[pick]
    return m


def spanwise(case_dir, snaps):
    """Blade+snapshot-averaged (r_R, M2cn, M2cc) + geometric sigma, R.
    `snaps` = candidate steps; the NEWEST TWO that exist are used."""
    have = [st for st in snaps if os.path.exists(os.path.join(
        case_dir, "vtk", "markers", f"markers_{st:08d}.vtp"))]
    if not have:
        print(f"  [warn] no marker VTPs in {case_dir}")
        return None
    prof = []
    sigma = R_lu = None
    for st in have[-2:]:
        p = os.path.join(case_dir, "vtk", "markers", f"markers_{st:08d}.vtp")
        if not os.path.exists(p):
            continue
        m = load_markers(p)
        if "chord_lu" not in m:
            if CHORD_DONOR:
                m = _donor_join(m)
            else:
                print(f"  [warn] {p}: legacy VTP without chord_lu — pass "
                      "--chord-donor or re-run with the current writer")
                continue
        act = m.get("active", np.ones_like(m["radius"])) > 0
        r_lu = m["radius"][act]
        rr = (m["r_R"][act] if "r_R" in m
              else r_lu / max(float(r_lu.max()), 1e-30))
        c_lu = m["chord_lu"][act]
        fn, ft = m["F_n"][act], m["F_theta"][act]
        blade = m["blade_id"][act]
        # per-blade marker spacing (uniform distribution)
        dr = np.empty_like(r_lu)
        for b in np.unique(blade):
            sel = blade == b
            rs = np.sort(r_lu[sel])
            d = np.median(np.diff(rs))
            dr[sel] = d
        q = 0.5 * (A_LU ** 2) * c_lu              # rho = 1
        m2cn = (fn / dr) / q
        m2cc = (ft / dr) / q
        prof.append((rr, m2cn, m2cc))
        if sigma is None:
            R_lu = float(r_lu.max())
            nb = len(np.unique(blade))
            one = blade == np.unique(blade)[0]
            sigma = nb * float(np.sum(c_lu[one] * dr[one])) / (np.pi * R_lu ** 2)
    if not prof:
        return None
    # bin-average across snapshots+blades on a common r/R grid
    rr_all = np.concatenate([p[0] for p in prof])
    cn_all = np.concatenate([p[1] for p in prof])
    cc_all = np.concatenate([p[2] for p in prof])
    grid = np.unique(np.round(rr_all, 4))
    cn = np.array([cn_all[np.isclose(rr_all, g, atol=2e-4)].mean() for g in grid])
    cc = np.array([cc_all[np.isclose(rr_all, g, atol=2e-4)].mean() for g in grid])
    return grid, cn, cc, sigma


def perf_last2rev(csv_path, last_step):
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    last_step = min(last_step, float(d["step"].max()))   # use what exists
    lo = last_step - 2 * REV
    msk = (d["step"] > lo) & (d["step"] <= last_step)
    if not msk.any():
        return None
    return {k: float(d[k][msk].mean()) for k in ("C_T", "C_P", "FM")} | \
           {f"{k}_std": float(d[k][msk].std()) for k in ("C_T", "C_P", "FM")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sweep_analysis")
    ap.add_argument("--chord-donor", default=None)
    a = ap.parse_args()
    global CHORD_DONOR
    CHORD_DONOR = a.chord_donor
    os.makedirs(a.out, exist_ok=True)

    rows, span = [], {}
    for deg, csv_p, rdir, snaps in CASES:
        have_csv = os.path.exists(csv_p)
        perf = perf_last2rev(csv_p, snaps[-1]) if have_csv else None
        sp = spanwise(rdir, snaps) if os.path.isdir(rdir) else None
        if perf is None and sp is None:
            print(f"[{deg:>2}deg] no data yet — skipped")
            continue
        sigma_geom = sp[3] if sp else np.nan
        if sp:
            span[deg] = sp
        rows.append({"deg": deg, "sigma": SIGMA_REF,
                     "sigma_geom": sigma_geom,
                     **(perf or {k: np.nan for k in
                                 ("C_T", "C_P", "FM", "C_T_std",
                                  "C_P_std", "FM_std")})})
        print(f"[{deg:>2}deg] CT={rows[-1]['C_T']:.6f}  CP={rows[-1]['C_P']:.6f}"
              f"  FM={rows[-1]['FM']:.4f}  sigma_ref={SIGMA_REF}"
              f"  (geom diag {sigma_geom:.5f})")

    if not rows:
        raise SystemExit("no case data found — check paths in CASES")

    # ── spanwise plots ──
    for key, idx, ylab in (("M2Cn", 1, r"$M^2 c_n$"),
                           ("M2Cc", 2, r"$M^2 c_c$")):
        plt.figure(figsize=(9, 5.2))
        for deg in sorted(span):
            g = span[deg]
            plt.plot(g[0], g[idx], "-o", ms=2.5, label=f"{deg}°")
        plt.xlabel("r/R"); plt.ylabel(ylab)
        plt.title(f"spanwise {key} — collective sweep (last-2-rev avg)")
        plt.grid(alpha=0.3); plt.legend()
        plt.savefig(os.path.join(a.out, f"spanwise_{key}_collective.png"),
                    dpi=130, bbox_inches="tight")
        plt.close()

    # spanwise CSV dump (consumed by compare_sweep_vs_refs.py overlays)
    for deg in sorted(span):
        g = span[deg]
        with open(os.path.join(a.out, f"spanwise_c{deg}.csv"), "w") as f:
            f.write("r_R,M2Cn,M2Cc\n")
            for i in range(len(g[0])):
                f.write(f"{g[0][i]:.5f},{g[1][i]:.6f},{g[2][i]:.6f}\n")

    # ── performance plots ──
    rows.sort(key=lambda r: r["deg"])
    deg = np.array([r["deg"] for r in rows])
    ct = np.array([r["C_T"] for r in rows])
    cp_ = np.array([r["C_P"] for r in rows])
    fm = np.array([r["FM"] for r in rows])
    sg = np.array([r["sigma"] for r in rows])

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(ct / sg, fm, "s-", ms=6)
    for d_, x_, y_ in zip(deg, ct / sg, fm):
        plt.annotate(f"{d_}°", (x_, y_), textcoords="offset points",
                     xytext=(6, 4))
    plt.xlabel(r"$C_T/\sigma$"); plt.ylabel("FM")
    plt.title("hover performance polar (last-2-rev avg)")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(a.out, "ct_sigma_vs_fom.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, y, ye, lab in ((axs[0], ct, "C_T_std", r"$C_T$"),
                           (axs[1], cp_, "C_P_std", r"$C_P$"),
                           (axs[2], fm, "FM_std", "FM")):
        err = np.array([r[ye] for r in rows])
        ax.errorbar(deg, y, yerr=err, fmt="o-", capsize=3)
        ax.set_xlabel("collective [deg]"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    fig.suptitle("collective sweep (last-2-rev mean ± std)")
    plt.savefig(os.path.join(a.out, "collective_vs_CT_CP_FM.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    with open(os.path.join(a.out, "sweep_summary.csv"), "w") as f:
        f.write("deg,sigma_ref,sigma_geom,C_T,C_T_std,C_P,C_P_std,"
                "FM,FM_std,CT_over_sigma\n")
        for r in rows:
            f.write(f"{r['deg']},{r['sigma']:.6f},{r['sigma_geom']:.6f},"
                    f"{r['C_T']:.7f},{r['C_T_std']:.2e},"
                    f"{r['C_P']:.7f},{r['C_P_std']:.2e},"
                    f"{r['FM']:.5f},{r['FM_std']:.2e},"
                    f"{r['C_T'] / r['sigma']:.5f}\n")
    print(f"wrote {a.out}/: spanwise x2, ct_sigma_vs_fom, "
          f"collective_vs_CT_CP_FM, sweep_summary.csv")


if __name__ == "__main__":
    main()
