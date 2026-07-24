"""HART-II archB+Shen collective sweep -> 3-column summary (Task 3).

Scans result_hart2_hover_c*_archB_shen_g030 folders (default: cwd, or pass
a directory), averages the LAST 5 REVS of rotor_performance.csv, and writes

    hart2_archB_sweep_summary.csv   columns: collective_deg, CT_sigma, FM

plus the FM vs CT/sigma figure (hart2_archB_fm_vs_ctsigma.png). Extra
diagnostics (CT, CP, thrust CV, completeness) go to stdout only.

    python summarize_hart2_sweep.py [results_dir]
"""
import csv
import glob
import os
import re
import sys

import numpy as np

# HART-II geometry (matches configs/hart2/_hart2_hover_base.py)
SIGMA     = 4 * 0.121 / (np.pi * 2.0)      # = 0.077028
STEPS_REV = 1257                            # D=40, u_tip=0.1
FULL_STEP = 25 * STEPS_REV                  # 25-rev production

base = sys.argv[1] if len(sys.argv) > 1 else "."
folders = sorted(glob.glob(os.path.join(
    base, "result_hart2_hover_c*_archB_shen_g030")))
if not folders:
    sys.exit(f"no result_hart2_hover_c*_archB_shen_g030 folders under {base!r}")

rows_out = []
print(f"{'coll':>5s} {'CT/sigma':>9s} {'FM':>7s}   "
      f"{'CT':>8s} {'CP':>9s} {'CV%':>4s}  last_step")
for f in folders:
    m = re.search(r"hover_c([0-9.]+)_", os.path.basename(f))
    coll = float(m.group(1))
    rp = os.path.join(f, "csv", "rotor_performance.csv")
    if not os.path.exists(rp):
        print(f"{coll:5.1f}  -- missing rotor_performance.csv — skipped")
        continue
    rows = list(csv.DictReader(open(rp)))
    a = np.array([[float(r["step"]), float(r["thrust_lu"]),
                   float(r["torque_lu"]), float(r["R_lu"])] for r in rows])
    R = a[0, 3]
    area, om = np.pi * R ** 2, 0.1 / R
    last = a[:, 0].max()
    mm = a[:, 0] >= last - 5 * STEPS_REV
    CTs = a[mm, 1] / (area * 0.01)
    CPs = a[mm, 2] * om / (area * 1e-3)
    CT, CP = CTs.mean(), CPs.mean()
    FM = CT ** 1.5 / (np.sqrt(2) * CP)
    cv = 100 * CTs.std() / CT
    note = "" if last >= FULL_STEP - 1 else f"  ⚠ INCOMPLETE ({last:.0f})"
    print(f"{coll:5.1f} {CT / SIGMA:9.4f} {FM:7.4f}   "
          f"{CT:8.5f} {CP:9.6f} {cv:4.1f}  {last:.0f}{note}")
    rows_out.append((coll, CT / SIGMA, FM))

out = os.path.join(base, "hart2_archB_sweep_summary.csv")
with open(out, "w") as fo:
    fo.write("collective_deg,CT_sigma,FM\n")
    for c, cts, fm in rows_out:
        fo.write(f"{c:.1f},{cts:.6f},{fm:.6f}\n")
print(f"\nwrote {out}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.array(rows_out)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(arr[:, 1], arr[:, 2], "o-", ms=7, lw=1.8,
            label="ALM-LBM archB + Shen g0.3 (mlg4)")
    for c, x, y in rows_out:
        ax.annotate(f"{c:.0f}°", (x, y), textcoords="offset points",
                    xytext=(6, -10), fontsize=8, color="gray")
    ax.set_xlabel(r"$C_T/\sigma$"); ax.set_ylabel("FM")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title(rf"HART-II hover — FM vs $C_T/\sigma$  ($\sigma$={SIGMA:.4f})")
    plt.tight_layout()
    png = os.path.join(base, "hart2_archB_fm_vs_ctsigma.png")
    plt.savefig(png, dpi=140)
    print(f"wrote {png}")
except Exception as e:                     # cluster nodes may lack matplotlib
    print(f"(plot skipped: {e})")
