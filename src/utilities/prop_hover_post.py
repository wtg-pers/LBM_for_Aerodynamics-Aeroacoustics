"""
Propeller/rotor hover post-processing — tail-averaged CT/CP in BOTH conventions.

Reads each run folder's `csv/rotor_performance.csv` (written every
logging_interval; dimensional lattice forces + per-row normalisation
constants), averages thrust/torque/power over the last `--avg-revs`
revolutions, and reports:

    rotorcraft:  C_T = T/(rho A (omega R)^2),  C_P = P/(rho A (omega R)^3)
    propeller :  C_T = T/(rho n^2 D^4),        C_P = P/(rho n^3 D^5)
    FM = C_T^1.5 / (sqrt(2) C_P)   [rotorcraft]

Window-convergence diagnostics (is the tail actually converged?):
    CV%        oscillation scatter of instantaneous C_T in the window
               (blade-passage/wake unsteadiness — NOT drift)
    drift%/rev linear C_T trend over the window, % of mean per revolution
    dhCT/dhCP% second-half minus first-half window mean, % of mean (drift sign)
    verdict    'conv' if |dhCT| and |dhCP| <= --tol-pct (default 0.5%),
               'DRIFT' otherwise, 'n/a' if too few samples
    --per-rev  per-revolution cycle means of C_T(prop) — see the approach shape

The [9] Rotor Performance block in the run log is the FINAL-STEP instantaneous
value — use this tool for converged tail averages instead.

Usage (main directory):
    python src/utilities/prop_hover_post.py \\
        --results "./result_apc18x8e_hover_*rpm_mlg4_notl" --avg-revs 5 --per-rev
"""
import argparse
import glob
import math
import os
import re
from typing import Optional

import numpy as np

# schema: src/solver/output_manager.py::log_rotor_performance_row
COLS = ("step", "time_lt", "time_phys", "revolutions", "thrust_lu",
        "torque_lu", "power_lu", "rho_ref", "area_lu", "tip_speed_lu",
        "omega_lu", "R_lu", "D_lu", "n_lu")


def tail_average(csv_path: str, avg_revs: float) -> Optional[dict]:
    """Average one rotor_performance.csv over its last `avg_revs` revolutions."""
    data = np.genfromtxt(csv_path, delimiter=",", names=COLS, skip_header=1)
    if data.size == 0:
        return None
    data = np.atleast_1d(data)
    rev = data["revolutions"]
    mask = rev >= (rev[-1] - avg_revs)
    if not mask.any():
        return None
    d = data[mask]
    rho, area = float(d["rho_ref"][-1]), float(d["area_lu"][-1])
    tip, n = float(d["tip_speed_lu"][-1]), float(d["n_lu"][-1])
    D = float(d["D_lu"][-1])
    T, Q, P = (float(np.mean(d[k])) for k in
               ("thrust_lu", "torque_lu", "power_lu"))
    ct_inst = d["thrust_lu"] / (rho * area * tip ** 2)
    cp_inst = d["power_lu"] / (rho * area * tip ** 3)
    rev_w = rev[mask]
    ct_mean = max(float(np.mean(ct_inst)), 1e-12)
    cp_mean = max(float(np.mean(cp_inst)), 1e-12)

    # ── window-convergence diagnostics ──
    if mask.sum() >= 4:
        # linear drift of C_T across the window, in % of mean per revolution
        slope = float(np.polyfit(rev_w, ct_inst, 1)[0])
        drift = 100.0 * slope / ct_mean
        # first-half vs second-half window means (drift with a sign, robust
        # to the 2/rev oscillation as long as each half holds >~1 rev)
        mid = 0.5 * (rev_w[0] + rev_w[-1])
        h1, h2 = rev_w < mid, rev_w >= mid
        dh_ct = 100.0 * (np.mean(ct_inst[h2]) - np.mean(ct_inst[h1])) / ct_mean
        dh_cp = 100.0 * (np.mean(cp_inst[h2]) - np.mean(cp_inst[h1])) / cp_mean
    else:
        drift = dh_ct = dh_cp = float("nan")

    # per-revolution cycle means of C_T(prop) — approach shape at a glance
    k_pr = (rho * n ** 2 * D ** 4) / (rho * area * tip ** 2)  # rc -> prop
    per_rev = []
    for k in range(int(np.floor(rev_w[0])), int(np.ceil(rev_w[-1]))):
        b = (rev_w >= k) & (rev_w < k + 1)
        if b.sum() >= 3:
            per_rev.append((k, float(np.mean(ct_inst[b])) / k_pr))

    return {
        "n_samples": int(mask.sum()),
        "rev_window": (float(rev_w[0]), float(rev[-1])),
        "CT_rc": T / (rho * area * tip ** 2),
        "CP_rc": P / (rho * area * tip ** 3),
        "CT_pr": T / (rho * n ** 2 * D ** 4),
        "CP_pr": P / (rho * n ** 3 * D ** 5),
        "FM": (T / (rho * area * tip ** 2)) ** 1.5
              / (math.sqrt(2.0) * P / (rho * area * tip ** 3)),
        "CV_pct": 100.0 * float(np.std(ct_inst)) / ct_mean,
        "drift_pct_rev": drift,
        "dh_ct_pct": dh_ct,
        "dh_cp_pct": dh_cp,
        "per_rev": per_rev,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results", required=True,
                    help="run-folder glob, e.g. './result_apc*_notl'")
    ap.add_argument("--avg-revs", type=float, default=5.0,
                    help="tail window in revolutions (default 5)")
    ap.add_argument("--tol-pct", type=float, default=0.5,
                    help="convergence tolerance on |dhalf CT/CP| [%% of mean]")
    ap.add_argument("--per-rev", action="store_true",
                    help="print per-revolution cycle means of CT(prop)")
    ap.add_argument("--out", default=None, help="optional summary CSV path")
    args = ap.parse_args()

    folders = sorted(glob.glob(args.results))
    if not folders:
        raise SystemExit(f"no folders match {args.results!r}")

    rows = []
    print(f"tail = last {args.avg_revs:g} revs, conv tol = {args.tol_pct:g}%\n")
    print(f"{'case':<44} {'rpm':>5}  {'CT_prop':>8} {'CP_prop':>8} {'FM':>6}"
          f"  {'CV%':>5} {'drift%/rev':>10} {'dhCT%':>6} {'dhCP%':>6}"
          f"  {'verdict':>7}  window")
    for folder in folders:
        path = os.path.join(folder, "csv", "rotor_performance.csv")
        if not os.path.isfile(path):
            print(f"{os.path.basename(folder):<44} -- missing {path}")
            continue
        r = tail_average(path, args.avg_revs)
        if r is None:
            print(f"{os.path.basename(folder):<44} -- empty/short csv")
            continue
        m = re.search(r"(\d+)rpm", folder)
        rpm = m.group(1) if m else "-"
        if math.isnan(r["dh_ct_pct"]):
            verdict = "n/a"
        elif (abs(r["dh_ct_pct"]) <= args.tol_pct
              and abs(r["dh_cp_pct"]) <= args.tol_pct):
            verdict = "conv"
        else:
            verdict = "DRIFT"
        print(f"{os.path.basename(folder):<44} {rpm:>5}  "
              f"{r['CT_pr']:8.4f} {r['CP_pr']:8.4f} {r['FM']:6.3f}  "
              f"{r['CV_pct']:5.1f} {r['drift_pct_rev']:10.2f} "
              f"{r['dh_ct_pct']:6.2f} {r['dh_cp_pct']:6.2f}  "
              f"{verdict:>7}  "
              f"[{r['rev_window'][0]:.1f},{r['rev_window'][1]:.1f}]"
              f" n={r['n_samples']}")
        if args.per_rev and r["per_rev"]:
            trail = "  ".join(f"{k}-{k+1}:{v:.4f}" for k, v in r["per_rev"])
            print(f"{'':<51} per-rev CT_prop  {trail}")
        rows.append((os.path.basename(folder), rpm, verdict, r))

    if args.out and rows:
        with open(args.out, "w") as fh:
            fh.write("case,rpm,CT_prop,CP_prop,CT_rc,CP_rc,FM,CV_pct,"
                     "drift_pct_rev,dh_ct_pct,dh_cp_pct,verdict,"
                     "rev_lo,rev_hi,n_samples\n")
            for name, rpm, verdict, r in rows:
                fh.write(f"{name},{rpm},{r['CT_pr']:.6f},{r['CP_pr']:.6f},"
                         f"{r['CT_rc']:.7f},{r['CP_rc']:.8f},{r['FM']:.4f},"
                         f"{r['CV_pct']:.2f},{r['drift_pct_rev']:.3f},"
                         f"{r['dh_ct_pct']:.3f},{r['dh_cp_pct']:.3f},"
                         f"{verdict},{r['rev_window'][0]:.2f},"
                         f"{r['rev_window'][1]:.2f},{r['n_samples']}\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
