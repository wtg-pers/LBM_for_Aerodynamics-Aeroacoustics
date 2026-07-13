"""2D end-to-end regression gate — cyl Re=100 IBB + MLG-3 (review #3, R3-3).

The 2D path was the gate ladder's blind spot: the fixed-order fallback
range(3) hardcode broke EVERY D2Q9 run (IndexError) and only a manual run
caught it. This gate drives the full production 2D chain through the real
entry point (main.py: D2Q9 cumulant + Bouzidi IBB + 3-level MLG + MEM force
+ output machinery) on a ~100 s shrunk cylinder case and asserts the
recorded Cd band.

Acceptance (pre-argued):
  - exit 0 and a populated force_history.csv (the end-to-end claim; any
    2D wiring regression dies loudly here);
  - tail-mean Cd (steps >= 10k) within the RECORDED band 1.181 +/- 0.05.
    12k steps = 37 convective times = still the developing steady wake
    (between Fornberg's steady 1.06 and the shedding mean ~1.35;
    shedding onset needs O(100+) CT). The band is a regression anchor
    for THIS config (recorded 2026-07-14, RTX 3090), not a literature
    comparison — the full-size literature case is
    configs/cylinders/cyl_Re100_ibb_mlg3.py (Cd 1.40 +/- 0.21 recorded);
  - tail-mean |Cl| < 0.05 (pre-shedding symmetry sanity).

Run from repo root:  python patch_notes/hpc_upgrade/gates/cyl2d_re100_gate.py
"""
import csv
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "..", ".."))
CFG = "configs/hpc_bench/cyl_re100_2d_gate.py"
CSV = "result_cyl_re100_2d_gate/csv/force_history.csv"
CD_BAND = (1.181 - 0.05, 1.181 + 0.05)


def main():
    r = subprocess.run([sys.executable, "main.py", "--config", CFG],
                       cwd=ROOT, capture_output=True, text=True,
                       timeout=1200)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        print(f"  exit={r.returncode}")
        print("  RESULT: FAIL")
        sys.exit(1)

    path = os.path.join(ROOT, CSV)
    rows = list(csv.DictReader(open(path)))
    tail = [(float(x["Cd"]), float(x["Cl"])) for x in rows
            if float(x["step"]) >= 10_000]
    cd = sum(c for c, _ in tail) / len(tail)
    cl = sum(l for _, l in tail) / len(tail)
    ok = (len(rows) > 1000 and CD_BAND[0] <= cd <= CD_BAND[1]
          and abs(cl) < 0.05)
    print(f"  exit=0  force rows={len(rows)}  tail-mean Cd={cd:.4f} "
          f"(band {CD_BAND[0]:.3f}..{CD_BAND[1]:.3f})  |Cl|={abs(cl):.4f}")
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
