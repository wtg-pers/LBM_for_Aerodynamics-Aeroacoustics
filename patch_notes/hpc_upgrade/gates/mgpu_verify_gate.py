"""G-verify — main_mpi.py --verify entry-point smoke (review #2 meta-lesson).

The gate ladder's blind spot is any entry point WITHOUT a gate: the M5b
build() signature change broke verify() (tuple unpack -> AttributeError ->
comm.Abort) and no gate walked that path — the restart gate drives main_mpi
but never --verify. This gate runs the actual runbook §1(a) command shape
and asserts (a) the process exits 0, (b) every level reports bit=True,
(c) RESULT: PASS. Also exercises --strict-bit (F-4), which lives inside
verify() and was equally unreachable while verify was broken.

Run from repo root:  python patch_notes/hpc_upgrade/gates/mgpu_verify_gate.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "..", ".."))
CFG = "configs/hpc_bench/bench5_purealm_m3.py"


def run(extra):
    cmd = ["mpirun", "-n", "2", sys.executable, "main_mpi.py",
           "--config", CFG, "--steps", "2", "--log-every", "0",
           "--verify"] + extra
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=560)
    return r


def main():
    ok = True
    for label, extra in (("plain", []), ("strict-bit", ["--strict-bit"])):
        r = run(extra)
        bits = re.findall(r"\[verify\] L\d: .*bit=(\w+)", r.stdout)
        passed = "RESULT: PASS" in r.stdout
        good = (r.returncode == 0 and passed
                and len(bits) == 5 and all(b == "True" for b in bits))
        print(f"  [{label}] exit={r.returncode} levels_bit={bits} "
              f"pass={passed}")
        if not good:
            print(r.stdout[-1500:])
            print(r.stderr[-1000:])
        ok = ok and good
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
