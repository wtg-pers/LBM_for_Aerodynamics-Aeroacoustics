"""G-restart — MPI checkpoint roundtrip vs uninterrupted run (bit).

bench5 pure-ALM (ramp ACTIVE at these steps -> maximal sensitivity to the
rotor fast-forward fix: a wrong _step_count/theta shows up immediately in
the ramp factor and blade positions):

  A: 4 coarse steps straight, checkpoint @4       -> reference npz
  B: 2 coarse steps, checkpoint @2
  C: restart from B's checkpoint, run to 4, ckpt @4

Gate: C's checkpoint f (all levels) == A's BIT-for-bit. Exactness chain:
std-f32 on disk is exact; scatter continues the esoteric parity (t0);
rotor.advance replay reproduces the fp-accumulated kinematics; bench5 ALM
partial sums are bit in practice (G-M3).

Run from repo root:  python patch_notes/hpc_upgrade/gates/mgpu_restart_gate.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CFG = "configs/hpc_bench/bench5_purealm_m3.py"
RES = os.path.join(ROOT, "result_hvab_hover_c10.0_M650_mlg5_D16_bench5_"
                         "bench5_purealm_m3")
CKPT = os.path.join(RES, "checkpoints")
SCRATCH = os.path.join(ROOT, "patch_notes", "hpc_upgrade", "gates",
                       "_restart_gate_tmp")


def run(extra):
    cmd = ["mpirun", "-n", "2", sys.executable, "main_mpi.py",
           "--config", CFG, "--log-every", "0"] + extra
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=560)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError("mpirun failed")


def stash(name):
    os.makedirs(SCRATCH, exist_ok=True)
    src = os.path.join(CKPT, name)
    dst = os.path.join(SCRATCH, name)
    shutil.move(src, dst)
    return dst


def main():
    if os.path.isdir(SCRATCH):
        shutil.rmtree(SCRATCH)
    for f in ("checkpoint_00000002.npz", "checkpoint_00000004.npz"):
        p = os.path.join(CKPT, f)
        if os.path.exists(p):
            os.remove(p)

    print("  [A] 4 steps straight, ckpt @4 ...")
    run(["--steps", "4", "--ckpt-every", "4"])
    ref = stash("checkpoint_00000004.npz")

    print("  [B] 2 steps, ckpt @2 ...")
    run(["--steps", "2", "--ckpt-every", "2"])
    b2 = stash("checkpoint_00000002.npz")

    print("  [C] restart from @2, run to 4, ckpt @4 ...")
    shutil.copy(b2, os.path.join(CKPT, "checkpoint_00000002.npz"))
    run(["--steps", "4", "--ckpt-every", "4",
         "--restart", os.path.join(CKPT, "checkpoint_00000002.npz")])

    a = np.load(ref, allow_pickle=True)
    c = np.load(os.path.join(CKPT, "checkpoint_00000004.npz"),
                allow_pickle=True)
    ok = True
    for key in ["f"] + [f"extra_f_level_{k}" for k in range(1, 5)]:
        bit = bool(np.array_equal(a[key], c[key]))
        md = float(np.max(np.abs(a[key] - c[key])))
        ok = ok and bit
        print(f"  {key}: bit={bit}  max|d|={md:.3e}")
    print("  RESULT:", "PASS" if ok else "FAIL")
    shutil.rmtree(SCRATCH)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
