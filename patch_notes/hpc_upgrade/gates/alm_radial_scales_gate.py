"""Gate — compute_radial_scales_batch vs the CPU reference (radial truncation).

D40-like geometry: 4 blades x 64 markers on radial lines, root-cut/tip per the
HVAB setup print (r 82.5..320 lu, eps 6.56 inboard tapering to ~4.2 at tip),
random azimuths. Batch (numpy and cupy) must match the reference to fp64
round-off (sum-order only). Also times reference vs GPU batch.

Run from repo root:  python patch_notes/hpc_upgrade/gates/alm_radial_scales_gate.py
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np
import cupy as cp
from src.actuator.spreading import compute_radial_scales, compute_radial_scales_batch

DOM = (260, 480, 480)          # generous box (markers well inside)
HUB = np.array([130.0, 240.0, 240.0])
AXIS = np.array([1.0, 0.0, 0.0])
R_ROOT, R_TIP = 80.6, 320.0
N_PER, NB = 64, 4


def geometry(seed):
    rng = np.random.default_rng(seed)
    r = np.linspace(82.5, 320.0, N_PER)
    eps = np.where(r < 304, 6.5564, np.linspace(6.5564, 4.17, N_PER)[None].ravel())
    pos, ep = [], []
    for b in range(NB):
        th = 2*np.pi*b/NB + rng.uniform(0, 2*np.pi/NB)
        ey = np.array([0, np.cos(th), np.sin(th)])
        pos.append(HUB[None, :] + r[:, None]*ey[None, :]
                   + rng.normal(0, 0.05, (N_PER, 3)))     # slight jitter
        ep.append(eps)
    return np.vstack(pos), np.hstack(ep)


ok = True
for seed in (0, 1, 2):
    pos, eps = geometry(seed)
    act = np.ones(len(pos), bool)
    ref = compute_radial_scales(DOM, pos, eps, act, AXIS, HUB, R_ROOT, R_TIP)
    b_np = compute_radial_scales_batch(np, DOM, pos, eps, act, AXIS, HUB, R_ROOT, R_TIP)
    b_cp = cp.asnumpy(compute_radial_scales_batch(cp, DOM, pos, eps, act, AXIS, HUB, R_ROOT, R_TIP))
    n_trunc = int(np.sum(ref != 1.0))
    d_np = float(np.max(np.abs(b_np - ref) / np.maximum(ref, 1e-30)))
    d_cp = float(np.max(np.abs(b_cp - ref) / np.maximum(ref, 1e-30)))
    print(f"  seed{seed}: truncated markers={n_trunc}/{len(pos)}  "
          f"max_rel: batch_np={d_np:.2e}  batch_cp={d_cp:.2e}  "
          f"scale range=[{ref.min():.4f},{ref.max():.4f}]")
    ok = ok and d_np < 1e-12 and d_cp < 1e-12 and n_trunc > 0

# timing (D40-like): reference host loop vs GPU batch
pos, eps = geometry(0); act = np.ones(len(pos), bool)
t0 = time.perf_counter()
for _ in range(10):
    compute_radial_scales(DOM, pos, eps, act, AXIS, HUB, R_ROOT, R_TIP)
t_ref = (time.perf_counter() - t0) / 10
_ = compute_radial_scales_batch(cp, DOM, pos, eps, act, AXIS, HUB, R_ROOT, R_TIP)
cp.cuda.runtime.deviceSynchronize()
t0 = time.perf_counter()
for _ in range(10):
    compute_radial_scales_batch(cp, DOM, pos, eps, act, AXIS, HUB, R_ROOT, R_TIP)
cp.cuda.runtime.deviceSynchronize()
t_gpu = (time.perf_counter() - t0) / 10
print(f"  timing: reference(host)={t_ref*1e3:.1f} ms  batch(gpu)={t_gpu*1e3:.1f} ms  "
      f"speedup {t_ref/t_gpu:.1f}x")
print("  RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
