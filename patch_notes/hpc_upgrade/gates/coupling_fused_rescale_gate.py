"""Gate — fused C->F rescale kernel == Python rescale (CV-band) + faster.

Compares CouplingRescaleKernelD3Q27.c2f against the exact Python pipeline it
replaces (coupling._compute_macroscopic + _compute_f_eq + f_neq/reconstruct,
incl. the half-step temporal interpolation), on ONE call with identical input.
Physical equivalence expected at fp32 last-bit (~1e-6); NOT bit-identical (the
fused kernel reduces rho/momentum in a different order and uses mul-by-inverse).
Also reports the speedup on an L4-coarse-sub-sized region.

Run from repo root on a CUDA GPU:
    python patch_notes/hpc_upgrade/gates/coupling_fused_rescale_gate.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import cupy as cp
from src.lattice.d3q27 import D3Q27
from src.kernels.coupling_rescale_d3q27 import CouplingRescaleKernelD3Q27


def py_rescale(f_sub, f_prev, half, c, w, cs2, factor):
    """Exact Python pipeline (matches coupling.coarse_to_fine steps 2-4+6)."""
    if half:
        f_sub = 0.5 * (f_prev + f_sub)
    rho = f_sub.sum(axis=0)
    mom = cp.einsum('dq,q...->d...', c, f_sub)
    u = mom / rho
    cu = cp.einsum('dq,d...->q...', c, u)
    usqr = (u * u).sum(axis=0)
    w_bc = w.reshape((-1,) + (1,) * 3)
    f_eq = w_bc * rho * (1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2)
                         - 0.5 * usqr / cs2)
    return f_eq + factor * (f_sub - f_eq)


def main():
    lat = D3Q27(cp)
    c = cp.asarray(lat.c, dtype=cp.float32)      # (3, 27)
    w = cp.asarray(lat.w, dtype=cp.float32)      # (27,)
    cs2 = float(lat.cs2)
    factor = 0.5312                              # representative c2f factor
    ker = CouplingRescaleKernelD3Q27(c, w, cs2)

    # near-equilibrium synthetic f: rho~1, small u, small f^neq perturbation
    Nx, Ny, Nz = 30, 150, 150                    # ~L4 coarse-sub size
    rng = cp.random.RandomState(0)
    base = w[:, None, None, None] * (1.0 + 0.05 * rng.rand(27, Nx, Ny, Nz).astype(cp.float32))
    base = base.astype(cp.float32)
    prev = (base * (1.0 + 0.02 * rng.rand(27, Nx, Ny, Nz).astype(cp.float32))).astype(cp.float32)

    worst = 0.0
    for half in (False, True):
        ref = py_rescale(base.copy(), prev.copy(), half, c, w, cs2, factor)
        got = ker.c2f(base.copy(), prev.copy() if half else None, half, factor)
        d = float(cp.abs(ref - got).max())
        rel = float((cp.abs(ref - got) / (cp.abs(ref) + 1e-30)).max())
        worst = max(worst, d)
        print(f"[gate] half_step={half!s:5s}  max|Δ|={d:.3e}  max rel={rel:.3e}")

    # timing (non-half): fused kernel vs Python chain
    def t_py():
        a = base.copy()
        for _ in range(3):
            py_rescale(a, None, False, c, w, cs2, factor)
        cp.cuda.runtime.deviceSynchronize(); t0 = time.perf_counter()
        for _ in range(50):
            py_rescale(a, None, False, c, w, cs2, factor)
        cp.cuda.runtime.deviceSynchronize(); return (time.perf_counter() - t0)/50*1e3

    def t_fused():
        for _ in range(3):
            ker.c2f(base.copy(), None, False, factor)
        cp.cuda.runtime.deviceSynchronize(); t0 = time.perf_counter()
        for _ in range(50):
            ker.c2f(base.copy(), None, False, factor)
        cp.cuda.runtime.deviceSynchronize(); return (time.perf_counter() - t0)/50*1e3

    tp, tf = t_py(), t_fused()
    print(f"[gate] rescale wall ({Nx}x{Ny}x{Nz}): python={tp:.3f} ms  fused={tf:.3f} ms"
          f"  speedup={tp/tf:.2f}x  (incl. base.copy in both)")

    ok = (worst < 1e-4) and (tf < tp)
    print("[gate] VERDICT:", "PASS (CV-band equivalent + faster)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
