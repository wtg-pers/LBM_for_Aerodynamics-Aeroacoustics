"""Phase 1c gate — Esoteric gather/scatter (MLG coupling bridge).

1. Roundtrip BIT-EXACT (pure permutation + roll) both parities:
   scatter(gather(mem, t), t) == mem and gather(scatter(phys, t), t) == phys.
2. Physical consistency vs evolution: run the esoteric cumulant kernel from a
   TGV IC; at each step, gather_std(mem, k) must equal the reference
   (standard collide + roll-stream) f at step-k entry to fp32 tolerance.
   This pins the gather parity convention to the real time evolution.

Run from repo root:  python patch_notes/hpc_upgrade/gates/eso_gather_scatter_gate.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp
from src.kernels.cumulant_d3q27 import CumulantCollideKernelD3Q27
from src.kernels.esoteric_cumulant_d3q27 import EsotericCumulantKernelD3Q27
from src.kernels.esoteric_d3q27 import (
    esoteric_gather_physical, esoteric_scatter_physical,
    esoteric_gather_std, esoteric_scatter_std)

_CXl = [0, 1,-1,0,0,0,0, 1,-1,1,-1,1,-1,1,-1, 0,0,0,0, 1,-1,1,-1,1,-1,1,-1]
_CYl = [0, 0,0,1,-1,0,0, 1,1,-1,-1,0,0,0,0, 1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1]
_CZl = [0, 0,0,0,0,1,-1, 0,0,0,0,1,1,-1,-1, 1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1]
CX = cp.asarray(_CXl, cp.float32); CY = cp.asarray(_CYl, cp.float32); CZ = cp.asarray(_CZl, cp.float32)
W = cp.asarray([8/27] + [2/27]*6 + [1/54]*12 + [1/216]*8, cp.float32)

N_AX, NSTEPS = 20, 24


def feq(rho, u):
    cu = (CX[:, None, None, None]*u[0] + CY[:, None, None, None]*u[1] + CZ[:, None, None, None]*u[2])
    usq = u[0]**2 + u[1]**2 + u[2]**2
    return (W[:, None, None, None]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)).astype(cp.float32)


def main():
    ok = True

    # ---- 1. roundtrip bit-exact ----
    rng = cp.random.default_rng(0)
    mem = rng.random((27, 12, 12, 12), dtype=cp.float32)
    for t in (0, 1, 6, 7):
        rt = esoteric_scatter_physical(cp, esoteric_gather_physical(cp, mem, t), t)
        exact = bool(cp.all(rt == mem))
        rt2 = esoteric_gather_std(cp, esoteric_scatter_std(cp, mem, t), t)
        exact2 = bool(cp.all(rt2 == mem))
        print(f"  [roundtrip t={t}] scatter∘gather bit-exact={exact}  "
              f"gather_std∘scatter_std bit-exact={exact2}")
        ok = ok and exact and exact2

    # ---- 2. gather vs reference evolution ----
    x = cp.arange(N_AX, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')
    k2 = 2*np.pi/N_AX
    u0 = cp.stack([0.05*cp.cos(k2*X)*cp.sin(k2*Y)*cp.sin(k2*Z),
                   -0.05*cp.sin(k2*X)*cp.cos(k2*Y)*cp.sin(k2*Z),
                   cp.zeros((N_AX,)*3, cp.float32)])
    f0 = feq(cp.ones((N_AX,)*3, cp.float32), u0)

    # reference: standard collide + roll-stream; record f at each step ENTRY
    Nn = N_AX**3
    coll = CumulantCollideKernelD3Q27(sgs_model="off")
    f = f0.copy(); fp = cp.empty_like(f)
    rho = cp.empty((N_AX,)*3, cp.float32); u = cp.empty((3,)+(N_AX,)*3, cp.float32)
    f_ref = []
    for _ in range(NSTEPS):
        f_ref.append(f.copy())
        coll.launch(f, fp, rho, u, None, 1.0, 1.0, 1.0, Nn)
        fnew = cp.empty_like(f)
        for q in range(27):
            fnew[q] = cp.roll(fp[q], (_CXl[q], _CYl[q], _CZl[q]), axis=(0, 1, 2))
        f = fnew

    # esoteric: gather at each step entry, compare
    mem = esoteric_scatter_std(cp, f0, 0)
    nt = cp.zeros(Nn, cp.int8); bc = cp.zeros(Nn, cp.float32); bcr = cp.ones(Nn, cp.float32)
    ker = EsotericCumulantKernelD3Q27()
    max_df = 0.0
    for k in range(NSTEPS):
        g = esoteric_gather_std(cp, mem, k)
        max_df = max(max_df, float(cp.max(cp.abs(g - f_ref[k]))))
        ker.launch(mem, rho, u, nt, bcr, bc, bc, bc, 1.0, 1.0, 1.0,
                   N_AX, N_AX, N_AX, t_step=k, force=None)
    print(f"  [gather-vs-ref] max|gather_std(mem,k) - f_ref[k]| over {NSTEPS} steps = {max_df:.3e}")
    ok = ok and (max_df < 5e-6)

    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
