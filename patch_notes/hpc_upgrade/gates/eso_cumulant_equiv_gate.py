"""Phase 1b gate (DECISIVE) — Esoteric Pull + cumulant vs standard fused cumulant.

The esoteric-cumulant kernel grafts the SAME cumulant collision (steps 4-8) into
the Esoteric Pull layout. Against the validated standard fused cumulant collide
(CumulantCollideKernelD3Q27) + periodic pull streaming, macroscopic (rho,u) must
match at every step entry to float32 last-bit. Any K-mapping / collision
transcription error gives O(1) divergence.

Run from repo root on a CUDA box:
    python patch_notes/hpc_upgrade/gates/eso_cumulant_equiv_gate.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp
from src.kernels.cumulant_d3q27 import CumulantCollideKernelD3Q27
from src.kernels.esoteric_cumulant_d3q27 import EsotericCumulantKernelD3Q27
from src.kernels.esoteric_d3q27 import convert_f_std_to_esoteric, init_f_esoteric

_CXl = [0, 1,-1,0,0,0,0, 1,-1,1,-1,1,-1,1,-1, 0,0,0,0, 1,-1,1,-1,1,-1,1,-1]
_CYl = [0, 0,0,1,-1,0,0, 1,1,-1,-1,0,0,0,0, 1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1]
_CZl = [0, 0,0,0,0,1,-1, 0,0,0,0,1,1,-1,-1, 1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1]
CX = cp.asarray(_CXl, cp.float32); CY = cp.asarray(_CYl, cp.float32); CZ = cp.asarray(_CZl, cp.float32)
W = cp.asarray([8/27] + [2/27]*6 + [1/54]*12 + [1/216]*8, cp.float32)

N_AX = 24
NSTEPS = 40
OMEGA1 = 1.0
OMEGA_BULK = 1.0
OMEGA_HIGH = 1.0
U0 = 0.05


def feq_std(rho, u):
    cu = (CX[:, None, None, None]*u[0] + CY[:, None, None, None]*u[1] + CZ[:, None, None, None]*u[2])
    usq = u[0]**2 + u[1]**2 + u[2]**2
    return (W[:, None, None, None]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)).astype(cp.float32)


def tgv_ic():
    x = cp.arange(N_AX, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')
    k = 2*np.pi/N_AX
    ux = U0*cp.cos(k*X)*cp.sin(k*Y)*cp.sin(k*Z)
    uy = -U0*cp.sin(k*X)*cp.cos(k*Y)*cp.sin(k*Z)
    uz = cp.zeros_like(ux)
    return cp.ones((N_AX,)*3, cp.float32), cp.stack([ux, uy, uz])


def run_reference(f0):
    """Standard fused cumulant collide + periodic pull streaming."""
    Nx = Ny = Nz = N_AX
    N = Nx*Ny*Nz
    coll = CumulantCollideKernelD3Q27(sgs_model="off")
    f = f0.copy()
    fp = cp.empty_like(f)
    rho = cp.empty((Nx, Ny, Nz), cp.float32)
    u = cp.empty((3, Nx, Ny, Nz), cp.float32)
    recs = []
    for _ in range(NSTEPS):
        coll.launch(f, fp, rho, u, None, OMEGA1, OMEGA_BULK, OMEGA_HIGH, N)
        recs.append((rho.copy(), u.copy()))
        fnew = cp.empty_like(f)
        for q in range(27):
            fnew[q] = cp.roll(fp[q], (_CXl[q], _CYl[q], _CZl[q]), axis=(0, 1, 2))
        f = fnew
    return recs


def run_esoteric(f0):
    Nx = Ny = Nz = N_AX
    N = Nx*Ny*Nz
    f_eso = convert_f_std_to_esoteric(cp, f0)
    f_mem = init_f_esoteric(cp, f_eso, t_start=0)
    nt = cp.zeros(N, cp.int8)
    bc = cp.zeros(N, cp.float32); bcr = cp.ones(N, cp.float32)
    rho = cp.empty((Nx, Ny, Nz), cp.float32)
    u = cp.empty((3, Nx, Ny, Nz), cp.float32)
    ker = EsotericCumulantKernelD3Q27()
    recs = []
    for k in range(NSTEPS):
        ker.launch(f_mem, rho, u, nt, bcr, bc, bc, bc,
                   OMEGA1, OMEGA_BULK, OMEGA_HIGH, Nx, Ny, Nz, t_step=k, force=None)
        recs.append((rho.copy(), u.copy()))
    return recs


def main():
    rho0, u0 = tgv_ic()
    f0 = feq_std(rho0, u0)
    m0 = float(cp.sum(f0))
    ref = run_reference(f0)
    eso = run_esoteric(f0)

    max_drho = max(float(cp.max(cp.abs(ref[k][0]-eso[k][0]))) for k in range(NSTEPS))
    max_du = max(float(cp.max(cp.abs(ref[k][1]-eso[k][1]))) for k in range(NSTEPS))
    print("  [trace] step : max|drho|   max|du|")
    for k in [0, 1, 2, 5, 10, NSTEPS-1]:
        print(f"          {k:4d} : {float(cp.max(cp.abs(ref[k][0]-eso[k][0]))):.3e}  "
              f"{float(cp.max(cp.abs(ref[k][1]-eso[k][1]))):.3e}")

    rho_last = eso[-1][0]
    mass = float(cp.sum(rho_last))
    nan = bool(cp.any(~cp.isfinite(rho_last))) or bool(cp.any(~cp.isfinite(eso[-1][1])))
    print(f"[eso_cumulant_equiv_gate] {N_AX}^3, {NSTEPS} steps, w1={OMEGA1}")
    print(f"  max|rho_eso-rho_ref|={max_drho:.3e}  max|u_eso-u_ref|={max_du:.3e}")
    print(f"  mass drift(eso)={(mass-m0)/m0:+.3e}  |u|max={float(cp.max(cp.abs(eso[-1][1]))):.4f}  NaN={nan}")
    ok = (not nan and max_drho < 5e-4 and max_du < 5e-4 and abs((mass-m0)/m0) < 1e-5)
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
