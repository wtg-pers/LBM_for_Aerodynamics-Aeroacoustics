"""Phase 1a gate — Esoteric Pull BGK kernel vs standard BGK reference.

Decisive correctness test for the resurrected Esoteric Pull mechanism
(paired-opposite ordering + even/odd LOAD/STORE). A periodic Taylor-Green box,
all-FLUID, no BC. Both schemes must produce the SAME macroscopic (rho, u) at
every step entry (they are the same BGK physics, different memory layout).
If the in-place slot mapping is wrong, fields diverge grossly (not last-bit).

Also checks mass conservation and NaN-freeness.

Run from repo root on a CUDA box:
    python patch_notes/hpc_upgrade/gates/eso_bgk_equiv_gate.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp
from src.kernels.esoteric_d3q27 import (
    EsotericBGKKernelD3Q27, convert_f_std_to_esoteric, init_f_esoteric)

# ---- standard D3Q27 lattice (matches src/lattice/d3q27.py) ----
_CXl = [0, 1,-1,0,0,0,0, 1,-1,1,-1,1,-1,1,-1, 0,0,0,0, 1,-1,1,-1,1,-1,1,-1]
_CYl = [0, 0,0,1,-1,0,0, 1,1,-1,-1,0,0,0,0, 1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1]
_CZl = [0, 0,0,0,0,1,-1, 0,0,0,0,1,1,-1,-1, 1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1]
CX = cp.asarray(_CXl, cp.float32); CY = cp.asarray(_CYl, cp.float32); CZ = cp.asarray(_CZl, cp.float32)
W = cp.asarray([8/27] + [2/27]*6 + [1/54]*12 + [1/216]*8, cp.float32)
CXi, CYi, CZi = _CXl, _CYl, _CZl   # python ints for roll shifts

N_AX = 24            # domain 24^3
NSTEPS = 40
OMEGA = 1.0         # tau = 1
U0 = 0.05           # low Mach TGV amplitude


def feq_std(rho, u):
    """f_eq in standard ordering. rho (Nx,Ny,Nz), u (3,Nx,Ny,Nz) -> (27,...)."""
    cu = (CX[:, None, None, None] * u[0] + CY[:, None, None, None] * u[1]
          + CZ[:, None, None, None] * u[2])
    usq = u[0]**2 + u[1]**2 + u[2]**2
    return (W[:, None, None, None] * rho
            * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)).astype(cp.float32)


def macro(f):
    rho = cp.sum(f, axis=0)
    ux = cp.sum(CX[:, None, None, None] * f, axis=0) / rho
    uy = cp.sum(CY[:, None, None, None] * f, axis=0) / rho
    uz = cp.sum(CZ[:, None, None, None] * f, axis=0) / rho
    return rho, cp.stack([ux, uy, uz])


def tgv_ic():
    x = cp.arange(N_AX, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')
    k = 2 * np.pi / N_AX
    ux = U0 * cp.cos(k*X) * cp.sin(k*Y) * cp.sin(k*Z)
    uy = -U0 * cp.sin(k*X) * cp.cos(k*Y) * cp.sin(k*Z)
    uz = cp.zeros_like(ux)
    rho = cp.ones((N_AX, N_AX, N_AX), dtype=cp.float32)
    return rho, cp.stack([ux, uy, uz])


def run_reference(f0):
    """Standard BGK: collide + periodic pull-stream. Record (rho,u) per step."""
    f = f0.copy()
    recs = []
    for _ in range(NSTEPS):
        rho, u = macro(f)
        recs.append((rho.copy(), u.copy()))
        feq = feq_std(rho, u)
        fp = f - OMEGA * (f - feq)
        # pull streaming (periodic): f_i(x) = fp_i(x - c_i) = roll(fp_i, +c_i)
        fnew = cp.empty_like(f)
        for q in range(27):
            fnew[q] = cp.roll(fp[q], (int(CXi[q]), int(CYi[q]), int(CZi[q])),
                              axis=(0, 1, 2))
        f = fnew
    return recs


def run_esoteric(f0):
    """Esoteric Pull kernel. Record kernel-output (rho,u) per launch."""
    xp = cp
    Nx = Ny = Nz = N_AX
    f_eso = convert_f_std_to_esoteric(xp, f0)   # esoteric direction ordering
    f_mem = init_f_esoteric(xp, f_eso, t_start=0)   # now fixed in the kernel file
    node_type = xp.zeros((Nx, Ny, Nz), dtype=xp.int8).ravel()   # all FLUID
    bc = xp.zeros(Nx*Ny*Nz, dtype=xp.float32)
    bc_rho = xp.ones(Nx*Ny*Nz, dtype=xp.float32)
    rho_out = xp.empty((Nx, Ny, Nz), dtype=xp.float32)
    u_out = xp.empty((3, Nx, Ny, Nz), dtype=xp.float32)
    ker = EsotericBGKKernelD3Q27()
    recs = []
    for k in range(NSTEPS):
        ker.launch(f_mem, rho_out, u_out, node_type,
                   bc_rho, bc, bc, bc, OMEGA, Nx, Ny, Nz, t_step=k)
        recs.append((rho_out.copy(), u_out.copy()))
    return recs


def main():
    rho0, u0 = tgv_ic()
    f0 = feq_std(rho0, u0)
    m0 = float(cp.sum(f0))

    ref = run_reference(f0)
    eso = run_esoteric(f0)

    max_drho = 0.0
    max_du = 0.0
    for k in range(NSTEPS):
        rr, ur = ref[k]
        re, ue = eso[k]
        max_drho = max(max_drho, float(cp.max(cp.abs(rr - re))))
        max_du = max(max_du, float(cp.max(cp.abs(ur - ue))))

    # DIAGNOSTIC: per-step trace (does divergence start at step 0 or accumulate?)
    print("  [trace] step : max|drho|   max|du|")
    for k in [0, 1, 2, 3, 5, 10, NSTEPS-1]:
        rr, ur = ref[k]; re, ue = eso[k]
        print(f"          {k:4d} : {float(cp.max(cp.abs(rr-re))):.3e}  {float(cp.max(cp.abs(ur-ue))):.3e}")
    # is eso[0] a one-cell shift of ref[0]? test roll of u_x in +/- each axis
    rr0, ur0 = ref[0]; re0, ue0 = eso[0]
    base = float(cp.max(cp.abs(ur0[0]-ue0[0])))
    shifts = {}
    for ax in (0,1,2):
        for s in (+1,-1):
            shifts[f"ax{ax}{s:+d}"] = float(cp.max(cp.abs(cp.roll(ur0[0], s, axis=ax) - ue0[0])))
    best = min(shifts, key=shifts.get)
    print(f"  [shift-test] eso[0].ux vs ref[0].ux: base={base:.3e}, best 1-cell shift {best}={shifts[best]:.3e}")

    # conservation + stability from the esoteric run
    rho_last = eso[-1][0]
    mass_last = float(cp.sum(rho_last))
    nan = bool(cp.any(~cp.isfinite(rho_last))) or bool(cp.any(~cp.isfinite(eso[-1][1])))
    umax = float(cp.max(cp.abs(eso[-1][1])))

    print(f"[eso_bgk_equiv_gate] domain {N_AX}^3, {NSTEPS} steps, omega={OMEGA}")
    print(f"  max|rho_eso - rho_ref| = {max_drho:.3e}")
    print(f"  max|u_eso   - u_ref|   = {max_du:.3e}")
    print(f"  mass drift (eso): {(mass_last - m0)/m0:+.3e}   |u|max={umax:.4f}   NaN={nan}")

    # float32 last-bit + O(Ma^2) eq-form agreement expected -> tol generous but
    # a slot/parity bug would give O(1) divergence, easily caught.
    ok = (not nan) and (max_drho < 5e-4) and (max_du < 5e-4) and (abs((mass_last-m0)/m0) < 1e-5)
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
