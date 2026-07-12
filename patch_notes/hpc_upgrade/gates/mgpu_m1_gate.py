"""G-M1 — multi-GPU M1: 2-rank domain split vs 1-rank, ALL THREE AXES.

Single-level esoteric cumulant periodic box (TGV IC), loopback transport
(in-process 2 "ranks", same code path as MPI except the wire). The v1
physical-band halo repeats only exact per-cell kernel work and exact
permutation primitives, so owned cells must match the single-rank run
BIT-FOR-BIT on the deterministic local 3090.

Run from repo root:  python patch_notes/hpc_upgrade/gates/mgpu_m1_gate.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp
from src.kernels.esoteric_cumulant_d3q27 import EsotericCumulantKernelD3Q27
from src.kernels.esoteric_d3q27 import esoteric_gather_std, esoteric_scatter_std
from src.parallel import Partition1D, HaloBandExchangerV1, LoopbackTransport

N_AX, NSTEPS, NR = 24, 16, 2
W1 = WB = WH = 1.0

_CXl = [0, 1,-1,0,0,0,0, 1,-1,1,-1,1,-1,1,-1, 0,0,0,0, 1,-1,1,-1,1,-1,1,-1]
_CYl = [0, 0,0,1,-1,0,0, 1,1,-1,-1,0,0,0,0, 1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1]
_CZl = [0, 0,0,0,0,1,-1, 0,0,0,0,1,1,-1,-1, 1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1]
CX = cp.asarray(_CXl, cp.float32); CY = cp.asarray(_CYl, cp.float32); CZ = cp.asarray(_CZl, cp.float32)
W = cp.asarray([8/27] + [2/27]*6 + [1/54]*12 + [1/216]*8, cp.float32)


def f0_global():
    x = cp.arange(N_AX, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')
    k = 2*np.pi/N_AX
    u = cp.stack([0.05*cp.cos(k*X)*cp.sin(k*Y)*cp.sin(k*Z),
                  -0.05*cp.sin(k*X)*cp.cos(k*Y)*cp.sin(k*Z),
                  cp.zeros((N_AX,)*3, cp.float32)])
    cu = (CX[:, None, None, None]*u[0] + CY[:, None, None, None]*u[1]
          + CZ[:, None, None, None]*u[2])
    usq = u[0]**2 + u[1]**2 + u[2]**2
    return (W[:, None, None, None]*1.0*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)
            ).astype(cp.float32)


def wrap_take(a, idx, axis):
    """a[..., idx, ...] with periodic wrap along spatial `axis` of (27,X,Y,Z)."""
    return cp.take(a, cp.asarray(idx) % a.shape[1 + axis], axis=1 + axis)


def run_single(f0):
    ker = EsotericCumulantKernelD3Q27()
    mem = esoteric_scatter_std(cp, f0, 0)
    n = N_AX
    nt = cp.zeros(n**3, cp.int8); bc = cp.zeros(n**3, cp.float32)
    bcr = cp.ones(n**3, cp.float32)
    rho = cp.empty((n,)*3, cp.float32); u = cp.empty((3,)+(n,)*3, cp.float32)
    for k in range(NSTEPS):
        ker.launch(mem, rho, u, nt, bcr, bc, bc, bc, W1, WB, WH,
                   n, n, n, t_step=k, force=None)
    return esoteric_gather_std(cp, mem, NSTEPS), rho, u


def run_split(f0, axis):
    parts = [Partition1D((N_AX,)*3, NR, r, axis=axis) for r in range(NR)]
    lb = LoopbackTransport()
    exch = [HaloBandExchangerV1(parts[r], lb, cp) for r in range(NR)]
    ker = EsotericCumulantKernelD3Q27()
    mems, rhos, us, aux = [], [], [], []
    for p in parts:
        # local physical slice = owned +- ghost, periodic wrap along axis
        lo = p.own_start - p.ghost
        idx = np.arange(lo, lo + p.local_shape[axis])
        f_loc = wrap_take(f0, idx, axis)
        mems.append(esoteric_scatter_std(cp, f_loc, 0))
        nx, ny, nz = p.local_shape
        Nl = nx * ny * nz
        rhos.append(cp.empty((nx, ny, nz), cp.float32))
        us.append(cp.empty((3, nx, ny, nz), cp.float32))
        aux.append((cp.zeros(Nl, cp.int8), cp.ones(Nl, cp.float32),
                    cp.zeros(Nl, cp.float32)))
    for k in range(NSTEPS):
        for r in range(NR):                      # phase A: gather+post (all)
            exch[r].post(mems[r], k)
        for r in range(NR):                      # phase B: recv+scatter (all)
            exch[r].complete(mems[r], k)
        for r in range(NR):                      # kernel (owned + ghosts)
            nx, ny, nz = parts[r].local_shape
            nt, bcr, bc = aux[r]
            ker.launch(mems[r], rhos[r], us[r], nt, bcr, bc, bc, bc,
                       W1, WB, WH, nx, ny, nz, t_step=k, force=None)
    # assemble owned regions in global order
    f_parts, rho_parts, u_parts = [], [], []
    for r in range(NR):
        sl = parts[r].owned_local()
        f_parts.append(esoteric_gather_std(cp, mems[r], NSTEPS)[(slice(None),) + sl])
        rho_parts.append(rhos[r][sl])
        u_parts.append(us[r][(slice(None),) + sl])
    cat = lambda lst, off: cp.concatenate(lst, axis=axis + off)
    return cat(f_parts, 1), cat(rho_parts, 0), cat(u_parts, 1)


def main():
    f0 = f0_global()
    f_ref, rho_ref, u_ref = run_single(f0)
    ok = True
    for axis, name in ((0, "x"), (1, "y"), (2, "z")):
        f_s, rho_s, u_s = run_split(f0, axis)
        bit_f = bool(cp.all(f_s == f_ref))
        bit_r = bool(cp.all(rho_s == rho_ref))
        bit_u = bool(cp.all(u_s == u_ref))
        md = float(cp.max(cp.abs(f_s - f_ref)))
        print(f"  axis={name}: f bit={bit_f}  rho bit={bit_r}  u bit={bit_u}"
              f"  (max|df|={md:.2e})")
        ok = ok and bit_f and bit_r and bit_u
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
