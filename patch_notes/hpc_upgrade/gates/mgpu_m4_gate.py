"""G-M4 — esoteric halo v2 (raw slot-subset, ghost=1) vs 1-rank, 3 axes.

Same single-level cumulant TGV as G-M1. v2 differences under test:
  - ghost = 1 (v1 needs 2), ghost planes marked SOLID -> never collided
    (pure transit mailboxes; v1 recomputes them redundantly);
  - exchange AFTER the kernel step: per face only the 9 axis-crossing
    populations/cell in two same-slot-copy groups (A: STORE deposits in our
    ghost -> neighbor's owned edge; B: residents at our edge that his next
    LOAD reads -> his ghost);
  - measured traffic vs v1 printed (expected 6x: 9 vs 2x27 slot-planes).

Owned cells must match the single-rank run BIT-FOR-BIT.

Run from repo root:  python patch_notes/hpc_upgrade/gates/mgpu_m4_gate.py
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import cupy as cp

_g = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mgpu_m1_gate.py")
_src = open(_g).read().replace('if __name__ == "__main__":\n    main()', '')
m1 = types.ModuleType("m1"); m1.__dict__['__file__'] = _g
exec(compile(_src, _g, "exec"), m1.__dict__)

from src.kernels.esoteric_cumulant_d3q27 import EsotericCumulantKernelD3Q27
from src.kernels.esoteric_d3q27 import esoteric_gather_std, esoteric_scatter_std
from src.parallel import Partition1D, LoopbackTransport
from src.parallel.halo_v2 import SlotHaloExchangerV2

N_AX, NSTEPS, NR = m1.N_AX, m1.NSTEPS, m1.NR


class CountingTransport(LoopbackTransport):
    def __init__(self):
        super().__init__()
        self.bytes = 0

    def post(self, src, dst, tag, arr):
        self.bytes += arr.nbytes
        super().post(src, dst, tag, arr)


def run_split_v2(f0, axis):
    parts = [Partition1D((N_AX,)*3, NR, r, axis=axis, ghost=1)
             for r in range(NR)]
    lb = CountingTransport()
    exch = [SlotHaloExchangerV2(parts[r], lb, cp) for r in range(NR)]
    ker = EsotericCumulantKernelD3Q27()
    mems, rhos, us, aux = [], [], [], []
    for p in parts:
        lo = p.own_start - p.ghost
        idx = np.arange(lo, lo + p.local_shape[axis])
        f_loc = m1.wrap_take(f0, idx, axis)
        mems.append(esoteric_scatter_std(cp, f_loc, 0))
        nx, ny, nz = p.local_shape
        # ghost planes SOLID: kernel skips them entirely (transit mailboxes)
        nt3 = cp.zeros((nx, ny, nz), cp.int8)
        gsl = [slice(None)] * 3
        gsl[axis] = 0
        nt3[tuple(gsl)] = 1
        gsl[axis] = p.local_shape[axis] - 1
        nt3[tuple(gsl)] = 1
        Nl = nx * ny * nz
        rhos.append(cp.empty((nx, ny, nz), cp.float32))
        us.append(cp.empty((3, nx, ny, nz), cp.float32))
        aux.append((nt3.ravel(), cp.ones(Nl, cp.float32),
                    cp.zeros(Nl, cp.float32)))
    for k in range(NSTEPS):
        for r in range(NR):                          # kernel first (v2!)
            nx, ny, nz = parts[r].local_shape
            nt, bcr, bc = aux[r]
            ker.launch(mems[r], rhos[r], us[r], nt, bcr, bc, bc, bc,
                       m1.W1, m1.WB, m1.WH, nx, ny, nz, t_step=k, force=None)
        for r in range(NR):                          # then slot exchange
            exch[r].post(mems[r], k)
        for r in range(NR):
            exch[r].complete(mems[r], k)
    f_parts, rho_parts, u_parts = [], [], []
    for r in range(NR):
        sl = parts[r].owned_local()
        f_parts.append(esoteric_gather_std(cp, mems[r], NSTEPS)
                       [(slice(None),) + sl])
        rho_parts.append(rhos[r][sl])
        u_parts.append(us[r][(slice(None),) + sl])
    cat = lambda lst, off: cp.concatenate(lst, axis=axis + off)
    return cat(f_parts, 1), cat(rho_parts, 0), cat(u_parts, 1), lb.bytes


def v1_bytes(f0, axis):
    """Measured v1 traffic on the same case (counting transport)."""
    from src.parallel import HaloBandExchangerV1
    parts = [Partition1D((N_AX,)*3, NR, r, axis=axis) for r in range(NR)]
    lb = CountingTransport()
    exch = [HaloBandExchangerV1(parts[r], lb, cp) for r in range(NR)]
    ker = EsotericCumulantKernelD3Q27()
    mems, aux = [], []
    for p in parts:
        lo = p.own_start - p.ghost
        idx = np.arange(lo, lo + p.local_shape[axis])
        mems.append(esoteric_scatter_std(cp, m1.wrap_take(f0, idx, axis), 0))
        nx, ny, nz = p.local_shape
        Nl = nx * ny * nz
        aux.append((cp.zeros(Nl, cp.int8), cp.ones(Nl, cp.float32),
                    cp.zeros(Nl, cp.float32),
                    cp.empty((nx, ny, nz), cp.float32),
                    cp.empty((3, nx, ny, nz), cp.float32)))
    for k in range(NSTEPS):
        for r in range(NR):
            exch[r].post(mems[r], k)
        for r in range(NR):
            exch[r].complete(mems[r], k)
        for r in range(NR):
            nx, ny, nz = parts[r].local_shape
            nt, bcr, bc, rho, u = aux[r]
            ker.launch(mems[r], rho, u, nt, bcr, bc, bc, bc,
                       m1.W1, m1.WB, m1.WH, nx, ny, nz, t_step=k, force=None)
    return lb.bytes


def main():
    f0 = m1.f0_global()
    f_ref, rho_ref, u_ref = m1.run_single(f0)
    ok = True
    for axis, name in ((0, "x"), (1, "y"), (2, "z")):
        f_s, rho_s, u_s, b2 = run_split_v2(f0, axis)
        b1 = v1_bytes(f0, axis)
        bit_f = bool(cp.all(f_s == f_ref))
        bit_r = bool(cp.all(rho_s == rho_ref))
        bit_u = bool(cp.all(u_s == u_ref))
        md = float(cp.max(cp.abs(f_s - f_ref)))
        print(f"  axis={name}: f bit={bit_f}  rho bit={bit_r}  u bit={bit_u}"
              f"  (max|df|={md:.2e})  traffic v1/v2={b1/b2:.2f}x"
              f"  ({b1/NSTEPS/1024:.0f} -> {b2/NSTEPS/1024:.0f} KiB/step)")
        ok = ok and bit_f and bit_r and bit_u
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
