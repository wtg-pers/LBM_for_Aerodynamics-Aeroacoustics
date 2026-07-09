"""Gate — cubic_interp_1d_z coalescing rewrite == original (bit-identical) + faster.

nsys (bench5_pure_lbm, 4090, 2026-07-08) showed cubic_interp_1d_z = 24.4% of the
pure-LBM step, ~5.7x its x/y peers on the SAME-size array. Cause: the fiber-per-
thread mapping (thread per (q,ix,iy), internal iz loop) put consecutive threads
on iy (stride Nz) -> uncoalesced global access. The rewrite maps thread per
(q,ix,iy,iz) with iz fastest -> coalesced. Arithmetic per node is unchanged, so
the result is bit-identical (no sha256 rebaseline).

This gate pins the reference = the ORIGINAL fiber kernel (kept inline here) vs
the CURRENT production kernel (imported), asserting max|Δ|==0 over L4-size random
data, and reports the speedup. Run from repo root on a CUDA GPU:

    python patch_notes/hpc_upgrade/gates/cubic_interp_z_coalesce_gate.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import cupy as cp
from src.kernels.interpolation_d3q27 import CubicInterpolationKernel3D

# --- reference: the ORIGINAL fiber-per-thread z-kernel (pre-coalescing) ------
_Z_OLD = r'''
extern "C" __global__
void z_old(float* __restrict__ f, const int Q, const int Nx, const int Ny, const int Nz){
    int total = Q * Nx * Ny;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int NyNz = Ny * Nz;
    long long NxNyNz = (long long)Nx * NyNz;
    int q = tid / (Nx * Ny);
    int rem = tid - q * Nx * Ny;
    int ix = rem / Ny;
    int iy = rem - ix * Ny;
    #define FZ(z) f[q * NxNyNz + ix * NyNz + iy * Nz + (z)]
    for (int iz = 3; iz <= Nz - 4; iz += 2)
        FZ(iz) = 0.5625f*(FZ(iz-1)+FZ(iz+1)) - 0.0625f*(FZ(iz-3)+FZ(iz+3));
    if (Nz>=5) FZ(1)=0.375f*FZ(0)+0.75f*FZ(2)-0.125f*FZ(4);
    else if (Nz>=3) FZ(1)=0.5f*(FZ(0)+FZ(2));
    if ((Nz-2)%2==1){ if(Nz>=5) FZ(Nz-2)=0.375f*FZ(Nz-1)+0.75f*FZ(Nz-3)-0.125f*FZ(Nz-5);
                      else if(Nz>=3) FZ(Nz-2)=0.5f*(FZ(Nz-3)+FZ(Nz-1)); }
    #undef FZ
}'''


def main():
    Q, Nx, Ny, Nz = 27, 57, 297, 297          # bench5 L4 (rotor slab) dims
    bs = 256
    kold = cp.RawKernel(_Z_OLD, 'z_old', options=('--use_fast_math',))
    knew = CubicInterpolationKernel3D()
    knew._compile()
    args = (cp.int32(Q), cp.int32(Nx), cp.int32(Ny), cp.int32(Nz))

    base = cp.random.rand(Q, Nx, Ny, Nz, dtype=cp.float32)   # even iz = coarse
    a_old, a_new = base.copy(), base.copy()
    n_old = Q * Nx * Ny
    n_new = Q * Nx * Ny * Nz
    kold(((n_old + bs - 1)//bs,), (bs,), (a_old,) + args)
    knew._kz(((n_new + bs - 1)//bs,), (bs,), (a_new,) + args)
    cp.cuda.runtime.deviceSynchronize()

    d = float(cp.abs(a_old - a_new).max())
    nbit = int((a_old.view(cp.int32) != a_new.view(cp.int32)).sum())

    def timeit(k, ntot):
        a = base.copy()
        for _ in range(3):
            k(((ntot + bs - 1)//bs,), (bs,), (a,) + args)
        cp.cuda.runtime.deviceSynchronize(); t = time.perf_counter()
        for _ in range(50):
            k(((ntot + bs - 1)//bs,), (bs,), (a,) + args)
        cp.cuda.runtime.deviceSynchronize()
        return (time.perf_counter() - t)/50*1e3

    to = timeit(kold, n_old)
    tn = timeit(knew._kz, n_new)
    print(f"[gate] cubic_interp_1d_z  bit: max|Δ|={d:.3e}  bits-differ={nbit}/{a_old.size}")
    print(f"[gate] wall (L4 dims): old={to:.3f} ms  new={tn:.3f} ms  speedup={to/tn:.2f}x")
    ok = (nbit == 0) and (tn < to)
    print("[gate] VERDICT:", "PASS (bit-identical + faster)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
