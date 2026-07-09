"""Fused D3Q27 coupling rescale kernel (Phase 1c Target 2).

The MLG coarse<->fine transfer rescales the non-equilibrium part of f between
levels. In Python that is a long chain of cupy elementwise kernels, each reading
and writing the full (Q, region) array to DRAM:

    rho   = sum_q f_q                        # cupy_sum
    mom_d = sum_q c[d,q] f_q                  # einsum -> cuBLAS sgemm
    u     = mom / rho                         # cupy_true_divide
    cu    = c . u                             # einsum -> sgemm
    f_eq  = w rho (1 + cu/cs2 + ...)          # several cupy add/mul/div
    out   = f_eq + factor (f - f_eq)          # subtract, multiply, add

nsys (4090, bench5_pure_lbm 2026-07-08) attributed ~39.5% of the pure-LBM step
to this unfused elementwise chain (add 1950, copy 4290, mul/div/sub ...). It is
bandwidth-bound: each op is ~1 FLOP per 2-3 DRAM words.

This kernel fuses the whole pointwise pipeline into ONE pass: read f (27) once,
compute rho/u/f_eq in registers, write out (27) once. Result differs from the
Python chain only at fp32 last-bit (reduction order, div-vs-mul) -> physically
equivalent (CV-band gate; 4090 pure-LBM sha is run-to-run non-deterministic
anyway, see 11 E.2). D3Q27 (Q=27) and float32 only; caller falls back to the
Python path otherwise.
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy.typing as npt


# NQ literal (compile-time) so the loops unroll and f[27] stays in registers.
_RESCALE_SRC = r'''
#define NQ 27
extern "C" __global__
void rescale_c2f(
    const float* __restrict__ f_now,   // (Q, N) contiguous
    const float* __restrict__ f_prev,  // (Q, N) contiguous (== f_now if !half)
    const int   half_step,             // 1 -> use 0.5*(f_prev + f_now)
    float* __restrict__ out,           // (Q, N); may alias f_now
    const float* __restrict__ cx,      // (Q)
    const float* __restrict__ cy,
    const float* __restrict__ cz,
    const float* __restrict__ ww,      // (Q) lattice weights
    const float cs2,
    const float factor,                // rescaling factor (c2f)
    const long long N
) {
    long long node = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;

    float f[NQ];
    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        long long idx = (long long)q * N + node;
        float fv = f_now[idx];
        if (half_step) fv = 0.5f * (f_prev[idx] + fv);   // temporal interpolation
        f[q] = fv;
        rho += fv;
        mx += cx[q] * fv;  my += cy[q] * fv;  mz += cz[q] * fv;
    }

    float inv_rho = 1.0f / rho;
    float ux = mx * inv_rho, uy = my * inv_rho, uz = mz * inv_rho;
    float usqr = ux*ux + uy*uy + uz*uz;
    float inv_cs2 = 1.0f / cs2;

    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float cu = cx[q]*ux + cy[q]*uy + cz[q]*uz;
        float feq = ww[q] * rho * (1.0f + cu*inv_cs2
                    + 0.5f*cu*cu*inv_cs2*inv_cs2 - 0.5f*usqr*inv_cs2);
        out[(long long)q * N + node] = feq + factor * (f[q] - feq);
    }
}
'''


class CouplingRescaleKernelD3Q27:
    """Fused per-node rescale for the D3Q27 MLG coupling (float32, cupy).

    Replaces coupling._compute_macroscopic + _compute_f_eq + f_neq/reconstruct
    (and the optional temporal interpolation) with a single kernel launch.
    """

    def __init__(self, c: 'npt.NDArray', w: 'npt.NDArray', cs2: float,
                 block_size: int = 256) -> None:
        import cupy as cp
        # c: (dim, Q); split into contiguous float32 per-axis arrays.
        self._cx = cp.ascontiguousarray(c[0].astype(cp.float32))
        self._cy = cp.ascontiguousarray(c[1].astype(cp.float32))
        self._cz = cp.ascontiguousarray(c[2].astype(cp.float32))
        self._w = cp.ascontiguousarray(w.astype(cp.float32))
        self._cs2 = float(cs2)
        self._bs = int(block_size)
        self._kernel = cp.RawKernel(_RESCALE_SRC, 'rescale_c2f',
                                    options=('--use_fast_math',))

    def c2f(self, f_now: 'npt.NDArray', f_prev: Optional['npt.NDArray'],
            half_step: bool, factor: float,
            out: Optional['npt.NDArray'] = None) -> 'npt.NDArray':
        """coarse_nodes = f_eq + factor*(f_sub - f_eq), where f_sub is f_now
        (or 0.5*(f_prev+f_now) if half_step). Writes into `out` (defaults to
        in-place on f_now) and returns it. Inputs must be (Q, ...) contiguous.
        """
        import cupy as cp
        if out is None:
            out = f_now
        Q = f_now.shape[0]
        N = f_now.size // Q                       # spatial node count
        prev = f_prev if (half_step and f_prev is not None) else f_now
        grid = (N + self._bs - 1) // self._bs
        self._kernel(
            (grid,), (self._bs,),
            (f_now, prev, cp.int32(1 if half_step else 0), out,
             self._cx, self._cy, self._cz, self._w,
             cp.float32(self._cs2), cp.float32(factor), cp.int64(N)),
        )
        return out
