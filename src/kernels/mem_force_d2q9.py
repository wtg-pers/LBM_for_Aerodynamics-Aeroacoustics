"""
CUDA Momentum Exchange Method (MEM) Force Kernels for D2Q9

2D analogue of `mem_force_d3q27.py`. Replaces the Python loops in
`force_calculator.py::MomentumExchangeForce.compute()` (HWBB) and
`._compute_ibb()` (Bouzidi linear IBB).

Two variants:

    HWBB:
        F_d = Σ_links 2 · c_i[d] · f̃_i(x_f)

    Bouzidi IBB (linear, stationary wall):
        q ≥ 1/2:
            F_d += c_i[d] · [ ((2q+1)/(2q)) · f̃_i(x_f)
                            + ((2q-1)/(2q)) · f̃_ī(x_f) ]
        q < 1/2:
            F_d += c_i[d] · [ (1+2q) · f̃_i(x_f)
                            + (1-2q) · f̃_i(x_f − c_i) ]

Both reduce to 2·c_i[d]·f̃_i at q=0.5, so the IBB kernel is a strict
super-set of the HWBB kernel — but we ship both because HWBB cases do
not allocate q_fraction.

Strategy:
    Each thread = one spatial node, accumulates per-direction contributions
    locally, then atomicAdd into the (2,) output.

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import numpy.typing as npt


_MEM_FORCE_D2Q9_KERNEL = r'''
extern "C" __global__
void mem_force_hwbb_d2q9(
    const float* __restrict__ f_post,        // (9, N)
    const bool*  __restrict__ needs_bounce,  // (9, N)
    float*       __restrict__ force_xy,      // (2,) atomic
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    const int cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

    float fx = 0.0f, fy = 0.0f;

    #pragma unroll
    for (int q = 1; q < 9; q++) {
        if (needs_bounce[q * N + idx]) {
            float fq = f_post[q * N + idx];
            fx += 2.0f * (float)cx[q] * fq;
            fy += 2.0f * (float)cy[q] * fq;
        }
    }

    if (fx != 0.0f) atomicAdd(&force_xy[0], fx);
    if (fy != 0.0f) atomicAdd(&force_xy[1], fy);
}

extern "C" __global__
void mem_force_ibb_d2q9(
    const float* __restrict__ f_post,        // (9, N)
    const bool*  __restrict__ needs_bounce,  // (9, N)
    const float* __restrict__ q_fraction,    // (9, N) q in (0, 1]
    float*       __restrict__ force_xy,      // (2,) atomic
    const int Nx, const int Ny
) {
    int N = Nx * Ny;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int ix = idx / Ny;
    int iy = idx - ix * Ny;

    const int cx[9]  = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    const int cy[9]  = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
    const int opp[9] = { 0,  3,  4,  1,  2,  7,  8,  5,  6 };

    float fx = 0.0f, fy = 0.0f;

    #pragma unroll
    for (int q = 1; q < 9; q++) {
        if (!needs_bounce[q * N + idx]) continue;

        float qv = q_fraction[q * N + idx];
        int   oq = opp[q];
        float fp_q = f_post[q * N + idx];

        float f_link;
        if (qv >= 0.5f) {
            // Local only
            float two_q = 2.0f * qv;
            float inv_2q = 1.0f / two_q;
            float fp_oq = f_post[oq * N + idx];
            f_link = (two_q + 1.0f) * inv_2q * fp_q
                   + (two_q - 1.0f) * inv_2q * fp_oq;
        } else {
            // Upstream f_post[q, x_f - c_i]
            int up_ix = (ix - cx[q] + Nx) % Nx;
            int up_iy = (iy - cy[q] + Ny) % Ny;
            int up_idx = up_ix * Ny + up_iy;
            float fp_q_up = f_post[q * N + up_idx];
            f_link = (1.0f + 2.0f * qv) * fp_q
                   + (1.0f - 2.0f * qv) * fp_q_up;
        }

        fx += (float)cx[q] * f_link;
        fy += (float)cy[q] * f_link;
    }

    if (fx != 0.0f) atomicAdd(&force_xy[0], fx);
    if (fy != 0.0f) atomicAdd(&force_xy[1], fy);
}
'''


class MEMForceKernelD2Q9:
    """CUDA HWBB MEM force kernel for D2Q9.

    Usage:
        >>> kernel = MEMForceKernelD2Q9()
        >>> Fx, Fy = kernel.compute(f_post, needs_bounce, N)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _MEM_FORCE_D2Q9_KERNEL,
            'mem_force_hwbb_d2q9',
            options=('--use_fast_math',),
        )

    def compute(
        self,
        f_post: 'npt.NDArray',
        needs_bounce: 'npt.NDArray',
        N: int,
    ) -> Tuple[float, float]:
        if self._kernel is None:
            self._compile()

        import cupy as cp

        force_xy = cp.zeros(2, dtype=cp.float32)
        grid = (N + self._block_size - 1) // self._block_size

        self._kernel(
            (grid,), (self._block_size,),
            (f_post, needs_bounce, force_xy, cp.int32(N)),
        )

        return (float(force_xy[0]), float(force_xy[1]))


class MEMForceIBBKernelD2Q9:
    """CUDA Bouzidi-linear IBB MEM force kernel for D2Q9.

    Usage:
        >>> kernel = MEMForceIBBKernelD2Q9()
        >>> Fx, Fy = kernel.compute(f_post, needs_bounce, q_fraction, Nx, Ny)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _MEM_FORCE_D2Q9_KERNEL,
            'mem_force_ibb_d2q9',
            options=('--use_fast_math',),
        )

    def compute(
        self,
        f_post: 'npt.NDArray',
        needs_bounce: 'npt.NDArray',
        q_fraction: 'npt.NDArray',
        Nx: int, Ny: int,
    ) -> Tuple[float, float]:
        if self._kernel is None:
            self._compile()

        import cupy as cp

        force_xy = cp.zeros(2, dtype=cp.float32)
        N = Nx * Ny
        grid = (N + self._block_size - 1) // self._block_size

        self._kernel(
            (grid,), (self._block_size,),
            (f_post, needs_bounce, q_fraction, force_xy,
             cp.int32(Nx), cp.int32(Ny)),
        )

        return (float(force_xy[0]), float(force_xy[1]))
