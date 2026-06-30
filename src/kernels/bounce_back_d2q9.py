"""
CUDA HalfwayBounceBack Kernel for D2Q9

2D analogue of `bounce_back_d3q27.py`. Replaces the Python Q=9 loop in
HalfwayBounceBack.apply() with a single GPU kernel.

Physical process:
    At fluid nodes adjacent to solid, the outgoing distribution
    toward the wall bounces back: f[i_opp, x] = f_post[i, x]

Lattice (D2Q9):
    opp[q] for q=0..8 → [0, 3, 4, 1, 2, 7, 8, 5, 6]

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy.typing as npt


_HWBB_D2Q9_KERNEL = r'''
extern "C" __global__
void hwbb_apply_d2q9(
    float*       __restrict__ f,            // (9, N) post-streaming
    const float* __restrict__ f_post,       // (9, N) post-collision source
    const bool*  __restrict__ needs_bounce, // (9, N) boundary link mask
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // D2Q9 opposite directions
    const int opp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

    #pragma unroll
    for (int q = 1; q < 9; q++) {
        if (needs_bounce[q * N + idx]) {
            int q_opp = opp[q];
            f[q_opp * N + idx] = f_post[q * N + idx];
        }
    }
}

extern "C" __global__
void hwbb_apply_inplace_d2q9(
    float*       __restrict__ f,            // (9, N) post-collision, src+dst
    const bool*  __restrict__ needs_bounce, // (9, N) boundary link mask
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int opp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

    // Save outgoing distributions first (direction-pair safety)
    float saved[9];
    #pragma unroll
    for (int q = 1; q < 9; q++) {
        if (needs_bounce[q * N + idx]) {
            saved[q] = f[q * N + idx];
        }
    }

    // Write bounced values
    #pragma unroll
    for (int q = 1; q < 9; q++) {
        if (needs_bounce[q * N + idx]) {
            int q_opp = opp[q];
            f[q_opp * N + idx] = saved[q];
        }
    }
}

extern "C" __global__
void hwbb_reset_solid_d2q9(
    float*       __restrict__ f,            // (9, N) modified
    const bool*  __restrict__ solid_mask,   // (N,) True=solid
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (!solid_mask[idx]) return;

    // Reset solid nodes to equilibrium (rho=1, u=0): f = w
    const float w[9] = {
        4.0f/9.0f,
        1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
    };

    #pragma unroll
    for (int q = 0; q < 9; q++) {
        f[q * N + idx] = w[q];
    }
}
'''


class HWBBKernelD2Q9:
    """CUDA HalfwayBounceBack kernel for D2Q9.

    Replaces Python Q-loop with a single GPU kernel launch. Provides
    both standard (separate f_post) and in-place modes.

    Usage:
        >>> kernel = HWBBKernelD2Q9()
        >>> kernel.apply(f, f_post, needs_bounce, N)
        >>> kernel.reset_solid(f, solid_mask, N)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel_apply = None
        self._kernel_inplace = None
        self._kernel_reset = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel_apply = cp.RawKernel(
            _HWBB_D2Q9_KERNEL, 'hwbb_apply_d2q9',
            options=('--use_fast_math',),
        )
        self._kernel_inplace = cp.RawKernel(
            _HWBB_D2Q9_KERNEL, 'hwbb_apply_inplace_d2q9',
            options=('--use_fast_math',),
        )
        self._kernel_reset = cp.RawKernel(
            _HWBB_D2Q9_KERNEL, 'hwbb_reset_solid_d2q9',
            options=('--use_fast_math',),
        )

    def apply(
        self,
        f: 'npt.NDArray',
        f_post: Optional['npt.NDArray'],
        needs_bounce: 'npt.NDArray',
        N: int,
    ) -> None:
        """Apply bounce-back.

        Args:
            f:            Post-streaming distribution (9, N) -- modified
            f_post:       Post-collision source (9, N), or None for in-place
            needs_bounce: Boundary link mask (9, N) bool
            N:            Total spatial nodes (Nx*Ny)
        """
        if self._kernel_apply is None:
            self._compile()

        import cupy as cp
        grid = (N + self._block_size - 1) // self._block_size

        if f_post is not None:
            self._kernel_apply(
                (grid,), (self._block_size,),
                (f, f_post, needs_bounce, cp.int32(N)),
            )
        else:
            self._kernel_inplace(
                (grid,), (self._block_size,),
                (f, needs_bounce, cp.int32(N)),
            )

    def reset_solid(
        self,
        f: 'npt.NDArray',
        solid_mask: 'npt.NDArray',
        N: int,
    ) -> None:
        """Reset solid nodes to equilibrium.

        Args:
            f:          Distribution (9, N) -- modified
            solid_mask: Boolean mask (N,) True=solid
            N:          Total nodes
        """
        if self._kernel_reset is None:
            self._compile()

        import cupy as cp
        grid = (N + self._block_size - 1) // self._block_size

        self._kernel_reset(
            (grid,), (self._block_size,),
            (f, solid_mask, cp.int32(N)),
        )
