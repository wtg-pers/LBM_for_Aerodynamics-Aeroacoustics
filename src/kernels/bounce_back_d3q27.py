"""
CUDA HalfwayBounceBack Kernel for D3Q27

Replaces Python loop (Q=27 iterations) with a single GPU kernel.
Each thread handles one spatial node, processing all 27 directions.

Physical process:
    At fluid nodes adjacent to solid, the outgoing distribution
    toward the wall bounces back:  f[i_opp, x] = f_post[i, x]

Memory traffic per boundary node (float32):
    Read:  needs_bounce (27 x 1B) + f_post source (~13 x 4B avg)
    Write: f destination (~13 x 4B avg)

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy.typing as npt


_HWBB_D3Q27_KERNEL = r'''
extern "C" __global__
void hwbb_apply_d3q27(
    float*       __restrict__ f,           // (27, N) post-streaming, modified
    const float* __restrict__ f_post,      // (27, N) post-collision source
    const bool*  __restrict__ needs_bounce, // (27, N) boundary link mask
    const long long N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // D3Q27 opposite direction mapping
    // opp[q] = direction index of -c_q
    const int opp[27] = {
        0,   2, 1, 4, 3, 6, 5,
        10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
        26, 25, 24, 23, 22, 21, 20, 19
    };

    // For each direction: if needs_bounce, apply bounce-back
    for (int q = 1; q < 27; q++) {  // skip rest direction
        if (needs_bounce[q * N + idx]) {
            int q_opp = opp[q];
            f[q_opp * N + idx] = f_post[q * N + idx];
        }
    }
}

extern "C" __global__
void hwbb_apply_inplace_d3q27(
    float*       __restrict__ f,           // (27, N) post-collision, source AND target
    const bool*  __restrict__ needs_bounce, // (27, N) boundary link mask
    const long long N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int opp[27] = {
        0,   2, 1, 4, 3, 6, 5,
        10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
        26, 25, 24, 23, 22, 21, 20, 19
    };

    // Save outgoing distributions first (direction-pair safety)
    float saved[27];
    for (int q = 1; q < 27; q++) {
        if (needs_bounce[q * N + idx]) {
            saved[q] = f[q * N + idx];
        }
    }

    // Write bounced values
    for (int q = 1; q < 27; q++) {
        if (needs_bounce[q * N + idx]) {
            int q_opp = opp[q];
            f[q_opp * N + idx] = saved[q];
        }
    }
}

extern "C" __global__
void hwbb_reset_solid_d3q27(
    float*       __restrict__ f,           // (27, N) modified
    const bool*  __restrict__ solid_mask,  // (N,) True=solid
    const long long N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (!solid_mask[idx]) return;

    // Reset solid nodes to equilibrium (rho=1, u=0): f = w
    const float w[27] = {
        8.0f/27.0f,
        2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f,
        2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f,
        1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,
        1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,
        1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,
        1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f,
        1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f
    };

    for (int q = 0; q < 27; q++) {
        f[q * N + idx] = w[q];
    }
}
'''


class HWBBKernelD3Q27:
    """CUDA HalfwayBounceBack kernel for D3Q27.

    Replaces Python Q-loop with a single GPU kernel launch.
    Provides both standard (separate f_post) and in-place modes.

    Usage:
        >>> kernel = HWBBKernelD3Q27()
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
            _HWBB_D3Q27_KERNEL, 'hwbb_apply_d3q27',
            options=('--use_fast_math',),
        )
        self._kernel_inplace = cp.RawKernel(
            _HWBB_D3Q27_KERNEL, 'hwbb_apply_inplace_d3q27',
            options=('--use_fast_math',),
        )
        self._kernel_reset = cp.RawKernel(
            _HWBB_D3Q27_KERNEL, 'hwbb_reset_solid_d3q27',
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
            f:            Post-streaming distribution (27, N) -- modified
            f_post:       Post-collision source (27, N), or None for in-place
            needs_bounce: Boundary link mask (27, N) bool
            N:            Total spatial nodes
        """
        if self._kernel_apply is None:
            self._compile()

        import cupy as cp
        grid = (N + self._block_size - 1) // self._block_size

        if f_post is not None:
            self._kernel_apply(
                (grid,), (self._block_size,),
                (f, f_post, needs_bounce, cp.int64(N)),
            )
        else:
            self._kernel_inplace(
                (grid,), (self._block_size,),
                (f, needs_bounce, cp.int64(N)),
            )

    def reset_solid(
        self,
        f: 'npt.NDArray',
        solid_mask: 'npt.NDArray',
        N: int,
    ) -> None:
        """Reset solid nodes to equilibrium.

        Args:
            f:          Distribution (27, N) -- modified
            solid_mask: Boolean mask (N,) True=solid
            N:          Total nodes
        """
        if self._kernel_reset is None:
            self._compile()

        import cupy as cp
        grid = (N + self._block_size - 1) // self._block_size

        self._kernel_reset(
            (grid,), (self._block_size,),
            (f, solid_mask, cp.int64(N)),
        )
