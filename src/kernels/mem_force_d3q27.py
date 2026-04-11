"""
CUDA Momentum Exchange Method (MEM) Force Kernel for D3Q27

Replaces Python double loop (dim × Q = 78 iterations) with a
GPU parallel reduction.

Physical formula:
    F_d = Σ_{boundary links} 2 * c_i[d] * f_i^post(x_fluid)

Strategy:
    Each thread handles one spatial node, accumulates force for all
    27 directions. Then uses atomicAdd to accumulate into global sums.

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import numpy.typing as npt


_MEM_FORCE_D3Q27_KERNEL = r'''
extern "C" __global__
void mem_force_d3q27(
    const float* __restrict__ f_post,       // (27, N)
    const bool*  __restrict__ needs_bounce,  // (27, N)
    float*       __restrict__ force_xyz,     // (3,) output — atomicAdd
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int cx[27] = {
        0,  1,-1, 0, 0, 0, 0,
        1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0,
        1,-1, 1,-1, 1,-1, 1,-1
    };
    const int cy[27] = {
        0,  0, 0, 1,-1, 0, 0,
        1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1,
        1, 1,-1,-1, 1, 1,-1,-1
    };
    const int cz[27] = {
        0,  0, 0, 0, 0, 1,-1,
        0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1,
        1, 1, 1, 1,-1,-1,-1,-1
    };

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (int q = 1; q < 27; q++) {
        if (needs_bounce[q * N + idx]) {
            float fq = f_post[q * N + idx];
            fx += 2.0f * (float)cx[q] * fq;
            fy += 2.0f * (float)cy[q] * fq;
            fz += 2.0f * (float)cz[q] * fq;
        }
    }

    // Accumulate to global force (atomicAdd for thread safety)
    if (fx != 0.0f) atomicAdd(&force_xyz[0], fx);
    if (fy != 0.0f) atomicAdd(&force_xyz[1], fy);
    if (fz != 0.0f) atomicAdd(&force_xyz[2], fz);
}
'''


class MEMForceKernelD3Q27:
    """CUDA MEM force kernel for D3Q27.

    Replaces Python dim×Q loop with a single GPU kernel.
    Uses atomicAdd for parallel reduction.

    Usage:
        >>> kernel = MEMForceKernelD3Q27()
        >>> Fx, Fy, Fz = kernel.compute(f_post, needs_bounce, N)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _MEM_FORCE_D3Q27_KERNEL,
            'mem_force_d3q27',
            options=('--use_fast_math',),
        )

    def compute(
        self,
        f_post: 'npt.NDArray',
        needs_bounce: 'npt.NDArray',
        N: int,
    ) -> Tuple[float, float, float]:
        """Compute MEM force.

        Args:
            f_post:       Post-collision distribution (27, N)
            needs_bounce: Boundary link mask (27, N) bool
            N:            Total spatial nodes

        Returns:
            (Fx, Fy, Fz) in lattice units
        """
        if self._kernel is None:
            self._compile()

        import cupy as cp

        force_xyz = cp.zeros(3, dtype=cp.float32)
        grid = (N + self._block_size - 1) // self._block_size

        self._kernel(
            (grid,), (self._block_size,),
            (f_post, needs_bounce, force_xyz, cp.int32(N)),
        )

        fx, fy, fz = float(force_xyz[0]), float(force_xyz[1]), float(force_xyz[2])
        return (fx, fy, fz)
