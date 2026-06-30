"""
CUDA Streaming Kernel for D2Q9

2D analogue of `streaming_d3q27.py`. Replaces CuPy advanced indexing
(`f_post[q_idx, src_x, src_y]`) with a direct CUDA kernel.

Pull scheme: f_next[q, x, y] = f_post[q, x-cx, y-cy]

Memory layout: (Q, Nx, Ny) C-contiguous → flat index q*N + ix*Ny + iy.

Memory traffic per node (float32):
    Read:  9 × 4B = 36B   (f_post from 9 neighbors)
    Write: 9 × 4B = 36B   (f_next at current node)
    Total: 72B

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_STREAMING_D2Q9_KERNEL = r'''
extern "C" __global__
void streaming_pull_d2q9(
    const float* __restrict__ f_post,
    float*       __restrict__ f_next,
    const int Nx, const int Ny
) {
    int N = Nx * Ny;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // C-contiguous (x, y): y varies fastest
    int ix = idx / Ny;
    int iy = idx - ix * Ny;

    const int cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    const int cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

    #pragma unroll
    for (int q = 0; q < 9; q++) {
        int sx = (ix - cx[q] + Nx) % Nx;
        int sy = (iy - cy[q] + Ny) % Ny;
        int src = sx * Ny + sy;
        f_next[q * N + idx] = f_post[q * N + src];
    }
}
'''


class StreamingKernelD2Q9:
    """CUDA pull-streaming kernel for D2Q9.

    Replaces CuPy advanced indexing — no precomputed index arrays needed.

    Usage:
        >>> kernel = StreamingKernelD2Q9()
        >>> kernel.launch(f_post, f_next, Nx, Ny)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _STREAMING_D2Q9_KERNEL,
            'streaming_pull_d2q9',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        f_post: 'npt.NDArray',
        f_next: 'npt.NDArray',
        Nx: int, Ny: int,
    ) -> None:
        """Launch streaming kernel.

        Args:
            f_post: Post-collision distribution (9, Nx*Ny) -- read
            f_next: Streamed output (9, Nx*Ny)             -- written
            Nx, Ny: Domain dimensions
        """
        if self._kernel is None:
            self._compile()

        import cupy as cp

        N = Nx * Ny
        grid_size = (N + self._block_size - 1) // self._block_size

        self._kernel(
            (grid_size,),
            (self._block_size,),
            (f_post, f_next, cp.int32(Nx), cp.int32(Ny)),
        )
