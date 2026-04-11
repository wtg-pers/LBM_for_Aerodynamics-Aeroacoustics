"""
CUDA Streaming Kernel for D3Q27

Replaces CuPy advanced indexing with a direct CUDA kernel.
Eliminates precomputed index arrays (saves 324 B/node of GPU memory).

Pull scheme: f_next[q, x, y, z] = f_post[q, x-cx, y-cy, z-cz]

Memory traffic per node (float32):
    Read:  27 × 4B = 108B  (f_post from 27 neighbors)
    Write: 27 × 4B = 108B  (f_next at current node)
    Total: 216B

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_STREAMING_D3Q27_KERNEL = r'''
extern "C" __global__
void streaming_pull_d3q27(
    const float* __restrict__ f_post,
    float*       __restrict__ f_next,
    const int Nx, const int Ny, const int Nz
) {
    int N = Nx * Ny * Nz;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // C-contiguous (x, y, z): z varies fastest
    // Linear index: idx = ix * Ny * Nz + iy * Nz + iz
    int ix = idx / (Ny * Nz);
    int rem = idx - ix * Ny * Nz;
    int iy = rem / Nz;
    int iz = rem - iy * Nz;

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

    for (int q = 0; q < 27; q++) {
        int sx = (ix - cx[q] + Nx) % Nx;
        int sy = (iy - cy[q] + Ny) % Ny;
        int sz = (iz - cz[q] + Nz) % Nz;
        int src = sx * Ny * Nz + sy * Nz + sz;
        f_next[q * N + idx] = f_post[q * N + src];
    }
}
'''


class StreamingKernelD3Q27:
    """CUDA pull-streaming kernel for D3Q27.

    Replaces CuPy advanced indexing. No precomputed index arrays needed.

    Usage:
        >>> kernel = StreamingKernelD3Q27()
        >>> kernel.launch(f_post, f_next, Nx, Ny, Nz)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _STREAMING_D3Q27_KERNEL,
            'streaming_pull_d3q27',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        f_post: 'npt.NDArray',
        f_next: 'npt.NDArray',
        Nx: int, Ny: int, Nz: int,
    ) -> None:
        """Launch streaming kernel.

        Args:
            f_post: Post-collision distribution (27, Nx*Ny*Nz) — read
            f_next: Streamed output (27, Nx*Ny*Nz) — written
            Nx, Ny, Nz: Domain dimensions
        """
        if self._kernel is None:
            self._compile()

        import cupy as cp

        N = Nx * Ny * Nz
        grid_size = (N + self._block_size - 1) // self._block_size

        self._kernel(
            (grid_size,),
            (self._block_size,),
            (f_post, f_next, cp.int32(Nx), cp.int32(Ny), cp.int32(Nz)),
        )
