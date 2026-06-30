"""Macroscopic moment kernel (D3Q27): f -> rho, u_x, u_y, u_z.

Used as a pre-pass before WALE-SGS. See macro_d2q9.py for rationale.

Lattice convention (src/lattice/d3q27.py):
    cx = [0,  1,-1, 0, 0, 0, 0,
              1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0,
              1,-1, 1,-1, 1,-1, 1,-1]
    cy = [0,  0, 0, 1,-1, 0, 0,
              1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1,
              1, 1,-1,-1, 1, 1,-1,-1]
    cz = [0,  0, 0, 0, 0, 1,-1,
              0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1,
              1, 1, 1, 1,-1,-1,-1,-1]
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_MACRO_D3Q27_KERNEL = r'''
extern "C" __global__
void macro_d3q27(
    const float* __restrict__ f,    // (27, N)
    float*       __restrict__ rho,  // (N,)
    float*       __restrict__ u_x,  // (N,)
    float*       __restrict__ u_y,  // (N,)
    float*       __restrict__ u_z,  // (N,)
    const long long N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Lattice constants (must match D3Q27 in src/lattice/d3q27.py)
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

    float r  = 0.0f;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        float fq = f[q * N + idx];
        r  += fq;
        mx += (float)cx[q] * fq;
        my += (float)cy[q] * fq;
        mz += (float)cz[q] * fq;
    }

    float inv_r = 1.0f / r;
    rho[idx] = r;
    u_x[idx] = mx * inv_r;
    u_y[idx] = my * inv_r;
    u_z[idx] = mz * inv_r;
}
'''


class MacroKernelD3Q27:
    """f -> (rho, u_x, u_y, u_z) global pre-pass for D3Q27."""

    def __init__(self, block_size: int = 128) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _MACRO_D3Q27_KERNEL, 'macro_d3q27',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        f: 'npt.NDArray',
        rho: 'npt.NDArray',
        u_x: 'npt.NDArray',
        u_y: 'npt.NDArray',
        u_z: 'npt.NDArray',
        N: int,
    ) -> None:
        if self._kernel is None:
            self._compile()
        import cupy as cp
        grid = (N + self._block_size - 1) // self._block_size
        self._kernel(
            (grid,), (self._block_size,),
            (f, rho, u_x, u_y, u_z, cp.int64(N)),
        )
