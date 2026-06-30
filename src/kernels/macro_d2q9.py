"""Macroscopic moment kernel (D2Q9): f -> rho, u_x, u_y.

Used as a pre-pass before WALE-SGS, where the velocity gradient tensor needs
u to be globally available before the cumulant collide kernel runs (Smag
computes |S| inline from local f^neq, no neighbor read needed).

Lattice convention (matches src/lattice/d2q9.py):
    c_x = [0, +1,  0, -1,  0, +1, -1, -1, +1]
    c_y = [0,  0, +1,  0, -1, +1, +1, -1, -1]
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_MACRO_D2Q9_KERNEL = r'''
extern "C" __global__
void macro_d2q9(
    const float* __restrict__ f,    // (9, N)
    float*       __restrict__ rho,  // (N,)
    float*       __restrict__ u_x,  // (N,)
    float*       __restrict__ u_y,  // (N,)
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float f0 = f[0*N + idx];
    float f1 = f[1*N + idx];
    float f2 = f[2*N + idx];
    float f3 = f[3*N + idx];
    float f4 = f[4*N + idx];
    float f5 = f[5*N + idx];
    float f6 = f[6*N + idx];
    float f7 = f[7*N + idx];
    float f8 = f[8*N + idx];

    float r = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    // c_x: 0,+1,0,-1,0,+1,-1,-1,+1
    float mx = (f1 - f3) + (f5 - f6) + (f8 - f7);
    // c_y: 0,0,+1,0,-1,+1,+1,-1,-1
    float my = (f2 - f4) + (f5 + f6) - (f7 + f8);

    float inv_r = 1.0f / r;
    rho[idx] = r;
    u_x[idx] = mx * inv_r;
    u_y[idx] = my * inv_r;
}
'''


class MacroKernelD2Q9:
    """f -> (rho, u_x, u_y) global pre-pass for D2Q9."""

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _MACRO_D2Q9_KERNEL, 'macro_d2q9',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        f: 'npt.NDArray',
        rho: 'npt.NDArray',
        u_x: 'npt.NDArray',
        u_y: 'npt.NDArray',
        N: int,
    ) -> None:
        """Compute (rho, u_x, u_y) from f. All arrays float32, C-contiguous."""
        if self._kernel is None:
            self._compile()
        import cupy as cp
        grid = (N + self._block_size - 1) // self._block_size
        self._kernel(
            (grid,), (self._block_size,),
            (f, rho, u_x, u_y, cp.int32(N)),
        )
