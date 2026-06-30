"""
CUDA 2D Cubic Interpolation Kernel for D2Q9 MLG Coupling

2D counterpart of `interpolation_d3q27.py`. Fills all odd indices along
x and y (z axis removed).

Stencil (Lagrava Eq. 4.23):
    Interior: g(x) = 9/16*[g(x+h)+g(x-h)] - 1/16*[g(x+3h)+g(x-3h)]
    Boundary: g(x) = 3/8*g(x-h) + 3/4*g(x+h) - 1/8*g(x+3h)

Input:  f[Q, Nx_f, Ny_f] with values at even indices (coarse nodes)
Output: f with all odd indices filled by interpolation

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_CUBIC_INTERP_2D_KERNEL = r'''
// 1D cubic interpolation along x, one fiber per (q, iy).
extern "C" __global__
void cubic_interp_1d_x(
    float* __restrict__ f,
    const int Q, const int Nx, const int Ny
) {
    int total = Q * Ny;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int q = tid / Ny;
    int iy = tid - q * Ny;

    int NxNy = Nx * Ny;
    #define FX(x) f[q * NxNy + (x) * Ny + iy]

    for (int ix = 3; ix <= Nx - 4; ix += 2) {
        FX(ix) = 0.5625f * (FX(ix-1) + FX(ix+1))
               - 0.0625f * (FX(ix-3) + FX(ix+3));
    }

    if (Nx >= 5) {
        FX(1) = 0.375f * FX(0) + 0.75f * FX(2) - 0.125f * FX(4);
    } else if (Nx >= 3) {
        FX(1) = 0.5f * (FX(0) + FX(2));
    }

    if ((Nx - 2) % 2 == 1) {
        if (Nx >= 5) {
            FX(Nx-2) = 0.375f * FX(Nx-1) + 0.75f * FX(Nx-3) - 0.125f * FX(Nx-5);
        } else if (Nx >= 3) {
            FX(Nx-2) = 0.5f * (FX(Nx-3) + FX(Nx-1));
        }
    }
    #undef FX
}

// 1D cubic interpolation along y, one fiber per (q, ix).
extern "C" __global__
void cubic_interp_1d_y(
    float* __restrict__ f,
    const int Q, const int Nx, const int Ny
) {
    int total = Q * Nx;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int q = tid / Nx;
    int ix = tid - q * Nx;

    int NxNy = Nx * Ny;
    #define FY(y) f[q * NxNy + ix * Ny + (y)]

    for (int iy = 3; iy <= Ny - 4; iy += 2) {
        FY(iy) = 0.5625f * (FY(iy-1) + FY(iy+1))
               - 0.0625f * (FY(iy-3) + FY(iy+3));
    }

    if (Ny >= 5) {
        FY(1) = 0.375f * FY(0) + 0.75f * FY(2) - 0.125f * FY(4);
    } else if (Ny >= 3) {
        FY(1) = 0.5f * (FY(0) + FY(2));
    }

    if ((Ny - 2) % 2 == 1) {
        if (Ny >= 5) {
            FY(Ny-2) = 0.375f * FY(Ny-1) + 0.75f * FY(Ny-3) - 0.125f * FY(Ny-5);
        } else if (Ny >= 3) {
            FY(Ny-2) = 0.5f * (FY(Ny-3) + FY(Ny-1));
        }
    }
    #undef FY
}
'''


class CubicInterpolationKernel2D:
    """CUDA cubic interpolation for 2D MLG coupling (D2Q9).

    Replaces 2 CuPy interpolate_1d calls with 2 CUDA kernel launches.
    Mirrors CubicInterpolationKernel3D exactly, sans z.

    Usage:
        >>> kernel = CubicInterpolationKernel2D()
        >>> kernel.interpolate(f, Q, Nx, Ny)  # modifies f in-place
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kx = None
        self._ky = None

    def _compile(self) -> None:
        import cupy as cp
        opts = ('--use_fast_math',)
        self._kx = cp.RawKernel(_CUBIC_INTERP_2D_KERNEL,
                                'cubic_interp_1d_x', options=opts)
        self._ky = cp.RawKernel(_CUBIC_INTERP_2D_KERNEL,
                                'cubic_interp_1d_y', options=opts)

    def interpolate(
        self,
        f: 'npt.NDArray',
        Q: int, Nx: int, Ny: int,
    ) -> None:
        """Cubic interpolation on both axes (in-place).

        Args:
            f:  Array (Q, Nx, Ny) with values at even indices.
                Odd indices filled in-place.
            Q, Nx, Ny: Dimensions.
        """
        if self._kx is None:
            self._compile()

        import cupy as cp
        bs = self._block_size
        args = (f, cp.int32(Q), cp.int32(Nx), cp.int32(Ny))

        # X-axis: threads = Q * Ny
        n = Q * Ny
        self._kx(((n + bs - 1) // bs,), (bs,), args)

        # Y-axis: threads = Q * Nx
        n = Q * Nx
        self._ky(((n + bs - 1) // bs,), (bs,), args)
