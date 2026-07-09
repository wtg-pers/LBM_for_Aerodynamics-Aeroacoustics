"""
CUDA 3D Cubic Interpolation Kernel for MLG Coupling

Replaces 3 sequential CuPy interpolate_1d calls with a single
CUDA kernel that fills all odd indices in 3D simultaneously.

Stencil (Lagrava Eq. 4.23):
    Interior: g(x) = 9/16*[g(x+h)+g(x-h)] - 1/16*[g(x+3h)+g(x-3h)]
    Boundary: g(x) = 3/8*g(x-h) + 3/4*g(x+h) - 1/8*g(x+3h)

Input:  f[Q, Nx_f, Ny_f, Nz_f] with values at even indices (coarse nodes)
Output: f with all odd indices filled by interpolation

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_CUBIC_INTERP_3D_KERNEL = r'''
extern "C" __global__
void cubic_interp_3d(
    float* __restrict__ f,     // (Q, Nx, Ny, Nz) -- modified in-place
    const int Q,
    const int Nx, const int Ny, const int Nz
) {
    // Each thread handles one (q, ix, iy, iz)
    int total = Q * Nx * Ny * Nz;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int NyNz = Ny * Nz;
    long long NxNyNz = (long long)Nx * NyNz;   // 64-bit: q*NxNyNz overflows int32 at N>~79.5M

    int q  = tid / NxNyNz;
    int rem = tid - q * NxNyNz;
    int ix = rem / NyNz;
    rem = rem - ix * NyNz;
    int iy = rem / Nz;
    int iz = rem - iy * Nz;

    // Only process odd indices (at least one of ix, iy, iz is odd)
    int ox = ix & 1, oy = iy & 1, oz = iz & 1;
    if (ox == 0 && oy == 0 && oz == 0) return;  // even node, skip

    // Helper: read f[q, x, y, z] with bounds check
    #define F(x, y, z) f[q * NxNyNz + (x) * NyNz + (y) * Nz + (z)]

    // 1D cubic interpolation weight function
    // For odd index i along axis with even neighbors:
    //   Interior (i-3>=0 && i+3<n): 9/16*(g[i-1]+g[i+1]) - 1/16*(g[i-3]+g[i+3])
    //   Left boundary (i=1, n>=5):  3/8*g[0] + 3/4*g[2] - 1/8*g[4]
    //   Right boundary (i=n-2):     3/8*g[n-1] + 3/4*g[n-3] - 1/8*g[n-5]
    //   Fallback (n<5):             0.5*(g[i-1]+g[i+1])

    // Strategy: interpolate sequentially x -> y -> z
    // But we need intermediate values at partially-interpolated nodes.
    // Instead, use the factored approach:
    //   For a node (ox, oy, oz):
    //     (1,0,0): interp x only, using even y,z neighbors
    //     (0,1,0): interp y only
    //     (0,0,1): interp z only
    //     (1,1,0): interp x first (using even-y values), then interp y...
    //              but we don't have intermediate values in a single pass.

    // For a single-kernel approach, we need the SEQUENTIAL interpolation
    // to work. This requires 3 passes (like the original), not 1 pass.
    // A single kernel can't do sequential axis interpolation because
    // odd-x values depend on even-x values, and odd-xy values depend
    // on odd-x values.

    // CONCLUSION: 3D interpolation is inherently sequential (axis by axis).
    // A single kernel can handle ONE axis at a time.
    // We provide a 1D kernel that's called 3 times.
    // This is still faster than CuPy because: 1 kernel launch per axis
    // instead of multiple CuPy operations per axis.

    // This kernel is NOT used -- see cubic_interp_1d below.
    return;
}

// --- 1D cubic interpolation kernel -------------------------
// Fills all odd indices along one axis in a single kernel launch.
// axis: 1=x, 2=y, 3=z (0=Q, skipped)

extern "C" __global__
void cubic_interp_1d_x(
    float* __restrict__ f,
    const int Q, const int Nx, const int Ny, const int Nz
) {
    // Thread per (q, iy, iz) -- process all odd ix for this fiber
    int NyNz = Ny * Nz;
    int total = Q * Ny * Nz;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int q = tid / NyNz;
    int rem = tid - q * NyNz;
    int iy = rem / Nz;
    int iz = rem - iy * Nz;

    long long NxNyNz = (long long)Nx * NyNz;   // 64-bit: q*NxNyNz overflows int32 at N>~79.5M
    #define FX(x) f[q * NxNyNz + (x) * NyNz + iy * Nz + iz]

    // Interior odd indices: 3, 5, ..., Nx-4
    for (int ix = 3; ix <= Nx - 4; ix += 2) {
        FX(ix) = 0.5625f * (FX(ix-1) + FX(ix+1))
               - 0.0625f * (FX(ix-3) + FX(ix+3));
    }

    // Left boundary: index 1
    if (Nx >= 5) {
        FX(1) = 0.375f * FX(0) + 0.75f * FX(2) - 0.125f * FX(4);
    } else if (Nx >= 3) {
        FX(1) = 0.5f * (FX(0) + FX(2));
    }

    // Right boundary: index Nx-2 (if odd)
    if ((Nx - 2) % 2 == 1) {
        if (Nx >= 5) {
            FX(Nx-2) = 0.375f * FX(Nx-1) + 0.75f * FX(Nx-3) - 0.125f * FX(Nx-5);
        } else if (Nx >= 3) {
            FX(Nx-2) = 0.5f * (FX(Nx-3) + FX(Nx-1));
        }
    }
    #undef FX
}

extern "C" __global__
void cubic_interp_1d_y(
    float* __restrict__ f,
    const int Q, const int Nx, const int Ny, const int Nz
) {
    int total = Q * Nx * Nz;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int NyNz = Ny * Nz;
    long long NxNyNz = (long long)Nx * NyNz;   // 64-bit: q*NxNyNz overflows int32 at N>~79.5M

    int q = tid / (Nx * Nz);
    int rem = tid - q * Nx * Nz;
    int ix = rem / Nz;
    int iz = rem - ix * Nz;

    #define FY(y) f[q * NxNyNz + ix * NyNz + (y) * Nz + iz]

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

extern "C" __global__
void cubic_interp_1d_z(
    float* __restrict__ f,
    const int Q, const int Nx, const int Ny, const int Nz
) {
    // Thread per (q, ix, iy, iz) with iz FASTEST across threads. z is the
    // contiguous axis, so a warp of consecutive iz touches a contiguous span
    // -> COALESCED global access. The previous fiber-per-thread mapping
    // (thread per (q,ix,iy), internal iz loop) put consecutive threads on iy
    // (stride Nz) -> uncoalesced, which made this kernel ~5.7x its x/y peers
    // for the SAME array (nsys 2026-07-08: z 876M vs x 152M ns). Arithmetic
    // per node is identical -> bit-identical result, only the mapping changed.
    long long total = (long long)Q * Nx * Ny * Nz;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int iz = (int)(tid % Nz);
    if ((iz & 1) == 0) return;                 // even node: coarse value, keep

    long long t2 = tid / Nz;
    int iy = (int)(t2 % Ny);
    long long t3 = t2 / Ny;
    int ix = (int)(t3 % Nx);
    int q  = (int)(t3 / Nx);

    int NyNz = Ny * Nz;
    long long NxNyNz = (long long)Nx * NyNz;   // 64-bit: q*NxNyNz overflows int32 at N>~79.5M
    #define FZ(z) f[q * NxNyNz + ix * NyNz + iy * Nz + (z)]

    if (iz == 1) {                             // left boundary
        if (Nz >= 5) FZ(1) = 0.375f * FZ(0) + 0.75f * FZ(2) - 0.125f * FZ(4);
        else if (Nz >= 3) FZ(1) = 0.5f * (FZ(0) + FZ(2));
    } else if (iz == Nz - 2) {                 // right boundary (Nz odd -> odd)
        if (Nz >= 5) FZ(Nz-2) = 0.375f * FZ(Nz-1) + 0.75f * FZ(Nz-3) - 0.125f * FZ(Nz-5);
        else if (Nz >= 3) FZ(Nz-2) = 0.5f * (FZ(Nz-3) + FZ(Nz-1));
    } else if (iz >= 3 && iz <= Nz - 4) {      // interior
        FZ(iz) = 0.5625f * (FZ(iz-1) + FZ(iz+1))
               - 0.0625f * (FZ(iz-3) + FZ(iz+3));
    }
    #undef FZ
}
'''


class CubicInterpolationKernel3D:
    """CUDA cubic interpolation for MLG coupling.

    Replaces 3 CuPy interpolate_1d calls with 3 CUDA kernel launches.
    Each kernel fills all odd indices along one axis.

    Usage:
        >>> kernel = CubicInterpolationKernel3D()
        >>> kernel.interpolate(f, Q, Nx, Ny, Nz)  # modifies f in-place
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kx = None
        self._ky = None
        self._kz = None

    def _compile(self) -> None:
        import cupy as cp
        opts = ('--use_fast_math',)
        self._kx = cp.RawKernel(_CUBIC_INTERP_3D_KERNEL, 'cubic_interp_1d_x', options=opts)
        self._ky = cp.RawKernel(_CUBIC_INTERP_3D_KERNEL, 'cubic_interp_1d_y', options=opts)
        self._kz = cp.RawKernel(_CUBIC_INTERP_3D_KERNEL, 'cubic_interp_1d_z', options=opts)

    def interpolate(
        self,
        f: 'npt.NDArray',
        Q: int, Nx: int, Ny: int, Nz: int,
    ) -> None:
        """Cubic interpolation on all 3 axes (in-place).

        Args:
            f:  Array (Q, Nx, Ny, Nz) with values at even indices.
                Odd indices filled in-place.
            Q, Nx, Ny, Nz: Dimensions.
        """
        if self._kx is None:
            self._compile()

        import cupy as cp
        bs = self._block_size

        args = (f, cp.int32(Q), cp.int32(Nx), cp.int32(Ny), cp.int32(Nz))

        # X-axis: threads = Q * Ny * Nz
        n = Q * Ny * Nz
        self._kx(((n + bs - 1) // bs,), (bs,), args)

        # Y-axis: threads = Q * Nx * Nz
        n = Q * Nx * Nz
        self._ky(((n + bs - 1) // bs,), (bs,), args)

        # Z-axis: thread per (q, ix, iy, iz), iz fastest -> coalesced.
        # threads = Q * Nx * Ny * Nz (even-iz lanes early-return).
        n = Q * Nx * Ny * Nz
        self._kz(((n + bs - 1) // bs,), (bs,), args)
