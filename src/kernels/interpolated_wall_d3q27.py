"""
CUDA Bouzidi-linear Interpolated Bounce-Back kernel for D3Q27 (sparse).

3D port of `interpolated_wall_d2q9.IBBKernelD2Q9`. Memory-efficient version
that iterates over a 1D list of boundary links instead of a dense
(Q, Nx, Ny, Nz) q_fraction array. For Re=3900 v2 setup this drops the
auxiliary memory from ~5.5 GB to ~18 MB.

Sparse layout (built in `InterpolatedBounceBack._build_sparse_links`):
    link_cell (n_links,) int32  -- flat cell index, C-contig (x, y, z)
    link_dir  (n_links,) int8   -- direction q in 1..26
    link_q    (n_links,) float32 -- Bouzidi q in (0, 1]

Bouzidi (2001) linear formula at each link:
    q >= 1/2:  f_ibar(x_f) = (1/(2q)) f_post[i, x_f]
                            + ((2q - 1)/(2q)) f_post[ibar, x_f]
    q <  1/2:  f_ibar(x_f) = 2 q f_post[i, x_f]
                            + (1 - 2 q) f_post[i, x_f - c_i]

ibar = opp[i]. No write-write race: links with the same fluid cell write
to different f[opp(q_i)] slots.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_IBB_D3Q27_KERNEL = r'''
extern "C" __global__
void ibb_apply_d3q27(
    float*       __restrict__ f,             // (27, N) post-streaming, modified
    const float* __restrict__ f_post,        // (27, N) post-collision source
    const int*   __restrict__ link_cell,     // (n_links,) flat cell idx
    const char*  __restrict__ link_dir,      // (n_links,) direction q in 1..26
    const float* __restrict__ link_q,        // (n_links,) q in (0, 1]
    const int n_links,
    const int Nx, const int Ny, const int Nz
) {
    int link_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (link_idx >= n_links) return;

    long long N = (long long)Nx * Ny * Nz;   // 64-bit: q*N overflows int32 at N>~79.5M
    int cell = link_cell[link_idx];
    int q    = (int)link_dir[link_idx];
    float qv = link_q[link_idx];

    // D3Q27 lattice
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
    const int opp[27] = {
        0,  2, 1, 4, 3, 6, 5,
        10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
        26, 25, 24, 23, 22, 21, 20, 19
    };

    int oq = opp[q];
    float fp_q  = f_post[q  * N + cell];
    float fp_oq = f_post[oq * N + cell];

    float result;
    if (qv >= 0.5f) {
        float two_q  = 2.0f * qv;
        float inv_2q = 1.0f / two_q;
        result = inv_2q * fp_q + (two_q - 1.0f) * inv_2q * fp_oq;
    } else {
        // Upstream: x_f - c_i, derived from the cell's (ix, iy, iz)
        int ix = cell / (Ny * Nz);
        int rem = cell - ix * (Ny * Nz);
        int iy = rem / Nz;
        int iz = rem - iy * Nz;
        int up_ix = (ix - cx[q] + Nx) % Nx;
        int up_iy = (iy - cy[q] + Ny) % Ny;
        int up_iz = (iz - cz[q] + Nz) % Nz;
        int up_cell = up_ix * (Ny * Nz) + up_iy * Nz + up_iz;
        float fp_q_up = f_post[q * N + up_cell];
        result = 2.0f * qv * fp_q + (1.0f - 2.0f * qv) * fp_q_up;
    }

    f[oq * N + cell] = result;
}

extern "C" __global__
void ibb_reset_solid_d3q27(
    float*       __restrict__ f,             // (27, N) modified
    const bool*  __restrict__ solid_mask,    // (N,) True = solid
    const long long N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (!solid_mask[idx]) return;

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

    #pragma unroll
    for (int q = 0; q < 27; q++) {
        f[q * N + idx] = w[q];
    }
}
'''


class IBBKernelD3Q27:
    """CUDA Bouzidi linear IBB kernel for D3Q27 (sparse, stationary wall).

    Usage:
        >>> kernel = IBBKernelD3Q27()
        >>> kernel.apply(f, f_post,
        ...              link_cell, link_dir, link_q, n_links,
        ...              Nx, Ny, Nz)
        >>> kernel.reset_solid(f, solid_mask, N)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel_apply = None
        self._kernel_reset = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel_apply = cp.RawKernel(
            _IBB_D3Q27_KERNEL, 'ibb_apply_d3q27',
            options=('--use_fast_math',),
        )
        self._kernel_reset = cp.RawKernel(
            _IBB_D3Q27_KERNEL, 'ibb_reset_solid_d3q27',
            options=('--use_fast_math',),
        )

    def apply(
        self,
        f: 'npt.NDArray',
        f_post: 'npt.NDArray',
        link_cell: 'npt.NDArray',
        link_dir: 'npt.NDArray',
        link_q: 'npt.NDArray',
        n_links: int,
        Nx: int, Ny: int, Nz: int,
    ) -> None:
        """Apply Bouzidi linear IBB over the sparse link list."""
        if self._kernel_apply is None:
            self._compile()
        if n_links == 0:
            return

        import cupy as cp
        grid = (n_links + self._block_size - 1) // self._block_size

        self._kernel_apply(
            (grid,), (self._block_size,),
            (f, f_post, link_cell, link_dir, link_q,
             cp.int32(n_links),
             cp.int32(Nx), cp.int32(Ny), cp.int32(Nz)),
        )

    def reset_solid(
        self,
        f: 'npt.NDArray',
        solid_mask: 'npt.NDArray',
        N: int,
    ) -> None:
        """Reset solid nodes to rest-state equilibrium (f_q = w_q)."""
        if self._kernel_reset is None:
            self._compile()

        import cupy as cp
        grid = (N + self._block_size - 1) // self._block_size

        self._kernel_reset(
            (grid,), (self._block_size,),
            (f, solid_mask, cp.int64(N)),
        )
