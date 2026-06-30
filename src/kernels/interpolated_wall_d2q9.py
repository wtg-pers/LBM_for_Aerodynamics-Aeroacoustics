"""
CUDA Bouzidi-linear Interpolated Bounce-Back Kernel for D2Q9

GPU port of `src/boundary/interpolated_wall.InterpolatedBounceBack.apply()`.
Replaces the Python loop over 8 directions (each branching on q≥0.5 vs
q<0.5 with masked CuPy ops) with a single fused kernel.

Bouzidi (2001) linear IBB, stationary wall, applied AFTER streaming using
post-collision f_post at time t:

    q ≥ 1/2:  f_ī(x_f) = (1/(2q))·f_post[i, x_f]
                       + ((2q-1)/(2q))·f_post[ī, x_f]
    q < 1/2:  f_ī(x_f) = 2q·f_post[i, x_f]
                       + (1 - 2q)·f_post[i, x_f - c_i]

where ī = opp[i]. The q<1/2 branch reads the post-collision value one
node further into the fluid (upstream of the link).

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_IBB_D2Q9_KERNEL = r'''
extern "C" __global__
void ibb_apply_d2q9(
    float*       __restrict__ f,             // (9, N) post-streaming, modified
    const float* __restrict__ f_post,        // (9, N) post-collision source
    const bool*  __restrict__ needs_bounce,  // (9, N) link mask
    const float* __restrict__ q_fraction,    // (9, N) q in (0, 1], 0.5 default
    const int Nx, const int Ny
) {
    int N = Nx * Ny;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // C-contiguous (Nx, Ny): y varies fastest
    int ix = idx / Ny;
    int iy = idx - ix * Ny;

    // D2Q9 lattice
    const int cx[9]  = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    const int cy[9]  = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
    const int opp[9] = { 0,  3,  4,  1,  2,  7,  8,  5,  6 };

    #pragma unroll
    for (int q = 1; q < 9; q++) {
        if (!needs_bounce[q * N + idx]) continue;

        float qv = q_fraction[q * N + idx];
        int   oq = opp[q];

        float fp_q   = f_post[q  * N + idx];
        float fp_oq  = f_post[oq * N + idx];

        float result;
        if (qv >= 0.5f) {
            // Local read at x_f
            float two_q = 2.0f * qv;
            float inv_2q = 1.0f / two_q;
            result = inv_2q * fp_q + (two_q - 1.0f) * inv_2q * fp_oq;
        } else {
            // Upstream read: f_post[q, x_f - c_i]  (c_i points into solid,
            // so x_f - c_i is one step further into the fluid).
            int up_ix = (ix - cx[q] + Nx) % Nx;
            int up_iy = (iy - cy[q] + Ny) % Ny;
            int up_idx = up_ix * Ny + up_iy;
            float fp_q_up = f_post[q * N + up_idx];
            result = 2.0f * qv * fp_q
                   + (1.0f - 2.0f * qv) * fp_q_up;
        }

        f[oq * N + idx] = result;
    }
}

extern "C" __global__
void ibb_reset_solid_d2q9(
    float*       __restrict__ f,             // (9, N) modified
    const bool*  __restrict__ solid_mask,    // (N,) True=solid
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (!solid_mask[idx]) return;

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


class IBBKernelD2Q9:
    """CUDA Bouzidi linear IBB kernel for D2Q9 (stationary wall).

    Usage:
        >>> kernel = IBBKernelD2Q9()
        >>> kernel.apply(f, f_post, needs_bounce, q_fraction, Nx, Ny)
        >>> kernel.reset_solid(f, solid_mask, N)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel_apply = None
        self._kernel_reset = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel_apply = cp.RawKernel(
            _IBB_D2Q9_KERNEL, 'ibb_apply_d2q9',
            options=('--use_fast_math',),
        )
        self._kernel_reset = cp.RawKernel(
            _IBB_D2Q9_KERNEL, 'ibb_reset_solid_d2q9',
            options=('--use_fast_math',),
        )

    def apply(
        self,
        f: 'npt.NDArray',
        f_post: 'npt.NDArray',
        needs_bounce: 'npt.NDArray',
        q_fraction: 'npt.NDArray',
        Nx: int, Ny: int,
    ) -> None:
        """Apply Bouzidi linear IBB.

        Args:
            f:            Post-streaming distribution (9, Nx*Ny) -- modified
            f_post:       Post-collision source (9, Nx*Ny)
            needs_bounce: Boundary link mask (9, Nx*Ny) bool
            q_fraction:   q ∈ (0, 1] per link (9, Nx*Ny) float32
            Nx, Ny:       Domain dimensions (for upstream wraparound)
        """
        if self._kernel_apply is None:
            self._compile()

        import cupy as cp
        N = Nx * Ny
        grid = (N + self._block_size - 1) // self._block_size

        self._kernel_apply(
            (grid,), (self._block_size,),
            (f, f_post, needs_bounce, q_fraction,
             cp.int32(Nx), cp.int32(Ny)),
        )

    def reset_solid(
        self,
        f: 'npt.NDArray',
        solid_mask: 'npt.NDArray',
        N: int,
    ) -> None:
        """Reset solid nodes to rest-state equilibrium (f_q = w_q).

        Args:
            f:          Distribution (9, N) -- modified
            solid_mask: Boolean mask (N,) True=solid
            N:          Total nodes (Nx*Ny)
        """
        if self._kernel_reset is None:
            self._compile()

        import cupy as cp
        grid = (N + self._block_size - 1) // self._block_size

        self._kernel_reset(
            (grid,), (self._block_size,),
            (f, solid_mask, cp.int32(N)),
        )
