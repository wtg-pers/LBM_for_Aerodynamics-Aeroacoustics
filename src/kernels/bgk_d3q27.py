"""
Fused BGK D3Q27 Collision Kernel (CUDA)

Replaces: macroscopic.compute() + Guo correction + bgk.collide()
with a single CUDA kernel launch.

Physical Process (per node):
    1. Compute rho = Sum f_i,  u = Sum c_i f_i / rho
    2. Guo correction: u += F/(2rho)  (if force present)
    3. Equilibrium: f_eq = w_i rho (1 + 3c*u + 4.5(c*u)^2 - 1.5|u|^2)
    4. BGK relaxation: f_post = f - omega(f - f_eq)
    5. Guo source term: f_post += S_i  (if force present)

Memory traffic per node (float32):
    Read:  27 x 4B = 108B  (f)  +  12B (force, if present)
    Write: 27 x 4B = 108B  (f_post)  +  4B (rho)  +  12B (u)
    Total: 232B  (vs ~1500B+ for CuPy multi-launch path)

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


# --- CUDA kernel source code ----------------------------------

_BGK_D3Q27_KERNEL = r'''
extern "C" __global__
void bgk_collide_d3q27(
    const float* __restrict__ f_in,       // (27, N) input distribution
    float*       __restrict__ f_post,     // (27, N) post-collision output
    float*       __restrict__ rho_out,    // (N,) density output
    float*       __restrict__ u_out,      // (3, N) velocity output
    const float* __restrict__ force,      // (3, N) body force or NULL
    const float omega,                     // relaxation rate 1/tau
    const long long N                      // total nodes = Nx*Ny*Nz (64-bit: q*N overflow)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // -- D3Q27 lattice constants (hardcoded for performance) --
    // Velocity vectors c[dim][Q]
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

    // Weights w[Q]
    const float w[27] = {
        8.0f/27.0f,                                               // rest
        2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f,                     // face (6)
        2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f,
        1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,         // edge (12)
        1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,
        1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,
        1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f,     // corner (8)
        1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f
    };

    // -- Step 1: Load f and compute macroscopic --------------
    float f_local[27];
    float rho = 0.0f;
    float mom_x = 0.0f, mom_y = 0.0f, mom_z = 0.0f;

    for (int q = 0; q < 27; q++) {
        float fq = f_in[q * N + idx];
        f_local[q] = fq;
        rho += fq;
        mom_x += cx[q] * fq;
        mom_y += cy[q] * fq;
        mom_z += cz[q] * fq;
    }

    float inv_rho = 1.0f / rho;
    float ux = mom_x * inv_rho;
    float uy = mom_y * inv_rho;
    float uz = mom_z * inv_rho;

    // -- Step 2: Guo velocity correction ---------------------
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    if (force != NULL) {
        Fx = force[0 * N + idx];
        Fy = force[1 * N + idx];
        Fz = force[2 * N + idx];
        float half_inv_rho = 0.5f * inv_rho;
        ux += Fx * half_inv_rho;
        uy += Fy * half_inv_rho;
        uz += Fz * half_inv_rho;
    }

    // -- Step 3: Write macroscopic output --------------------
    rho_out[idx] = rho;
    u_out[0 * N + idx] = ux;
    u_out[1 * N + idx] = uy;
    u_out[2 * N + idx] = uz;

    // -- Step 4: BGK collision -------------------------------
    //   f_eq = w_i rho (1 + 3(c*u) + 4.5(c*u)^2 - 1.5|u|^2)
    //   f_post = f - omega(f - f_eq)
    float usqr = ux*ux + uy*uy + uz*uz;

    for (int q = 0; q < 27; q++) {
        float cu = (float)cx[q]*ux + (float)cy[q]*uy + (float)cz[q]*uz;
        float f_eq = w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usqr);
        f_local[q] = f_local[q] - omega * (f_local[q] - f_eq);
    }

    // -- Step 5: Guo forcing source term ---------------------
    //   S_i = (1 - 1/(2tau)) w_i [(c_i-u)/cs^2 + (c_i*u)c_i/cs^4] * F
    if (force != NULL) {
        float prefactor = 1.0f - 0.5f * omega;  // (1 - 1/(2tau))

        for (int q = 0; q < 27; q++) {
            float cxq = (float)cx[q], cyq = (float)cy[q], czq = (float)cz[q];

            // (c_i * F) and (c_i * u)
            float ci_dot_F = cxq*Fx + cyq*Fy + czq*Fz;
            float ci_dot_u = cxq*ux + cyq*uy + czq*uz;
            float u_dot_F  = ux*Fx + uy*Fy + uz*Fz;

            // term1 = (c_i*F - u*F) * 3.0  (inv_cs2 = 3)
            // term2 = (c_i*u)(c_i*F) * 9.0  (inv_cs4 = 9)
            float S_i = prefactor * w[q] * (
                (ci_dot_F - u_dot_F) * 3.0f +
                ci_dot_u * ci_dot_F * 9.0f
            );

            f_local[q] += S_i;
        }
    }

    // -- Step 6: Write post-collision output -----------------
    for (int q = 0; q < 27; q++) {
        f_post[q * N + idx] = f_local[q];
    }
}
'''


# --- FP16 shifted collision kernel ---------------------------

_BGK_D3Q27_FP16S_KERNEL = r'''
#include <cuda_fp16.h>

extern "C" __global__
void bgk_collide_d3q27_fp16s(
    const __half* __restrict__ f_in,      // (27, N) FP16 shifted (f - w)
    __half*       __restrict__ f_post,    // (27, N) FP16 shifted output
    float*        __restrict__ rho_out,   // (N,) FP32
    float*        __restrict__ u_out,     // (3, N) FP32
    const float*  __restrict__ force,     // (3, N) or NULL, FP32
    const float omega,
    const long long N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // -- D3Q27 lattice constants -----------------------------
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

    // -- Step 1: Load FP16 shifted -> restore to FP32 --------
    //   f_stored = FP16(f - w),  f = FP32(f_stored) + w
    float f_local[27];
    float rho = 0.0f;
    float mom_x = 0.0f, mom_y = 0.0f, mom_z = 0.0f;

    for (int q = 0; q < 27; q++) {
        float fq = __half2float(f_in[q * N + idx]) + w[q];  // restore
        f_local[q] = fq;
        rho += fq;
        mom_x += cx[q] * fq;
        mom_y += cy[q] * fq;
        mom_z += cz[q] * fq;
    }

    float inv_rho = 1.0f / rho;
    float ux = mom_x * inv_rho;
    float uy = mom_y * inv_rho;
    float uz = mom_z * inv_rho;

    // -- Step 2: Guo velocity correction ---------------------
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    if (force != NULL) {
        Fx = force[0 * N + idx];
        Fy = force[1 * N + idx];
        Fz = force[2 * N + idx];
        float half_inv_rho = 0.5f * inv_rho;
        ux += Fx * half_inv_rho;
        uy += Fy * half_inv_rho;
        uz += Fz * half_inv_rho;
    }

    // -- Step 3: Write macroscopic (FP32) --------------------
    rho_out[idx] = rho;
    u_out[0 * N + idx] = ux;
    u_out[1 * N + idx] = uy;
    u_out[2 * N + idx] = uz;

    // -- Step 4: BGK collision (all FP32) --------------------
    float usqr = ux*ux + uy*uy + uz*uz;
    for (int q = 0; q < 27; q++) {
        float cu = (float)cx[q]*ux + (float)cy[q]*uy + (float)cz[q]*uz;
        float f_eq = w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usqr);
        f_local[q] = f_local[q] - omega * (f_local[q] - f_eq);
    }

    // -- Step 5: Guo source term (FP32) ----------------------
    if (force != NULL) {
        float prefactor = 1.0f - 0.5f * omega;
        for (int q = 0; q < 27; q++) {
            float cxq = (float)cx[q], cyq = (float)cy[q], czq = (float)cz[q];
            float ci_dot_F = cxq*Fx + cyq*Fy + czq*Fz;
            float ci_dot_u = cxq*ux + cyq*uy + czq*uz;
            float u_dot_F  = ux*Fx + uy*Fy + uz*Fz;
            float S_i = prefactor * w[q] * (
                (ci_dot_F - u_dot_F) * 3.0f + ci_dot_u * ci_dot_F * 9.0f
            );
            f_local[q] += S_i;
        }
    }

    // -- Step 6: Write FP32 -> FP16 shifted -------------------
    //   f_stored = FP16(f_post - w)
    for (int q = 0; q < 27; q++) {
        f_post[q * N + idx] = __float2half(f_local[q] - w[q]);
    }
}
'''


# --- Fused pull-collide kernel (streaming integrated) --------

_BGK_D3Q27_STREAM_COLLIDE_KERNEL = r'''
extern "C" __global__
void bgk_stream_collide_d3q27(
    const float* __restrict__ f_src,      // (27, Nx*Ny*Nz) source (prev step)
    float*       __restrict__ f_dst,      // (27, Nx*Ny*Nz) destination (this step)
    float*       __restrict__ rho_out,    // (N,) density
    float*       __restrict__ u_out,      // (3, N) velocity
    const float* __restrict__ force,      // (3, N) or NULL
    const float omega,
    const int Nx, const int Ny, const int Nz
) {
    long long N = (long long)Nx * Ny * Nz;   // 64-bit: q*N overflows int32 at N>~79.5M
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // 1D -> 3D index
    int iz = idx / (Nx * Ny);
    int rem = idx - iz * Nx * Ny;
    int iy = rem / Nx;
    int ix = rem - iy * Nx;

    // -- D3Q27 lattice constants -----------------------------
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

    // -- Step 1: Pull from neighbors (streaming) -------------
    float f_local[27];
    float rho = 0.0f;
    float mom_x = 0.0f, mom_y = 0.0f, mom_z = 0.0f;

    for (int q = 0; q < 27; q++) {
        int sx = (ix - cx[q] + Nx) % Nx;
        int sy = (iy - cy[q] + Ny) % Ny;
        int sz = (iz - cz[q] + Nz) % Nz;
        int src_idx = sz * Nx * Ny + sy * Nx + sx;

        float fq = f_src[q * N + src_idx];
        f_local[q] = fq;
        rho += fq;
        mom_x += cx[q] * fq;
        mom_y += cy[q] * fq;
        mom_z += cz[q] * fq;
    }

    float inv_rho = 1.0f / rho;
    float ux = mom_x * inv_rho;
    float uy = mom_y * inv_rho;
    float uz = mom_z * inv_rho;

    // -- Step 2: Guo velocity correction ---------------------
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    if (force != NULL) {
        Fx = force[0 * N + idx];
        Fy = force[1 * N + idx];
        Fz = force[2 * N + idx];
        float half_inv_rho = 0.5f * inv_rho;
        ux += Fx * half_inv_rho;
        uy += Fy * half_inv_rho;
        uz += Fz * half_inv_rho;
    }

    // -- Step 3: Write macroscopic ---------------------------
    rho_out[idx] = rho;
    u_out[0 * N + idx] = ux;
    u_out[1 * N + idx] = uy;
    u_out[2 * N + idx] = uz;

    // -- Step 4: BGK collision -------------------------------
    float usqr = ux*ux + uy*uy + uz*uz;
    for (int q = 0; q < 27; q++) {
        float cu = (float)cx[q]*ux + (float)cy[q]*uy + (float)cz[q]*uz;
        float f_eq = w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usqr);
        f_local[q] = f_local[q] - omega * (f_local[q] - f_eq);
    }

    // -- Step 5: Guo source term -----------------------------
    if (force != NULL) {
        float prefactor = 1.0f - 0.5f * omega;
        for (int q = 0; q < 27; q++) {
            float cxq = (float)cx[q], cyq = (float)cy[q], czq = (float)cz[q];
            float ci_dot_F = cxq*Fx + cyq*Fy + czq*Fz;
            float ci_dot_u = cxq*ux + cyq*uy + czq*uz;
            float u_dot_F  = ux*Fx + uy*Fy + uz*Fz;
            float S_i = prefactor * w[q] * (
                (ci_dot_F - u_dot_F) * 3.0f + ci_dot_u * ci_dot_F * 9.0f
            );
            f_local[q] += S_i;
        }
    }

    // -- Step 6: Write post-collision to destination ---------
    //   f_dst = post-collision (serves as f_post for BC)
    for (int q = 0; q < 27; q++) {
        f_dst[q * N + idx] = f_local[q];
    }
}
'''


# --- Python wrappers -----------------------------------------

class BGKCollideKernelD3Q27:
    """Fused BGK collision kernel for D3Q27.

    Replaces macroscopic.compute() + bgk.collide() with a single
    CUDA kernel launch. Streaming and BC remain unchanged.

    Usage:
        >>> kernel = BGKCollideKernelD3Q27(block_size=256)
        >>> kernel.launch(f, f_post, rho, u, force, omega, N)
    """

    def __init__(self, block_size: int = 256) -> None:
        """Initialize kernel (compiles on first use).

        Args:
            block_size: CUDA threads per block (default 256).
        """
        self._block_size = block_size
        self._kernel = None  # lazy compilation

    def _compile(self) -> None:
        """Compile CUDA kernel via CuPy RawKernel (first call only)."""
        import cupy as cp
        self._kernel = cp.RawKernel(
            _BGK_D3Q27_KERNEL,
            'bgk_collide_d3q27',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        f_in: 'npt.NDArray',
        f_post: 'npt.NDArray',
        rho_out: 'npt.NDArray',
        u_out: 'npt.NDArray',
        force: Optional['npt.NDArray'],
        omega: float,
        N: int,
    ) -> None:
        """Launch the fused collision kernel.

        Args:
            f_in:    Input distribution (27, N) -- read only
            f_post:  Post-collision output (27, N) -- written
            rho_out: Density output (N,) -- written
            u_out:   Velocity output (3, N) -- written
            force:   Body force (3, N) or None
            omega:   Relaxation rate 1/tau
            N:       Total number of spatial nodes
        """
        if self._kernel is None:
            self._compile()

        import cupy as cp

        grid_size = (N + self._block_size - 1) // self._block_size

        # CuPy RawKernel: pass 0 (nullptr) for NULL pointer
        force_arg = force if force is not None else cp.int32(0)

        self._kernel(
            (grid_size,),
            (self._block_size,),
            (
                f_in, f_post, rho_out, u_out,
                force_arg,
                cp.float32(omega),
                cp.int64(N),
            ),
        )


class BGKStreamCollideKernelD3Q27:
    """Fused pull-collide kernel for D3Q27 BGK.

    Combines streaming (pull from neighbors) + macroscopic + collision
    into a single CUDA kernel launch. Uses ping-pong buffers:
    f_src (previous step) -> f_dst (this step).

    f_dst contains post-collision values = f_post for BC.

    Usage:
        >>> kernel = BGKStreamCollideKernelD3Q27()
        >>> kernel.launch(f_src, f_dst, rho, u, force, omega, Nx, Ny, Nz)
        >>> # BC operates on f_dst (f_dst = f_post)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _BGK_D3Q27_STREAM_COLLIDE_KERNEL,
            'bgk_stream_collide_d3q27',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        f_src: 'npt.NDArray',
        f_dst: 'npt.NDArray',
        rho_out: 'npt.NDArray',
        u_out: 'npt.NDArray',
        force: Optional['npt.NDArray'],
        omega: float,
        Nx: int, Ny: int, Nz: int,
    ) -> None:
        """Launch fused pull-collide kernel.

        Args:
            f_src: Source distribution (27, N) -- previous step, read only
            f_dst: Destination distribution (27, N) -- this step, written
                   (= post-collision, serves as f_post for BC)
            rho_out: Density output (N,)
            u_out:   Velocity output (3, N)
            force:   Body force (3, N) or None
            omega:   Relaxation rate 1/tau
            Nx, Ny, Nz: Domain dimensions
        """
        if self._kernel is None:
            self._compile()

        import cupy as cp

        N = Nx * Ny * Nz
        grid_size = (N + self._block_size - 1) // self._block_size
        force_arg = force if force is not None else cp.int32(0)

        self._kernel(
            (grid_size,),
            (self._block_size,),
            (
                f_src, f_dst, rho_out, u_out,
                force_arg,
                cp.float32(omega),
                cp.int32(Nx), cp.int32(Ny), cp.int32(Nz),
            ),
        )


class BGKCollideKernelD3Q27_FP16S:
    """FP16 shifted BGK collision kernel for D3Q27.

    f arrays are stored as float16(f - w). Internal collision uses FP32.
    Memory: 54 B/node per f array (vs 108 for FP32).

    Usage:
        >>> kernel = BGKCollideKernelD3Q27_FP16S()
        >>> kernel.launch(f_fp16s, f_post_fp16s, rho, u, force, omega, N)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _BGK_D3Q27_FP16S_KERNEL,
            'bgk_collide_d3q27_fp16s',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        f_in: 'npt.NDArray',
        f_post: 'npt.NDArray',
        rho_out: 'npt.NDArray',
        u_out: 'npt.NDArray',
        force: Optional['npt.NDArray'],
        omega: float,
        N: int,
    ) -> None:
        """Launch FP16S collision kernel.

        Args:
            f_in:    FP16 shifted input (27, N) -- float16(f - w)
            f_post:  FP16 shifted output (27, N) -- float16(f_post - w)
            rho_out: Density (N,) -- FP32
            u_out:   Velocity (3, N) -- FP32
            force:   Body force (3, N) FP32 or None
            omega:   Relaxation rate
            N:       Total nodes
        """
        if self._kernel is None:
            self._compile()

        import cupy as cp

        grid_size = (N + self._block_size - 1) // self._block_size
        force_arg = force if force is not None else cp.int32(0)

        self._kernel(
            (grid_size,),
            (self._block_size,),
            (
                f_in, f_post, rho_out, u_out,
                force_arg,
                cp.float32(omega),
                cp.int64(N),
            ),
        )
