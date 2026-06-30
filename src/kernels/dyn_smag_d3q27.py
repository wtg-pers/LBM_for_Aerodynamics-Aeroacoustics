"""Dynamic Smagorinsky eddy viscosity kernel (D3Q27): u -> nu_t.

3D analogue of dyn_smag_d2q9.py. Uses 5×5×5 u stencil per thread (heavy
register pressure but adequate; reduce block size if needed).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_DYN_SMAG_D3Q27_KERNEL = r'''
extern "C" __global__
void dyn_smag_d3q27(
    const float* __restrict__ u_x,
    const float* __restrict__ u_y,
    const float* __restrict__ u_z,
    float*       __restrict__ nu_t,
    const int Nx, const int Ny, const int Nz,
    const float dx,
    const float Cs_max_sq,
    const float alpha_sq,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long N = (long long)Nx * Ny * Nz;   // 64-bit: q*N overflows int32 at N>~79.5M
    if (idx >= N) return;
    int Nyz = Ny * Nz;
    int ix = idx / Nyz;
    int rem = idx - ix * Nyz;
    int iy = rem / Nz;
    int iz = rem - iy * Nz;

    // 5x5x5 u stencil with periodic wrap.  Each axis offset di in [-2..2].
    // Indexing convention: ux[di+2][dj+2][dk+2].
    float ux[5][5][5], uy[5][5][5], uz[5][5][5];
    #pragma unroll
    for (int di = -2; di <= 2; ++di) {
        int ixn = ((ix + di) % Nx + Nx) % Nx;
        #pragma unroll
        for (int dj = -2; dj <= 2; ++dj) {
            int iyn = ((iy + dj) % Ny + Ny) % Ny;
            #pragma unroll
            for (int dk = -2; dk <= 2; ++dk) {
                int izn = ((iz + dk) % Nz + Nz) % Nz;
                int id = ixn * Nyz + iyn * Nz + izn;
                ux[di+2][dj+2][dk+2] = u_x[id];
                uy[di+2][dj+2][dk+2] = u_y[id];
                uz[di+2][dj+2][dk+2] = u_z[id];
            }
        }
    }

    // |S| at center
    float gxx_c = 0.5f * (ux[3][2][2] - ux[1][2][2]);
    float gxy_c = 0.5f * (ux[2][3][2] - ux[2][1][2]);
    float gxz_c = 0.5f * (ux[2][2][3] - ux[2][2][1]);
    float gyx_c = 0.5f * (uy[3][2][2] - uy[1][2][2]);
    float gyy_c = 0.5f * (uy[2][3][2] - uy[2][1][2]);
    float gyz_c = 0.5f * (uy[2][2][3] - uy[2][2][1]);
    float gzx_c = 0.5f * (uz[3][2][2] - uz[1][2][2]);
    float gzy_c = 0.5f * (uz[2][3][2] - uz[2][1][2]);
    float gzz_c = 0.5f * (uz[2][2][3] - uz[2][2][1]);
    float sxx_c = gxx_c, syy_c = gyy_c, szz_c = gzz_c;
    float sxy_c = 0.5f * (gxy_c + gyx_c);
    float sxz_c = 0.5f * (gxz_c + gzx_c);
    float syz_c = 0.5f * (gyz_c + gzy_c);
    float s_dd_c = sxx_c*sxx_c + syy_c*syy_c + szz_c*szz_c
                 + 2.0f*(sxy_c*sxy_c + sxz_c*sxz_c + syz_c*syz_c);
    float s_mag_c = sqrtf(2.0f * s_dd_c);

    // Sums for test filter (3x3x3 box, 27 cells)
    float S_xx_h = 0, S_yy_h = 0, S_zz_h = 0;
    float S_xy_h = 0, S_xz_h = 0, S_yz_h = 0;
    float SS_xx_h = 0, SS_yy_h = 0, SS_zz_h = 0;
    float SS_xy_h = 0, SS_xz_h = 0, SS_yz_h = 0;
    float ux_h = 0, uy_h = 0, uz_h = 0;
    float uu_xx = 0, uu_yy = 0, uu_zz = 0;
    float uu_xy = 0, uu_xz = 0, uu_yz = 0;

    #pragma unroll
    for (int bi = 0; bi < 3; ++bi) {
        #pragma unroll
        for (int bj = 0; bj < 3; ++bj) {
            #pragma unroll
            for (int bk = 0; bk < 3; ++bk) {
                int i = bi + 1, j = bj + 1, k = bk + 1;
                float gxx = 0.5f * (ux[i+1][j][k] - ux[i-1][j][k]);
                float gxy = 0.5f * (ux[i][j+1][k] - ux[i][j-1][k]);
                float gxz = 0.5f * (ux[i][j][k+1] - ux[i][j][k-1]);
                float gyx = 0.5f * (uy[i+1][j][k] - uy[i-1][j][k]);
                float gyy = 0.5f * (uy[i][j+1][k] - uy[i][j-1][k]);
                float gyz = 0.5f * (uy[i][j][k+1] - uy[i][j][k-1]);
                float gzx = 0.5f * (uz[i+1][j][k] - uz[i-1][j][k]);
                float gzy = 0.5f * (uz[i][j+1][k] - uz[i][j-1][k]);
                float gzz = 0.5f * (uz[i][j][k+1] - uz[i][j][k-1]);
                float sx = gxx, sy = gyy, sz = gzz;
                float sxy = 0.5f * (gxy + gyx);
                float sxz = 0.5f * (gxz + gzx);
                float syz = 0.5f * (gyz + gzy);
                float sd = sx*sx + sy*sy + sz*sz
                         + 2.0f*(sxy*sxy + sxz*sxz + syz*syz);
                float sm = sqrtf(2.0f * sd);
                S_xx_h += sx;  S_yy_h += sy;  S_zz_h += sz;
                S_xy_h += sxy; S_xz_h += sxz; S_yz_h += syz;
                SS_xx_h += sm*sx;  SS_yy_h += sm*sy;  SS_zz_h += sm*sz;
                SS_xy_h += sm*sxy; SS_xz_h += sm*sxz; SS_yz_h += sm*syz;
                float uxc = ux[i][j][k], uyc = uy[i][j][k], uzc = uz[i][j][k];
                ux_h += uxc; uy_h += uyc; uz_h += uzc;
                uu_xx += uxc*uxc; uu_yy += uyc*uyc; uu_zz += uzc*uzc;
                uu_xy += uxc*uyc; uu_xz += uxc*uzc; uu_yz += uyc*uzc;
            }
        }
    }
    const float inv27 = 1.0f / 27.0f;
    S_xx_h *= inv27; S_yy_h *= inv27; S_zz_h *= inv27;
    S_xy_h *= inv27; S_xz_h *= inv27; S_yz_h *= inv27;
    SS_xx_h *= inv27; SS_yy_h *= inv27; SS_zz_h *= inv27;
    SS_xy_h *= inv27; SS_xz_h *= inv27; SS_yz_h *= inv27;
    ux_h *= inv27; uy_h *= inv27; uz_h *= inv27;
    uu_xx *= inv27; uu_yy *= inv27; uu_zz *= inv27;
    uu_xy *= inv27; uu_xz *= inv27; uu_yz *= inv27;

    // |S_hat|
    float S_h_dd = S_xx_h*S_xx_h + S_yy_h*S_yy_h + S_zz_h*S_zz_h
                 + 2.0f*(S_xy_h*S_xy_h + S_xz_h*S_xz_h + S_yz_h*S_yz_h);
    float S_mag_h = sqrtf(2.0f * S_h_dd);

    // Leonard L_ij
    float L_xx = uu_xx - ux_h * ux_h;
    float L_yy = uu_yy - uy_h * uy_h;
    float L_zz = uu_zz - uz_h * uz_h;
    float L_xy = uu_xy - ux_h * uy_h;
    float L_xz = uu_xz - ux_h * uz_h;
    float L_yz = uu_yz - uy_h * uz_h;
    float Lkk = L_xx + L_yy + L_zz;
    float third = (1.0f / 3.0f) * Lkk;
    float Ld_xx = L_xx - third;
    float Ld_yy = L_yy - third;
    float Ld_zz = L_zz - third;

    // M_ij = 2[(|S|S_ij)_hat - alpha^2 |S_hat| S_hat_ij]
    float M_xx = 2.0f * (SS_xx_h - alpha_sq * S_mag_h * S_xx_h);
    float M_yy = 2.0f * (SS_yy_h - alpha_sq * S_mag_h * S_yy_h);
    float M_zz = 2.0f * (SS_zz_h - alpha_sq * S_mag_h * S_zz_h);
    float M_xy = 2.0f * (SS_xy_h - alpha_sq * S_mag_h * S_xy_h);
    float M_xz = 2.0f * (SS_xz_h - alpha_sq * S_mag_h * S_xz_h);
    float M_yz = 2.0f * (SS_yz_h - alpha_sq * S_mag_h * S_yz_h);

    float LM = Ld_xx*M_xx + Ld_yy*M_yy + Ld_zz*M_zz
             + 2.0f * (L_xy*M_xy + L_xz*M_xz + L_yz*M_yz);
    float MM = M_xx*M_xx + M_yy*M_yy + M_zz*M_zz
             + 2.0f * (M_xy*M_xy + M_xz*M_xz + M_yz*M_yz);

    float Cs_sq = (MM > eps) ? (LM / (MM * dx * dx)) : 0.0f;
    Cs_sq = fmaxf(0.0f, fminf(Cs_max_sq, Cs_sq));

    nu_t[idx] = Cs_sq * dx * dx * s_mag_c;
}
'''


class DynSmagKernelD3Q27:
    """u -> nu_t (Dynamic Smagorinsky) for D3Q27."""

    def __init__(self, block_size: int = 64) -> None:
        # Smaller block size due to heavy register pressure (5×5×5 u stencil).
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _DYN_SMAG_D3Q27_KERNEL, 'dyn_smag_d3q27',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        u_x: 'npt.NDArray',
        u_y: 'npt.NDArray',
        u_z: 'npt.NDArray',
        nu_t: 'npt.NDArray',
        Nx: int, Ny: int, Nz: int,
        dx: float = 1.0,
        Cs_max: float = 0.5,
        alpha_sq: float = 3.0,
        eps: float = 1e-12,
    ) -> None:
        if self._kernel is None:
            self._compile()
        import cupy as cp
        N = Nx * Ny * Nz
        grid = (N + self._block_size - 1) // self._block_size
        self._kernel(
            (grid,), (self._block_size,),
            (u_x, u_y, u_z, nu_t,
             cp.int32(Nx), cp.int32(Ny), cp.int32(Nz),
             cp.float32(dx), cp.float32(Cs_max * Cs_max),
             cp.float32(alpha_sq), cp.float32(eps)),
        )
