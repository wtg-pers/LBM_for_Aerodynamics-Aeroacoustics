"""WALE eddy viscosity kernel (D3Q27): u -> nu_t.

3D analogue of `wale_d2q9.py`. Central FD with periodic wrap on all three
axes. CPU reference: src.turbulence.sgs.nu_t_wale_3d.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_WALE_D3Q27_KERNEL = r'''
extern "C" __global__
void wale_d3q27(
    const float* __restrict__ u_x,    // (N,) where N = Nx*Ny*Nz, C-order (ix*Ny*Nz + iy*Nz + iz)
    const float* __restrict__ u_y,
    const float* __restrict__ u_z,
    float*       __restrict__ nu_t,
    const int Nx, const int Ny, const int Nz,
    const float Cw, const float dx,
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

    int ixp = (ix + 1) % Nx;
    int ixm = (ix - 1 + Nx) % Nx;
    int iyp = (iy + 1) % Ny;
    int iym = (iy - 1 + Ny) % Ny;
    int izp = (iz + 1) % Nz;
    int izm = (iz - 1 + Nz) % Nz;

    int i_xp = ixp * Nyz + iy  * Nz + iz;
    int i_xm = ixm * Nyz + iy  * Nz + iz;
    int i_yp = ix  * Nyz + iyp * Nz + iz;
    int i_ym = ix  * Nyz + iym * Nz + iz;
    int i_zp = ix  * Nyz + iy  * Nz + izp;
    int i_zm = ix  * Nyz + iy  * Nz + izm;

    // gradient tensor g[a][b] = du_a/dx_b
    float gxx = 0.5f * (u_x[i_xp] - u_x[i_xm]);
    float gxy = 0.5f * (u_x[i_yp] - u_x[i_ym]);
    float gxz = 0.5f * (u_x[i_zp] - u_x[i_zm]);
    float gyx = 0.5f * (u_y[i_xp] - u_y[i_xm]);
    float gyy = 0.5f * (u_y[i_yp] - u_y[i_ym]);
    float gyz = 0.5f * (u_y[i_zp] - u_y[i_zm]);
    float gzx = 0.5f * (u_z[i_xp] - u_z[i_xm]);
    float gzy = 0.5f * (u_z[i_yp] - u_z[i_ym]);
    float gzz = 0.5f * (u_z[i_zp] - u_z[i_zm]);

    // S = sym(g)
    float sxx = gxx;
    float syy = gyy;
    float szz = gzz;
    float sxy = 0.5f * (gxy + gyx);
    float sxz = 0.5f * (gxz + gzx);
    float syz = 0.5f * (gyz + gzy);
    float s_dd = sxx*sxx + syy*syy + szz*szz
               + 2.0f*(sxy*sxy + sxz*sxz + syz*syz);

    // g^2 = g . g
    float g2_xx = gxx*gxx + gxy*gyx + gxz*gzx;
    float g2_yy = gyx*gxy + gyy*gyy + gyz*gzy;
    float g2_zz = gzx*gxz + gzy*gyz + gzz*gzz;
    float g2_xy = gxx*gxy + gxy*gyy + gxz*gzy;
    float g2_yx = gyx*gxx + gyy*gyx + gyz*gzx;
    float g2_xz = gxx*gxz + gxy*gyz + gxz*gzz;
    float g2_zx = gzx*gxx + gzy*gyx + gzz*gzx;
    float g2_yz = gyx*gxz + gyy*gyz + gyz*gzz;
    float g2_zy = gzx*gxy + gzy*gyy + gzz*gzy;

    float trace = g2_xx + g2_yy + g2_zz;
    float third = trace * (1.0f/3.0f);
    float sd_xx = g2_xx - third;
    float sd_yy = g2_yy - third;
    float sd_zz = g2_zz - third;
    float sd_xy = 0.5f * (g2_xy + g2_yx);
    float sd_xz = 0.5f * (g2_xz + g2_zx);
    float sd_yz = 0.5f * (g2_yz + g2_zy);
    float sd_dd = sd_xx*sd_xx + sd_yy*sd_yy + sd_zz*sd_zz
                + 2.0f*(sd_xy*sd_xy + sd_xz*sd_xz + sd_yz*sd_yz);

    float num = powf(sd_dd, 1.5f);
    float den = powf(s_dd, 2.5f) + powf(sd_dd, 1.25f) + eps;
    nu_t[idx] = (Cw * dx) * (Cw * dx) * num / den;
}
'''


class WALEKernelD3Q27:
    """u -> nu_t (WALE) global kernel for D3Q27 (periodic central FD)."""

    def __init__(self, block_size: int = 128) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _WALE_D3Q27_KERNEL, 'wale_d3q27',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        u_x: 'npt.NDArray',
        u_y: 'npt.NDArray',
        u_z: 'npt.NDArray',
        nu_t: 'npt.NDArray',
        Nx: int, Ny: int, Nz: int,
        Cw: float = 0.5,
        dx: float = 1.0,
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
             cp.float32(Cw), cp.float32(dx), cp.float32(eps)),
        )
