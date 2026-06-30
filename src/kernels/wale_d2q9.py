"""WALE eddy viscosity kernel (D2Q9): u -> nu_t.

Computes WALE (Nicoud & Ducros 1999) eddy viscosity per cell using central
finite differences of the velocity field.

Formula:
    g_ab    = du_a / dx_b                       (velocity gradient tensor)
    S_ab    = 0.5 (g_ab + g_ba)                 (rate of strain)
    g2_ab   = sum_c g_ac g_cb                   (squared gradient tensor)
    Sd_ab   = 0.5 (g2_ab + g2_ba) - (1/3) delta_ab tr(g2)   (traceless symm)
    nu_t    = (Cw * Delta)^2 * (Sd:Sd)^{3/2}
                                / [(S:S)^{5/2} + (Sd:Sd)^{5/4} + eps]

Derivative is central FD with periodic wrap on both axes:
    du/dx |_{i,j} = (u_{i+1, j} - u_{i-1, j}) / 2
    du/dy |_{i,j} = (u_{i, j+1} - u_{i, j-1}) / 2

For non-periodic faces the wrap effectively gives a one-cell-shifted central
FD; the user should mask out values at edge cells in post-processing or use
periodic BCs (the typical LES test case is periodic).

CPU reference: src.turbulence.sgs.nu_t_wale_2d.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_WALE_D2Q9_KERNEL = r'''
extern "C" __global__
void wale_d2q9(
    const float* __restrict__ u_x,    // (Nx*Ny,)  C-contiguous (x-fast, y-slow inverted: idx = ix*Ny + iy)
    const float* __restrict__ u_y,
    float*       __restrict__ nu_t,
    const int Nx, const int Ny,
    const float Cw, const float dx,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Nx * Ny;
    if (idx >= N) return;

    // Unflatten to (ix, iy) in C-order (ix slow, iy fast):
    //   idx = ix * Ny + iy
    int ix = idx / Ny;
    int iy = idx - ix * Ny;

    int ixp = (ix + 1) % Nx;
    int ixm = (ix - 1 + Nx) % Nx;
    int iyp = (iy + 1) % Ny;
    int iym = (iy - 1 + Ny) % Ny;

    int idx_xp = ixp * Ny + iy;
    int idx_xm = ixm * Ny + iy;
    int idx_yp = ix  * Ny + iyp;
    int idx_ym = ix  * Ny + iym;

    // Central FD gradients (a = component, b = derivative direction)
    // gxx = du_x/dx, gxy = du_x/dy, gyx = du_y/dx, gyy = du_y/dy
    float gxx = 0.5f * (u_x[idx_xp] - u_x[idx_xm]);
    float gxy = 0.5f * (u_x[idx_yp] - u_x[idx_ym]);
    float gyx = 0.5f * (u_y[idx_xp] - u_y[idx_xm]);
    float gyy = 0.5f * (u_y[idx_yp] - u_y[idx_ym]);

    // Symmetric strain rate
    float sxx = gxx;
    float syy = gyy;
    float sxy = 0.5f * (gxy + gyx);
    float s_dd = sxx*sxx + syy*syy + 2.0f*sxy*sxy;

    // g^2_ab = sum_c g_ac g_cb
    float g2_xx = gxx*gxx + gxy*gyx;
    float g2_yy = gyx*gxy + gyy*gyy;
    float g2_xy = gxx*gxy + gxy*gyy;
    float g2_yx = gyx*gxx + gyy*gyx;
    float trace = g2_xx + g2_yy;
    float sd_xx = g2_xx - trace * (1.0f/3.0f);
    float sd_yy = g2_yy - trace * (1.0f/3.0f);
    float sd_xy = 0.5f * (g2_xy + g2_yx);
    float sd_dd = sd_xx*sd_xx + sd_yy*sd_yy + 2.0f*sd_xy*sd_xy;

    float num = powf(sd_dd, 1.5f);
    float den = powf(s_dd, 2.5f) + powf(sd_dd, 1.25f) + eps;
    nu_t[idx] = (Cw * dx) * (Cw * dx) * num / den;
}
'''


class WALEKernelD2Q9:
    """u -> nu_t (WALE) global kernel for D2Q9 (periodic central FD)."""

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _WALE_D2Q9_KERNEL, 'wale_d2q9',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        u_x: 'npt.NDArray',
        u_y: 'npt.NDArray',
        nu_t: 'npt.NDArray',
        Nx: int,
        Ny: int,
        Cw: float = 0.5,
        dx: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        """Compute nu_t from (u_x, u_y) on a Nx x Ny grid (periodic wrap)."""
        if self._kernel is None:
            self._compile()
        import cupy as cp
        N = Nx * Ny
        grid = (N + self._block_size - 1) // self._block_size
        self._kernel(
            (grid,), (self._block_size,),
            (u_x, u_y, nu_t,
             cp.int32(Nx), cp.int32(Ny),
             cp.float32(Cw), cp.float32(dx), cp.float32(eps)),
        )
