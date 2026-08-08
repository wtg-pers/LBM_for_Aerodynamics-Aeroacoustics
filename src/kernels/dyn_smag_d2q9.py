"""Dynamic Smagorinsky eddy viscosity kernel (D2Q9): u -> nu_t.

Germano-Lilly (1991/1992) dynamic procedure with per-cell Cs² estimate
(no spatial averaging beyond the test filter; relies on clipping for
stability). Test filter is the tensor-product 3-point box (1+1+1)/3 per
axis, giving (Δ̂/Δ)² ≈ 3.

Algorithm per cell:
    1. Read u in 5×5 stencil (centered at the cell).
    2. Compute S_ij(b) and |S|(b) at each cell b in the 3×3 box around centre
       (ie {(ix-1..ix+1)}×{(iy-1..iy+1)}); inner cells use central FD on the
       outer 5×5 u stencil.
    3. Test-filter (3×3 box average, /9) — gives û, |Ŝ|, Ŝ_ij,
       (u_i u_j)_hat, (|S| S_ij)_hat.
    4. Leonard L_ij = (u_i u_j)_hat − û_i û_j; deviatoric L^d_ij.
    5. M_ij = 2 [(|S| S_ij)_hat − α² |Ŝ| Ŝ_ij].
    6. Cs² = max(0, min(Cs_max², ⟨L^d:M⟩ / (Δ² ⟨M:M⟩))) per cell.
    7. ν_t = Cs² Δ² |S|(centre).

Reference: src.turbulence.sgs has CPU helper for the same arithmetic in
the test file.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


_DYN_SMAG_D2Q9_KERNEL = r'''
extern "C" __global__
void dyn_smag_d2q9(
    const float* __restrict__ u_x,
    const float* __restrict__ u_y,
    float*       __restrict__ nu_t,
    const int Nx, const int Ny,
    const float dx,
    const float Cs_max_sq,    // (Cs_max)^2 clipping
    const float alpha_sq,     // (Dhat/D)^2
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Nx * Ny;
    if (idx >= N) return;
    int ix = idx / Ny;
    int iy = idx - ix * Ny;

    // 5x5 u stencil (periodic wrap on (ix,iy))
    float ux[5][5], uy[5][5];
    #pragma unroll
    for (int di = -2; di <= 2; ++di) {
        int ixn = ((ix + di) % Nx + Nx) % Nx;
        #pragma unroll
        for (int dj = -2; dj <= 2; ++dj) {
            int iyn = ((iy + dj) % Ny + Ny) % Ny;
            int id = ixn * Ny + iyn;
            ux[di+2][dj+2] = u_x[id];
            uy[di+2][dj+2] = u_y[id];
        }
    }

    // |S| at center (for nu_t = Cs^2 D^2 |S|)
    float gxx_c = 0.5f * (ux[3][2] - ux[1][2]);
    float gxy_c = 0.5f * (ux[2][3] - ux[2][1]);
    float gyx_c = 0.5f * (uy[3][2] - uy[1][2]);
    float gyy_c = 0.5f * (uy[2][3] - uy[2][1]);
    float sxx_c = gxx_c, syy_c = gyy_c, sxy_c = 0.5f * (gxy_c + gyx_c);
    float s_dd_c = sxx_c*sxx_c + syy_c*syy_c + 2.0f*sxy_c*sxy_c;
    float s_mag_c = sqrtf(2.0f * s_dd_c);

    // Compute S_ij, |S|, |S|*S_ij at all 3x3 box cells around center
    // Box index (bi, bj) in [0..2] maps to 5x5 index (bi+1, bj+1) in [1..3].
    float SS_xx_hat = 0.f, SS_yy_hat = 0.f, SS_xy_hat = 0.f;
    float S_xx_hat = 0.f, S_yy_hat = 0.f, S_xy_hat = 0.f;
    float ux_hat = 0.f, uy_hat = 0.f;
    float uu_xx = 0.f, uu_yy = 0.f, uu_xy = 0.f;

    #pragma unroll
    for (int bi = 0; bi < 3; ++bi) {
        #pragma unroll
        for (int bj = 0; bj < 3; ++bj) {
            int i = bi + 1, j = bj + 1;
            float gxx = 0.5f * (ux[i+1][j] - ux[i-1][j]);
            float gxy = 0.5f * (ux[i][j+1] - ux[i][j-1]);
            float gyx = 0.5f * (uy[i+1][j] - uy[i-1][j]);
            float gyy = 0.5f * (uy[i][j+1] - uy[i][j-1]);
            float sx = gxx, sy = gyy, sxy = 0.5f * (gxy + gyx);
            float sd = sx*sx + sy*sy + 2.0f*sxy*sxy;
            float sm = sqrtf(2.0f * sd);
            S_xx_hat += sx;  S_yy_hat += sy;  S_xy_hat += sxy;
            SS_xx_hat += sm * sx;  SS_yy_hat += sm * sy;  SS_xy_hat += sm * sxy;
            float uxc = ux[i][j], uyc = uy[i][j];
            ux_hat += uxc;  uy_hat += uyc;
            uu_xx += uxc * uxc;
            uu_yy += uyc * uyc;
            uu_xy += uxc * uyc;
        }
    }
    const float inv9 = 1.0f / 9.0f;
    S_xx_hat *= inv9;  S_yy_hat *= inv9;  S_xy_hat *= inv9;
    SS_xx_hat *= inv9; SS_yy_hat *= inv9; SS_xy_hat *= inv9;
    ux_hat *= inv9;    uy_hat *= inv9;
    uu_xx *= inv9;     uu_yy *= inv9;     uu_xy *= inv9;

    // |Shat| from test-filtered S
    float S_hat_dd = S_xx_hat*S_xx_hat + S_yy_hat*S_yy_hat
                    + 2.0f*S_xy_hat*S_xy_hat;
    float S_mag_hat = sqrtf(2.0f * S_hat_dd);

    // Leonard stress and deviatoric
    float L_xx = uu_xx - ux_hat * ux_hat;
    float L_yy = uu_yy - uy_hat * uy_hat;
    float L_xy = uu_xy - ux_hat * uy_hat;
    // 2D deviatoric: trace/dim with dim=2.  Using Lkk/2 here.
    float Lkk = L_xx + L_yy;
    float Ld_xx = L_xx - 0.5f * Lkk;
    float Ld_yy = L_yy - 0.5f * Lkk;

    // M_ij = 2 [ (|S|S_ij)_hat - alpha^2 |Shat| Shat_ij ]
    float M_xx = 2.0f * (SS_xx_hat - alpha_sq * S_mag_hat * S_xx_hat);
    float M_yy = 2.0f * (SS_yy_hat - alpha_sq * S_mag_hat * S_yy_hat);
    float M_xy = 2.0f * (SS_xy_hat - alpha_sq * S_mag_hat * S_xy_hat);

    float LM = Ld_xx * M_xx + Ld_yy * M_yy + 2.0f * L_xy * M_xy;
    float MM = M_xx * M_xx + M_yy * M_yy + 2.0f * M_xy * M_xy;

    float Cs_sq = (MM > eps) ? (LM / (MM * dx * dx)) : 0.0f;
    Cs_sq = fmaxf(0.0f, fminf(Cs_max_sq, Cs_sq));

    nu_t[idx] = Cs_sq * dx * dx * s_mag_c;
}
'''


class DynSmagKernelD2Q9:
    """u -> nu_t (Dynamic Smagorinsky) for D2Q9."""

    def __init__(self, block_size: int = 128) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _DYN_SMAG_D2Q9_KERNEL, 'dyn_smag_d2q9',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        u_x: 'npt.NDArray',
        u_y: 'npt.NDArray',
        nu_t: 'npt.NDArray',
        Nx: int, Ny: int,
        dx: float = 1.0,
        Cs_max: float = 0.5,
        alpha_sq: float = 3.0,
        eps: float = 1e-12,
    ) -> None:
        if self._kernel is None:
            self._compile()
        import cupy as cp
        N = Nx * Ny
        grid = (N + self._block_size - 1) // self._block_size
        self._kernel(
            (grid,), (self._block_size,),
            (u_x, u_y, nu_t,
             cp.int32(Nx), cp.int32(Ny),
             cp.float32(dx), cp.float32(Cs_max * Cs_max),
             cp.float32(alpha_sq), cp.float32(eps)),
        )
