"""
Fused Cumulant D2Q9 Collision Kernel (CUDA)

Replaces: macroscopic.compute() + Guo correction + cumulant_d2q9.collide()
with a single CUDA kernel launch.

Mirrors the structure of `cumulant_d3q27.py` but implements the D2Q9
cumulant math from `src/collision/cumulant_d2q9.py` (PALABOS/Geier-style
trace-traceless decomposition + Galilean correction + bulk viscosity).

Physical Process (per node):
    1. Load f, compute rho and raw momentum (Sigma f, Sigma c*f)
    2. Guo velocity correction: u += F/(2*rho)
    3. Forward: f -> normalized central moments K_hat
    4. K_hat -> cumulants C (only C22 non-trivial)
    [SGS Smagorinsky: Pi^neq from K_ab, solve tau_total quadratic for w1]
    5. Velocity gradients (Chapman-Enskog 1st order, 2D) using w1
    6. Galilean-corrected trace/deviator relaxation of C20, C02
       Simple relaxation of C11, C21, C12, C22
    7. Inject force half-step (K10, K01)
    8. Backward: cumulants -> central -> raw moments
    9. Inverse M -> f*

SGS variants are compiled as separate kernels via `_build_kernel(sgs_model)`.
The "off" variant produces the byte-identical kernel string of the original
implementation (no SGS-related instructions emitted), giving bit-exact
regression against pre-SGS results.

Lattice convention (D2Q9):
    q:  0    1    2    3    4    5    6    7    8
    cx: 0    1    0   -1    0    1   -1   -1    1
    cy: 0    0    1    0   -1    1    1   -1   -1
    w:  4/9  1/9  1/9  1/9  1/9  1/36 1/36 1/36 1/36
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy.typing as npt


# Base kernel template. {{KERNEL_NAME}}, {{SGS_PARAM}}, {{SGS_BLOCK}} are
# substituted by `_build_kernel`.
_CUMULANT_D2Q9_TEMPLATE = r'''
extern "C" __global__
void {{KERNEL_NAME}}(
    const float* __restrict__ f_in,
    float*       __restrict__ f_post,
    float*       __restrict__ rho_out,
    float*       __restrict__ u_out,
    const float* __restrict__ force,
    const float omega_1,         // shear: 1/tau
    const float omega_bulk,      // bulk viscosity rate
    const float omega_3,         // 3rd-order cumulants (C21, C12)
    const float omega_4,         // 4th-order cumulant (C22)
    const int N{{SGS_PARAM}}
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // -- D2Q9 lattice constants ----------------------------------
    const int cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    const int cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

    const float cs2     = 1.0f / 3.0f;
    const float inv_cs2 = 3.0f;
    const float q025_inv_cs2 = 0.75f;   // 0.25 * inv_cs2

    // ============================================================
    // Step 1: Load f, compute macroscopic
    // ============================================================
    float f_local[9];
    float rho = 0.0f;
    float mom_x = 0.0f, mom_y = 0.0f;

    #pragma unroll
    for (int q = 0; q < 9; q++) {
        float fq = f_in[q * N + idx];
        f_local[q] = fq;
        rho   += fq;
        mom_x += (float)cx[q] * fq;
        mom_y += (float)cy[q] * fq;
    }

    float inv_rho = 1.0f / rho;
    float ux = mom_x * inv_rho;
    float uy = mom_y * inv_rho;

    // ============================================================
    // Step 2: Guo velocity correction
    // ============================================================
    float Fx = 0.0f, Fy = 0.0f;
    if (force != NULL) {
        Fx = force[0 * N + idx];
        Fy = force[1 * N + idx];
        float half_inv_rho = 0.5f * inv_rho;
        ux += Fx * half_inv_rho;
        uy += Fy * half_inv_rho;
    }

    // Write macroscopic
    rho_out[idx] = rho;
    u_out[0 * N + idx] = ux;
    u_out[1 * N + idx] = uy;

    // ============================================================
    // Step 3: Normalized central moments K_hat_{mn}
    //   K20h = sum f*dx*dx   K02h = sum f*dy*dy   K11h = sum f*dx*dy
    //   K21h = sum f*dx*dx*dy   K12h = sum f*dx*dy*dy   K22h = sum f*dx*dx*dy*dy
    //   then divide all by rho
    // ============================================================
    float K20 = 0.0f, K02 = 0.0f, K11 = 0.0f;
    float K21 = 0.0f, K12 = 0.0f, K22 = 0.0f;

    #pragma unroll
    for (int q = 0; q < 9; q++) {
        float fq = f_local[q];
        float dx = (float)cx[q] - ux;
        float dy = (float)cy[q] - uy;
        float fdx   = fq * dx;
        float fdy   = fq * dy;
        float fdxdy = fdx * dy;
        K20 += fdx * dx;
        K02 += fdy * dy;
        K11 += fdxdy;
        K21 += fdxdy * dx;
        K12 += fdxdy * dy;
        K22 += fdxdy * dx * dy;
    }

    K20 *= inv_rho;
    K02 *= inv_rho;
    K11 *= inv_rho;
    K21 *= inv_rho;
    K12 *= inv_rho;
    K22 *= inv_rho;

    // ============================================================
    // Step 4: Forward cumulant transform (orders <= 3 are identity)
    // ============================================================
    float C20 = K20;
    float C02 = K02;
    float C11 = K11;
    float C21 = K21;
    float C12 = K12;
    float C22 = K22 - K20 * K02 - 2.0f * K11 * K11;

    // ============================================================
    // Step 5: Velocity gradients (C-E 1st order, 2D)
    //   C20 - C02         = -2*cs2/omega_1 * (d_x ux - d_y uy)
    //   C20 + C02 - 2*cs2 = -2*cs2/omega_b * (d_x ux + d_y uy)
    // ============================================================
    float w1 = omega_1;
    float w2 = omega_bulk;
    float w3 = omega_3;
    float w4 = omega_4;
{{SGS_BLOCK}}
    float shear_dev = C20 - C02;
    float bulk_dev  = C20 + C02 - 2.0f * cs2;
    float Dxu = -q025_inv_cs2 * (w1 * shear_dev + w2 * bulk_dev);
    float Dyv =  q025_inv_cs2 *  w1 * shear_dev
                - q025_inv_cs2 *  w2 * bulk_dev;

    // ============================================================
    // Step 6: Galilean correction (normalized)
    // ============================================================
    float fac_shear = -inv_cs2 * (1.0f - 0.5f * w1);
    float fac_bulk  = -inv_cs2 * (1.0f - 0.5f * w2);
    float Gal_xx  = fac_shear * ux * ux * Dxu;
    float Gal_yy  = fac_shear * uy * uy * Dyv;
    float Gal_bxx = fac_bulk  * ux * ux * Dxu;
    float Gal_byy = fac_bulk  * uy * uy * Dyv;

    // - Trace/deviator relaxation of C20, C02 -------------------
    float trace_eq = 2.0f * cs2;
    float trace    = w2 * trace_eq
                     + (1.0f - w2) * (C20 + C02)
                     + (Gal_bxx + Gal_byy);
    float deviator = (1.0f - w1) * (C20 - C02)
                     + (Gal_xx - Gal_yy);

    float C20s = 0.5f * (trace + deviator);
    float C02s = 0.5f * (trace - deviator);

    // - Simple relaxation of remaining cumulants ----------------
    float C11s = (1.0f - w1) * C11;     // shear rate
    float C21s = (1.0f - w3) * C21;     // 3rd order
    float C12s = (1.0f - w3) * C12;
    float C22s = (1.0f - w4) * C22;     // 4th order

    // - Force half-step: K10, K01 -------------------------------
    float K10s = 0.0f, K01s = 0.0f;
    if (force != NULL) {
        K10s = 0.5f * Fx * inv_rho;
        K01s = 0.5f * Fy * inv_rho;
    }

    // ============================================================
    // Step 7: Backward cumulant transform
    // ============================================================
    float K20s = C20s;
    float K02s = C02s;
    float K11s = C11s;
    float K21s = C21s;
    float K12s = C12s;
    float K22s = C22s + C20s * C02s + 2.0f * C11s * C11s;

    // ============================================================
    // Step 8: Central -> raw moments (binomial)
    // ============================================================
    float ux2 = ux * ux;
    float uy2 = uy * uy;

    float mu10p = K10s + ux;
    float mu01p = K01s + uy;
    float mu20p = K20s + 2.0f * ux * K10s + ux2;
    float mu02p = K02s + 2.0f * uy * K01s + uy2;
    float mu11p = K11s + ux * K01s + uy * K10s + ux * uy;
    float mu21p = K21s
                  + 2.0f * ux * K11s + uy * K20s
                  + ux2 * K01s + 2.0f * ux * uy * K10s
                  + ux2 * uy;
    float mu12p = K12s
                  + ux * K02s + 2.0f * uy * K11s
                  + 2.0f * ux * uy * K01s + uy2 * K10s
                  + ux * uy2;
    float mu22p = K22s
                  + 2.0f * uy * K21s + 2.0f * ux * K12s
                  + uy2 * K20s + ux2 * K02s + 4.0f * ux * uy * K11s
                  + 2.0f * ux * uy2 * K10s + 2.0f * ux2 * uy * K01s
                  + ux2 * uy2;

    // Scale by rho
    float mu00 = rho;
    float mu10 = rho * mu10p;
    float mu01 = rho * mu01p;
    float mu20 = rho * mu20p;
    float mu02 = rho * mu02p;
    float mu11 = rho * mu11p;
    float mu21 = rho * mu21p;
    float mu12 = rho * mu12p;
    float mu22 = rho * mu22p;

    // ============================================================
    // Step 9: Inverse M -> f*
    // ============================================================
    f_post[0 * N + idx] = mu00 - mu20 - mu02 + mu22;
    f_post[1 * N + idx] = 0.5f  * ( mu10 + mu20 - mu12 - mu22);
    f_post[2 * N + idx] = 0.5f  * ( mu01 + mu02 - mu21 - mu22);
    f_post[3 * N + idx] = 0.5f  * (-mu10 + mu20 + mu12 - mu22);
    f_post[4 * N + idx] = 0.5f  * (-mu01 + mu02 + mu21 - mu22);
    f_post[5 * N + idx] = 0.25f * ( mu11 + mu21 + mu12 + mu22);
    f_post[6 * N + idx] = 0.25f * (-mu11 + mu21 - mu12 + mu22);
    f_post[7 * N + idx] = 0.25f * ( mu11 - mu21 - mu12 + mu22);
    f_post[8 * N + idx] = 0.25f * (-mu11 - mu21 + mu12 + mu22);
}
'''


# SGS branch templates (substituted for the "{{SGS_BLOCK}}" slot).

_SGS_BLOCK_OFF = ""

_SGS_BLOCK_WALE = r'''
    // -- SGS WALE: read pre-computed nu_t from buffer ------------
    //   tau_total = tau_0 + 3 * nu_t        (linear, no quadratic)
    //   w1        = 1 / tau_total
    {
        float tau0_wale = 1.0f / omega_1;
        float nu_t_in   = (nu_t_in_buf != NULL) ? nu_t_in_buf[idx] : 0.0f;
        float tau_t_wale = tau0_wale + 3.0f * nu_t_in;
        w1 = 1.0f / tau_t_wale;
        if (nu_t_out != NULL) {
            nu_t_out[idx] = nu_t_in;
        }
    }
'''

_SGS_BLOCK_SMAG = r'''
    // -- SGS Smagorinsky inline (Stiebler 2011 Eq. 12) -----------
    //   Pi^neq_xx = rho * (K20 - cs2)
    //   Pi^neq_yy = rho * (K02 - cs2)
    //   Pi^neq_xy = rho * K11
    //   Q         = sqrt( 2 * Pi:Pi )
    //   tau_total = ( tau_0 + sqrt(tau_0^2 + 18 Cs^2 Q) ) / 2
    //   w1        = 1 / tau_total
    //   nu_t (lu) = (tau_total - tau_0) / 3       (Stiebler decomposition)
    {
        float Pi_xx_sgs = rho * (K20 - cs2);
        float Pi_yy_sgs = rho * (K02 - cs2);
        float Pi_xy_sgs = rho * K11;
        float Q_sgs = sqrtf(2.0f * (Pi_xx_sgs * Pi_xx_sgs
                                    + Pi_yy_sgs * Pi_yy_sgs
                                    + 2.0f * Pi_xy_sgs * Pi_xy_sgs));
        float tau0_sgs = 1.0f / omega_1;
        float tau_t_sgs = 0.5f * (tau0_sgs
                          + sqrtf(tau0_sgs * tau0_sgs + 18.0f * Cs * Cs * Q_sgs));
        w1 = 1.0f / tau_t_sgs;
        if (nu_t_out != NULL) {
            nu_t_out[idx] = (tau_t_sgs - tau0_sgs) * (1.0f / 3.0f);
        }
    }
'''


def _build_kernel(sgs_model: str) -> tuple[str, str]:
    """Return (kernel_source, kernel_name) for a given SGS variant.

    sgs_model:
        "off"         -- byte-identical to pre-SGS implementation.
        "smagorinsky" -- inline Pi^neq -> tau_total quadratic.
        "wale"        -- not yet implemented (stage B).
    """
    if sgs_model == "off":
        name = "cumulant_collide_d2q9"
        sgs_param = ""
        sgs_block = _SGS_BLOCK_OFF
    elif sgs_model == "smagorinsky":
        name = "cumulant_collide_d2q9_smag"
        # Two extra params on top of the SGS-off signature: Cs and the
        # optional nu_t output buffer (NULL -> skip writing).
        sgs_param = ",\n    const float Cs,\n    float* __restrict__ nu_t_out"
        sgs_block = _SGS_BLOCK_SMAG
    elif sgs_model == "wale":
        name = "cumulant_collide_d2q9_wale"
        sgs_param = (",\n    const float* __restrict__ nu_t_in_buf"
                     ",\n    float* __restrict__ nu_t_out")
        sgs_block = _SGS_BLOCK_WALE
    else:
        raise ValueError(f"Unsupported sgs_model: {sgs_model!r}")

    src = (_CUMULANT_D2Q9_TEMPLATE
           .replace("{{KERNEL_NAME}}", name)
           .replace("{{SGS_PARAM}}", sgs_param)
           .replace("{{SGS_BLOCK}}", sgs_block))
    return src, name


class CumulantCollideKernelD2Q9:
    """Fused Cumulant collision kernel for D2Q9.

    Replaces macroscopic.compute() + Guo correction + cumulant_d2q9.collide()
    with a single CUDA kernel launch. Per-instance SGS variant is fixed at
    construction; switching requires building a new instance.

    Usage:
        >>> kernel = CumulantCollideKernelD2Q9()                         # SGS off
        >>> kernel = CumulantCollideKernelD2Q9(sgs_model="smagorinsky")  # SGS on
        >>> kernel.launch(f, f_post, rho, u, force,
        ...               omega_1, omega_bulk, omega_3, omega_4, N,
        ...               Cs=0.18)   # only used when sgs_model == "smagorinsky"
    """

    def __init__(self, sgs_model: str = "off", block_size: int = 256) -> None:
        if sgs_model not in ("off", "smagorinsky", "wale"):
            raise ValueError(f"Unsupported sgs_model: {sgs_model!r}")
        self._sgs_model = sgs_model
        self._block_size = block_size
        self._kernel = None
        self._kernel_name: Optional[str] = None

    @property
    def sgs_model(self) -> str:
        return self._sgs_model

    def _compile(self) -> None:
        import cupy as cp
        src, name = _build_kernel(self._sgs_model)
        self._kernel = cp.RawKernel(src, name, options=('--use_fast_math',))
        self._kernel_name = name

    def launch(
        self,
        f_in: 'npt.NDArray',
        f_post: 'npt.NDArray',
        rho_out: 'npt.NDArray',
        u_out: 'npt.NDArray',
        force: Optional['npt.NDArray'],
        omega_1: float,
        omega_bulk: float,
        omega_3: float,
        omega_4: float,
        N: int,
        Cs: float = 0.0,
        nu_t_out: Optional['npt.NDArray'] = None,
        nu_t_in: Optional['npt.NDArray'] = None,
    ) -> None:
        """Launch the fused cumulant collision kernel.

        Args:
            f_in, f_post, rho_out, u_out: as in the SGS-off path.
            Cs: Smagorinsky constant (only consumed when sgs_model == "smagorinsky").
            nu_t_out: Optional eddy-viscosity output buffer (N,) [lu^2/lt].
                      Smag and WALE paths; None -> skip writing.
            nu_t_in: Pre-computed nu_t from WALE pre-pass (N,) [lu^2/lt].
                     Required when sgs_model == "wale" (None -> nu_t treated as 0).
        """
        if self._kernel is None:
            self._compile()

        import cupy as cp

        grid_size = (N + self._block_size - 1) // self._block_size
        force_arg = force if force is not None else cp.int32(0)

        base_args = (
            f_in, f_post, rho_out, u_out,
            force_arg,
            cp.float32(omega_1),
            cp.float32(omega_bulk),
            cp.float32(omega_3),
            cp.float32(omega_4),
            cp.int32(N),
        )
        if self._sgs_model == "smagorinsky":
            nu_t_arg = nu_t_out if nu_t_out is not None else cp.int32(0)
            args = base_args + (cp.float32(Cs), nu_t_arg)
        elif self._sgs_model == "wale":
            nu_t_in_arg  = nu_t_in  if nu_t_in  is not None else cp.int32(0)
            nu_t_out_arg = nu_t_out if nu_t_out is not None else cp.int32(0)
            args = base_args + (nu_t_in_arg, nu_t_out_arg)
        else:
            args = base_args

        self._kernel((grid_size,), (self._block_size,), args)
