"""Esoteric Pull + Cumulant collision kernel for D3Q27.

Single-buffer in-place streaming (Lehmann 2022 Esoteric Pull) fused with the
Geier (2015) cumulant collision + Guo forcing + in-kernel BC. Re-implemented
for Phase 1b (patch_notes/hpc_upgrade/15): the original esoteric_cumulant was
never committed. Built by grafting the CURRENT fused cumulant collision
(src/kernels/cumulant_d3q27.py, steps 3-9) into the Esoteric Pull LOAD/STORE/BC
scaffolding (src/kernels/esoteric_d3q27.py).

Why this is clean: the cumulant collision (Chimera forward -> cumulant transform
-> relax -> backward) operates purely on the central-moment tensor K[3][3][3]
indexed by K[cx+1][cy+1][cz+1]. That binning is velocity-component-based, so it
is IDENTICAL for any distribution ordering. Only the LOAD, macroscopic sum, K
gather, and K scatter reference the per-slot velocity, so using the Esoteric
paired-ordering velocities (CX_ESO...) makes the whole kernel consistent.

Direction ordering: Esoteric paired opposites (i, i+1) for i = 1,3,...,25.

Reference: Lehmann, Computation 10(6) 2022; Geier et al., Comput. Math. Appl. 2015.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy.typing as npt

# Reuse the Esoteric paired-ordering lattice constants + formatter from the BGK
# Esoteric kernel (single source of truth for the ordering).
from src.kernels.esoteric_d3q27 import (
    CX_ESO, CY_ESO, CZ_ESO, W_ESO, _fmt_array,
    NODE_FLUID, NODE_SOLID, NODE_EQ_BC, NODE_NEUMANN, NODE_SPONGE,
)
# SGS branch templates: SINGLE SOURCE shared with the standard fused cumulant
# kernel (identical blocks -> identical SGS physics in both layouts).
from src.kernels.cumulant_d3q27 import (
    _SGS_BLOCK_OFF, _SGS_BLOCK_WALE, _SGS_BLOCK_SMAG,
)


_ESOTERIC_CUMULANT_TEMPLATE = r'''
extern "C" __global__
void esoteric_cumulant_d3q27(
    float*        __restrict__ f,          // (27, N) single buffer, in-place (Esoteric layout)
    float*        __restrict__ rho_out,    // (N,)
    float*        __restrict__ u_out,      // (3, N)
    const char*   __restrict__ node_type,  // (N,) 0=fluid 1=solid 2=eq 3=neumann 4=sponge
    const float*  __restrict__ bc_rho,     // (N,)
    const float*  __restrict__ bc_ux,      // (N,)
    const float*  __restrict__ bc_uy,      // (N,)
    const float*  __restrict__ bc_uz,      // (N,)  (reused as sigma for sponge)
    const float*  __restrict__ force,      // (3, N) body force or NULL
    const float omega_1,                   // shear: 1/tau
    const float omega_bulk,                // bulk viscosity rate
    const float omega_high,                // higher-order rate (omega_3-10)
    const int Nx, const int Ny, const int Nz,
    const int t_step{{SGS_PARAM}}
) {
    long long N = (long long)Nx * (long long)Ny * (long long)Nz;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int type = node_type[idx];
    if (type == 1) return;   // SOLID: skip (implicit bounce-back)

    long long ix = idx / ((long long)Ny * Nz);
    long long rem = idx - ix * (long long)Ny * Nz;
    long long iy = rem / Nz;
    long long iz = rem - iy * Nz;

    // Esoteric D3Q27 lattice constants (paired ordering)
    const int cx[27] = {''' + _fmt_array(CX_ESO) + r'''};
    const int cy[27] = {''' + _fmt_array(CY_ESO) + r'''};
    const int cz[27] = {''' + _fmt_array(CZ_ESO) + r'''};
    const float w[27] = {''' + _fmt_array(W_ESO) + r'''};

    int is_odd = t_step & 1;

    // ========================================================
    // LOAD (Esoteric Pull streaming part 2/2)
    // ========================================================
    float fhn[27];
    fhn[0] = f[0 * N + idx];
    for (int p = 0; p < 13; p++) {
        int i = 2 * p + 1;
        long long nx = (ix + cx[i] + Nx) % Nx;
        long long ny = (iy + cy[i] + Ny) % Ny;
        long long nz = (iz + cz[i] + Nz) % Nz;
        long long j_i = nx * (long long)Ny * Nz + ny * (long long)Nz + nz;
        if (is_odd) {
            fhn[i]   = f[(long long)i     * N + idx];
            fhn[i+1] = f[(long long)(i+1) * N + j_i];
        } else {
            fhn[i]   = f[(long long)(i+1) * N + idx];
            fhn[i+1] = f[(long long)i     * N + j_i];
        }
    }

    // ========================================================
    // MACROSCOPIC (+ Guo velocity correction)
    // ========================================================
    float rho = 0.0f, mom_x = 0.0f, mom_y = 0.0f, mom_z = 0.0f;
    for (int q = 0; q < 27; q++) {
        float fq = fhn[q];
        rho += fq; mom_x += cx[q]*fq; mom_y += cy[q]*fq; mom_z += cz[q]*fq;
    }
    float inv_rho = 1.0f / rho;
    float ux = mom_x * inv_rho, uy = mom_y * inv_rho, uz = mom_z * inv_rho;

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    if (force != NULL) {
        Fx = force[0 * N + idx]; Fy = force[1 * N + idx]; Fz = force[2 * N + idx];
        float h = 0.5f * inv_rho;
        ux += Fx * h; uy += Fy * h; uz += Fz * h;
    }

    if (type == 2) {
        // EQUILIBRIUM BC: override rho,u with target -> f_eq
        rho = bc_rho[idx]; ux = bc_ux[idx]; uy = bc_uy[idx]; uz = bc_uz[idx];
        float usqr = ux*ux + uy*uy + uz*uz;
        for (int q = 0; q < 27; q++) {
            float cu = (float)cx[q]*ux + (float)cy[q]*uy + (float)cz[q]*uz;
            fhn[q] = w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usqr);
        }
    }
    else if (type == 0 || type == 4) {
        // FLUID or SPONGE: cumulant collision (Geier 2015).
        // ---- Step 3: bin f -> K[3][3][3] (central-moment tensor slots) ----
        float K[3][3][3];
        for (int q = 0; q < 27; q++) K[cx[q]+1][cy[q]+1][cz[q]+1] = fhn[q];

        // ==== Steps 4-8 VERBATIM from cumulant_d3q27.py (K-only, order-agnostic) ====
        // Step 4: Forward Chimera (f -> central moments); z -> y -> x
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            float dm = K[i][j][0], d0 = K[i][j][1], dp = K[i][j][2];
            float m0 = dm + d0 + dp;
            float m1 = -dm + dp - uz * m0;
            float m2 = dm + dp - 2.0f*uz*m1 - uz*uz*m0;
            K[i][j][0] = m0; K[i][j][1] = m1; K[i][j][2] = m2;
        }
        for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) {
            float dm = K[i][0][k], d0 = K[i][1][k], dp = K[i][2][k];
            float m0 = dm + d0 + dp;
            float m1 = -dm + dp - uy * m0;
            float m2 = dm + dp - 2.0f*uy*m1 - uy*uy*m0;
            K[i][0][k] = m0; K[i][1][k] = m1; K[i][2][k] = m2;
        }
        for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) {
            float dm = K[0][j][k], d0 = K[1][j][k], dp = K[2][j][k];
            float m0 = dm + d0 + dp;
            float m1 = -dm + dp - ux * m0;
            float m2 = dm + dp - 2.0f*ux*m1 - ux*ux*m0;
            K[0][j][k] = m0; K[1][j][k] = m1; K[2][j][k] = m2;
        }

        // Step 5: Forward cumulant transform (orders 4-6)
        float k220_o = K[2][2][0], k202_o = K[2][0][2], k022_o = K[0][2][2];
        float k211_o = K[2][1][1], k121_o = K[1][2][1], k112_o = K[1][1][2];
        K[2][1][1] -= (K[2][0][0]*K[0][1][1] + 2.0f*K[1][1][0]*K[1][0][1]) * inv_rho;
        K[1][2][1] -= (K[0][2][0]*K[1][0][1] + 2.0f*K[1][1][0]*K[0][1][1]) * inv_rho;
        K[1][1][2] -= (K[0][0][2]*K[1][1][0] + 2.0f*K[1][0][1]*K[0][1][1]) * inv_rho;
        K[2][2][0] -= (K[2][0][0]*K[0][2][0] + 2.0f*K[1][1][0]*K[1][1][0]) * inv_rho;
        K[2][0][2] -= (K[2][0][0]*K[0][0][2] + 2.0f*K[1][0][1]*K[1][0][1]) * inv_rho;
        K[0][2][2] -= (K[0][2][0]*K[0][0][2] + 2.0f*K[0][1][1]*K[0][1][1]) * inv_rho;
        K[1][2][2] -= (K[0][0][2]*K[1][2][0] + K[0][2][0]*K[1][0][2]
                       + 4.0f*K[0][1][1]*K[1][1][1]
                       + 2.0f*(K[1][0][1]*K[0][2][1] + K[1][1][0]*K[0][1][2])) * inv_rho;
        K[2][1][2] -= (K[0][0][2]*K[2][1][0] + K[2][0][0]*K[0][1][2]
                       + 4.0f*K[1][0][1]*K[1][1][1]
                       + 2.0f*(K[0][1][1]*K[2][0][1] + K[1][1][0]*K[1][0][2])) * inv_rho;
        K[2][2][1] -= (K[0][2][0]*K[2][0][1] + K[2][0][0]*K[0][2][1]
                       + 4.0f*K[1][1][0]*K[1][1][1]
                       + 2.0f*(K[0][1][1]*K[2][1][0] + K[1][0][1]*K[1][2][0])) * inv_rho;
        float inv_rho2 = inv_rho * inv_rho;
        K[2][2][2] -= (
            4.0f*K[1][1][1]*K[1][1][1]
            + K[2][0][0]*k022_o + K[0][2][0]*k202_o + K[0][0][2]*k220_o
            + 4.0f*(K[0][1][1]*k211_o + K[1][0][1]*k121_o + K[1][1][0]*k112_o)
            + 2.0f*(K[1][2][0]*K[1][0][2] + K[2][1][0]*K[0][1][2] + K[2][0][1]*K[0][2][1])
        ) * inv_rho;
        K[2][2][2] += (
            16.0f*K[1][1][0]*K[1][0][1]*K[0][1][1]
            + 4.0f*(K[1][0][1]*K[1][0][1]*K[0][2][0]
                    + K[0][1][1]*K[0][1][1]*K[2][0][0]
                    + K[1][1][0]*K[1][1][0]*K[0][0][2])
            + 2.0f*K[2][0][0]*K[0][2][0]*K[0][0][2]
        ) * inv_rho2;

        // Step 6: Relax cumulants (Galilean-corrected)
        float w1 = omega_1, w2 = omega_bulk;
        float w3 = omega_high, w4 = omega_high, w5 = omega_high;
        float w6 = omega_high, w7 = omega_high, w8 = omega_high;
        float w9 = omega_high, w10 = omega_high;
{{SGS_BLOCK}}
        float C200 = K[2][0][0], C020 = K[0][2][0], C002 = K[0][0][2];
        float Dxu = -w1*0.5f*inv_rho*(2.0f*C200 - C020 - C002)
                    -w2*0.5f*inv_rho*(C200 + C020 + C002 - rho);
        float Dyv = Dxu + 1.5f*w1*inv_rho*(C200 - C020);
        float Dzw = Dxu + 1.5f*w1*inv_rho*(C200 - C002);
        K[1][1][0] *= (1.0f - w1);
        K[1][0][1] *= (1.0f - w1);
        K[0][1][1] *= (1.0f - w1);
        float Gxx = -3.0f*rho*(1.0f - 0.5f*w1)*ux*ux*Dxu;
        float Gyy = -3.0f*rho*(1.0f - 0.5f*w1)*uy*uy*Dyv;
        float Gzz = -3.0f*rho*(1.0f - 0.5f*w1)*uz*uz*Dzw;
        float Gbxx = -3.0f*rho*(1.0f - 0.5f*w2)*ux*ux*Dxu;
        float Gbyy = -3.0f*rho*(1.0f - 0.5f*w2)*uy*uy*Dyv;
        float Gbzz = -3.0f*rho*(1.0f - 0.5f*w2)*uz*uz*Dzw;
        float diff_xy = (1.0f - w1)*(C200 - C020) + (Gxx - Gyy);
        float diff_xz = (1.0f - w1)*(C200 - C002) + (Gxx - Gzz);
        float trace = w2*rho + (1.0f - w2)*(C200 + C020 + C002) + (Gbxx + Gbyy + Gbzz);
        K[2][0][0] = (trace + diff_xy + diff_xz) / 3.0f;
        K[0][2][0] = K[2][0][0] - diff_xy;
        K[0][0][2] = K[2][0][0] - diff_xz;
        float s120 = (1.0f-w3)*(K[1][2][0] + K[1][0][2]);
        float s210 = (1.0f-w3)*(K[2][1][0] + K[0][1][2]);
        float s201 = (1.0f-w3)*(K[2][0][1] + K[0][2][1]);
        float d120 = (1.0f-w4)*(K[1][2][0] - K[1][0][2]);
        float d210 = (1.0f-w4)*(K[2][1][0] - K[0][1][2]);
        float d201 = (1.0f-w4)*(K[2][0][1] - K[0][2][1]);
        K[1][2][0] = 0.5f*(s120 + d120);
        K[1][0][2] = 0.5f*(s120 - d120);
        K[2][1][0] = 0.5f*(s210 + d210);
        K[0][1][2] = 0.5f*(s210 - d210);
        K[2][0][1] = 0.5f*(s201 + d201);
        K[0][2][1] = 0.5f*(s201 - d201);
        K[1][1][1] *= (1.0f - w5);
        float tC220 = K[2][2][0], tC202 = K[2][0][2], tC022 = K[0][2][2];
        float cb1 = (1.0f-w6)*(tC220 - 2.0f*tC202 + tC022);
        float cb2 = (1.0f-w6)*(tC220 + tC202 - 2.0f*tC022);
        float cb3 = (1.0f-w7)*(tC220 + tC202 + tC022);
        K[2][2][0] = (cb1 + cb2 + cb3) / 3.0f;
        K[2][0][2] = (cb3 - cb1) / 3.0f;
        K[0][2][2] = (cb3 - cb2) / 3.0f;
        K[2][1][1] *= (1.0f - w8);
        K[1][2][1] *= (1.0f - w8);
        K[1][1][2] *= (1.0f - w8);
        K[2][2][1] *= (1.0f - w9);
        K[2][1][2] *= (1.0f - w9);
        K[1][2][2] *= (1.0f - w9);
        K[2][2][2] *= (1.0f - w10);

        // Step 7: Backward cumulant transform
        inv_rho2 = inv_rho * inv_rho;
        K[2][2][0] += (K[2][0][0]*K[0][2][0] + 2.0f*K[1][1][0]*K[1][1][0]) * inv_rho;
        K[2][0][2] += (K[2][0][0]*K[0][0][2] + 2.0f*K[1][0][1]*K[1][0][1]) * inv_rho;
        K[0][2][2] += (K[0][2][0]*K[0][0][2] + 2.0f*K[0][1][1]*K[0][1][1]) * inv_rho;
        K[2][1][1] += (K[2][0][0]*K[0][1][1] + 2.0f*K[1][1][0]*K[1][0][1]) * inv_rho;
        K[1][2][1] += (K[0][2][0]*K[1][0][1] + 2.0f*K[1][1][0]*K[0][1][1]) * inv_rho;
        K[1][1][2] += (K[0][0][2]*K[1][1][0] + 2.0f*K[1][0][1]*K[0][1][1]) * inv_rho;
        K[1][2][2] += (K[0][0][2]*K[1][2][0] + K[0][2][0]*K[1][0][2]
                       + 4.0f*K[0][1][1]*K[1][1][1]
                       + 2.0f*(K[1][0][1]*K[0][2][1] + K[1][1][0]*K[0][1][2])) * inv_rho;
        K[2][1][2] += (K[0][0][2]*K[2][1][0] + K[2][0][0]*K[0][1][2]
                       + 4.0f*K[1][0][1]*K[1][1][1]
                       + 2.0f*(K[0][1][1]*K[2][0][1] + K[1][1][0]*K[1][0][2])) * inv_rho;
        K[2][2][1] += (K[0][2][0]*K[2][0][1] + K[2][0][0]*K[0][2][1]
                       + 4.0f*K[1][1][0]*K[1][1][1]
                       + 2.0f*(K[0][1][1]*K[2][1][0] + K[1][0][1]*K[1][2][0])) * inv_rho;
        K[2][2][2] += (
            4.0f*K[1][1][1]*K[1][1][1]
            + K[2][0][0]*K[0][2][2] + K[0][2][0]*K[2][0][2] + K[0][0][2]*K[2][2][0]
            + 4.0f*(K[0][1][1]*K[2][1][1] + K[1][0][1]*K[1][2][1] + K[1][1][0]*K[1][1][2])
            + 2.0f*(K[1][2][0]*K[1][0][2] + K[2][1][0]*K[0][1][2] + K[2][0][1]*K[0][2][1])
        ) * inv_rho;
        K[2][2][2] -= (
            16.0f*K[1][1][0]*K[1][0][1]*K[0][1][1]
            + 4.0f*(K[1][0][1]*K[1][0][1]*K[0][2][0]
                    + K[0][1][1]*K[0][1][1]*K[2][0][0]
                    + K[1][1][0]*K[1][1][0]*K[0][0][2])
            + 2.0f*K[2][0][0]*K[0][2][0]*K[0][0][2]
        ) * inv_rho2;

        // Step 7b: Force sign-flip (1st-order moments)
        if (force != NULL) {
            K[1][0][0] = -K[1][0][0];
            K[0][1][0] = -K[0][1][0];
            K[0][0][1] = -K[0][0][1];
        }

        // Step 8: Backward Chimera (central moments -> f*); x -> y -> z
        float ux2 = ux * ux;
        for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) {
            float m0 = K[0][j][k], m1 = K[1][j][k], m2 = K[2][j][k];
            K[1][j][k] = m0*(1.0f - ux2) - 2.0f*ux*m1 - m2;
            K[0][j][k] = 0.5f*(m0*(ux2 - ux) + m1*(2.0f*ux - 1.0f) + m2);
            K[2][j][k] = 0.5f*(m0*(ux2 + ux) + m1*(2.0f*ux + 1.0f) + m2);
        }
        float uy2 = uy * uy;
        for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) {
            float m0 = K[i][0][k], m1 = K[i][1][k], m2 = K[i][2][k];
            K[i][1][k] = m0*(1.0f - uy2) - 2.0f*uy*m1 - m2;
            K[i][0][k] = 0.5f*(m0*(uy2 - uy) + m1*(2.0f*uy - 1.0f) + m2);
            K[i][2][k] = 0.5f*(m0*(uy2 + uy) + m1*(2.0f*uy + 1.0f) + m2);
        }
        float uz2 = uz * uz;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            float m0 = K[i][j][0], m1 = K[i][j][1], m2 = K[i][j][2];
            K[i][j][1] = m0*(1.0f - uz2) - 2.0f*uz*m1 - m2;
            K[i][j][0] = 0.5f*(m0*(uz2 - uz) + m1*(2.0f*uz - 1.0f) + m2);
            K[i][j][2] = 0.5f*(m0*(uz2 + uz) + m1*(2.0f*uz + 1.0f) + m2);
        }
        // ==== end verbatim ====

        // Step 9: scatter K -> fhn (+ Guo source term)
        if (force != NULL) {
            float pf = 1.0f - 0.5f * omega_1;
            for (int q = 0; q < 27; q++) {
                int i = cx[q]+1, j = cy[q]+1, k = cz[q]+1;
                float cxq = (float)cx[q], cyq = (float)cy[q], czq = (float)cz[q];
                float ci_F = cxq*Fx + cyq*Fy + czq*Fz;
                float ci_u = cxq*ux + cyq*uy + czq*uz;
                float u_F  = ux*Fx + uy*Fy + uz*Fz;
                float Si = pf * w[q] * ((ci_F - u_F)*3.0f + ci_u*ci_F*9.0f);
                fhn[q] = K[i][j][k] + Si;
            }
        } else {
            for (int q = 0; q < 27; q++)
                fhn[q] = K[cx[q]+1][cy[q]+1][cz[q]+1];
        }

        if (type == 4) {
            // SPONGE: blend post-collision toward target equilibrium.
            float sigma = bc_uz[idx];
            float rho_t = bc_rho[idx];
            float ux_t = bc_ux[idx], uy_t = bc_uy[idx];
            float usqr_t = ux_t*ux_t + uy_t*uy_t;
            for (int q = 0; q < 27; q++) {
                float cu_t = (float)cx[q]*ux_t + (float)cy[q]*uy_t;
                float ft = w[q] * rho_t * (1.0f + 3.0f*cu_t + 4.5f*cu_t*cu_t - 1.5f*usqr_t);
                fhn[q] = (1.0f - sigma)*fhn[q] + sigma*ft;
            }
        }
    }
    // type == 3 (NEUMANN): passthrough (fhn unchanged)

    // Write macroscopic
    rho_out[idx] = rho;
    u_out[0 * N + idx] = ux;
    u_out[1 * N + idx] = uy;
    u_out[2 * N + idx] = uz;

    // ========================================================
    // STORE (Esoteric Pull streaming part 1/2)
    // ========================================================
    f[0 * N + idx] = fhn[0];
    for (int p = 0; p < 13; p++) {
        int i = 2 * p + 1;
        long long nx = (ix + cx[i] + Nx) % Nx;
        long long ny = (iy + cy[i] + Ny) % Ny;
        long long nz = (iz + cz[i] + Nz) % Nz;
        long long j_i = nx * (long long)Ny * Nz + ny * (long long)Nz + nz;
        if (is_odd) {
            f[(long long)(i+1) * N + j_i] = fhn[i];
            f[(long long)i     * N + idx] = fhn[i+1];
        } else {
            f[(long long)i     * N + j_i] = fhn[i];
            f[(long long)(i+1) * N + idx] = fhn[i+1];
        }
    }
}
'''


def _build_eso_cumulant_kernel(sgs_model: str) -> str:
    """Substitute the SGS variant into the template (mirrors cumulant_d3q27).

    sgs_model: "off" | "smagorinsky" (inline, local) | "wale" (nu_t from a
    pre-pass buffer; also serves dyn_smag, which reuses the WALE branch).
    """
    if sgs_model == "off":
        sgs_param = ""
        sgs_block = _SGS_BLOCK_OFF
    elif sgs_model == "smagorinsky":
        sgs_param = ",\n    const float Cs,\n    float* __restrict__ nu_t_out"
        sgs_block = _SGS_BLOCK_SMAG
    elif sgs_model == "wale":
        sgs_param = (",\n    const float* __restrict__ nu_t_in_buf"
                     ",\n    float* __restrict__ nu_t_out")
        sgs_block = _SGS_BLOCK_WALE
    else:
        raise ValueError(f"Unsupported sgs_model: {sgs_model!r}")
    return (_ESOTERIC_CUMULANT_TEMPLATE
            .replace("{{SGS_PARAM}}", sgs_param)
            .replace("{{SGS_BLOCK}}", sgs_block))


class EsotericCumulantKernelD3Q27:
    """Esoteric Pull + cumulant collision + BC, single launch, in-place."""

    def __init__(self, sgs_model: str = "off", block_size: int = 256) -> None:
        if sgs_model not in ("off", "smagorinsky", "wale"):
            raise ValueError(f"Unsupported sgs_model: {sgs_model!r}")
        self._sgs_model = sgs_model
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _build_eso_cumulant_kernel(self._sgs_model),
            "esoteric_cumulant_d3q27",
            options=('--use_fast_math',))

    def launch(
        self,
        f: 'npt.NDArray',
        rho_out: 'npt.NDArray',
        u_out: 'npt.NDArray',
        node_type: 'npt.NDArray',
        bc_rho: 'npt.NDArray',
        bc_ux: 'npt.NDArray',
        bc_uy: 'npt.NDArray',
        bc_uz: 'npt.NDArray',
        omega_1: float,
        omega_bulk: float,
        omega_high: float,
        Nx: int, Ny: int, Nz: int,
        t_step: int,
        force: Optional['npt.NDArray'] = None,
        Cs: float = 0.0,
        nu_t_out: Optional['npt.NDArray'] = None,
        nu_t_in: Optional['npt.NDArray'] = None,
    ) -> None:
        if self._kernel is None:
            self._compile()
        import cupy as cp
        N = Nx * Ny * Nz
        grid = (N + self._block_size - 1) // self._block_size
        force_arg = force if force is not None else cp.int32(0)
        base_args = (
            f, rho_out, u_out, node_type,
            bc_rho, bc_ux, bc_uy, bc_uz, force_arg,
            cp.float32(omega_1), cp.float32(omega_bulk), cp.float32(omega_high),
            cp.int32(Nx), cp.int32(Ny), cp.int32(Nz), cp.int32(t_step),
        )
        if self._sgs_model == "smagorinsky":
            nu_t_arg = nu_t_out if nu_t_out is not None else cp.int32(0)
            args = base_args + (cp.float32(Cs), nu_t_arg)
        elif self._sgs_model == "wale":
            nu_t_in_arg = nu_t_in if nu_t_in is not None else cp.int32(0)
            nu_t_out_arg = nu_t_out if nu_t_out is not None else cp.int32(0)
            args = base_args + (nu_t_in_arg, nu_t_out_arg)
        else:
            args = base_args
        self._kernel((grid,), (self._block_size,), args)
