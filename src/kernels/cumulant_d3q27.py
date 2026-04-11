"""
Fused Cumulant D3Q27 Collision Kernel (CUDA)

Replaces: macroscopic.compute() + Guo correction + cumulant.collide()
with a single CUDA kernel launch.

Physical Process (per node, Geier et al. 2015):
    1. Compute ρ, u from f
    2. Guo correction: u += F/(2ρ)
    3. f → K[3][3][3] via Chimera forward (Eqs. 43-45)
    4. Central moments → cumulants (Eqs. 46-54)
    5. Relax cumulants with Galilean correction (Eqs. 55-80)
    6. Cumulants → central moments (Eqs. 81-84)
    7. Force sign-flip of 1st moments (Eqs. 85-87)
    8. K[3][3][3] → f* via Chimera backward (Eqs. 88-96)
    9. Add Guo source term S_i

Register budget (float32):
    f_local[27] = 27, K[27] = 27 (reuse f_local), rho/u/omega = ~6
    Chimera temps = ~9, Relaxation temps = ~20
    Total: ~90 registers (OK for RTX 3090, may reduce occupancy)

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


_CUMULANT_D3Q27_KERNEL = r'''
extern "C" __global__
void cumulant_collide_d3q27(
    const float* __restrict__ f_in,
    float*       __restrict__ f_post,
    float*       __restrict__ rho_out,
    float*       __restrict__ u_out,
    const float* __restrict__ force,
    const float omega_1,        // shear: 1/tau
    const float omega_bulk,     // bulk viscosity rate
    const float omega_high,     // higher-order rate (omega_3-10)
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ── D3Q27 lattice constants ─────────────────────────────
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

    // q → (i,j,k) mapping: i=cx+1, j=cy+1, k=cz+1
    // ijk_to_q[i][j][k] → flat index q
    // Built from cx,cy,cz arrays above.
    int ijk_to_q[3][3][3];
    for (int q = 0; q < 27; q++)
        ijk_to_q[cx[q]+1][cy[q]+1][cz[q]+1] = q;

    // ══════════════════════════════════════════════════════════
    // Step 1: Load f, compute macroscopic
    // ══════════════════════════════════════════════════════════
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

    // ══════════════════════════════════════════════════════════
    // Step 2: Guo velocity correction
    // ══════════════════════════════════════════════════════════
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

    // Write macroscopic
    rho_out[idx] = rho;
    u_out[0 * N + idx] = ux;
    u_out[1 * N + idx] = uy;
    u_out[2 * N + idx] = uz;

    // ══════════════════════════════════════════════════════════
    // Step 3: Reshape f_local to K[3][3][3] (flat → ijk)
    // ══════════════════════════════════════════════════════════
    float K[3][3][3];
    for (int q = 0; q < 27; q++) {
        int i = cx[q] + 1, j = cy[q] + 1, k = cz[q] + 1;
        K[i][j][k] = f_local[q];
    }

    // ══════════════════════════════════════════════════════════
    // Step 4: Forward Chimera transform — f → central moments
    //   z-direction → y-direction → x-direction
    //   Each: m0 = d_m + d_0 + d_p
    //         m1 = -d_m + d_p - v*m0
    //         m2 = d_m + d_p - 2v*m1 - v²*m0
    // ══════════════════════════════════════════════════════════

    // z-direction (vel = uz)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float dm = K[i][j][0], d0 = K[i][j][1], dp = K[i][j][2];
            float m0 = dm + d0 + dp;
            float m1 = -dm + dp - uz * m0;
            float m2 = dm + dp - 2.0f*uz*m1 - uz*uz*m0;
            K[i][j][0] = m0; K[i][j][1] = m1; K[i][j][2] = m2;
        }
    }
    // y-direction (vel = uy)
    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < 3; k++) {
            float dm = K[i][0][k], d0 = K[i][1][k], dp = K[i][2][k];
            float m0 = dm + d0 + dp;
            float m1 = -dm + dp - uy * m0;
            float m2 = dm + dp - 2.0f*uy*m1 - uy*uy*m0;
            K[i][0][k] = m0; K[i][1][k] = m1; K[i][2][k] = m2;
        }
    }
    // x-direction (vel = ux)
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            float dm = K[0][j][k], d0 = K[1][j][k], dp = K[2][j][k];
            float m0 = dm + d0 + dp;
            float m1 = -dm + dp - ux * m0;
            float m2 = dm + dp - 2.0f*ux*m1 - ux*ux*m0;
            K[0][j][k] = m0; K[1][j][k] = m1; K[2][j][k] = m2;
        }
    }
    // Now K[a][b][c] = κ_{abc} (central moments)

    // ══════════════════════════════════════════════════════════
    // Step 5: Forward cumulant transform (Eqs. 46-54)
    //   Orders 0-3: C = κ (no change)
    //   Order 4+: subtract lower-order products
    // ══════════════════════════════════════════════════════════

    // Save originals for order 6 (Eq. 54)
    float k220_o = K[2][2][0], k202_o = K[2][0][2], k022_o = K[0][2][2];
    float k211_o = K[2][1][1], k121_o = K[1][2][1], k112_o = K[1][1][2];

    // Order 4: C_{211} and perms (Eq. 51)
    K[2][1][1] -= (K[2][0][0]*K[0][1][1] + 2.0f*K[1][1][0]*K[1][0][1]) * inv_rho;
    K[1][2][1] -= (K[0][2][0]*K[1][0][1] + 2.0f*K[1][1][0]*K[0][1][1]) * inv_rho;
    K[1][1][2] -= (K[0][0][2]*K[1][1][0] + 2.0f*K[1][0][1]*K[0][1][1]) * inv_rho;

    // C_{220} and perms (Eq. 52)
    K[2][2][0] -= (K[2][0][0]*K[0][2][0] + 2.0f*K[1][1][0]*K[1][1][0]) * inv_rho;
    K[2][0][2] -= (K[2][0][0]*K[0][0][2] + 2.0f*K[1][0][1]*K[1][0][1]) * inv_rho;
    K[0][2][2] -= (K[0][2][0]*K[0][0][2] + 2.0f*K[0][1][1]*K[0][1][1]) * inv_rho;

    // Order 5: C_{122} and perms (Eq. 53)
    K[1][2][2] -= (K[0][0][2]*K[1][2][0] + K[0][2][0]*K[1][0][2]
                   + 4.0f*K[0][1][1]*K[1][1][1]
                   + 2.0f*(K[1][0][1]*K[0][2][1] + K[1][1][0]*K[0][1][2])) * inv_rho;
    K[2][1][2] -= (K[0][0][2]*K[2][1][0] + K[2][0][0]*K[0][1][2]
                   + 4.0f*K[1][0][1]*K[1][1][1]
                   + 2.0f*(K[0][1][1]*K[2][0][1] + K[1][1][0]*K[1][0][2])) * inv_rho;
    K[2][2][1] -= (K[0][2][0]*K[2][0][1] + K[2][0][0]*K[0][2][1]
                   + 4.0f*K[1][1][0]*K[1][1][1]
                   + 2.0f*(K[0][1][1]*K[2][1][0] + K[1][0][1]*K[1][2][0])) * inv_rho;

    // Order 6: C_{222} (Eq. 54) — uses ORIGINAL 4th-order values
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

    // ══════════════════════════════════════════════════════════
    // Step 6: Relax cumulants (Eqs. 55-80, Galilean correction)
    // ══════════════════════════════════════════════════════════
    float w1 = omega_1;
    float w2 = omega_bulk;
    float w3 = omega_high, w4 = omega_high, w5 = omega_high;
    float w6 = omega_high, w7 = omega_high, w8 = omega_high;
    float w9 = omega_high, w10 = omega_high;

    float C200 = K[2][0][0], C020 = K[0][2][0], C002 = K[0][0][2];

    // Galilean correction (Eqs. 58-60)
    float Dxu = -w1*0.5f*inv_rho*(2.0f*C200 - C020 - C002)
                -w2*0.5f*inv_rho*(C200 + C020 + C002 - rho);
    float Dyv = Dxu + 1.5f*w1*inv_rho*(C200 - C020);
    float Dzw = Dxu + 1.5f*w1*inv_rho*(C200 - C002);

    // 2nd order off-diagonal (Eqs. 55-57)
    K[1][1][0] *= (1.0f - w1);
    K[1][0][1] *= (1.0f - w1);
    K[0][1][1] *= (1.0f - w1);

    // 2nd order diagonal with Galilean (Eqs. 61-63)
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

    // 3rd order (Eqs. 64-70)
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

    // 4th order (Eqs. 71-76)
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

    // 5th order (Eqs. 77-79)
    K[2][2][1] *= (1.0f - w9);
    K[2][1][2] *= (1.0f - w9);
    K[1][2][2] *= (1.0f - w9);

    // 6th order (Eq. 80)
    K[2][2][2] *= (1.0f - w10);

    // ══════════════════════════════════════════════════════════
    // Step 7: Backward cumulant transform (Eqs. 81-84)
    // ══════════════════════════════════════════════════════════
    inv_rho2 = inv_rho * inv_rho;

    // Order 4 (Eq. 82)
    K[2][2][0] += (K[2][0][0]*K[0][2][0] + 2.0f*K[1][1][0]*K[1][1][0]) * inv_rho;
    K[2][0][2] += (K[2][0][0]*K[0][0][2] + 2.0f*K[1][0][1]*K[1][0][1]) * inv_rho;
    K[0][2][2] += (K[0][2][0]*K[0][0][2] + 2.0f*K[0][1][1]*K[0][1][1]) * inv_rho;
    // (Eq. 81)
    K[2][1][1] += (K[2][0][0]*K[0][1][1] + 2.0f*K[1][1][0]*K[1][0][1]) * inv_rho;
    K[1][2][1] += (K[0][2][0]*K[1][0][1] + 2.0f*K[1][1][0]*K[0][1][1]) * inv_rho;
    K[1][1][2] += (K[0][0][2]*K[1][1][0] + 2.0f*K[1][0][1]*K[0][1][1]) * inv_rho;

    // Order 5 (Eq. 83)
    K[1][2][2] += (K[0][0][2]*K[1][2][0] + K[0][2][0]*K[1][0][2]
                   + 4.0f*K[0][1][1]*K[1][1][1]
                   + 2.0f*(K[1][0][1]*K[0][2][1] + K[1][1][0]*K[0][1][2])) * inv_rho;
    K[2][1][2] += (K[0][0][2]*K[2][1][0] + K[2][0][0]*K[0][1][2]
                   + 4.0f*K[1][0][1]*K[1][1][1]
                   + 2.0f*(K[0][1][1]*K[2][0][1] + K[1][1][0]*K[1][0][2])) * inv_rho;
    K[2][2][1] += (K[0][2][0]*K[2][0][1] + K[2][0][0]*K[0][2][1]
                   + 4.0f*K[1][1][0]*K[1][1][1]
                   + 2.0f*(K[0][1][1]*K[2][1][0] + K[1][0][1]*K[1][2][0])) * inv_rho;

    // Order 6 (Eq. 84)
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

    // ══════════════════════════════════════════════════════════
    // Step 7b: Force sign-flip (Eqs. 85-87)
    // ══════════════════════════════════════════════════════════
    if (force != NULL) {
        K[1][0][0] = -K[1][0][0];
        K[0][1][0] = -K[0][1][0];
        K[0][0][1] = -K[0][0][1];
    }

    // ══════════════════════════════════════════════════════════
    // Step 8: Backward Chimera — central moments → f*
    //   x-direction → y-direction → z-direction
    //   f_0   = m0(1-v²) - 2v·m1 - m2
    //   f_{-} = 0.5(m0(v²-v) + m1(2v-1) + m2)
    //   f_{+} = 0.5(m0(v²+v) + m1(2v+1) + m2)
    // ══════════════════════════════════════════════════════════

    // x-direction (vel = ux)
    float ux2 = ux * ux;
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            float m0 = K[0][j][k], m1 = K[1][j][k], m2 = K[2][j][k];
            K[1][j][k] = m0*(1.0f - ux2) - 2.0f*ux*m1 - m2;
            K[0][j][k] = 0.5f*(m0*(ux2 - ux) + m1*(2.0f*ux - 1.0f) + m2);
            K[2][j][k] = 0.5f*(m0*(ux2 + ux) + m1*(2.0f*ux + 1.0f) + m2);
        }
    }
    // y-direction (vel = uy)
    float uy2 = uy * uy;
    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < 3; k++) {
            float m0 = K[i][0][k], m1 = K[i][1][k], m2 = K[i][2][k];
            K[i][1][k] = m0*(1.0f - uy2) - 2.0f*uy*m1 - m2;
            K[i][0][k] = 0.5f*(m0*(uy2 - uy) + m1*(2.0f*uy - 1.0f) + m2);
            K[i][2][k] = 0.5f*(m0*(uy2 + uy) + m1*(2.0f*uy + 1.0f) + m2);
        }
    }
    // z-direction (vel = uz)
    float uz2 = uz * uz;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float m0 = K[i][j][0], m1 = K[i][j][1], m2 = K[i][j][2];
            K[i][j][1] = m0*(1.0f - uz2) - 2.0f*uz*m1 - m2;
            K[i][j][0] = 0.5f*(m0*(uz2 - uz) + m1*(2.0f*uz - 1.0f) + m2);
            K[i][j][2] = 0.5f*(m0*(uz2 + uz) + m1*(2.0f*uz + 1.0f) + m2);
        }
    }
    // Now K[i][j][k] = f*_{ijk} (post-collision)

    // ══════════════════════════════════════════════════════════
    // Step 9: Write f_post (ijk → flat) + Guo source term
    // ══════════════════════════════════════════════════════════
    if (force != NULL) {
        float pf = 1.0f - 0.5f * omega_1;  // Guo prefactor
        for (int q = 0; q < 27; q++) {
            int i = cx[q]+1, j = cy[q]+1, k = cz[q]+1;
            float cxq = (float)cx[q], cyq = (float)cy[q], czq = (float)cz[q];
            float ci_F = cxq*Fx + cyq*Fy + czq*Fz;
            float ci_u = cxq*ux + cyq*uy + czq*uz;
            float u_F  = ux*Fx + uy*Fy + uz*Fz;
            float Si = pf * w[q] * ((ci_F - u_F)*3.0f + ci_u*ci_F*9.0f);
            f_post[q * N + idx] = K[i][j][k] + Si;
        }
    } else {
        for (int q = 0; q < 27; q++) {
            int i = cx[q]+1, j = cy[q]+1, k = cz[q]+1;
            f_post[q * N + idx] = K[i][j][k];
        }
    }
}
'''


# ─── Streaming-fused version: pull from neighbors + collide ──

_CUMULANT_D3Q27_STREAM_COLLIDE_KERNEL = (
    _CUMULANT_D3Q27_KERNEL
    .replace(
        'void cumulant_collide_d3q27(',
        'void cumulant_stream_collide_d3q27(',
    )
    .replace(
        '    const int N\n'
        ') {\n'
        '    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n'
        '    if (idx >= N) return;',
        '    const int Nx, const int Ny, const int Nz\n'
        ') {\n'
        '    int N = Nx * Ny * Nz;\n'
        '    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n'
        '    if (idx >= N) return;\n'
        '\n'
        '    // 1D → 3D index\n'
        '    int iz = idx / (Nx * Ny);\n'
        '    int rem = idx - iz * Nx * Ny;\n'
        '    int iy = rem / Nx;\n'
        '    int ix = rem - iy * Nx;',
    )
    .replace(
        # Replace local read with neighbor pull
        '    for (int q = 0; q < 27; q++) {\n'
        '        float fq = f_in[q * N + idx];\n'
        '        f_local[q] = fq;',
        '    for (int q = 0; q < 27; q++) {\n'
        '        int sx = (ix - cx[q] + Nx) % Nx;\n'
        '        int sy = (iy - cy[q] + Ny) % Ny;\n'
        '        int sz = (iz - cz[q] + Nz) % Nz;\n'
        '        int src_idx = sz * Nx * Ny + sy * Nx + sx;\n'
        '        float fq = f_in[q * N + src_idx];\n'
        '        f_local[q] = fq;',
    )
)


class CumulantStreamCollideKernelD3Q27:
    """Fused pull-collide kernel for D3Q27 Cumulant.

    Combines streaming + macroscopic + cumulant collision in 1 launch.
    Uses ping-pong buffers. f_dst = post-collision (serves as f_post for BC).
    """

    def __init__(self, block_size: int = 128) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _CUMULANT_D3Q27_STREAM_COLLIDE_KERNEL,
            'cumulant_stream_collide_d3q27',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        f_src: 'npt.NDArray',
        f_dst: 'npt.NDArray',
        rho_out: 'npt.NDArray',
        u_out: 'npt.NDArray',
        force: Optional['npt.NDArray'],
        omega_1: float,
        omega_bulk: float,
        omega_high: float,
        Nx: int, Ny: int, Nz: int,
    ) -> None:
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
                cp.float32(omega_1),
                cp.float32(omega_bulk),
                cp.float32(omega_high),
                cp.int32(Nx), cp.int32(Ny), cp.int32(Nz),
            ),
        )


class CumulantCollideKernelD3Q27:
    """Fused Cumulant collision kernel for D3Q27.

    Replaces macroscopic.compute() + cumulant.collide() with a single
    CUDA kernel launch.

    Usage:
        >>> kernel = CumulantCollideKernelD3Q27()
        >>> kernel.launch(f, f_post, rho, u, force, omega_1, omega_bulk, omega_high, N)
    """

    def __init__(self, block_size: int = 128) -> None:
        """Initialize kernel.

        Args:
            block_size: CUDA threads per block. Lower than BGK (128 vs 256)
                       due to higher register pressure (~90 registers).
        """
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _CUMULANT_D3Q27_KERNEL,
            'cumulant_collide_d3q27',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        f_in: 'npt.NDArray',
        f_post: 'npt.NDArray',
        rho_out: 'npt.NDArray',
        u_out: 'npt.NDArray',
        force: Optional['npt.NDArray'],
        omega_1: float,
        omega_bulk: float,
        omega_high: float,
        N: int,
    ) -> None:
        """Launch the fused cumulant collision kernel.

        Args:
            f_in:       Input distribution (27, N)
            f_post:     Post-collision output (27, N)
            rho_out:    Density output (N,)
            u_out:      Velocity output (3, N)
            force:      Body force (3, N) or None
            omega_1:    Shear relaxation rate 1/τ
            omega_bulk: Bulk viscosity rate ω₂
            omega_high: Higher-order rate ω₃-ω₁₀
            N:          Total spatial nodes
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
                cp.float32(omega_1),
                cp.float32(omega_bulk),
                cp.float32(omega_high),
                cp.int32(N),
            ),
        )
