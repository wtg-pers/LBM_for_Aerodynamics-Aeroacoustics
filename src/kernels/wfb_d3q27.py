"""CUDA Wall-Function Bounce (WFB) fused sparse pass for D3Q27 — W2.

Runs AFTER the obstacle bounce (HWBB/IBB kernel) on the standard path.
Per boundary link (fluid cell x_f, std direction d into the solid), one
thread executes the fused chain of patch_notes/wall_model/PLAN.md W2:

    trilinear u at the link's sample point x_s
      -> tangential decomposition against the link's wall normal
      -> wall-law u_tau (runtime law_id branch; laws mirrored 1:1 by
         src/boundary/wall_function.py host classes)
      -> per-link gate  (enable flag  AND  U_t > tiny  AND  y+ > y+_min)
      -> virtual wall velocity  u_w = (U_t - u_tau^2 y_s / nu_eff) t_hat
      -> bounce-slot correction f[opp(d), x_f] -= coef * rho * (c_d.u_w)
      -> REAL-EXCHANGE force  F += c_d (f*_d + f_returned)   (f64 atomics)

nu_eff of the u_w relation is pluggable (W3b, the "nu_eff definition"
question of PLAN W3):
    mode 0  scalar nu_eff (default: molecular — W1b-exact laminar limit)
    mode 1  nu + nu_t(x_s), nu_t trilinearly sampled from the SGS field
            with the SAME corner weights as u
    mode 2  nu + kappa u_tau y_s  (equilibrium mixing length, kappa=0.41
            = Musker's; field-free -> u_w ~ U_t - u_tau/kappa in the log
            layer, the classic slip closure)
    mode 3  nu + kappa u_tau y_s [1 - exp(-y+/A+)]^2, A+ = 26 — the
            van-Driest-damped mixing length of CAMWA 2024 Eq.(25)
            (Liang et al., Comput. Math. Appl. 158, 21-35), i.e. mode 2
            plus the damping the reference prescribes. Pairs with the
            near-wall eddy-viscosity reconstruction of their Eq.(39)
            (src/turbulence/near_wall_sgs.py) — patch_notes 09/10.
u_w DETERMINATION (uw_mode) — the W3c stage-2 fork:
    mode 0 "gradient"  u_w = U_t - u_tau^2 y_s / nu_eff   (the CAMWA /
        classic (c) relation). Linearises the wall-to-sample profile
        with a single nu_eff; both published prescriptions for that
        nu_eff missed our Delta+ = 31 channel by 3.4x (too small, mode
        1) and 8.4x (too large, mode 3) — patch_notes 07 / 12.
    mode 1 "target"    *** EXPERIMENTALLY UNSTABLE AS WRITTEN — see
        patch_notes/wall_model/12. Kept because the mechanism is
        gate-verified (kernel == host) and it becomes usable the moment
        u_w is solved PER WALL CELL instead of per link. ***
        u_w is a per-link CONTROL state driven so that
        the link's own fluid cell reaches the wall law's own profile
        value:  u_w <- u_w + gain (u~(y1, u_tau) - U_1),
        with y1 the wall-normal distance to that cell centre and U_1 its
        tangential velocity. The fixed point u(y1) = u~(y1, u_tau) is
        exactly the condition Malaspinas-Sagaut (JCP 275 (2014) 25-40)
        impose by overwriting the boundary node; here it is reached
        through the bounce instead, so it also works on curved IBB
        walls. NO nu_eff and no gap linearisation — whatever the real
        near-wall transport is, the controller lands the profile.
        FAILURE MODE (measured): the 9 links of one wall cell each
        integrate the SAME cell-velocity error independently, so their
        u_w values drift apart in the null space of that observable —
        within-cell spread grows to 2.2x the between-cell spread and the
        run diverges (~1500 steps). The observable is per cell; the
        control variable must be too.
    mode 2 "flux"      FLUX TARGETING with a per-wall-cell reduction —
        the resolution of the above (patch_notes/wall_model/12 §5).
        The tangential momentum a link hands the body is linear in u_w,
            m = m0 - S_l u_w,   m0 = (c_d.t^)(f*_d + f_bb),
            S_l = coef rho (c_d.t^)^2,
        so summing over the links of one wall cell and demanding that
        the cell deliver exactly the modelled stress gives an ALGEBRAIC
        single-step solution — no nu_eff, no integrator, hence no
        stability question:
            u_w(cell) = (SUM m0 - SUM 3 u_tau^2 S_l) / SUM S_l.
        The effective wall area is measured by the cell's own exchange
        sensitivity (SUM S_l = rho c_s^2 A), which reduces to A = 1 on a
        flat halfway wall and generalises to curved IBB links without an
        area bookkeeping of its own. Runs as two passes: `wfb_apply_d3q27`
        with uw_mode = 2 only reduces (no f write, no force), then
        `wfb_flux_apply_d3q27` writes and accumulates the force.

u_w is symmetrically clamped to +-uw_clamp*U_t; the default +inf is
bit-neutral (IEEE fmin/fmax with infinities), so W2/W3a results are
unchanged.

Wall-model INPUT TIME FILTER (W3b, Kawai-Larsson practice): the sampled
(u, nu_t) enter through a per-link exponential filter
    s_hat(t) = (1 - eps) s_hat(t-1) + eps s(t),   state in wl_uf (n,4),
because the instantaneous turbulent sample makes the u_w correction
noisy at ~(S/gN ~ 5e2) amplification (w3b smoke measured +-500 gN force
swings in "sgs" mode). eps = 1 (default) is bit-neutral:
(1-1)*prev + 1*s == s exactly.

The correction is the moving-wall BB term applied to the already-bounced
slot; coef = 6 w_d (q<1/2) or 6 w_d/(2q) (q>=1/2, Bouzidi moving-wall
interpolation weight) is precomputed host-side (src/boundary/wfb.py, also
the "uniform" knob for the W3 (b)/(c) discrimination). A gated-off link
writes NOTHING — the existing HWBB/IBB result stays bit-identical (the
fallback contract of review 03) — but its exchange still enters the force
sum, so the pass returns the total body MEM force (the W1b gate [C-force]
lesson: once slots are rewritten, only the fused sum c_d(f*_d + f_ret) is
faithful; a post-hoc 2c*f formula is not).

All per-link arithmetic is double precision with --fmad=false so the
numpy host reference (WallFunctionBounce.host_reference) reproduces the
laminar-law pass bit-for-bit; the Musker branch differs only by libm-vs-
CUDA atan/log ulps (W2 gate tolerance). ASCII-only source (nvrtc POSIX
locale rule).

Author: LBM Development Team
Date: 2026-08 (wall_model track W2)
"""

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

from src.boundary.wall_function import _MUSKER_C0

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


def _kernel_source() -> str:
    # Musker additive constant baked from the host module (repr round-trips
    # the exact f64 bits) so kernel and host share the anchor.
    c0 = f"{float(_MUSKER_C0)!r}"
    return r"""
static __device__ const int CXS[27] = {
    0,  1,-1, 0, 0, 0, 0,
    1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0,
    1,-1, 1,-1, 1,-1, 1,-1
};
static __device__ const int CYS[27] = {
    0,  0, 0, 1,-1, 0, 0,
    1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1,
    1, 1,-1,-1, 1, 1,-1,-1
};
static __device__ const int CZS[27] = {
    0,  0, 0, 0, 0, 1,-1,
    0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1,
    1, 1, 1, 1,-1,-1,-1,-1
};
static __device__ const int OPPS[27] = {
    0,  2, 1, 4, 3, 6, 5,
    10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
    26, 25, 24, 23, 22, 21, 20, 19
};

__device__ __forceinline__ int wfb_wrap(int i, const int n, const int per)
{
    if (per) { i %= n; if (i < 0) i += n; return i; }
    return (i < 0) ? 0 : ((i >= n) ? (n - 1) : i);
}

__device__ __forceinline__ double wfb_musker_uplus(const double yp)
{
    // Mirror of wall_function.musker_uplus (u+(0)=0 exact anchor).
    return 5.424 * atan((2.0 * yp - 8.15) / 16.7)
         + 4.1693 * log(yp + 10.6)
         - 0.8686 * log(yp * yp - 8.15 * yp + 86.0)
         - (""" + c0 + r""");
}

extern "C" __global__ void wfb_apply_d3q27(
    float*       __restrict__ f,        // (27, N) post-stream/BC/bounce
    const float* __restrict__ f_post,   // (27, N) post-collision
    const float* __restrict__ u,        // (3, N) macro velocity
    const float* __restrict__ rho,      // (N,)
    const float* __restrict__ nu_t_f,   // (N,) SGS nu_t (read iff mode 1)
    const int*   __restrict__ wl_cell,  // (n,) flat fluid cell (C-order)
    const signed char* __restrict__ wl_dir,   // (n,) std dir 1..26
    const float* __restrict__ wl_coef,  // (n,) bounce-correction coef
    const float* __restrict__ wl_nrm,   // (n,3) unit wall normal (fluid side)
    const float* __restrict__ wl_ncell, // (n,3) CELL-mean wall normal (static)
    const float* __restrict__ wl_xs,    // (n,3) sample point [level lu]
    float*       __restrict__ wl_uf,    // (n,4) filtered (u, nu_t) state
    float*       __restrict__ wl_utau,  // (n,) out: per-link u_tau (0=off)
    const float* __restrict__ wl_y1,    // (n,) wall->own-cell-centre distance
    float*       __restrict__ wl_uw,    // (n,) u_w control state (mode 1)
    float*       __restrict__ wl_that,  // (n,3) out: unit tangent (mode 2)
    unsigned char* __restrict__ wl_gate,// (n,) out: link was gated (mode 2)
    const int*   __restrict__ wl_grp,   // (n,) wall-cell group id (mode 2)
    double*      __restrict__ grp_m0,   // (n_grp,) accum: no-slip exchange
    double*      __restrict__ grp_S,    // (n_grp,) accum: sensitivity
    double*      __restrict__ grp_T,    // (n_grp,) accum: 3 u_tau^2 S_l
    double*      __restrict__ grp_U,    // (n_grp,) accum: S_l U_t (sample)
    double*      __restrict__ grp_V,    // (n_grp,) accum: S_l U_1 (own cell)
    const unsigned char* __restrict__ wl_enable,  // (n,) build/sensor gate
    double*      __restrict__ force_out,          // (3,) f64 or NULL
    const int n_links,
    const int Nx, const int Ny, const int Nz,
    const double y_s,                   // wall->sample distance [lu]
    const double nu,                    // molecular viscosity [lu^2/lt]
    const double nu_eff,                // mode-0 scalar u_w viscosity
    const int nu_eff_mode,              // 0 scalar | 1 nu+nu_t(x_s) | 2 mixing
    const double uw_clamp,              // |u_w| <= uw_clamp*U_t (inf = off)
    const double s_min,                 // per-cell tangential-coupling floor
    const double uf_eps,                // input-filter blend (1 = off)
    const int uw_mode,                  // 0 gradient | 1 law-target control
    const double uw_gain,               // control gain (mode 1)
    const double yplus_min,             // gate threshold (~30; 0 = off)
    const int law_id,                   // 0 = viscous/laminar, 1 = Musker
    const double law_p0,                // law constant (laminar: alpha)
    const int law_iters,                // Musker fixed-point sweeps
    const int per_x, const int per_y, const int per_z,
    const int cx0, const int cx1,       // owned clip [x0, x1) etc. (MPI)
    const int cy0, const int cy1,
    const int cz0, const int cz1)
{
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= n_links) return;

    const long long N = (long long)Nx * Ny * Nz;
    const int cell = wl_cell[l];
    const int d = (int)wl_dir[l];
    const int iz = cell % Nz;
    const int iy = (cell / Nz) % Ny;
    const int ix = cell / (Ny * Nz);

    const long long addr = (long long)OPPS[d] * N + cell;
    const double f_old = (double)f[addr];
    const double f_inc = (double)f_post[(long long)d * N + cell];
    double f_ret = f_old;               // returned population (default: keep)
    wl_utau[l] = 0.0f;                  // overwritten iff the link is gated
    if (uw_mode == 2) wl_gate[l] = 0;

    if (wl_enable[l]) {
        // -- trilinear velocity sample at x_s (s_ok guaranteed the 2x2x2
        //    neighbourhood is fluid; wrap on periodic axes, clamp otherwise)
        const double xs0 = (double)wl_xs[3 * l];
        const double xs1 = (double)wl_xs[3 * l + 1];
        const double xs2 = (double)wl_xs[3 * l + 2];
        const double fx = xs0 - floor(xs0);
        const double fy = xs1 - floor(xs1);
        const double fz = xs2 - floor(xs2);
        const int ib[2] = { wfb_wrap((int)floor(xs0), Nx, per_x),
                            wfb_wrap((int)floor(xs0) + 1, Nx, per_x) };
        const int jb[2] = { wfb_wrap((int)floor(xs1), Ny, per_y),
                            wfb_wrap((int)floor(xs1) + 1, Ny, per_y) };
        const int kb[2] = { wfb_wrap((int)floor(xs2), Nz, per_z),
                            wfb_wrap((int)floor(xs2) + 1, Nz, per_z) };
        const double wa[2] = { 1.0 - fx, fx };
        const double wb[2] = { 1.0 - fy, fy };
        const double wc[2] = { 1.0 - fz, fz };
        const int want_nut = (nu_eff_mode == 1);
        double sx = 0.0, sy = 0.0, sz = 0.0, snt = 0.0;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
                for (int c = 0; c < 2; c++) {
                    const double wgt = wa[a] * wb[b] * wc[c];
                    const long long cid =
                        ((long long)ib[a] * Ny + jb[b]) * Nz + kb[c];
                    sx += wgt * (double)u[cid];
                    sy += wgt * (double)u[N + cid];
                    sz += wgt * (double)u[2 * N + cid];
                    if (want_nut) snt += wgt * (double)nu_t_f[cid];
                }

        // -- per-link exponential input filter (state; eps=1 -> raw)
        {
            const long long b4 = 4LL * l;
            sx = (1.0 - uf_eps) * (double)wl_uf[b4] + uf_eps * sx;
            sy = (1.0 - uf_eps) * (double)wl_uf[b4 + 1] + uf_eps * sy;
            sz = (1.0 - uf_eps) * (double)wl_uf[b4 + 2] + uf_eps * sz;
            snt = (1.0 - uf_eps) * (double)wl_uf[b4 + 3] + uf_eps * snt;
            wl_uf[b4] = (float)sx;
            wl_uf[b4 + 1] = (float)sy;
            wl_uf[b4 + 2] = (float)sz;
            wl_uf[b4 + 3] = (float)snt;
        }

        // -- tangential decomposition (local frame, W1 convention).
        // uw_mode 2 reduces over a WALL CELL, so every link of that cell
        // must project onto the SAME tangent, else the summed momentum
        // is not a projection of one vector. Per-link tangents differ by
        // ~2 deg median on a sphere but by up to 177 deg on links whose
        // tangential projection nearly vanishes (measured, patch 18) --
        // enough to corrupt the cell sum. So mode 2 builds the tangent
        // from the CELL-mean normal (static geometry) and the cell's own
        // velocity, which is identical for all links by construction.
        double nx_, ny_, nz_, tx, ty, tz;
        if (uw_mode == 2) {
            nx_ = (double)wl_ncell[3 * l];
            ny_ = (double)wl_ncell[3 * l + 1];
            nz_ = (double)wl_ncell[3 * l + 2];
            const double cx_ = (double)u[cell];
            const double cy_ = (double)u[N + cell];
            const double cz_ = (double)u[2 * N + cell];
            const double cn = cx_ * nx_ + cy_ * ny_ + cz_ * nz_;
            tx = cx_ - cn * nx_;
            ty = cy_ - cn * ny_;
            tz = cz_ - cn * nz_;
            const double tm = sqrt(tx * tx + ty * ty + tz * tz);
            if (tm > 1e-14) { tx /= tm; ty /= tm; tz /= tm; }
            // magnitude still from the SAMPLE point, projected on it
            const double Us = sx * tx + sy * ty + sz * tz;
            tx *= Us; ty *= Us; tz *= Us;
        } else {
            nx_ = (double)wl_nrm[3 * l];
            ny_ = (double)wl_nrm[3 * l + 1];
            nz_ = (double)wl_nrm[3 * l + 2];
            const double un = sx * nx_ + sy * ny_ + sz * nz_;
            tx = sx - un * nx_;
            ty = sy - un * ny_;
            tz = sz - un * nz_;
        }
        const double Ut = sqrt(tx * tx + ty * ty + tz * tz);

        if (Ut > 1e-14) {
            // -- wall law u_tau (host mirrors: wall_function.WALL_LAWS)
            double ut;
            if (law_id == 0) {
                ut = sqrt(law_p0 * nu * Ut / y_s);
            } else {
                const double tiny = 1e-14;
                double u0 = sqrt(fmax(nu * Ut / y_s, 0.0));
                ut = fmax(u0, tiny);
                for (int it = 0; it < law_iters; it++) {
                    const double yp = y_s * ut / nu;
                    const double up = wfb_musker_uplus(fmax(yp, 0.0));
                    const double utn =
                        (up > tiny) ? (Ut / fmax(up, tiny)) : 0.0;
                    ut = ut + 0.5 * (utn - ut);
                }
            }

            // -- per-link gate: y+ threshold (transition-sensor slot is
            //    the wl_enable byte, cleared host-side)
            const double yplus = y_s * ut / nu;
            if (yplus > yplus_min) {
                const double inv = 1.0 / Ut;
                const double thx = tx * inv;
                const double thy = ty * inv;
                const double thz = tz * inv;
                wl_utau[l] = (float)ut;
                if (uw_mode == 2) {
                    // -- reduce only: hand the wall cell its share of the
                    //    no-slip exchange and of the exchange sensitivity
                    const double cdt = (double)CXS[d] * thx
                                     + (double)CYS[d] * thy
                                     + (double)CZS[d] * thz;
                    const double Sl = (double)wl_coef[l] * (double)rho[cell]
                                    * cdt * cdt;
                    const int g = wl_grp[l];
                    atomicAdd(&grp_m0[g], cdt * (f_inc + f_old));
                    atomicAdd(&grp_S[g], Sl);
                    atomicAdd(&grp_T[g], 3.0 * ut * ut * Sl);
                    atomicAdd(&grp_U[g], Sl * Ut);
                    // own-cell tangential velocity: u_w = U_1 is the
                    // zero-tangential-exchange (free-slip) reference, so
                    // the SMALL wall-model correction is measured from
                    // there instead of from no-slip (conditioning).
                    atomicAdd(&grp_V[g], Sl * ((double)u[cell] * thx
                                             + (double)u[N + cell] * thy
                                             + (double)u[2 * N + cell] * thz));
                    wl_that[3 * l]     = (float)thx;
                    wl_that[3 * l + 1] = (float)thy;
                    wl_that[3 * l + 2] = (float)thz;
                    wl_gate[l] = 1;
                    return;             // pass 2 writes f and the force
                }
                double uw;
                if (uw_mode == 1) {
                    // law-target control (M-S condition through the bounce)
                    const double y1 = (double)wl_y1[l];
                    const double U_target =
                        ut * wfb_musker_uplus(y1 * ut / nu);
                    const double U1 = (double)u[cell] * thx
                                    + (double)u[N + cell] * thy
                                    + (double)u[2 * N + cell] * thz;
                    uw = (double)wl_uw[l] + uw_gain * (U_target - U1);
                } else {
                    double nueff = nu_eff;
                    if (nu_eff_mode == 1) nueff = nu + snt;
                    else if (nu_eff_mode == 2) nueff = nu + 0.41 * ut * y_s;
                    else if (nu_eff_mode == 3) {
                        // CAMWA 2024 Eq.(25): van Driest damped mixing length
                        const double vd = 1.0 - exp(-yplus / 26.0);
                        nueff = nu + 0.4184 * ut * y_s * vd * vd;
                    }
                    uw = Ut - ut * ut * y_s / nueff;
                }
                const double uwmax = uw_clamp * Ut;
                uw = fmin(fmax(uw, -uwmax), uwmax);
                // persist the (clamped) control state
                if (uw_mode == 1) wl_uw[l] = (float)uw;
                const double cdott = (double)CXS[d] * thx
                                   + (double)CYS[d] * thy
                                   + (double)CZS[d] * thz;
                const double corr =
                    (double)wl_coef[l] * (double)rho[cell] * cdott * uw;
                const float vf = (float)(f_old - corr);
                f[addr] = vf;
                f_ret = (double)vf;
            }
        }
    }

    // mode 2 links that never entered the gate: pass 2 owns their force
    if (uw_mode == 2) return;

    // -- real-exchange MEM force on the body (owned-clip for MPI ranks)
    if (force_out != NULL &&
        ix >= cx0 && ix < cx1 && iy >= cy0 && iy < cy1 &&
        iz >= cz0 && iz < cz1) {
        const double m = f_inc + f_ret;
        if (CXS[d] != 0) atomicAdd(&force_out[0], m * (double)CXS[d]);
        if (CYS[d] != 0) atomicAdd(&force_out[1], m * (double)CYS[d]);
        if (CZS[d] != 0) atomicAdd(&force_out[2], m * (double)CZS[d]);
    }
}

extern "C" __global__ void wfb_flux_apply_d3q27(
    float*       __restrict__ f,
    const float* __restrict__ f_post,
    const float* __restrict__ rho,
    const int*   __restrict__ wl_cell,
    const signed char* __restrict__ wl_dir,
    const float* __restrict__ wl_coef,
    const float* __restrict__ wl_that,
    const unsigned char* __restrict__ wl_gate,
    const int*   __restrict__ wl_grp,
    const double* __restrict__ grp_m0,
    const double* __restrict__ grp_S,
    const double* __restrict__ grp_T,
    const double* __restrict__ grp_U,
    const double* __restrict__ grp_V,
    float*       __restrict__ wl_uw,    // (n,) in/out: RELAXED delta state
    double*      __restrict__ force_out,
    const int n_links,
    const double s_min,
    const double uw_relax,              // 1 = deadbeat, <1 = damped
    const double uw_clamp,              // |u_w| <= uw_clamp * U_t(cell)
    const int Nx, const int Ny, const int Nz,
    const int cx0, const int cx1,
    const int cy0, const int cy1,
    const int cz0, const int cz1)
{
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= n_links) return;

    const long long N = (long long)Nx * Ny * Nz;
    const int cell = wl_cell[l];
    const int d = (int)wl_dir[l];
    const int iz = cell % Nz;
    const int iy = (cell / Nz) % Ny;
    const int ix = cell / (Ny * Nz);

    const long long addr = (long long)OPPS[d] * N + cell;
    const double f_old = (double)f[addr];
    double f_ret = f_old;

    if (wl_gate[l]) {
        const int g = wl_grp[l];
        const double S = grp_S[g];
        // Span-direction degeneracy guard (CAMWA's 'spanwise weight
        // coefficient = 0 -> divergence'): a cell whose links are
        // nearly perpendicular to the flow has S -> 0 and u_w blows
        // up. Measured on a sphere: 27% of cells sit below 5% of the
        // flat-wall value rho*c_s^2 = rho/3, yet they carry only ~1%
        // of the drag capacity, so falling those back to no-slip is
        // nearly free (patch 18).
        if (S > s_min) {
            // Deliver the modelled stress over this wall cell.
            //
            // The zero-tangential-exchange wall velocity is EXACT and
            // needs no proxy: delivered(u_w) = m0 - S u_w vanishes at
            //     u_w0 = m0 / S.
            // (patch 14 approximated that reference by the cell's macro
            // velocity U_1; measured, the two differ by ~25% of U_1
            // because m0's convective part is a direction-weighted mean
            // of the POPULATIONS, not the macroscopic velocity. Near a
            // wall that bias is ~700x the target and it set u_w -- the
            // reason the 600k channel run drifted to zero drag, patch 19.)
            //
            // u_w0 carries the turbulent noise, so IT is what gets
            // time-filtered; the small deterministic offset stays
            // outside the filter:
            //     u_w = <m0/S>_filtered - 3 u_tau^2 ,   3 u_tau^2 = T/S
            // Mean delivered = T exactly, with the instantaneous stress
            // free to fluctuate as it physically should.
            const double uw0 = grp_m0[g] / S;
            double st = (1.0 - uw_relax) * (double)wl_uw[l]
                      + uw_relax * uw0;
            wl_uw[l] = (float)st;              // filtered reference state
            double uw = st - grp_T[g] / S;
            const double Utc = grp_U[g] / S;
            const double uwmax = uw_clamp * fabs(Utc);
            uw = fmin(fmax(uw, -uwmax), uwmax);
            const double thx = (double)wl_that[3 * l];
            const double thy = (double)wl_that[3 * l + 1];
            const double thz = (double)wl_that[3 * l + 2];
            const double cdt = (double)CXS[d] * thx
                             + (double)CYS[d] * thy
                             + (double)CZS[d] * thz;
            const double corr =
                (double)wl_coef[l] * (double)rho[cell] * cdt * uw;
            const float vf = (float)(f_old - corr);
            f[addr] = vf;
            f_ret = (double)vf;
        }
    } else {
        wl_uw[l] = 0.0f;
    }

    if (force_out != NULL &&
        ix >= cx0 && ix < cx1 && iy >= cy0 && iy < cy1 &&
        iz >= cz0 && iz < cz1) {
        const double m = (double)f_post[(long long)d * N + cell] + f_ret;
        if (CXS[d] != 0) atomicAdd(&force_out[0], m * (double)CXS[d]);
        if (CYS[d] != 0) atomicAdd(&force_out[1], m * (double)CYS[d]);
        if (CZS[d] != 0) atomicAdd(&force_out[2], m * (double)CZS[d]);
    }
}
"""


class WFBKernelD3Q27:
    """Thin launcher for the fused WFB pass (one launch per step)."""

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None
        self._k_flux = None

    def _compile(self) -> None:
        import cupy as cp
        src = _kernel_source()
        assert all(ord(ch) < 128 for ch in src)  # nvrtc POSIX locale rule
        # --fmad=false: IEEE mul/add order == numpy host reference
        # (bit-reproducible laminar branch; sparse pass, cost negligible).
        self._kernel = cp.RawKernel(
            src, 'wfb_apply_d3q27', options=('--fmad=false',))
        self._k_flux = cp.RawKernel(
            src, 'wfb_flux_apply_d3q27', options=('--fmad=false',))

    def flux_apply(self, args, n_links: int) -> None:
        """Launch pass 2 of the flux-targeting path (uw_mode = 2)."""
        if self._k_flux is None:
            self._compile()
        grid = ((n_links + self._block_size - 1) // self._block_size,)
        self._k_flux(grid, (self._block_size,), args)

    def apply(
        self,
        f: 'npt.NDArray',
        f_post: 'npt.NDArray',
        u: 'npt.NDArray',
        rho: 'npt.NDArray',
        nu_t_f: 'npt.NDArray',
        wl_cell: 'npt.NDArray',
        wl_dir: 'npt.NDArray',
        wl_coef: 'npt.NDArray',
        wl_nrm: 'npt.NDArray',
        wl_ncell: 'npt.NDArray',
        wl_xs: 'npt.NDArray',
        wl_uf: 'npt.NDArray',
        wl_utau: 'npt.NDArray',
        wl_y1: 'npt.NDArray',
        wl_uw: 'npt.NDArray',
        wl_that: 'npt.NDArray',
        wl_gate: 'npt.NDArray',
        wl_grp: 'npt.NDArray',
        grp_m0: 'npt.NDArray',
        grp_S: 'npt.NDArray',
        grp_T: 'npt.NDArray',
        grp_U: 'npt.NDArray',
        grp_V: 'npt.NDArray',
        wl_enable: 'npt.NDArray',
        force_out: Optional['npt.NDArray'],
        n_links: int,
        domain_shape: Tuple[int, int, int],
        y_s: float,
        nu: float,
        nu_eff: float,
        nu_eff_mode: int,
        uw_clamp: float,
        s_min: float,
        uf_eps: float,
        uw_mode: int,
        uw_gain: float,
        yplus_min: float,
        law_id: int,
        law_p0: float,
        law_iters: int,
        periodic: Tuple[bool, bool, bool],
        clip_bounds: Sequence[Tuple[int, int]],
    ) -> None:
        if n_links == 0:
            return
        if self._kernel is None:
            self._compile()
        import cupy as cp
        nx, ny, nz = domain_shape
        grid = ((n_links + self._block_size - 1) // self._block_size,)
        cb = clip_bounds
        self._kernel(grid, (self._block_size,), (
            f, f_post, u, rho, nu_t_f,
            wl_cell, wl_dir, wl_coef, wl_nrm, wl_ncell, wl_xs, wl_uf, wl_utau,
            wl_y1, wl_uw, wl_that, wl_gate, wl_grp,
            grp_m0, grp_S, grp_T, grp_U, grp_V, wl_enable, force_out,
            cp.int32(n_links),
            cp.int32(nx), cp.int32(ny), cp.int32(nz),
            cp.float64(y_s), cp.float64(nu), cp.float64(nu_eff),
            cp.int32(nu_eff_mode), cp.float64(uw_clamp),
            cp.float64(s_min),
            cp.float64(uf_eps),
            cp.int32(uw_mode), cp.float64(uw_gain),
            cp.float64(yplus_min),
            cp.int32(law_id), cp.float64(law_p0), cp.int32(law_iters),
            cp.int32(int(periodic[0])), cp.int32(int(periodic[1])),
            cp.int32(int(periodic[2])),
            cp.int32(cb[0][0]), cp.int32(cb[0][1]),
            cp.int32(cb[1][0]), cp.int32(cb[1][1]),
            cp.int32(cb[2][0]), cp.int32(cb[2][1])))
