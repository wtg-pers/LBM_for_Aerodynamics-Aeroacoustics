"""CUDA kernels for the surfel facet boundary — surfel track S3k.

Mirrors `src.boundary.surfel.SurfelFacets` expression-for-expression; that
host class is the bit-comparison anchor (gate s3k). Four passes:

    surfel_gather      Gamma_in[a,i]  = sum_x V_i^a(x) n'_i(x)          (8)
    surfel_scatter     Eq. (10) | (16) [+ (25)] then (21,22), per facet
    surfel_distribute  Q_i(x) += [V_i^a(x)/Vsum] Gamma_out[a,i]         (6)
    surfel_advect      N_i(y) = [dV - g_i](y-c_i) n'_i(y-c_i) + Q_i(y)  (3)

Layout choices, and why:
  * gather/distribute are threaded over (facet, direction PAIR). Exactly one
    of the pair's two directions is in-going for a given facet (c.n has one
    sign), so a thread handles one direction and the CSR range is the same
    for both -- this is the S2 "one prism per pair" fact turned into a
    thread mapping.
  * the population field is float32 (production layout) but every facet
    accumulation is float64: Eq. (16) subtracts Gamma_in from a comparable
    V*(feq+feq), and Eq. (22) is a difference of shell sums, so f32
    accumulation would eat the very cancellation the scheme relies on.
  * scatter is one thread per facet with a 27-entry local Gamma_out, because
    the shell mass fix (21,22) needs the uncorrected out-going set first.

ASCII-only source (nvrtc under a POSIX locale).

Author: LBM Development Team
Date: 2026-08 (surfel track S3k)
"""

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from src.boundary.surfel_transport import C27, N_PAIR, OPP27, PAIR_DIR

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

_W27 = np.array([8 / 27] + [2 / 27] * 6 + [1 / 54] * 12 + [1 / 216] * 8)
#: additive constant of wall_function.musker_uplus, embedded at full f64
#: precision so the device law is the same function as the host one
_MUSKER_C0 = float(5.424 * np.arctan(-8.15 / 16.7)
                   + 4.1693 * np.log(10.6) - 0.8686 * np.log(86.0))
_CSQ = (C27 * C27).sum(axis=1)
#: (13, 2) direction indices of each pair: (+v_p, -v_p)
PAIR_IDX = np.zeros((N_PAIR, 2), dtype=np.int32)
for _p in range(N_PAIR):
    _ip = int(np.nonzero((C27 == PAIR_DIR[_p][None, :]).all(axis=1))[0][0])
    PAIR_IDX[_p] = (_ip, int(OPP27[_ip]))


def _census_dump(tag: str, top: int = 40) -> None:
    """Print every live device array grouped by (dtype, shape) — the
    mem-census doctrine's run-state instrument (64 sec. 17). gc-global:
    in-flight locals of enclosing frames are visible too."""
    import gc
    import os
    import sys
    import cupy as cp
    mp = cp.get_default_memory_pool()
    rows, seen = {}, set()
    for o in gc.get_objects():
        if isinstance(o, cp.ndarray):
            base = o.base if o.base is not None else o
            if id(base) in seen:
                continue
            seen.add(id(base))
            key = (str(base.dtype), base.shape)
            n, b = rows.get(key, (0, 0))
            rows[key] = (n + 1, b + base.nbytes)
    pid = os.getpid()
    lines = [f"[census pid{pid}] {tag}: used {mp.used_bytes() / 2**30:.2f}"
             f" / held {mp.total_bytes() / 2**30:.2f} GiB"]
    tot = 0
    for (dt, shp), (n, b) in sorted(rows.items(), key=lambda kv: -kv[1][1])[:top]:
        tot += b
        lines.append(f"[census pid{pid}]   {b / 2**30:7.3f} GiB  x{n:<3d} "
                     f"{dt:<9s} {shp}")
    lines.append(f"[census pid{pid}]   top-{top} sum {tot / 2**30:.2f} GiB")
    print("\n".join(lines), file=sys.stderr, flush=True)


def _ints(a):
    return ", ".join(str(int(v)) for v in np.asarray(a).ravel())


def _dbls(a):
    return ", ".join(repr(float(v)) for v in np.asarray(a).ravel())


def _source() -> str:
    tables = f"""
__constant__ int  CX[27] = {{{_ints(C27[:, 0])}}};
__constant__ int  CY[27] = {{{_ints(C27[:, 1])}}};
__constant__ int  CZ[27] = {{{_ints(C27[:, 2])}}};
__constant__ int  OPP[27] = {{{_ints(OPP27)}}};
__constant__ int  CSQ[27] = {{{_ints(_CSQ)}}};
__constant__ double WT[27] = {{{_dbls(_W27)}}};
__constant__ int  PIDX[{2 * N_PAIR}] = {{{_ints(PAIR_IDX)}}};
#define N_PAIR {N_PAIR}
#define THETA (1.0/3.0)
#define MUSKER_C0 ({_MUSKER_C0!r})
"""
    return tables + r"""
__device__ __forceinline__ double musker_uplus(const double yp)
{   // mirrors wall_function.musker_uplus (libm ulps, not bit)
    return 5.424*atan((2.0*yp - 8.15)/16.7)
         + 4.1693*log(yp + 10.6)
         - 0.8686*log(yp*yp - 8.15*yp + 86.0) - MUSKER_C0;
}

__device__ __forceinline__ double solve_utau_d(
    const double U, const double y, const double nu,
    const int law_id, const int iters)
{   // mirrors wall_function.solve_utau with tol = 0 (fixed sweep), which is
    // exactly why that function runs the full max_iter: no batch-coupled
    // early exit, so host and kernel take the same path.
    const double tiny = 1e-14;
    if (!(U > tiny)) return 0.0;
    if (law_id == 0) return sqrt(fmax(nu*U/y, 0.0));      // viscous
    double ut = fmax(sqrt(fmax(nu*U/y, 0.0)), tiny);
    for (int k = 0; k < iters; ++k) {
        const double yp = y*ut/nu;
        const double up = musker_uplus(fmax(yp, 0.0));
        const double un = (up > tiny) ? (U/fmax(up, tiny)) : 0.0;
        ut = ut + 0.5*(un - ut);
    }
    return ut;
}

__device__ __forceinline__ double feq_i(
    const int i, const double rho, const double ux,
    const double uy, const double uz)
{
    const double cu = CX[i]*ux + CY[i]*uy + CZ[i]*uz;
    const double usq = ux*ux + uy*uy + uz*uz;
    return rho * WT[i] * (1.0 + cu/THETA + 0.5*(cu/THETA)*(cu/THETA)
                          - 0.5*usq/THETA);
}

extern "C" __global__ void surfel_gather(
    const float*  __restrict__ n_post,     // (27, N) f32 post-collision
    const long long* __restrict__ indptr,  // (n_f*N_PAIR+1,)
    const int*    __restrict__ cell,       // (nnz,)
    const double* __restrict__ wgt,        // (nnz,)
    const double* __restrict__ nrm,        // (n_f, 3)
    double*       __restrict__ G_in,       // (n_f, 27) out (pre-zeroed)
    const int n_f, const long long N)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_f * N_PAIR) return;
    const int a = t / N_PAIR;
    const int p = t - a * N_PAIR;

    const int ip = PIDX[2*p], im = PIDX[2*p+1];
    const double nx = nrm[3*a], ny = nrm[3*a+1], nz = nrm[3*a+2];
    const double cn = CX[ip]*nx + CY[ip]*ny + CZ[ip]*nz;
    if (cn == 0.0) return;                 // tangent pair: V == 0
    const int din = (cn < 0.0) ? ip : im;  // exactly one member is in-going

    const long long b = indptr[t], e = indptr[t+1];
    double acc = 0.0;
    for (long long k = b; k < e; ++k)
        acc += wgt[k] * (double)n_post[(long long)din * N + cell[k]];
    G_in[(long long)a * 27 + din] = acc;
}

__device__ __forceinline__ void sample_state(
    const float* rho, const float* uf, const unsigned char* live,
    const double* cen, const double* nrm, const int a,
    const double h, const int Nx, const int Ny, const int Nz,
    const int px, const int py, const int pz,
    double* out)      // out[0..3] = rho, u; out[4] = stencil weight
{
    const long long N = (long long)Nx * Ny * Nz;
    const double sx = cen[3*a]   + h * nrm[3*a];
    const double sy = cen[3*a+1] + h * nrm[3*a+1];
    const double sz = cen[3*a+2] + h * nrm[3*a+2];
    const double bx = floor(sx), by = floor(sy), bz = floor(sz);
    const double fx = sx - bx, fy = sy - by, fz = sz - bz;
    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0, wsum = 0.0;
    for (int da = 0; da < 2; ++da)
    for (int db = 0; db < 2; ++db)
    for (int dc = 0; dc < 2; ++dc) {
        double w = (da ? fx : 1.0-fx) * (db ? fy : 1.0-fy)
                 * (dc ? fz : 1.0-fz);
        int i = (int)bx + da, j = (int)by + db, k = (int)bz + dc;
        if (px) { i %= Nx; if (i < 0) i += Nx; }
        else    { i = min(max(i, 0), Nx-1); }
        if (py) { j %= Ny; if (j < 0) j += Ny; }
        else    { j = min(max(j, 0), Ny-1); }
        if (pz) { k %= Nz; if (k < 0) k += Nz; }
        else    { k = min(max(k, 0), Nz-1); }
        const long long c = ((long long)i * Ny + j) * Nz + k;
        // masked to live cells: a fully solid cell holds no populations, so
        // its zeros must not enter the facet state (host doc, S3 defect d)
        w *= (double)live[c];
        acc0 += w * (double)rho[c];
        acc1 += w * (double)uf[c];
        acc2 += w * (double)uf[N + c];
        acc3 += w * (double)uf[2*N + c];
        wsum += w;
    }
    const double inv = 1.0 / fmax(wsum, 1e-300);
    out[0] = acc0*inv; out[1] = acc1*inv;
    out[2] = acc2*inv; out[3] = acc3*inv; out[4] = wsum;
}

extern "C" __global__ void surfel_scatter(
    const double* __restrict__ G_in,       // (n_f, 27)
    const double* __restrict__ nrm,        // (n_f, 3)
    const double* __restrict__ area,       // (n_f,)
    const double* __restrict__ cen,        // (n_f, 3)
    const float*  __restrict__ rho,        // (N,)
    const float*  __restrict__ uf,         // (3, N)
    const unsigned char* __restrict__ live,// (N,)
    const double* __restrict__ tau_w,      // (n_f,) input when law_id < 0
    double*       __restrict__ G_out,      // (n_f, 27) out
    double*       __restrict__ tau_out,    // (n_f,) tau_w actually used
    unsigned char* __restrict__ fb_out,    // (n_f,) 1 = fell back to Eq.10
    const int n_f, const int mode, const double h,
    const double h_law, const double nu, const double y_plus_min,
    const int law_id, const int law_iters,
    const int fric_dir, const int fb_mode, const double sample_h,
    double*       __restrict__ u_wm,       // (n_f, 3) Yang Eq.(2) state
    double*       __restrict__ utau_prev,  // (n_f,) previous step, wm_mode 1
    const int wm_mode, const double wm_tf, const int wm_seed,
    const int Nx, const int Ny, const int Nz,
    const int px, const int py, const int pz,
    double*       __restrict__ rho_out)    // (n_f,) rho^a of the facet
                                           // state sample (p_state =
                                           // rho^a theta, patch 70)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_f) return;
    const double nx = nrm[3*a], ny = nrm[3*a+1], nz = nrm[3*a+2];
    const double A = area[a];
    const long long base = (long long)a * 27;

    double go[27];
    double cn[27], V[27];
    for (int i = 0; i < 27; ++i) {
        cn[i] = CX[i]*nx + CY[i]*ny + CZ[i]*nz;
        V[i]  = fabs(cn[i]) * A;
        go[i] = 0.0;
    }

    if (mode == 0) {                                   // Eq. (10) no-slip
        for (int i = 1; i < 27; ++i)
            if (cn[i] > 0.0) go[i] = G_in[base + OPP[i]];
        for (int i = 0; i < 27; ++i) G_out[base + i] = go[i];
        if (tau_out) tau_out[a] = 0.0;
        if (fb_out)  fb_out[a] = 1;
        if (rho_out) rho_out[a] = 0.0;     // no sample in no-slip (host: NaN)
        return;
    }

    double st[5];
    sample_state(rho, uf, live, cen, nrm, a, h, Nx, Ny, Nz, px, py, pz, st);
    double ur = st[0], ux = st[1], uy = st[2], uz = st[3];
    if (rho_out) rho_out[a] = ur;
    const double un = ux*nx + uy*ny + uz*nz;           // Eq. (12): u_n = 0
    ux -= un*nx; uy -= un*ny; uz -= un*nz;

    double kf = 0.0, tw = 0.0;
    int fallback = 0;
    // velocity Eq. (25) builds the friction from; u^a in Chen's structure,
    // the log-layer tangent when fric_dir = 1 (DEVIATION, patch 10)
    double fx = ux, fy = uy, fz = uz;
    if (mode == 2) {
        const double umag = sqrt(ux*ux + uy*uy + uz*uz);
        double fmag = umag;
        if (law_id >= 0) {
            // LOG-LAYER sample: a SECOND point at h_law, distinct from the
            // facet-state sample at h.
            double sl[5];
            sample_state(rho, uf, live, cen, nrm, a, h_law,
                         Nx, Ny, Nz, px, py, pz, sl);
            double lx = sl[1], ly = sl[2], lz = sl[3];
            const double lnn = lx*nx + ly*ny + lz*nz;
            lx -= lnn*nx; ly -= lnn*ny; lz -= lnn*nz;
            // Yang, Park & Moin PRF 2, 104601 (2017) Eq. (2): exponential
            // time filter on the wall-model INPUT. Mirrors the host
            // SurfelFacets._filter_input -- keep the two in step.
            // *** eps comes from the CURRENT UNFILTERED sample, never from the
            // filter state: the self-referential version deadlocks at rest
            // (u_tau = 0 -> eps = 0 -> state frozen -> u_tau = 0). Measured,
            // gate s11 [D]. Host _filter_input carries the same note.
            if (wm_mode != 0) {
                double* s = &u_wm[3*a];
                if (wm_seed) { s[0] = lx; s[1] = ly; s[2] = lz; }
                else {
                    const double vt = sqrt(lx*lx + ly*ly + lz*lz);
                    double e;
                    if (wm_mode == 1) {                  // T_f = h/(kappa ut)
                        const double us = solve_utau_d(vt, h_law, nu,
                                                       law_id, law_iters);
                        utau_prev[a] = us;
                        e = 0.41*us/h_law;
                    } else if (wm_mode == 2) {           // T_f = dx/|u|, dx=1
                        e = vt;
                    } else {                             // T_f given
                        e = 1.0/wm_tf;
                    }
                    e = fmin(fmax(e, 0.0), 1.0);
                    s[0] += e*(lx - s[0]);
                    s[1] += e*(ly - s[1]);
                    s[2] += e*(lz - s[2]);
                }
                lx = s[0]; ly = s[1]; lz = s[2];
            }
            const double lmag = sqrt(lx*lx + ly*ly + lz*lz);
            double Ut;
            if (fric_dir == 1) {
                Ut = lmag;               // own tangent speed: never < 0
                fx = lx; fy = ly; fz = lz; fmag = lmag;
            } else {
                // magnitude = PROJECTION on the facet-state direction, not
                // |u_t| (wall_model patch 18 sec. 3)
                Ut = (umag > 1e-14) ? (lx*ux + ly*uy + lz*uz)/umag : 0.0;
            }
            const double ut = solve_utau_d(fabs(Ut), h_law, nu,
                                           law_id, law_iters);
            const double yp = h_law*ut/nu;
            const int clean = (sl[4] > 0.5);
            if (!(clean && yp > y_plus_min && Ut > 0.0)) fallback = 1;
            if (!fallback)          tw = sl[0]*ut*ut;
            else if (fb_mode == 1) {                 // viscous sublayer
                if (clean) {
                    tw = sl[0]*nu*fabs(Ut)/h_law;
                } else {                             // stencil unusable ->
                    tw = ur*nu*umag/fmax(sample_h, 1e-300);   // facet state
                    fx = ux; fy = uy; fz = uz; fmag = umag;
                }
            } else tw = 0.0;                         // freeslip / noslip
        } else {
            tw = tau_w[a];
        }
        const double pa = ur * THETA;
        kf = (fmag > 1e-14) ? (tw / fmax(pa*fmag, 1e-300)) : 0.0;
    }
    if (tau_out) tau_out[a] = tw;
    if (fb_out)  fb_out[a] = (unsigned char)fallback;

    for (int i = 0; i < 27; ++i) {
        if (!(cn[i] > 0.0)) continue;
        const int ib = OPP[i];
        const double fi = feq_i(i,  ur, ux, uy, uz);
        const double fb = feq_i(ib, ur, ux, uy, uz);
        double v = V[i]*(fi + fb) - G_in[base + ib];   // Eq. (16)
        if (mode == 2) {                                       // Eq. (25)
            // feq_i - feq_i* is exactly 2 rho w_i (c_i.u)/theta, so this is
            // exactly linear in the friction velocity: its magnitude
            // cancels against kf and only the direction survives.
            const double gi = (fric_dir == 1 || fb_mode == 1)
                ? feq_i(i,  ur, fx, fy, fz) : fi;
            const double gb = (fric_dir == 1 || fb_mode == 1)
                ? feq_i(ib, ur, fx, fy, fz) : fb;
            v += -kf * V[i] * cn[i] * (gi - gb);
        }
        go[i] = v;
    }

    for (int j = 1; j <= 3; ++j) {                     // Eq. (21)-(22)
        double num = 0.0, den = 0.0;
        for (int i = 0; i < 27; ++i) {
            if (CSQ[i] != j) continue;
            if (cn[i] < 0.0) { num += G_in[base + i]; den += V[i]; }
            else if (cn[i] > 0.0) { num -= go[i]; }
        }
        const double df = num / fmax(den, 1e-300);
        for (int i = 0; i < 27; ++i)
            if (CSQ[i] == j && cn[i] > 0.0) go[i] += V[i] * df;
    }
    if (fallback && fb_mode == 0) {   // -> Eq. (10); no mass fix, no dp
        for (int i = 0; i < 27; ++i) go[i] = 0.0;
        for (int i = 1; i < 27; ++i)
            if (cn[i] > 0.0) go[i] = G_in[base + OPP[i]];
    }
    for (int i = 0; i < 27; ++i) G_out[base + i] = go[i];
}

extern "C" __global__ void surfel_distribute(
    const double* __restrict__ G_out,      // (n_f, 27)
    const double* __restrict__ Vsum,       // (n_f, 27)
    const long long* __restrict__ indptr,
    const int*    __restrict__ cell,
    const double* __restrict__ wgt,
    const double* __restrict__ nrm,
    double*       __restrict__ Q,          // (27, N) out (pre-zeroed)
    const int n_f, const long long N)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_f * N_PAIR) return;
    const int a = t / N_PAIR;
    const int p = t - a * N_PAIR;

    const int ip = PIDX[2*p], im = PIDX[2*p+1];
    const double nx = nrm[3*a], ny = nrm[3*a+1], nz = nrm[3*a+2];
    const double cn = CX[ip]*nx + CY[ip]*ny + CZ[ip]*nz;
    if (cn == 0.0) return;
    const int dout = (cn > 0.0) ? ip : im;

    const double vs = Vsum[(long long)a*27 + dout];
    if (!(vs > 0.0)) return;
    const double frac = G_out[(long long)a*27 + dout] / vs;

    const long long b = indptr[t], e = indptr[t+1];
    for (long long k = b; k < e; ++k)
        atomicAdd(&Q[(long long)dout * N + cell[k]], wgt[k] * frac);
}

extern "C" __global__ void surfel_advect(
    const float*  __restrict__ n_post,     // (27, N)
    const double* __restrict__ g_field,    // (27, n_sup) COMPACT
    const double* __restrict__ Q,          // (27, n_sup) COMPACT
    const int*    __restrict__ qmap,       // (N,) dense -> support, -1 off
    const long long n_sup,
    const double* __restrict__ dV,         // (N,)
    const unsigned char* __restrict__ live,// (N,)
    float*        __restrict__ n_new,      // (27, N) out
    const int Nx, const int Ny, const int Nz)
{
    // g/Q live on the facet-support band only (a few % of the box); the
    // dense (27, N) f64 pair was 7.6 GiB of the span16 L3 advance peak
    // (patch 64 sec. 18). Off-band both are exactly 0.0 by construction
    // (zeros-init + on-band writes), so the -1 branch is bit-identical.
    const long long N = (long long)Nx * Ny * Nz;
    const long long y = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= N) return;
    const int kz = (int)(y % Nz);
    const int jy = (int)((y / Nz) % Ny);
    const int ix = (int)(y / ((long long)Nz * Ny));
    const int lv = (int)live[y];
    const double idv = lv ? (1.0 / dV[y]) : 0.0;
    const int my = qmap[y];

    for (int i = 0; i < 27; ++i) {
        int sx = ix - CX[i]; sx += (sx < 0) ? Nx : ((sx >= Nx) ? -Nx : 0);
        int sy = jy - CY[i]; sy += (sy < 0) ? Ny : ((sy >= Ny) ? -Ny : 0);
        int sz = kz - CZ[i]; sz += (sz < 0) ? Nz : ((sz >= Nz) ? -Nz : 0);
        const long long s = ((long long)sx * Ny + sy) * Nz + sz;
        const long long off = (long long)i * N;
        const int ms = qmap[s];
        const double gf = (ms >= 0)
            ? g_field[(long long)i * n_sup + ms] : 0.0;
        const double qq = (lv && my >= 0)
            ? Q[(long long)i * n_sup + my] : 0.0;
        const double src = (dV[s] - gf) * (double)n_post[off + s];
        n_new[off + y] = (float)(lv ? (src + qq) * idv : 0.0);
    }
}
"""


class SurfelKernelD3Q27:
    """Device-side facet passes; mirrors SurfelFacets (gate s3k anchor)."""

    _MODE = {"noslip": 0, "freeslip": 1, "wallmodel": 2}
    _FRIC = {"state": 0, "log": 1}
    _FB = {"noslip": 0, "viscous": 1, "freeslip": 2}

    def __init__(self, facets, block: int = 128) -> None:
        import cupy as cp
        src = _source()
        assert all(ord(c) < 128 for c in src)      # nvrtc POSIX locale rule
        opts = ('--fmad=false',)
        self._k = {n: cp.RawKernel(src, n, options=opts) for n in
                   ('surfel_gather', 'surfel_scatter', 'surfel_distribute',
                    'surfel_advect')}
        self.block = int(block)
        self.f = facets
        self.n_f = facets.n_f
        self.shape = facets.shape
        self.N = int(np.prod(facets.shape))
        self.mode = self._MODE[facets.mode]

        # rebuild the CSR in (facet, pair) key order -- the host class keeps
        # it re-sorted by pair for its per-pair numpy loops
        self.indptr = cp.asarray(self._rebuild_indptr(facets))
        self.cell = cp.asarray(self._csr_cell.astype(np.int32))
        self.wgt = cp.asarray(self._csr_w)
        self.nrm = cp.asarray(np.ascontiguousarray(facets.normal))
        self.area = cp.asarray(np.ascontiguousarray(facets.area))
        self.cen = cp.asarray(np.ascontiguousarray(facets.centroid))
        self.Vsum = cp.asarray(np.ascontiguousarray(facets.Vsum))
        # g/Q support compaction (64 sec. 18): both fields live ONLY on
        # the transport-table cells — g is pair_cell_sums over the SAME
        # tables (structurally zero elsewhere), Q's atomics target the
        # same CSR. Dense (27, N) f64 was ~93% zeros at span16 scale.
        sup = np.unique(self._csr_cell).astype(np.int64)
        self.sup = sup                       # host: slab builds slice it
        self.n_sup = int(sup.size)
        qmap = np.full(self.N, -1, dtype=np.int32)
        qmap[sup] = np.arange(self.n_sup, dtype=np.int32)
        self.qmap = cp.asarray(qmap)
        self.cellc = cp.asarray(qmap[self._csr_cell])    # compact CSR
        self.g_field = cp.asarray(np.ascontiguousarray(
            facets.g_field.reshape(27, -1)[:, sup]))     # (27, n_sup)
        self.G_in = cp.zeros((self.n_f, 27), dtype=cp.float64)
        self.G_out = cp.zeros((self.n_f, 27), dtype=cp.float64)
        # Q is PER-STEP scratch on the support band (216 B/support-cell
        # since 64 sec. 18) — allocated on first apply, not at build: the
        # MPI replicated build holds every level's surfel state at once
        # (64 sec. 13). Allocation timing only — apply always precedes
        # advect (gate s13 pins the chain).
        self.Q = None
        self.tau_out = cp.zeros(self.n_f, dtype=cp.float64)
        self.fb_out = cp.zeros(self.n_f, dtype=cp.uint8)
        # rho^a of the facet-state sample (patch 70): the surface
        # writer's p_state = rho^a theta — the pressure Eq. (20)
        # intends, free of the Eq. (24) dp the normal traction carries.
        self.rho_out = cp.zeros(self.n_f, dtype=cp.float64)
        self._per = tuple(int(p) for p in facets.periodic)
        # Physics scalars are REQUIRED facet attributes (surfel_boundary
        # always sets them): a getattr default here (nu=1/6 ~ tau=1!) would
        # silently poison the wall model if the adapter drifted.
        law = getattr(facets, 'law', None)
        self.law_id = -1 if law is None else int(law.kernel_id)
        self.law_iters = int(getattr(law, 'kernel_iters', 0) or 0)
        self.h_law = float(facets.h_law)
        self.nu = float(facets.nu)
        self.y_plus_min = float(facets.y_plus_min)
        self.fric_dir = self._FRIC[facets.friction_dir]
        self.fb_mode = self._FB[facets.fallback]
        # Yang Eq. (2) wall-model input filter (patch_notes/surfel/13).
        # wm_mode 0 = off (default) -> the kernel skips the block entirely,
        # so an unfiltered run stays bit-identical to before the knob.
        wmf = getattr(facets, 'wm_filter', None)
        if wmf is None:
            self.wm_mode, self.wm_tf = 0, 0.0
        elif wmf == 'ti':
            self.wm_mode, self.wm_tf = 1, 0.0
        elif wmf == 'dtc':
            self.wm_mode, self.wm_tf = 2, 0.0
        else:
            self.wm_mode, self.wm_tf = 3, float(wmf)
        self.u_wm = cp.zeros((self.n_f, 3), dtype=cp.float64)
        self.utau_prev = cp.zeros(self.n_f, dtype=cp.float64)
        #: seed the filter state from the first sample, not from zero
        self._wm_seed = 1

    def _rebuild_indptr(self, facets):
        """CSR keyed by facet*N_PAIR + pair (the kernel's thread index)."""
        n_key = facets.n_f * N_PAIR
        pair_of_row = np.zeros(facets._t_fac.size, dtype=np.int64)
        for p in range(N_PAIR):
            b, e = facets._t_beg[p], facets._t_end[p]
            pair_of_row[b:e] = p
        key = facets._t_fac * N_PAIR + pair_of_row
        order = np.argsort(key, kind='stable')
        self._csr_cell = facets._t_cell[order]
        self._csr_w = facets._t_w[order]
        counts = np.bincount(key[order], minlength=n_key)
        return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    # ------------------------------------------------------------------
    def apply(self, n_post, rho, u, live, tau_w=None):
        """Run gather -> scatter -> distribute. Returns (Q, force_on_body)."""
        import cupy as cp
        gp = ((self.n_f * N_PAIR + self.block - 1) // self.block,)
        gf = ((self.n_f + self.block - 1) // self.block,)
        blk = (self.block,)
        nx, ny, nz = self.shape

        self.G_in.fill(0)
        self._k['surfel_gather'](gp, blk, (
            n_post, self.indptr, self.cell, self.wgt, self.nrm, self.G_in,
            cp.int32(self.n_f), cp.int64(self.N)))

        if tau_w is not None:
            tw = tau_w
        else:
            tw = getattr(self, '_d_tw0', None)
            if tw is None:
                tw = cp.zeros(1, dtype=cp.float64)
                self._d_tw0 = tw
        self._k['surfel_scatter'](gf, blk, (
            self.G_in, self.nrm, self.area, self.cen, rho, u, live, tw,
            self.G_out, self.tau_out, self.fb_out,
            cp.int32(self.n_f), cp.int32(self.mode),
            cp.float64(self.f.sample_h),
            cp.float64(self.h_law), cp.float64(self.nu),
            cp.float64(self.y_plus_min),
            cp.int32(self.law_id), cp.int32(self.law_iters),
            cp.int32(self.fric_dir), cp.int32(self.fb_mode),
            cp.float64(self.f.sample_h),
            self.u_wm, self.utau_prev,
            cp.int32(self.wm_mode), cp.float64(self.wm_tf),
            cp.int32(self._wm_seed),
            cp.int32(nx), cp.int32(ny), cp.int32(nz),
            cp.int32(self._per[0]), cp.int32(self._per[1]),
            cp.int32(self._per[2]), self.rho_out))

        self._wm_seed = 0          # only the first pass seeds the filter

        if self.Q is None:
            import os
            if (os.environ.get('LBM_MEM_CENSUS') == '2'
                    and not getattr(self, '_census_done', False)):
                # measurement instrument (64 sec. 17): dump every live
                # device array at the exact span16 death instant (pre-Q
                # alloc) — enclosing-frame locals included via gc.
                self._census_done = True
                _census_dump(f"pre-Q N={self.N} n_f={self.n_f}")
            self.Q = cp.zeros((27, self.n_sup), dtype=cp.float64)
        else:
            self.Q.fill(0)
        # distribute writes Q through COMPACT indices (64 sec. 18): the
        # kernel is unchanged — cellc in the cell slot, n_sup as N.
        self._k['surfel_distribute'](gp, blk, (
            self.G_out, self.Vsum, self.indptr, self.cellc, self.wgt,
            self.nrm, self.Q, cp.int32(self.n_f), cp.int64(self.n_sup)))

        # device-cached constants (64 sec. 19b): cp.asarray of the HOST
        # cdotn here was a synchronous (n_f, 27) f64 H2D upload on EVERY
        # apply — ~650 MB/coarse step at span16, each blocking the
        # stream. Lazy getattr: slab clones are built via __new__.
        cdotn = getattr(self, '_d_cdotn', None)
        if cdotn is None:
            cdotn = cp.asarray(self.f.cdotn)
            self._d_cdotn = cdotn
        C27T = getattr(self, '_d_C27T', None)
        if C27T is None:
            C27T = cp.asarray(C27.astype(np.float64)).T
            self._d_C27T = C27T
        force = (C27T
                 @ (self.G_in * (cdotn < 0) - self.G_out * (cdotn > 0)).sum(
                     axis=0))
        return self.Q, force

    def advect(self, n_post, dV, live, n_new):
        import cupy as cp
        nx, ny, nz = self.shape
        grid = ((self.N + self.block - 1) // self.block,)
        self._k['surfel_advect'](grid, (self.block,), (
            n_post, self.g_field, self.Q, self.qmap,
            cp.int64(self.n_sup), dV, live, n_new,
            cp.int32(nx), cp.int32(ny), cp.int32(nz)))
