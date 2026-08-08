"""Wall-Function Bounce (WFB) — std-path sparse pass driver (W2).

Owns the per-link arrays and knobs of the fused CUDA pass in
src/kernels/wfb_d3q27.py and its bit-mirroring numpy reference. Built
from the two validated sources whose link ORDER is proven identical
(W0 gate [O]):

    InterpolatedBounceBack   sparse triple (link_cell, link_dir, link_q)
                             — the bounce actually applied each step
    compute_link_wall_geometry / canonicalize_span_through dict
                             — per-link wall normal, sample point, s_ok

Scheme (02 + 03, and the choices fixed here):

  * Correction = moving-wall BB term on the already-bounced slot:
        f[opp(d), x_f] -= coef * rho(x_f) * (c_d . u_w),
    u_w = (U_t - u_tau^2 y_s / nu_eff) t_hat  (02's virtual-wall form;
    nu_eff is a scalar knob, molecular by default — W3 decides).
  * coef ("bfl" mode, default): 6 w_d for q < 1/2 and 6 w_d / (2q) for
    q >= 1/2 — the Bouzidi moving-wall interpolation weight of the
    bounced part, degenerating to the W1b-validated 6 w_d at q = 1/2.
    DECIDED (W3a gate, patch 05): the flat q=0.8 and inclined-torus
    laminar slip benchmarks confirm bfl (= scheme (c)) — analytic slip
    within 0.4~0.9%, while the (b)-flavoured "uniform" 6 w_d misses by
    60~290% off the halfway point. "uniform" is kept as a diagnostic
    knob only.
    q is taken from ibb.link_q — the interpolation the bounce kernel
    ACTUALLY used — not from the sibling geometry q (bit-equal for a
    shared STL source anyway, W0 gate [Q]).
  * rho in the correction = the link's fluid-cell density (weakly
    compressible; ~1).
  * Per-link gate: link_enable (build-time hit AND s_ok; runtime
    transition-sensor hook — clear bytes to force fallback) AND
    U_t > 1e-14 AND y+ > y_plus_min. A gated-off link is NOT written:
    the existing HWBB/IBB result stays bit-identical (03 fallback
    contract), only its real exchange joins the force sum.
  * last_force() = total body MEM force of the pass, F = sum_links
    c_d (f*_d + f_returned), f64 atomics with an MPI owned clip — the
    only formula that stays faithful once slots are rewritten (W1b [C]).

host_reference() mirrors the kernel expression-for-expression in f64
(kernel is compiled --fmad=false): bit-exact for the laminar law, libm
ulps for Musker. It is the W2 gate's comparison anchor, not a compute
path.

Author: LBM Development Team
Date: 2026-08 (wall_model track W2)
"""

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np

from src.boundary.wall_function import WallLaw

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

# Std D3Q27 tables (== src.lattice and the CUDA kernel; asserted by gate)
_CX = np.array([0, 1, -1, 0, 0, 0, 0,
                1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0,
                1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int64)
_CY = np.array([0, 0, 0, 1, -1, 0, 0,
                1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1,
                1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int64)
_CZ = np.array([0, 0, 0, 0, 0, 1, -1,
                0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1,
                1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int64)
_OPP = np.array([0, 2, 1, 4, 3, 6, 5,
                 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
                 26, 25, 24, 23, 22, 21, 20, 19], dtype=np.int64)
_W = np.array([8.0 / 27.0] + [2.0 / 27.0] * 6 + [1.0 / 54.0] * 12
              + [1.0 / 216.0] * 8, dtype=np.float64)

_TINY_UT = 1e-14


def _to_np(a) -> np.ndarray:
    return a.get() if hasattr(a, 'get') else np.asarray(a)


class WallFunctionBounce:
    """Fused WFB sparse pass over one level's boundary links (std path)."""

    def __init__(
        self,
        xp: 'ModuleType',
        ibb,
        geom: Dict[str, np.ndarray],
        law: WallLaw,
        *,
        h_cells: float,
        nu: float,
        nu_eff: Optional[float] = None,
        nu_eff_mode: str = "scalar",
        uw_rel_clamp: float = np.inf,
        uf_eps: float = 1.0,
        uw_mode: str = "gradient",
        uw_gain: float = 0.05,
        uw_relax: float = 0.05,
        s_min_frac: float = 0.05,
        y_plus_min: float = 30.0,
        coef_mode: str = "bfl",
        periodic_axes: Tuple[int, ...] = (),
        clip_bounds: Optional[Sequence[Tuple[int, int]]] = None,
        block_size: int = 256,
    ) -> None:
        """
        Args:
            xp:        cupy for the kernel path (numpy allowed for a
                       host_reference-only instance).
            ibb:       InterpolatedBounceBack with the sparse link triple.
            geom:      dict from compute_link_wall_geometry (optionally
                       canonicalize_span_through) in the SAME link order.
            law:       wall_function.WALL_LAWS instance (Musker / laminar).
            h_cells:   wall->sample distance y_s used to build geom
                       [this level's lu].
            nu:        molecular viscosity of this level [lu^2/lt].
            nu_eff:    mode-"scalar" u_w viscosity (default: nu — the
                       W1b-exact laminar limit).
            nu_eff_mode: u_w relation viscosity (W3b candidates, the
                       PLAN W3 "nu_eff definition" discriminator):
                       "scalar" | "sgs" (nu + nu_t(x_s), needs the nu_t
                       field in apply()) | "mixing" (nu + kappa u_tau
                       y_s, kappa = 0.41, field-free equilibrium) |
                       "camwa" (mixing + van Driest damping, A+ = 26 —
                       CAMWA 2024 Eq.25; pair with the Eq.39 near-wall
                       reconstruction, src/turbulence/near_wall_sgs.py).
            uw_rel_clamp: |u_w| <= uw_rel_clamp * U_t safety clamp
                       (default inf = off, bit-neutral).
            uf_eps:    per-link exponential input filter on the sampled
                       (u, nu_t): eps = 1/T_steps; 1.0 = off
                       (bit-neutral). Kawai-Larsson-style noise control.
            uw_mode:   "gradient" (default, the nu_eff relation) or
                       "target" (W3c stage 2: per-link control driving
                       the link's own cell onto the law profile
                       u~(y1, u_tau) — the Malaspinas-Sagaut condition
                       imposed through the bounce; no nu_eff).
            uw_gain:   control gain of uw_mode="target" [per step].
            uw_relax:  under-relaxation of uw_mode="flux" (1 = deadbeat,
                       which is unstable on the instantaneous turbulent
                       exchange; the fixed point is relax-independent).
            y_plus_min: gate threshold (~30); 0 disables the y+ gate.
            coef_mode: "bfl" (q-interpolated, default) | "uniform".
            periodic_axes: axes on which the sample stencil wraps
                       (e.g. span_through z of the infinite wing).
            clip_bounds: force-ownership clip [(x0,x1),(y0,y1),(z0,z1)]
                       (MPI ranks; default whole domain).
        """
        self.xp = xp
        self.law = law
        self.domain_shape = tuple(int(s) for s in ibb.solid_mask.shape)
        if len(self.domain_shape) != 3:
            raise ValueError("WallFunctionBounce is 3D-only (W2).")
        nx, ny, nz = self.domain_shape
        self.N = nx * ny * nz

        cell = _to_np(ibb.link_cell).astype(np.int64)
        dirs = _to_np(ibb.link_dir).astype(np.int64)
        link_q = _to_np(ibb.link_q).astype(np.float64)
        self.n_links = int(cell.size)

        # Link-order contract: geometry rows must be the bounce links.
        if (geom['link_cell'].shape[0] != self.n_links
                or not np.array_equal(geom['link_cell'], cell)
                or not np.array_equal(
                    geom['link_dir'].astype(np.int64), dirs)):
            raise ValueError(
                "geom link order != ibb sparse link order (both must come "
                "from the same needs_bounce; see W0 gate [O]).")

        self.y_s = float(h_cells)
        self.nu = float(nu)
        self.nu_eff = float(nu) if nu_eff is None else float(nu_eff)
        if nu_eff_mode not in ("scalar", "sgs", "mixing", "camwa"):
            raise ValueError(
                f"nu_eff_mode must be scalar|sgs|mixing|camwa: "
                f"{nu_eff_mode}")
        self.nu_eff_mode = nu_eff_mode
        self.uw_rel_clamp = float(uw_rel_clamp)
        if not 0.0 < uf_eps <= 1.0:
            raise ValueError(f"uf_eps must be in (0, 1]: {uf_eps}")
        self.uf_eps = float(uf_eps)
        if uw_mode not in ("gradient", "target", "flux"):
            raise ValueError(
                f"uw_mode must be gradient|target|flux: {uw_mode}")
        self.uw_mode = uw_mode
        self.uw_gain = float(uw_gain)
        if not 0.0 < uw_relax <= 1.0:
            raise ValueError(f"uw_relax must be in (0, 1]: {uw_relax}")
        self.uw_relax = float(uw_relax)
        # per-cell tangential-coupling floor, as a fraction of the
        # flat-wall analytic value rho c_s^2 = rho/3 (rho ~ 1 in the
        # weakly compressible limit). Cells below it fall back to
        # no-slip: on a sphere that is 27% of cells but only ~1% of
        # the drag capacity (patch 18).
        self.s_min_frac = float(s_min_frac)
        self.s_min = float(s_min_frac) / 3.0
        self.y_plus_min = float(y_plus_min)
        if coef_mode not in ("bfl", "uniform"):
            raise ValueError(f"coef_mode must be bfl|uniform: {coef_mode}")
        self.coef_mode = coef_mode
        self.periodic = tuple(int(a) in periodic_axes for a in range(3))
        if clip_bounds is None:
            clip_bounds = [(0, s) for s in self.domain_shape]
        self.clip_bounds = [(int(a), int(b)) for a, b in clip_bounds]

        w_d = _W[dirs]
        if coef_mode == "bfl":
            coef = np.where(link_q < 0.5, 6.0 * w_d,
                            6.0 * w_d / (2.0 * link_q))
        else:
            coef = 6.0 * w_d

        # Host copies (host_reference + build introspection)
        self._cell_np = cell
        self._dir_np = dirs
        self._coef_np = coef.astype(np.float32)
        self._nrm_np = np.ascontiguousarray(
            geom['normal'], dtype=np.float32)
        self._xs_np = np.ascontiguousarray(
            geom['sample'], dtype=np.float32)
        enable = (np.asarray(geom['hit'], dtype=bool)
                  & np.asarray(geom['s_ok'], dtype=bool))
        self._enable_np = enable.astype(np.uint8)
        self.slot_addr = _OPP[dirs] * self.N + cell   # f slot per link

        # Device arrays (kernel path)
        self.wl_cell = xp.ascontiguousarray(
            xp.asarray(cell.astype(np.int32)))
        self.wl_dir = xp.ascontiguousarray(
            xp.asarray(dirs.astype(np.int8)))
        self.wl_coef = xp.asarray(self._coef_np)
        self.wl_nrm = xp.asarray(self._nrm_np)
        self.wl_xs = xp.asarray(self._xs_np)
        #: Runtime gate byte per link (transition-sensor hook): clear to
        #: force the no-write fallback on a link.
        self.link_enable = xp.asarray(self._enable_np)
        #: per-link exponential-filter state, (n, 4) = (ux, uy, uz, nu_t)
        self.wl_uf = xp.zeros((self.n_links, 4), dtype=xp.float32)
        #: per-link u_tau of the last apply() (0 on ungated links) —
        #: diagnostic + driver for the near-wall SGS reconstruction
        self.wl_utau = xp.zeros(self.n_links, dtype=xp.float32)
        # wall-normal distance from the wall point to the link's OWN
        # fluid cell centre (uw_mode="target" evaluates the law there)
        nrm64 = np.asarray(geom['normal'], dtype=np.float64)
        ctr = np.stack([cell // (ny * nz), (cell // nz) % ny, cell % nz],
                       axis=1).astype(np.float64)
        y1 = np.einsum('ij,ij->i', ctr - np.asarray(geom['wall']), nrm64)
        self._y1_np = np.maximum(y1, 1e-6).astype(np.float32
                                                   ).astype(np.float64)
        self.wl_y1 = xp.asarray(self._y1_np.astype(np.float32))
        #: per-link u_w control state ("target") / applied u_w ("flux")
        self.wl_uw = xp.zeros(self.n_links, dtype=xp.float32)

        # ── wall-cell grouping (uw_mode="flux"): u_w is a PER-CELL
        # unknown, so the links of one fluid cell must solve it jointly.
        # Solving it per link is ill-posed — the links share one
        # observable and drift in its null space (patch 12 §4).
        uniq, grp = np.unique(cell, return_inverse=True)
        self.n_groups = int(uniq.size)
        self.wl_grp = xp.asarray(grp.astype(np.int32))
        # CELL-mean wall normal: the reduction needs ONE tangent per
        # cell (per-link tangents differ by up to 177 deg where the
        # tangential projection vanishes -> corrupts the cell sum).
        acc = np.zeros((self.n_groups, 3))
        for a in range(3):
            acc[:, a] = np.bincount(grp, nrm64[:, a], self.n_groups)
        acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True),
                          1e-300)
        self.wl_ncell = xp.asarray(
            np.ascontiguousarray(acc[grp], dtype=np.float32))
        self._grp_np = grp.astype(np.int64)
        self.wl_that = xp.zeros((self.n_links, 3), dtype=xp.float32)
        self.wl_gate = xp.zeros(self.n_links, dtype=xp.uint8)
        self._grp_m0 = xp.zeros(self.n_groups, dtype=xp.float64)
        self._grp_S = xp.zeros(self.n_groups, dtype=xp.float64)
        self._grp_T = xp.zeros(self.n_groups, dtype=xp.float64)
        self._grp_U = xp.zeros(self.n_groups, dtype=xp.float64)
        self._grp_V = xp.zeros(self.n_groups, dtype=xp.float64)
        self._force = xp.zeros(3, dtype=xp.float64)
        # valid-but-unread pointer for the kernel when mode != "sgs"
        self._nut_dummy = xp.zeros(1, dtype=xp.float32) \
            if xp.__name__ == 'cupy' else None
        self._kernel = None
        self._flux_primed = False
        self._block_size = block_size

    _NU_EFF_MODE_ID = {"scalar": 0, "sgs": 1, "mixing": 2, "camwa": 3}
    _UW_MODE_ID = {"gradient": 0, "target": 1, "flux": 2}

    # ------------------------------------------------------------------
    @property
    def n_enabled(self) -> int:
        return int(self._enable_np.sum())

    def apply(
        self,
        f: 'npt.NDArray',
        f_post: 'npt.NDArray',
        u: 'npt.NDArray',
        rho: 'npt.NDArray',
        nu_t: Optional['npt.NDArray'] = None,
    ) -> None:
        """Run the fused pass on device (call right after the obstacle BC).

        Args:
            f:      post-streaming/BC/bounce distribution (27, *shape),
                    float32 — bounced slots corrected in place.
            f_post: post-collision distribution (27, *shape) float32.
            u:      macro velocity (3, *shape) (cast to float32 if needed).
            rho:    macro density (*shape,)   (cast to float32 if needed).
            nu_t:   SGS eddy-viscosity field (*shape,) float32 — required
                    iff nu_eff_mode == "sgs" (Simulation passes sim.nu_t).
        """
        xp = self.xp
        if xp.__name__ != 'cupy':
            raise RuntimeError(
                "WallFunctionBounce.apply is the CUDA path; use "
                "host_reference() on numpy.")
        if self.n_links == 0:
            return
        if self.nu_eff_mode == "sgs":
            if nu_t is None:
                raise ValueError(
                    "nu_eff_mode='sgs' needs the nu_t field (enable an "
                    "SGS model so Simulation provides sim.nu_t).")
            nut32 = nu_t if nu_t.dtype == xp.float32 \
                else nu_t.astype(xp.float32)
            nut32 = xp.ascontiguousarray(nut32)
        else:
            nut32 = self._nut_dummy
        if self._kernel is None:
            from src.kernels.wfb_d3q27 import WFBKernelD3Q27
            self._kernel = WFBKernelD3Q27(self._block_size)
        u32 = u if u.dtype == xp.float32 else u.astype(xp.float32)
        rho32 = rho if rho.dtype == xp.float32 else rho.astype(xp.float32)
        rho32 = xp.ascontiguousarray(rho32)
        self._force.fill(0)
        if self.uw_mode == "flux":
            self._grp_m0.fill(0)
            self._grp_S.fill(0)
            self._grp_T.fill(0)
            self._grp_U.fill(0)
            self._grp_V.fill(0)
        self._kernel.apply(
            f, f_post, xp.ascontiguousarray(u32),
            xp.ascontiguousarray(rho32), nut32,
            self.wl_cell, self.wl_dir, self.wl_coef, self.wl_nrm,
            self.wl_ncell, self.wl_xs, self.wl_uf, self.wl_utau, self.wl_y1, self.wl_uw,
            self.wl_that, self.wl_gate, self.wl_grp,
            self._grp_m0, self._grp_S, self._grp_T, self._grp_U,
            self._grp_V, self.link_enable, self._force,
            self.n_links, self.domain_shape,
            self.y_s, self.nu, self.nu_eff,
            self._NU_EFF_MODE_ID[self.nu_eff_mode], self.uw_rel_clamp,
            self.s_min,
            self.uf_eps,
            self._UW_MODE_ID[self.uw_mode], self.uw_gain,
            self.y_plus_min,
            int(self.law.kernel_id), float(self.law.kernel_p0),
            int(self.law.kernel_iters),
            self.periodic, self.clip_bounds)

        if self.uw_mode == "flux":
            import cupy as cp
            cb = self.clip_bounds
            nx, ny, nz = self.domain_shape
            self._kernel.flux_apply((
                f, f_post, rho32,
                self.wl_cell, self.wl_dir, self.wl_coef, self.wl_that,
                self.wl_gate, self.wl_grp,
                self._grp_m0, self._grp_S, self._grp_T, self._grp_U,
                self._grp_V, self.wl_uw, self._force,
                cp.int32(self.n_links),
                cp.float64(self.s_min),
                # first pass runs deadbeat so the filtered
                # reference starts AT m0/S instead of at 0
                cp.float64(1.0 if not self._flux_primed
                           else self.uw_relax), cp.float64(self.uw_rel_clamp),
                cp.int32(nx), cp.int32(ny), cp.int32(nz),
                cp.int32(cb[0][0]), cp.int32(cb[0][1]),
                cp.int32(cb[1][0]), cp.int32(cb[1][1]),
                cp.int32(cb[2][0]), cp.int32(cb[2][1])), self.n_links)
            self._flux_primed = True

    def gate_stats(self):
        """(gated fraction, mean y+ at the sample) — cheap device
        reductions for the per-step trace. wl_utau is written only on
        links the pass actually gated, so its support IS the gate."""
        xp = self.xp
        nz = int(xp.count_nonzero(self.wl_utau))
        frac = nz / self.n_links if self.n_links else 0.0
        return frac, self.y_s * self.mean_utau() / self.nu

    def target_sum(self) -> float:
        """SUM over cells of the modelled tangential momentum target
        (uw_mode='flux'); 0 otherwise. Averaged over the same window as
        the force, <F> vs <target> tests whether the scheme delivers its
        own target in the MEAN (patch 20 hypothesis iii)."""
        if self.uw_mode != "flux":
            return 0.0
        return float(self.xp.sum(self._grp_T))

    def reference_residual(self):
        """(|sigma|/T) per cell — validity of the zero-exchange
        reference. sigma = m0 - SUM S_l U_1 is what patch 14 used
        AS the reference; it must be SMALL compared with the
        target T, else u_w is set by a numerical residual rather
        than by the wall model (patch 19). Diagnostic only."""
        S = _to_np(self._grp_S); ok = S > 0
        if not ok.any():
            return np.zeros(0)
        sig = (_to_np(self._grp_m0) - _to_np(self._grp_V))[ok]
        T = _to_np(self._grp_T)[ok]
        return np.abs(sig) / np.maximum(np.abs(T), 1e-300)

    def last_force(self) -> np.ndarray:
        """Body MEM force of the most recent apply(), host f64 (3,)."""
        return _to_np(self._force)

    def mean_utau(self) -> float:
        """Mean u_tau over the links the last apply() actually gated.

        Drives the near-wall SGS reconstruction (a scalar is adequate on
        a statistically homogeneous wall; a per-cell field from the
        nearest link is the curved-body generalisation, W4)."""
        xp = self.xp
        ut = self.wl_utau
        n = int(xp.count_nonzero(ut))
        return float(xp.sum(ut) / n) if n else 0.0

    # ------------------------------------------------------------------
    def _host_sample(
        self,
        uN: np.ndarray,
        nutN: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trilinear (u, nu_t) at the sample points, f64, kernel corner
        order (a-major x, y, z) — shared by host_reference and
        prime_filter."""
        nx, ny, nz = self.domain_shape
        xs = self._xs_np.astype(np.float64)
        base = np.floor(xs)
        frac = xs - base
        idx0, idx1 = [], []
        for a, n in enumerate((nx, ny, nz)):
            b0 = base[:, a].astype(np.int64)
            if self.periodic[a]:
                i0, i1 = b0 % n, (b0 + 1) % n
            else:
                i0 = np.clip(b0, 0, n - 1)
                i1 = np.clip(b0 + 1, 0, n - 1)
            idx0.append(i0)
            idx1.append(i1)
        wa = (1.0 - frac[:, 0], frac[:, 0])
        wb = (1.0 - frac[:, 1], frac[:, 1])
        wc = (1.0 - frac[:, 2], frac[:, 2])
        ib = (idx0[0], idx1[0])
        jb = (idx0[1], idx1[1])
        kb = (idx0[2], idx1[2])
        sx = np.zeros(self.n_links)
        sy = np.zeros(self.n_links)
        sz = np.zeros(self.n_links)
        snt = np.zeros(self.n_links)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    wgt = wa[a] * wb[b] * wc[c]
                    cid = (ib[a] * ny + jb[b]) * nz + kb[c]
                    sx += wgt * uN[0, cid]
                    sy += wgt * uN[1, cid]
                    sz += wgt * uN[2, cid]
                    if nutN is not None:
                        snt += wgt * nutN[cid]
        return sx, sy, sz, snt

    def prime_filter(
        self,
        u: 'npt.NDArray',
        nu_t: Optional['npt.NDArray'] = None,
    ) -> None:
        """Initialize the input-filter state from the CURRENT fields.

        Avoids the cold-start transient of the exponential filter (state
        starts at 0 -> one filter time of spurious u_w; the w3b smoke
        measured a +1e3 gN force spike at step 0 without priming). Call
        after the initial condition is set, before advancing.
        """
        uN = _to_np(u).astype(np.float64).reshape(3, -1)
        nutN = None
        if nu_t is not None:
            nutN = _to_np(nu_t).astype(np.float64).ravel()
        sx, sy, sz, snt = self._host_sample(uN, nutN)
        st = np.stack([sx, sy, sz, snt], axis=1).astype(np.float32)
        self.wl_uf = self.xp.asarray(np.ascontiguousarray(st))

    # ------------------------------------------------------------------
    def host_reference(
        self,
        f: 'npt.NDArray',
        f_post: 'npt.NDArray',
        u: 'npt.NDArray',
        rho: 'npt.NDArray',
        nu_t: Optional['npt.NDArray'] = None,
    ) -> Dict[str, np.ndarray]:
        """Pure numpy mirror of one kernel pass (no mutation) — gate anchor.

        Returns per-link arrays:
            slot     int64 flat index into f (opp(d)*N + cell)
            val      float32 slot value AFTER the pass (written or kept)
            gated    bool   link actually corrected
            u_tau, yplus, U_t, uw  float64 diagnostics (uw = 0 if ungated)
            force    (3,) float64 real-exchange body force (ordered sum;
                     kernel atomics differ in order -> compare with tol)
        """
        nx, ny, nz = self.domain_shape
        fN = _to_np(f).reshape(27, -1)
        fpN = _to_np(f_post).reshape(27, -1)
        uN = _to_np(u).astype(np.float64).reshape(3, -1)
        rhoN = _to_np(rho).astype(np.float64).ravel()
        want_nut = self.nu_eff_mode == "sgs"
        if want_nut:
            if nu_t is None:
                raise ValueError("nu_eff_mode='sgs' needs the nu_t field")
            nutN = _to_np(nu_t).astype(np.float64).ravel()

        slot = self.slot_addr
        f_old32 = fN[_OPP[self._dir_np], self._cell_np]
        f_old = f_old32.astype(np.float64)
        enable = _to_np(self.link_enable).astype(bool)

        sx, sy, sz, snt = self._host_sample(uN, nutN if want_nut else None)

        # -- input-filter blend (reads the state, does NOT update it —
        #    host_reference is pure; the kernel owns the state advance)
        uf = _to_np(self.wl_uf).astype(np.float64)
        eps = self.uf_eps
        sx = (1.0 - eps) * uf[:, 0] + eps * sx
        sy = (1.0 - eps) * uf[:, 1] + eps * sy
        sz = (1.0 - eps) * uf[:, 2] + eps * sz
        snt = (1.0 - eps) * uf[:, 3] + eps * snt

        # -- tangential decomposition
        nrm = self._nrm_np.astype(np.float64)
        nx_, ny_, nz_ = nrm[:, 0], nrm[:, 1], nrm[:, 2]
        un = sx * nx_ + sy * ny_ + sz * nz_
        tx = sx - un * nx_
        ty = sy - un * ny_
        tz = sz - un * nz_
        Ut = np.sqrt(tx * tx + ty * ty + tz * tz)
        live = Ut > _TINY_UT

        # -- wall law + gate
        ut = self.law.utau(Ut, self.y_s, self.nu)
        yplus = self.y_s * ut / self.nu
        gated = enable & live & (yplus > self.y_plus_min)

        # -- correction (evaluated everywhere, applied where gated)
        Ut_safe = np.where(live, Ut, 1.0)
        inv = 1.0 / Ut_safe
        thx, thy, thz = tx * inv, ty * inv, tz * inv
        if self.uw_mode == "target":
            from src.boundary.wall_function import musker_uplus
            y1 = self._y1_np
            U_target = ut * musker_uplus(y1 * ut / self.nu)
            U1 = (uN[0, self._cell_np] * thx + uN[1, self._cell_np] * thy
                  + uN[2, self._cell_np] * thz)
            uw = _to_np(self.wl_uw).astype(np.float64) \
                + self.uw_gain * (U_target - U1)
        else:
            if self.nu_eff_mode == "sgs":
                nueff = self.nu + snt
            elif self.nu_eff_mode == "mixing":
                nueff = self.nu + 0.41 * ut * self.y_s
            elif self.nu_eff_mode == "camwa":
                vd = 1.0 - np.exp(-yplus / 26.0)
                nueff = self.nu + 0.4184 * ut * self.y_s * vd * vd
            else:
                nueff = self.nu_eff
            uw = Ut - ut * ut * self.y_s / nueff
        uwmax = self.uw_rel_clamp * Ut_safe   # == clamp*Ut on live links
        uw = np.minimum(np.maximum(uw, -uwmax), uwmax)
        cxd = _CX[self._dir_np].astype(np.float64)
        cyd = _CY[self._dir_np].astype(np.float64)
        czd = _CZ[self._dir_np].astype(np.float64)
        cdott = cxd * thx + cyd * thy + czd * thz
        coef = self._coef_np.astype(np.float64)
        corr = coef * rhoN[self._cell_np] * cdott * uw
        val = np.where(gated, (f_old - corr).astype(np.float32), f_old32)

        # -- real-exchange force with ownership clip
        gx = self._cell_np // (ny * nz)
        gy = (self._cell_np // nz) % ny
        gz = self._cell_np % nz
        cb = self.clip_bounds
        own = ((gx >= cb[0][0]) & (gx < cb[0][1])
               & (gy >= cb[1][0]) & (gy < cb[1][1])
               & (gz >= cb[2][0]) & (gz < cb[2][1]))
        m = (fpN[self._dir_np, self._cell_np].astype(np.float64)
             + val.astype(np.float64)) * own
        force = np.array([(m * cxd).sum(), (m * cyd).sum(),
                          (m * czd).sum()])

        return {
            'slot': slot, 'val': val, 'gated': gated, 'u_tau': ut,
            'yplus': yplus, 'U_t': Ut, 'uw': np.where(gated, uw, 0.0),
            'force': force,
        }

    # ------------------------------------------------------------------
    def get_info(self) -> str:
        lines = [
            "Wall-Function Bounce (WFB, std path):",
            f"  law={type(self.law).__name__}, coef_mode={self.coef_mode}",
            f"  links: {self.n_links:,} (enabled {self.n_enabled:,})",
            f"  y_s={self.y_s}, nu={self.nu:.3e}, "
            f"nu_eff[{self.nu_eff_mode}]={self.nu_eff:.3e}, "
            f"y+_min={self.y_plus_min}, uw_clamp={self.uw_rel_clamp}",
        ]
        return "\n".join(lines)
