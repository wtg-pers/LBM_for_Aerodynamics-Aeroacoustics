"""Surfel facet boundary algorithm — surfel track S3.

Host reference of the Chen-Teixeira-Molvig facet dynamics on top of the S1
geometry and the S2 swept-volume tables. Every expression below is one of
the paper's ([C1] = Int. J. Mod. Phys. C 9(8) (1998) 1281-1292):

    gather      Gamma_in[a,i]  = sum_x V_i^a(x) n'_i(x)      (8, with the
                                 dV of P_i^a = V_i^a/dV cancelling N' = n' dV)
    no-slip     Gamma_out[a,i] = Gamma_in[a,i*]                        (10)
    free-slip   Gamma_out[a,i] = V_i(feq_i + feq_i*) - Gamma_in[a,i*]  (16)
    mass fix    += V_i * df_j            per SPEED SHELL j          (21,22)
    friction    += -Kf V_i (c_i.n)(feq_i - feq_i*)                     (25)
    scatter     Q_i(x) = sum_a [V_i^a(x)/V_i^a] Gamma_out[a,i]          (6)
    force       F = sum_a [sum_out c Gamma_out - sum_in c Gamma_in]      (9)
                (that is the force ON THE FLUID; the body feels -F)

The facet state (rho^a, u^a) is sampled at x_a + h n and its NORMAL component
is zeroed (12), which is what makes Eq. (19) collapse to F = p n (20).

Structural notes carried from S0/S2:
  * i and i* share one prism, so the tables are indexed by direction PAIR.
  * the mass fix must be per speed shell: sum_{OUT,shell} c_i V_i is parallel
    to n, so a per-shell correction cannot disturb the tangential force.
  * no iteration anywhere -- the whole scatter is closed form.

Author: LBM Development Team
Date: 2026-08 (surfel track S3)
"""

from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np

from src.boundary.surfel_transport import (
    C27, N_PAIR, OPP27, PAIR_DIR, PAIR_OF, pair_cell_sums,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

W27 = np.array([8 / 27] + [2 / 27] * 6 + [1 / 54] * 12 + [1 / 216] * 8)
CSQ27 = (C27 * C27).sum(axis=1)
THETA = 1.0 / 3.0
_SHELLS = (1, 2, 3)


def traction_kinematics(G_in, G_out, is_in, is_out, area, normal):
    """Per-facet (force, traction, p = -n.t, tau) from the G buffers.

    The facet_traction primitive, factored so a facet SUBSET (the MPI
    slab's owned facets, patch 68) evaluates the identical arithmetic —
    same expressions in the same order, so the rank-0 assembly of owned
    rows is bitwise the single-GPU full evaluation.
    """
    # Row-independent reductions ONLY: a BLAS matmul / einsum changes
    # its blocking with the row count, so a facet subset would differ
    # from the full evaluation at the ulp (measured: MPI surface files
    # rel 5e-9 off the 1-GPU ones). Explicit per-column sums in a fixed
    # order make every row's result independent of which rows are
    # present.
    G = G_in * is_in - G_out * is_out                     # (n, 27)
    f_body = np.zeros((G.shape[0], 3))
    for d in range(3):
        acc = np.zeros(G.shape[0])
        for i in range(27):
            c = C27[i, d]
            if c != 0.0:
                acc = acc + G[:, i] * c
        f_body[:, d] = acc
    a = np.maximum(area, 1e-300)[:, None]
    trac = f_body / a
    pn = (trac[:, 0] * normal[:, 0] + trac[:, 1] * normal[:, 1]
          + trac[:, 2] * normal[:, 2])
    tau = trac - pn[:, None] * normal
    return f_body, trac, pn, tau


def equilibrium(rho, u):
    """Second-order equilibrium; satisfies Eq. (13) exactly (gate s0 [Q]).

    rho: (...,), u: (..., 3) -> (..., 27)
    """
    cu = u @ C27.T
    usq = (u * u).sum(axis=-1)[..., None]
    return (rho[..., None] * W27
            * (1.0 + cu / THETA + 0.5 * (cu / THETA) ** 2 - 0.5 * usq / THETA))


def tble_u_of_tw(tw, bk, h, nu, n_pts=24, growth=1.35):
    """u(h) for the kinematic linear-stress TBLE (patch 81 sec. 1).

    tau_k(y) = tw + bk*y, du/dy = tau_k/(nu + nu_t), nu_t = kappa y
    u_tau_local D^2 with LOCAL stress scaling u_tau_local =
    sqrt(|tau_k(y)|) and van Driest damping D = 1 - exp(-y u_tau_local
    /(nu A)), kappa = 0.41, A = 19 (Wang & Moin 2002 simplified TBLE,
    pressure-gradient term only). Geometric y-grid, midpoint rule —
    the CUDA device mirror (tble_intu) keeps the identical arithmetic
    order; keep the two in step.
    """
    tw = np.asarray(tw, dtype=np.float64)
    bk = np.asarray(bk, dtype=np.float64)
    y1 = h * (growth - 1.0) / (growth ** n_pts - 1.0)
    u = np.zeros(np.broadcast(tw, bk).shape, dtype=np.float64)
    y0, dy = 0.0, y1
    for _ in range(n_pts):
        ym = y0 + 0.5 * dy
        tk = tw + bk * ym
        utl = np.sqrt(np.abs(tk))
        dvd = 1.0 - np.exp(-ym * utl / (nu * 19.0))
        nut = 0.41 * ym * utl * dvd * dvd
        u = u + tk / (nu + nut) * dy
        y0 += dy
        dy *= growth
    return u


def tble_solve_tw(U, h, nu, bk, tw_seed):
    """Bisection for u(h; tw) = U — vectorized host mirror of the CUDA
    tble_solve_tw (fixed 40-sweep, no early exit: determinism, the
    solve_utau convention). Returns tw >= 0; 0 = incipient separation
    under a strong adverse gradient (Q5 counts these)."""
    U = np.asarray(U, dtype=np.float64)
    bk = np.asarray(bk, dtype=np.float64)
    hi = (np.maximum(16.0 * np.asarray(tw_seed, dtype=np.float64),
                     4.0 * nu * U / h) + 4.0 * np.abs(bk) * h + 1e-30)
    for _ in range(3):
        low = tble_u_of_tw(hi, bk, h, nu) < U
        hi = np.where(low, hi * 4.0, hi)
    sep = tble_u_of_tw(np.zeros_like(hi), bk, h, nu) >= U
    lo = np.zeros_like(hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        below = tble_u_of_tw(mid, bk, h, nu) < U
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.where(sep, 0.0, 0.5 * (lo + hi))


def build_facet_intermittency(centroid, normal, spec):
    """Per-facet intermittency gamma from chord-frame geometry (patch 80).

    Forced/natural transition at a KNOWN station lets gamma be geometry
    (the experimental trip-strip logic) instead of a transport equation:
    tau_w = (1-g) tau_lam + g tau_Musker with g = smoothstep in x/c.

    Chord frame comes from the facet cloud itself — no config angle:
    e_c = principal axis of the centroids' airfoil-plane (x, y) scatter,
    oriented downstream (+x); e_n = the left normal (lift-up for flow
    along +x at positive alpha). Side is classified by the OUTWARD
    facet normal's e_n sign, which stays correct around the nose where
    the centroid offset changes sign. x/c is the e_c projection
    normalized to the cloud's own [min, max] = the true chord extent.

    spec: {"suction"|"pressure": {"x_tr": float, "width": float}} in
    x/c units; gamma ramps 0 -> 1 over [x_tr - w/2, x_tr + w/2]. A side
    without an entry stays fully turbulent (g = 1).
    """
    unknown = set(spec) - {'suction', 'pressure'}
    if unknown:
        raise ValueError(f"intermittency: unknown sides {sorted(unknown)}")
    xy = np.asarray(centroid, dtype=np.float64)[:, :2]
    d = xy - xy.mean(axis=0)
    _, vecs = np.linalg.eigh(d.T @ d)
    e_c = vecs[:, -1]
    if e_c[0] < 0.0:
        e_c = -e_c
    e_n = np.array([-e_c[1], e_c[0]])
    s = d @ e_c
    xc = (s - s.min()) / max(float(s.max() - s.min()), 1e-300)
    upper = np.asarray(normal, dtype=np.float64)[:, :2] @ e_n >= 0.0
    gam = np.ones(xy.shape[0], dtype=np.float64)
    for side, sel in (('suction', upper), ('pressure', ~upper)):
        ps = spec.get(side)
        if ps is None:
            continue
        bad = set(ps) - {'x_tr', 'width'}
        if bad:
            raise ValueError(f"intermittency.{side}: unknown keys "
                             f"{sorted(bad)}")
        w = max(float(ps['width']), 1e-300)
        t = np.clip((xc[sel] - (float(ps['x_tr']) - 0.5 * w)) / w, 0.0, 1.0)
        gam[sel] = t * t * (3.0 - 2.0 * t)
    return gam


class SurfelFacets:
    """Facet dynamics over one level's surfels (host numpy reference)."""

    def __init__(
        self,
        surfels: Dict[str, np.ndarray],
        tables: Dict[str, np.ndarray],
        shape: Sequence[int],
        *,
        mode: str = "noslip",
        sample_h: float = 0.5,
        periodic: Sequence[bool] = (True, True, True),
        live: Optional['npt.NDArray'] = None,
        vol_correction: bool = False,
        law=None,
        h_law: float = 3.0,
        nu: float = 1.0 / 6.0,
        y_plus_min: float = 30.0,
        friction_dir: str = "log",
        fallback: str = "viscous",
        wm_filter=None,
    ) -> None:
        """
        Args:
            surfels: `build_surfels(..., return_polygons=True)` output.
            tables:  `build_prism_tables(...)` output (mode="exact").
            shape:   (Nx, Ny, Nz).
            mode:    "noslip" (Eq. 10) | "freeslip" (Eq. 16) |
                     "wallmodel" (Eq. 16 + 25; supply tau_w to apply()).
            sample_h: facet-state sample distance x_a + h n [cells];
                     the paper uses dl/2. Unused by "noslip".
            periodic: axes on which the sample stencil wraps.
            law:  wall_function.WALL_LAWS instance (Musker / viscous). When
                given, mode="wallmodel" computes tau_w ITSELF from the
                log-layer sample instead of taking it from apply().
            h_law: log-layer sample height x_a + h_law n [cells]. Distinct
                from `sample_h`: that one sets the facet state (pressure and
                the free-slip base, [C1] Eq. 12) and sits at ~dl/2, this one
                must sit in the log layer. Conflating them was flagged in
                00_design sec. 5.
            nu:   molecular viscosity of this level [lu^2/lt] (wall law).
            y_plus_min: facets whose sampled y+ falls below this, or whose
                sample stencil is not clean, are gated off the log law and
                handled by `fallback`. Set 0 to disable the y+ test: it
                exists to keep a LOG law out of the region where it is
                invalid, and Musker is a composite profile valid down to
                y+ = 0, so the test is only meaningful for a pure log law
                (same argument the s6k gate already applied to ViscousLaw).
            friction_dir: which sample sets the Eq. (25) friction DIRECTION.
                "state" = u^a, i.e. Chen's structure literally: Eq. (25)
                    builds the friction out of feq(rho^a, u^a), so the
                    direction is whatever the dl/2 facet state says.
                "log"   = the log-layer sample's tangent (DEFAULT).
                    ***DEVIATION FROM [C1]*** -- see patch_notes/surfel/10.
                    Measured cause: the wall model drains the very layer it
                    samples, so u^a collapses (|u^a| 1.05e-2 free-slip ->
                    1.9e-4 wall-model at the same station) and its DIRECTION
                    becomes noise -- a median 85 deg off the flow on the
                    facets that then trip the gate. Magnitudes are unaffected
                    either way: |u^a| cancels against K_f exactly (and
                    feq_i - feq_i* is exactly 2 rho w_i (c_i.u)/theta, i.e.
                    exactly linear in u), so this changes the DIRECTION and
                    nothing else. Measured on the driven channel:
                        gated-off facets   31%  ("state")  ->  0%  ("log")
                        samples with F_x <= 0   11%        ->  0%
                        delivered/target      1.41         ->  1.00
                    "state" is kept selectable so that measurement stays
                    reproducible and Chen's literal path stays runnable.
            fallback: what a gated-off facet does.
                "noslip"   = Eq. (10). ***MEASURED CATASTROPHIC*** in a
                    wall-modelled (slipping) flow: Eq. (10) returns the full
                    one-sided flux, F = 2 sum_in Gamma_in c ~ rho theta u A,
                    which is ~10^3 x the intended tau_w A (measured median
                    3.5e4) and carries the sign of the near-wall shell
                    rather than of the drag -- one facet in 5776 removed 14%
                    of the total wall force. Kept selectable, never default.
                "viscous"  = stay on Eq. (16)+(25) and take tau_w from the
                    VISCOUS law on the same sample (DEFAULT). No operator
                    switch, so no O(rho theta u A) jump, and it is not slip:
                    it is the molecular stress, the correct y+ -> 0 limit,
                    which is exactly what the y+_min guard was for.
                "freeslip" = stay on Eq. (16) with tau_w = 0. Continuous
                    like "viscous" but delivers NO stress on an
                    unresolved-but-viscous wall, which is the error the
                    original gating design set out to avoid; "viscous"
                    dominates it on that same criterion. NOT decided by the
                    channel testbed -- once friction_dir="log" removes the
                    self-inflicted trigger, the gate stops firing there at
                    all, so this choice rests on the argument, not a
                    measurement (patch 10 sec. 6).
            wm_filter: temporal filter on the WALL-MODEL INPUT velocity,
                the log-layer-mismatch (LLM) remedy of Yang, Park & Moin,
                Phys. Rev. Fluids 2, 104601 (2017), their Eq. (2):

                    u_wm^n = (1 - eps) u_wm^(n-1) + eps u_LES^n,
                    eps = dt / T_f          (dt = 1 lattice step here)

                ***IMPLEMENTATION OF A PUBLISHED REMEDY, NOT OURS.***
                Their diagnosis (their Eq. 3): feeding the wall model the
                INSTANTANEOUS sampled velocity makes tau_w' proportional to
                u' (tau_w' ~ C rho u_tau u', measured C ~ 0.2 in coarse
                WMLES vs < 0.1 in high-Re DNS at the same wall distance).
                That unphysical correlation LOCKS the sampled layer, damping
                its fluctuations, so the resolved Reynolds shear stress
                -<u'v'> comes out too small; the mean momentum balance then
                raises d<u>/dy to compensate, which IS the positive LLM.
                Filtering in time breaks the correlation, and their Fig. 3
                shows it removes LLM *without* moving the sample away from
                the wall -- the property we are after (patch_notes/surfel/12).

                Accepted values:
                  None / "none" -- OFF (DEFAULT). eps = 1, i.e. the plain
                      instantaneous sample: bit-identical to every run made
                      before this knob existed.
                  "ti"   -- T_f = h_law / (kappa u_tau), their T_i, the
                      wall-normal turbulent transport time at the sample.
                      Their strongest case (LLM ~1%). u_tau is this facet's
                      value from the PREVIOUS step: it is an output of the
                      very solve the filter feeds, so there is no
                      simultaneous value to use, and it only sets a time
                      scale their Fig. 3 shows the answer is insensitive to
                      (LLM stays negligible even for T_f >> T_i).
                  "dtc"  -- T_f = dx / |u| at the sample, their convective
                      Dt_c (LLM ~2%). dx = 1 lu here.
                  float  -- T_f directly, in lattice steps.

                ★What is filtered is the tangential velocity VECTOR, matching
                their Eq. (2), which filters the components u_wm and w_wm.
                Filtering the MAGNITUDE instead would leave the direction
                unfiltered and would rectify cross-flow fluctuations into a
                high-biased u_tau -- the bias already measured and fixed in
                the link formulation (wall_model patch 18 sec. 3).

                ★The state is seeded with the first sample, not with zero: a
                zero seed would drive y+ below y_plus_min on every facet for
                the first ~T_f steps and fire the fallback spuriously.

                ★This does NOT threaten the delivery identity. The filter
                changes WHAT tau_w is asked for; the scheme still delivers
                exactly that (per-facet |F_t| = tau_w A to 4.2e-13, patch 10),
                so <delivered>/<target> = 1 holds by construction. This is
                why the WFB filter dilemma (patch 20: delivery collapsed
                0.82 -> 0.11 under filtering) does not transfer -- WFB was a
                CORRECTION scheme whose filter ate the transfer itself.
            live: (Nx,Ny,Nz) bool, cells carrying fluid (dV > 0). The facet
                state sample MUST exclude fully solid cells: they hold no
                populations, so an unmasked trilinear stencil at
                x_a + h n drags their zeros into rho^a and biases the
                free-slip construction (measured 8e-8 in a uniform flow that
                should be preserved exactly). Masked + renormalised, a
                uniform field is sampled exactly. Defaults to all-live.
        """
        if mode not in ("noslip", "freeslip", "wallmodel"):
            raise ValueError(
                f"mode must be noslip|freeslip|wallmodel: {mode}")
        self.mode = mode
        #: Eq. (23)-(24) volume correction -- REFUTED guess, keep False
        self.vol_correction = bool(vol_correction)
        self.law = law
        #: per-facet intermittency (patch 80). None = pure Musker; an
        #: (n_f,) array blends tau_w = (1-g) tau_lam + g tau_Musker in
        #: wall_law_tau (kernel mirror in surfel_d3q27). Set by
        #: SurfelBoundary from the config's `intermittency` spec BEFORE
        #: the kernel wrapper is built.
        self.gamma = None
        #: per-facet crease flag (robin/02 sec. 7.12): True -> Eq. (10) for
        #: that facet regardless of `mode`. None = off (bit-identical).
        self.crease = None
        #: pressure-gradient wall function (patch 81): None = off; a
        #: float = the tangential half-spacing ds [cells] of the two
        #: extra rho samples that build dp/ds. Set by SurfelBoundary
        #: BEFORE the kernel wrapper is built.
        self.pg_ds = None
        self.h_law = float(h_law)
        self.nu = float(nu)
        self.y_plus_min = float(y_plus_min)
        if friction_dir not in ("state", "log"):
            raise ValueError(f"friction_dir must be state|log: {friction_dir}")
        if fallback not in ("noslip", "viscous", "freeslip"):
            raise ValueError(
                f"fallback must be noslip|viscous|freeslip: {fallback}")
        self.friction_dir = friction_dir
        self.fallback = fallback
        if wm_filter in (None, "none"):
            self.wm_filter = None
        elif wm_filter in ("ti", "dtc"):
            self.wm_filter = wm_filter
        else:
            self.wm_filter = float(wm_filter)
            if self.wm_filter <= 0.0:
                raise ValueError(
                    f"wm_filter as a time scale must be > 0: {wm_filter}")
        #: Yang Eq. (2) state: filtered tangential sample, one vector per
        #: facet. None until the first wall_law_tau call seeds it.
        self._u_wm = None
        #: previous-step u_tau per facet, only for the "ti" time scale.
        self._u_tau_prev = None
        self.shape = tuple(int(s) for s in shape)
        self.periodic = tuple(bool(p) for p in periodic)
        self.sample_h = float(sample_h)
        self.n_f = int(surfels['cell'].size)
        self.normal = np.ascontiguousarray(surfels['normal'])
        self.area = np.ascontiguousarray(surfels['area'])
        self.centroid = np.ascontiguousarray(surfels['centroid'])

        cdotn = self.normal @ C27.T                       # (n_f, 27)
        #: V_i^a = |c_i . n| A, zero on tangent directions and on rest
        self.V = np.abs(cdotn) * self.area[:, None]
        self.is_in = cdotn < 0.0
        self.is_out = cdotn > 0.0
        self.cdotn = cdotn

        # per-shell IN-side V sums (Eq. 22 denominator); > 0 for any normal
        self._shell_mask = [CSQ27 == j for j in _SHELLS]
        self._shell_den = [np.maximum(
            (self.V * (self.is_in & m)).sum(axis=1), 1e-300)
            for m in self._shell_mask]

        # flatten the CSR into per-pair slices (gather/scatter inner loops)
        indptr = tables['indptr']
        cnt = np.diff(indptr)
        key = np.repeat(np.arange(cnt.size), cnt)
        fac = key // N_PAIR
        pair = key % N_PAIR
        order = np.argsort(pair, kind='stable')
        self._t_fac = fac[order]
        self._t_cell = tables['cell'][order]
        self._t_w = tables['weight'][order]
        bounds = np.searchsorted(pair[order], np.arange(N_PAIR + 1))
        self._t_beg, self._t_end = bounds[:-1], bounds[1:]

        # Actual per-(facet, direction) table sum. Eq. (6) divides by V_i^a;
        # using the TABLE's own sum instead makes sum_x Q_i == Gamma_out
        # identically, so mass cannot leak even if a prism is clipped by a
        # non-periodic domain edge (Eq. 7's "<= 1" case).
        self.Vsum = np.zeros((self.n_f, 27))
        for p in range(N_PAIR):
            b, e = self._t_beg[p], self._t_end[p]
            if b == e:
                continue
            tot = np.bincount(self._t_fac[b:e], self._t_w[b:e], self.n_f)
            for i in np.nonzero(PAIR_OF == p)[0]:
                self.Vsum[:, int(i)] = tot

        # per-direction outflow field g_i(x) (Eq. 5 numerator)
        self.g_field = np.zeros((27,) + self.shape)
        for p in range(N_PAIR):
            S_minus, S_plus = pair_cell_sums(tables, surfels, shape, p)
            for i in range(1, 27):
                if PAIR_OF[i] != p:
                    continue
                same = bool((C27[i] == PAIR_DIR[p]).all())
                self.g_field[i] = S_minus if same else S_plus

        self.live = (np.ones(self.shape, dtype=bool) if live is None
                     else np.asarray(live, dtype=bool))
        #: b_j = number of lattice directions in speed shell j (Eq. 24)
        self._shell_b = np.array([int(m.sum()) for m in self._shell_mask])
        self._force = np.zeros(3)
        self._df = np.zeros((self.n_f, len(_SHELLS)))
        self._state = (None, None)
        self._fallback = np.zeros(self.n_f, dtype=bool)
        self._u_tau = np.zeros(self.n_f)
        self._y_plus = np.zeros(self.n_f)
        self._tau_w = np.zeros(self.n_f)
        self._last = {}

    # ------------------------------------------------------------------
    def gather(self, n_post: 'npt.NDArray') -> np.ndarray:
        """Gamma_in[a, i] = sum_x V_i^a(x) n'_i(x)   (Eq. 8). (n_f, 27)."""
        flat = n_post.reshape(27, -1)
        G = np.zeros((self.n_f, 27))
        for p in range(N_PAIR):
            b, e = self._t_beg[p], self._t_end[p]
            if b == e:
                continue
            fac, cell, w = self._t_fac[b:e], self._t_cell[b:e], self._t_w[b:e]
            for i in (int(np.nonzero(PAIR_OF == p)[0][0]),
                      int(np.nonzero(PAIR_OF == p)[0][1])):
                G[:, i] = np.bincount(fac, w * flat[i, cell], self.n_f)
        return G * self.is_in

    def scatter(
        self,
        G_in: np.ndarray,
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
        tau_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Gamma_out[a, i] from Eq. (10) or (16) + (21,22) [+ (25)]."""
        G_out = np.zeros_like(G_in)
        opp = OPP27
        if self.mode == "noslip":
            G_out[:, 1:] = G_in[:, opp[1:]]
            G_out *= self.is_out
            # Eq. (10) conserves mass per facet AND per speed shell by
            # itself (i -> i* is a bijection preserving |c|), so there is no
            # Eq. (21)-(22) correction and hence NO dp term: no-slip surface
            # pressure carries none of the deferred Eq. (23)-(24) error.
            self._state = (None, None)
            self._df = np.zeros((self.n_f, len(_SHELLS)))
            self._fallback = np.ones(self.n_f, dtype=bool)
            self._tau_w = np.zeros(self.n_f)
            return G_out

        rho_a, u_a, u_n = self.sample_state(rho, u, want_un=True)
        fallback = np.zeros(self.n_f, dtype=bool)
        #: velocity the Eq. (25) friction is BUILT from. Same direction as
        #: u_a in Chen's structure; `friction_dir="log"` swaps the direction
        #: (and only the direction) for the log-layer tangent.
        u_fric = u_a
        if self.mode == "wallmodel" and self.law is not None:
            tau_w, gated, self._u_tau, self._y_plus, u_fric = \
                self.wall_law_tau(rho, u, u_a, rho_a)
            fallback = ~gated
            if self.fallback == "noslip":
                tau_w = np.where(gated, tau_w, 0.0)
        #: tau_w this pass ACTUALLY used (per facet). Diagnostics must read
        #: this, not a recomputation from a later macroscopic state.
        self._tau_w = (np.zeros(self.n_f) if tau_w is None
                       else np.broadcast_to(np.asarray(tau_w, dtype=float),
                                            (self.n_f,)).copy())
        feq = equilibrium(rho_a, u_a)                       # (n_f, 27)
        V = self.V
        if self.vol_correction:
            # Eq. (23)-(24): V_i -> (1 + beta (n.c_i)^2) V_i with
            # beta = (u_n / (theta dt)) (3/2 - tau), tau = 1/2 from Eq. (16).
            #
            # *** REFUTED RECONSTRUCTION -- KEEP OFF (default False). ***
            # The paper states the substitution but defers its derivation to
            # an internal document (ref. 19), so WHERE V~ enters is not
            # given. This guess -- facet algebra only, never the transport
            # weights (those carry the S2 conservation identity) -- was
            # MEASURED on a sphere and makes the spurious dp WORSE:
            #     dp std / q :  1.53  (off)  ->  4.90  (on)
            # It is kept, wired and labelled, so the same guess is not
            # retried. Getting Eq. (23)-(24) right needs the missing
            # derivation, not another placement guess.
            beta = u_n / THETA
            V = V * (1.0 + beta[:, None] * (self.cdotn ** 2))
        G_out = (V * (feq + feq[:, opp]) - G_in[:, opp]) * self.is_out

        if self.mode == "wallmodel":
            if tau_w is None:
                raise ValueError("mode='wallmodel' needs tau_w (n_f,)")
            # Eq. (25) with Kf = tau_w / (p |u|), written so that the
            # delivered friction is tau_w * A along -u_fric even as |u| -> 0.
            # feq_i - feq_i* is EXACTLY 2 rho w_i (c_i.u)/theta (the even
            # orders cancel), so this is exactly linear in u_fric: the
            # magnitude cancels against Kf and only the direction survives.
            p_a = rho_a * THETA
            umag = np.linalg.norm(u_fric, axis=1)
            kf = np.where(umag > 1e-14,
                          np.asarray(tau_w) / np.maximum(p_a * umag, 1e-300),
                          0.0)
            feq_f = (feq if u_fric is u_a else equilibrium(rho_a, u_fric))
            G_out += (-kf[:, None] * V * self.cdotn
                      * (feq_f - feq_f[:, opp])) * self.is_out

        # Eq. (21)-(22): per-speed-shell mass/energy correction
        self._state = (rho_a, u_a)
        self._df = np.zeros((self.n_f, len(_SHELLS)))
        for jj, (j, m) in enumerate(zip(_SHELLS, self._shell_mask)):
            num = ((G_in * (self.is_in & m)).sum(axis=1)
                   - (G_out * (self.is_out & m)).sum(axis=1))
            den = (np.maximum((V * (self.is_in & m)).sum(axis=1), 1e-300)
                   if self.vol_correction else self._shell_den[jj])
            df = num / den
            self._df[:, jj] = df
            G_out += df[:, None] * V * (self.is_out & m)

        if fallback.any() and self.fallback == "noslip":
            # gated-off facets revert to Eq. (10): no mass fix, hence no dp
            bb = np.zeros_like(G_out)
            bb[:, 1:] = G_in[:, opp[1:]]
            G_out = np.where(fallback[:, None], bb * self.is_out, G_out)
            self._df[fallback] = 0.0
        if self.crease is not None and np.any(self.crease):
            # concave-crease facets: Eq. (10) (robin/02 sec. 7.12)
            cr = np.asarray(self.crease, dtype=bool)
            bb = np.zeros_like(G_out)
            bb[:, 1:] = G_in[:, opp[1:]]
            G_out = np.where(cr[:, None], bb * self.is_out, G_out)
            self._df[cr] = 0.0
            if tau_w is not None:
                tau_w = np.where(cr, 0.0, np.asarray(tau_w))
            fallback = fallback | cr
        self._fallback = fallback
        return G_out

    def distribute(self, G_out: np.ndarray) -> np.ndarray:
        """Q_i(x) = sum_a [V_i^a(x)/V_i^a] Gamma_out[a, i]   (Eq. 6)."""
        nx, ny, nz = self.shape
        Q = np.zeros((27, nx * ny * nz))
        frac = np.where(self.Vsum > 0.0,
                        G_out / np.maximum(self.Vsum, 1e-300), 0.0)
        for p in range(N_PAIR):
            b, e = self._t_beg[p], self._t_end[p]
            if b == e:
                continue
            fac, cell, w = self._t_fac[b:e], self._t_cell[b:e], self._t_w[b:e]
            for i in np.nonzero(PAIR_OF == p)[0]:
                i = int(i)
                np.add.at(Q[i], cell, w * frac[fac, i])
        return Q.reshape((27,) + self.shape)

    def force(self, G_in: np.ndarray, G_out: np.ndarray) -> np.ndarray:
        """Force ON THE BODY = -(Eq. 9 summed over facets, times A dt)."""
        return ((G_in * self.is_in - G_out * self.is_out) @ C27)\
            .sum(axis=0)

    # ------------------------------------------------------------------
    def sample_state(self, rho, u, want_un=False):
        """(rho^a, u^a) at x_a + h n, with the NORMAL velocity zeroed (12)."""
        xs = self.centroid + self.sample_h * self.normal
        fields = np.concatenate([rho[None, ...], u], axis=0)
        got = self._trilinear(fields, xs)
        r, v = got[0], got[1:].T
        un = np.einsum('ij,ij->i', v, self.normal)
        v = v - un[:, None] * self.normal
        return (r, v, un) if want_un else (r, v)

    def _trilinear(self, fields, xs, want_w=False):
        """Trilinear interpolation MASKED to live cells and renormalised.

        With want_w, also returns the surviving stencil weight: < 1 means the
        sample straddles solid, 0 means it is unusable (gating input).
        """
        dims = self.shape
        base = np.floor(xs).astype(np.int64)
        frac = xs - base
        out = np.zeros((fields.shape[0], xs.shape[0]))
        wsum = np.zeros(xs.shape[0])
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    wgt = ((1 - frac[:, 0] if a == 0 else frac[:, 0])
                           * (1 - frac[:, 1] if b == 0 else frac[:, 1])
                           * (1 - frac[:, 2] if c == 0 else frac[:, 2]))
                    idx = []
                    for ax, off in enumerate((a, b, c)):
                        k = base[:, ax] + off
                        if self.periodic[ax]:
                            k = k % dims[ax]
                        else:
                            k = np.clip(k, 0, dims[ax] - 1)
                        idx.append(k)
                    wgt = wgt * self.live[idx[0], idx[1], idx[2]]
                    out += wgt * fields[:, idx[0], idx[1], idx[2]]
                    wsum += wgt
        res = out / np.maximum(wsum, 1e-300)
        return (res, wsum) if want_w else res

    # ------------------------------------------------------------------
    def wall_law_tau(self, rho, u, u_a, rho_a=1.0):
        """tau_w and the gate from the LOG-LAYER sample (mode="wallmodel").

        Direction comes from the facet state u^a (that is what Eq. 25 builds
        the friction from); the MAGNITUDE comes from the log-layer sample,
        PROJECTED on that direction rather than taken as |u_t|. Using the
        magnitude rectifies cross-flow fluctuations and biases u_tau high --
        measured and fixed once already in the link formulation
        (wall_model patch 18 sec. 3).

        With friction_dir="log" the direction is the log-layer tangent
        instead (DEVIATION, see the constructor docstring): then U_t is that
        sample's own tangential speed, which is non-negative by construction
        so the U_t > 0 test stops being a fluctuation trigger.

        Returns (tau_w, gated, u_tau, y_plus, u_fric).

        With `wm_filter` set, the log-layer sample is passed through the
        Yang et al. (2017) Eq. (2) exponential time filter before it reaches
        the wall law -- see the constructor docstring.
        """
        xs = self.centroid + self.h_law * self.normal
        fields = np.concatenate([rho[None, ...], u], axis=0)
        got, wsum = self._trilinear(fields, xs, want_w=True)
        rho_l, v = got[0], got[1:].T
        v = v - (np.einsum('ij,ij->i', v, self.normal)[:, None] * self.normal)
        if self.wm_filter is not None:
            v = self._filter_input(v)
        mag = np.linalg.norm(u_a, axis=1)
        if self.friction_dir == "log":
            lmag = np.linalg.norm(v, axis=1)
            that = np.where(lmag[:, None] > 1e-14,
                            v / np.maximum(lmag, 1e-300)[:, None], 0.0)
            U_t = lmag
            u_fric = that * lmag[:, None]          # = v, but explicit
        else:
            that = np.where(mag[:, None] > 1e-14, u_a / np.maximum(
                mag, 1e-300)[:, None], 0.0)
            U_t = np.einsum('ij,ij->i', v, that)
            u_fric = u_a
        u_tau = self.law.utau(np.abs(U_t), self.h_law, self.nu)
        y_plus = self.h_law * u_tau / self.nu
        clean = wsum > 0.5
        gated = clean & (y_plus > self.y_plus_min) & (U_t > 0.0)
        tau_w = rho_l * u_tau ** 2
        if self.pg_ds is not None:
            # pressure-gradient ratio (patch 81, kernel mirror):
            # tau_turb = tau_Musker * R, R = tau_TBLE(beta)/tau_TBLE(0).
            # Ratio form keeps the Musker family baseline — R == 1
            # exactly at beta == 0 (same integrator over itself).
            ds = float(self.pg_ds)
            # tangent projected off the span axis (z): the periodic
            # direction has zero mean dp/ds — kernel mirror.
            t2 = u_fric.copy()
            t2[:, 2] = 0.0
            lm = np.linalg.norm(t2, axis=1)
            that_pg = np.where(lm[:, None] > 1e-14,
                               t2 / np.maximum(lm, 1e-300)[:, None],
                               0.0)
            base = self.centroid + self.h_law * self.normal
            rp, wp = self._trilinear(rho[None], base + ds * that_pg,
                                     want_w=True)
            rm, wm = self._trilinear(rho[None], base - ds * that_pg,
                                     want_w=True)
            ok = (wp > 0.5) & (wm > 0.5) & (lm > 1e-14)
            beta = np.where(ok, (rp[0] - rm[0]) / (3.0 * 2.0 * ds), 0.0)
            tw_b = tble_solve_tw(np.abs(U_t), self.h_law, self.nu,
                                 beta, u_tau ** 2)
            tw_0 = tble_solve_tw(np.abs(U_t), self.h_law, self.nu,
                                 np.zeros_like(beta), u_tau ** 2)
            R = np.where(tw_0 > 1e-300, tw_b / np.maximum(tw_0, 1e-300),
                         1.0)
            tau_w = tau_w * np.where(beta == 0.0, 1.0, R)
        if self.gamma is not None:
            # intermittency blend (patch 80): tau_w = (1-g) tau_lam +
            # g tau_Musker, tau_lam = the same linear viscous stress the
            # fallback uses, at the same h_law sample. g >= 1 rows keep
            # the pure Musker product bit-exactly (kernel mirror).
            g = self.gamma
            t_lam = rho_l * self.nu * np.abs(U_t) / self.h_law
            tau_w = np.where(g >= 1.0, tau_w, (1.0 - g) * t_lam + g * tau_w)
        if self.fallback == "viscous":
            # molecular stress from the same sample where the log law is not
            # usable; falls to 0 continuously as U_t -> 0. Where the stencil
            # itself is unusable, drop to the facet state, the one sample
            # that always exists.
            tv = np.where(clean,
                          rho_l * self.nu * np.abs(U_t) / self.h_law,
                          rho_a * self.nu * mag / max(self.sample_h, 1e-300))
            u_fric = np.where(clean[:, None], u_fric, u_a)
            tau_w = np.where(gated, tau_w, tv)
        elif self.fallback == "freeslip":
            tau_w = np.where(gated, tau_w, 0.0)
        return tau_w, gated, u_tau, y_plus, u_fric

    #: von Karman constant, only for the "ti" filter time scale. The wall
    #: LAW carries its own (Musker is a fitted composite); this is the
    #: 0.41 named in wall_function's module docstring.
    KAPPA = 0.41

    def _filter_input(self, v):
        """Yang et al. (2017) Eq. (2) on the wall-model input velocity.

            u_wm^n = (1 - eps) u_wm^(n-1) + eps v^n,   eps = dt / T_f

        dt = 1 lattice step. Seeded with the first sample (see constructor:
        a zero seed would gate every facet off for the first ~T_f steps).
        eps is clipped to [0, 1]: eps > 1 would overshoot past the new
        sample and oscillate, and T_f < dt is not a filter at all.

        ★THE TIME SCALE IS BUILT FROM THE CURRENT *UNFILTERED* SAMPLE `v`,
        never from the filter's own state or its own previous u_tau. That is
        a deliberate departure from the most literal reading, and it is not
        cosmetic -- the literal version DEADLOCKS. Measured (gate s11 [D],
        first attempt): with eps taken from the filtered state, a channel
        started from rest has u_tau = 0, hence eps = 0, hence u_wm frozen at
        0 forever, hence u_tau = 0 -- tau_w collapsed to EXACTLY zero on
        every facet and stayed there for 3000 steps. Both adaptive scales
        ("ti" and "dtc") self-lock this way. Driving eps from the incoming
        sample breaks the loop: as the real flow develops, v grows whatever
        the filter state is doing, so the filter always has a live input.
        Yang et al. do not meet this because their T_i uses the channel's
        MEAN u_tau, known a priori in their configuration.
        """
        v = np.ascontiguousarray(v, dtype=np.float64)
        if self._u_wm is None or self._u_wm.shape != v.shape:
            self._u_wm = v.copy()
            return self._u_wm
        if self.wm_filter == "ti":
            # T_f = h / (kappa u_tau)  =>  eps = kappa u_tau / h, with u_tau
            # from the UNFILTERED sample (extra wall-law solve per step).
            vt = np.linalg.norm(v, axis=1)
            u_tau_s = self.law.utau(vt, self.h_law, self.nu)
            self._u_tau_prev = u_tau_s
            eps = self.KAPPA * u_tau_s / max(self.h_law, 1e-300)
        elif self.wm_filter == "dtc":
            # T_f = dx / |u| with dx = 1 lu  =>  eps = |v|
            eps = np.linalg.norm(v, axis=1)
        else:
            eps = np.full(v.shape[0], 1.0 / self.wm_filter)
        eps = np.clip(np.asarray(eps, dtype=np.float64), 0.0, 1.0)[:, None]
        self._u_wm = self._u_wm + eps * (v - self._u_wm)
        return self._u_wm

    def apply(self, n_post, rho, u, tau_w=None):
        """One facet pass: returns (Q, force_on_body)."""
        G_in = self.gather(n_post)
        G_out = self.scatter(G_in, rho, u, tau_w)
        self._force = self.force(G_in, G_out)
        self._last = {'G_in': G_in, 'G_out': G_out}
        return self.distribute(G_out), self._force

    def last_force(self) -> np.ndarray:
        return self._force

    # ------------------------------------------------------------------
    def facet_traction(self, G_in=None, G_out=None,
                       rho_a=None) -> Dict[str, np.ndarray]:
        """Per-facet surface load, on the GEOMETRY (not the lattice).

        Eq. (9) IS the momentum flux through a facet, so this is the
        algorithm's primitive output; the total force is the derived sum.
        (A staircase/link scheme has it the other way round: link exchanges
        are primitive and any surface distribution is a re-binning.)

        Returns per-facet arrays:
            force    (n_f, 3)  force ON THE BODY [lu]
            traction (n_f, 3)  force per unit area
            p        (n_f,)    surface pressure = -traction . n
            tau      (n_f, 3)  wall shear vector (tangential traction)
            tau_mag  (n_f,)    |tau|
            dp       (n_f,)    Eq. (24) pressure offset carried by the
                               Eq. (21)-(22) mass fix. Identically 0 for
                               "noslip" (Eq. 10 needs no mass fix).
            p_state  (n_f,)    rho^a * theta -- the pressure Eq. (20)
                               INTENDS, taken straight from the facet state
                               (Eq. 12) instead of from the normal traction.
                               NaN in "noslip" (which does not sample).
            p_use    (n_f,)    the one to trust per mode: `p` for "noslip",
                               `p_state` for the slip modes.

        WHICH PRESSURE TO USE. Eq. (23) says F = (p + dp) n, and this is
        verified to 2.4e-16 here: p - p_state - dp == 0. So the normal
        traction carries the spurious dp, while p_state does not.
            noslip            dp == 0, so `p` and `p_state` agree; use `p`.
            freeslip / wallmodel
                              dp varies by ~79% of the pressure signal
                              (patch 07), so `p` is unusable for Cp while
                              `p_state` is clean -- measured std/q 0.49 vs
                              1.94 on a sphere. Use `p_state`.
        Caveat on p_state: it is sampled at x_a + h n, i.e. it is the
        NEAR-WALL pressure at the model's own sampling height, so it carries
        an h-dependent offset. That is a bounded discretisation error, not
        the 79%-of-signal artefact, but it is not yet validated against a
        reference (S4(ii)).
        """
        if G_in is None or G_out is None:
            if not self._last:
                raise RuntimeError("call apply()/scatter() first, or pass "
                                   "G_in and G_out explicitly")
            G_in, G_out = self._last['G_in'], self._last['G_out']
        f_body, trac, pn, tau = traction_kinematics(
            G_in, G_out, self.is_in, self.is_out, self.area, self.normal)
        dp = (self._df * self._shell_b * np.array(_SHELLS)).sum(axis=1) / 6.0
        # rho_a: pass EXPLICITLY from the CUDA kernel's rho_out (patch
        # 70) — self._state is the python-pass state the kernel never
        # writes (the same trap as G_in/G_out above). Without it the
        # production surface Cp silently fell back to -pn (68 sec. 0).
        if rho_a is None:
            rho_a = self._state[0]
        if rho_a is None:
            p_state = np.full(self.n_f, np.nan)
            p_use = -pn
        else:
            p_state = np.asarray(rho_a, dtype=np.float64) * THETA
            p_use = p_state
        return {'force': f_body, 'traction': trac, 'p': -pn, 'tau': tau,
                'tau_mag': np.linalg.norm(tau, axis=1), 'dp': dp,
                'p_state': p_state, 'p_use': p_use}
