"""Tamaki-Kawai model turbulent shear stress ("tau-model") for WMLES.

[IMPLEMENTATION -- no novelty claimed]

Source
    Y. Tamaki, S. Kawai, Phys. Rev. Fluids 6, 114603 (2021): the near-wall
    total shear-stress balance is maintained by ADDING a model turbulent
    shear stress -- SHEAR COMPONENT ONLY, diagonal components untouched --
    so the resolved wall-normal turbulent mixing is not damped. Isotropic
    nu_t reinforcement fails exactly there (FODIBM mu-model -8.42% ==
    our uniform-Cs -7.0%, patch_notes/surfel/24 sec. 1).

    Practical form + LBM validation: Mao, Tamaki et al., "A fully one-sided
    diffuse-interface immersed boundary method for wall-modeled LES",
    arXiv:2607.07443 (2026) ("FODIBM"), sec. 2.4.3. TWO equivalent-in-intent
    forms of the stress scale:

        tau^model_ij = tau_w (xi_i eta_j + eta_i xi_j) f_s      (Eq. 50)
        tau^model_ij = rho (kappa d_ref)^2 (du_xi/deta) |du_xi/deta|
                       (xi_i eta_j + eta_i xi_j) f_s            (Eq. 54)
        f_s = max((d_ref - eta)/d_ref, 0)                       (Eq. 49)

    xi = wall-parallel, eta = wall-normal unit vector, eta also the wall
    distance; d_ref = reference height (identical to the wall-model sample
    height in their construction -- tie the two unless probing).

    Eq. 50 is the PRIMARY Tamaki-Kawai form; Eq. 54 is FODIBM's local-
    gradient ESTIMATE of tau_w, introduced only "to avoid searching for the
    nearest wall from every fluid point" (their sec. 2.4.3). ★Our default is
    Eq. 50 (`tau_model_shear_wall`): the surfel boundary already holds the
    wall-model tau_w and the wall geometry PER FACET, so their detour is
    unnecessary here -- and the detour is precisely what broke on cut cells
    (patch 26 sec. 4: the first band cell's FD gradient reads the
    un-collapsible cut-layer segment, overproducing sigma^m 2.2~2.8x, the
    excess draining through a sink outside the facet force accounting,
    true friction +77~142%). Eq. 50 bounds sigma^m <= tau_w by
    construction. Eq. 54 (`tau_model_shear`) is kept as the A/B control.

Deviations from the source (patch_notes/surfel/25 owns the record):
    1. INJECTION FACTOR. FODIBM injects through the 2nd-order Hermite
       moment of the FORCING term of a regularized (HRR-family) scheme,
       a^{model,(2)} = -tau^model/tau_LB with tau_LB = nu_tot/c_s^2 (their
       Eq. 55). That factor belongs to their forcing + regularization
       bookkeeping. Our cumulant collision CARRIES its non-equilibrium
       moments, so we inject POST-COLLISION in population space; injecting
       dPi every step reaches the fixed point

           carried = tau_bar * dPi      (pre* = (1 - 1/tau_bar) pre* + dPi)

       and the transported face flux equals the carried moment for this
       momentum-free Hermite-2 term (direct lattice sum: each c_z = +-1
       half carries sum w_i c_x^2 = 1/18, so J_extra = carried). Hence

           dPi = -sigma^m / tau_bar,

       NOT -sigma^m/tau_LB -- the two differ by ~10^3 at tau_bar -> 1/2.
       Gate s12 [C] pins the 1/tau_bar law at tau_bar = 0.8 AND 0.5001.
    2. tau_bar is passed as the scalar molecular tau_0; the exact factor is
       the per-cell tau_total, but 3 nu_t << 1/2 keeps the error <= ~0.3%.
    3. SGS = Smagorinsky Cs = 0.1 (production-kernel parity, patch 21
       verdict), not their Vreman. The model ADDS to the SGS stress (their
       Eq. 47) -- it does not replace it.
    4. At tau_bar -> 1/2 the carried moment relaxes with (1 - 1/tau_bar)
       ~ -1, so it OSCILLATES between 0 and 2 sigma^m step-to-step with the
       correct TIME MEAN. Amplitude ~1e-5 = O(1e-3) of the viscous Pi^neq;
       accepted, monitored through rho_min/rho_max.
    5. BAND STARTS AT THE FIRST FULL CELL with a complete wall-normal
       stencil. Injecting into a wall-adjacent CUT cell feeds the term's
       c_z lanes into the boundary transport under the volumetric (dV < 1)
       scaling, breaking its zero-net-momentum property -- measured as a
       secular local runaway (gate s12 [S] first run, e-fold ~220 steps,
       upper wall only where the cut cell centre lies in the fluid). The
       cut layer belongs to the wall exchange (surfel); FODIBM likewise
       keeps its IB region separate from the band points.

This module is the geometry-free math. The channel testbed wires xi = x,
eta = z-distance (patch_notes/surfel/channel_wmles_surfel.py); the STL path
(S8) reuses these functions with per-facet xi/eta from the surfel geometry.
Everything is namespace-agnostic (numpy or cupy arrays).
"""

from __future__ import annotations

import numpy as np

#: von Karman constant, FODIBM sec. 2.4 (their kappa = 0.41 throughout).
KAPPA = 0.41


#: Measured-closure blending table (patch_notes/surfel/31, 2026-08-05).
#: f_s at integer eta [cells]; linear interpolation between nodes, 1 below
#: the first, 0 beyond the last. ★ZERO tunable parameters -- every node is
#: a MEASUREMENT INVERSION: at each height y, the three measured
#: (model-supply, Xi) states [supply 0 = zoff05_cs010 (OFF); low = taum3
#: h3d3_z05 (linear f_s, d 3); high = taum3 h5d5_z05 (linear f_s, d 5)]
#: are interpolated to the supply at which Xi = (y/u_tau) dU/dy equals the
#: log-layer value 1/kappa (kappa = 0.41 -- a constant, not a dial). The
#: band depth (zero-crossing at eta = 5) comes out of the data too.
#: DEVIATION from FODIBM Eq. 49, which the source itself marks as
#: convenience ("this simple linear form is adopted for convenience,
#: despite the possibility of alternative choices"); our resolved
#: turbulence is damaged more deeply than their healthy-LES premise, and
#: linear f_s was measured unable to satisfy h = 3 and h = 5 at once
#: (patch 30: h3 under- at y 2~4, h5 over-supplied at y 1~3).
#: SCOPE: measured at Delta+ ~ 115, Cs 0.1, this scheme, in CELL units --
#: same portability status as the damage depth itself (Delta+ sweep
#: pending). Iteration protocol: if the band Xi still misses 1/kappa,
#: ADD the new measured (supply, Xi) points and re-invert -- convergence
#: by measurement, never by hand-tuning (patch 31 sec. 3).
#: v2 (patch 32): second measurement inversion, taum4's aligned states
#: added. Node update = per-node scaling by its face's delivered/target
#: ratio (uniform rule, no thresholds; the exact tridiagonal inversion is
#: ill-conditioned and returns non-physical nodes -- measured, rejected).
#: This is the LAST inversion round per the patch 31 sec. 2 protocol.
#: v1 was (1.0, 0.739, 0.627, 0.355, 0.199, 0.0).
FS_TABLE_ETA = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
FS_TABLE_VAL = (1.0, 0.690, 0.635, 0.290, 0.202, 0.0)


def f_s_measured(eta):
    """Measured-closure blending profile (table above), FODIBM Eq. 49 slot.

    numpy only -- the profile is precomputed on the host once per run.
    """
    return np.interp(eta, FS_TABLE_ETA, FS_TABLE_VAL, left=1.0, right=0.0)


#: y1-CONDITIONAL cut-class tables (patch 37 C1, 2026-08-06). Key = the
#: wall's first-fluid-node distance y1 -- the ONE sub-cell parameter of
#: this geometry (patch 20: dV(node cell) = min(1, y1 + 1/2), forced).
#: Every node value is a measurement inversion (target Xi = 1/kappa) from
#: the per-wall (supply, Xi) map of patch 37 sec. 2, converted from FACE
#: targets to CELL nodes by forward substitution from the wall:
#:   y1 = 0.3: well-posed -- the cut cell is excluded so c(0.3) = 0 is
#:       KNOWN; the resulting allocation is deliberately non-monotone
#:       (0.39 at eta 1.3, 0.87 at 2.3): the eta = 1.3 cell feeds the
#:       0.8 face, which the map measured as ALREADY over-flattened.
#:   y1 = 0.7: anchored at f_s = 1 on the wall side (Eq. 46).
#:   y1 = 0.5: the aligned FS_TABLE above (identical object -- aligned
#:       runs are bit-compatible with the taum5/dsw rounds).
#: v1 is EXTRAPOLATION-heavy (only the y1=0.3 wall's first face was
#: bracketed); the C2 round re-inverts by the same iteration protocol as
#: the aligned loop (patch 31 sec. 2). Lookup interpolates linearly in y1
#: and CLAMPS outside [0.3, 0.7] until C3 widens the coverage.
#: cut-class v2 (patch 38): SECOND inversion -- the cut2 states supplied
#: true Xi = 1/kappa BRACKETS on nearly every face (v1's extrapolation
#: overshot into flow REVERSAL, Xi < 0, because the cut-wall response
#: slope is ~4x the extrapolated one), so v2 is bracket interpolation.
#: cut-class v3 (patch 40, the registered patch 39 sec. 2 LOCAL
#: REFINEMENT -- the last allowed round): derivation is executable,
#: patch_notes/surfel/fs_invert_round3.py. Two repairs at once:
#:   (a) CONVERSION BUG closed -- the v1/v2 forward substitution turned
#:       face supply targets into f_s with a CONSTANT gamma ~ 0.9667
#:       (the y = 1.2 face's factor), where the exact factor is
#:       (1 - y/delta)/D_wall, y-dependent. cut3's delivered supply
#:       therefore missed the REGISTERED round-2 targets by +5~13%
#:       growing outward -- and matches v2-imposed x D/(1 - y/delta) to
#:       0.2% on all 8 faces (D per wall constant to <0.01% across
#:       faces: the delivery model is exact). A mechanical share of the
#:       patch-39 "ripple" was this bug, not bracket contamination. The
#:       v2 clamp narrative ("face 1.8 wants s* > 1, deliverable max
#:       0.76") also dissolves: with healthy targets and exact
#:       conversion the chain is clamp-free (c(2.3) = 0.969).
#:   (b) HEALTHY BRACKETS only (cut3 = over-flat side, taum5 = starved
#:       side; reversal-contaminated cut2 points excluded). 8/10 faces
#:       truly bracketed; x-check: taum3-h5d5 measured (0.405, Xi 2.44)
#:       AT the log target on lo 3.2, vs inverted s* = 0.402.
#: TAIL RULE (new, forced): the bidiagonal face<-node chain leaves one
#: node serving both the outermost registered face and the super-band
#: face, whose measured response lines demand f_s values ~3x apart (a
#: supply cliff at the band edge that mean-of-adjacent-nodes cannot
#: produce without a negative node). c_tail = minimax of the two
#: predicted |Xi - 1/kappa| defects on the measured c3<->t5 response
#: lines, under a REVERSAL VETO and a STABLE-FLOOD FLOOR (min in-band
#: Xi measured stable, 0.739). All inputs measured; zero free
#: parameters. The outer faces (lo 4.2, up 3.8) then ride on NONLOCAL
#: RECOVERY -- the patch-39-validated mechanism (up 1.8's "impossible"
#: requirement dropped to a met 0.77 once the neighbouring reversal was
#: gone); their discriminator bands are pre-registered in patch 40.
#: v1 was 0.3: (0.386, 0.870, 0.792, 0.348), 0.7: (1.0, 0.836, 0.729,
#: 0.121, 0.150). v2 was 0.3: (0.471, 1.0(clamp), 0.366, 0.452),
#: 0.7: (1.0, 1.0, 0.623, 0.709, 0.108, 0.156).
FS_CLASSES = {
    0.3: ((1.3, 2.3, 3.3, 4.3, 5.3), (0.500, 0.969, 0.254, 0.445, 0.0)),
    0.5: (FS_TABLE_ETA, FS_TABLE_VAL),
    0.7: ((0.0, 0.7, 1.7, 2.7, 3.7, 4.7, 5.7),
          (1.0, 1.0, 0.557, 0.685, 0.030, 0.157, 0.0)),
}

#: header provenance (patch 29's re-copied-stale-table defence): the
#: runner prints this WITH the node values -- values are data, labels lie.
FS_CLASSES_VERSION = "v3 (patch 40)"


def f_s_class(eta, y1: float):
    """y1-conditional measured-closure f_s (patch 37 C1).

    Linear interpolation between class tables in y1, clamped to the
    covered range. Left extension = first node value (for the 0.3 class
    that is 0.500, NOT 1 -- the allocation shift is the design; cells
    below its first node are cut cells the band mask excludes anyway).
    """
    keys = sorted(FS_CLASSES)
    y1c = min(max(float(y1), keys[0]), keys[-1])
    lo = max(k for k in keys if k <= y1c)
    hi = min(k for k in keys if k >= y1c)

    def _ev(k):
        e, v = FS_CLASSES[k]
        return np.interp(eta, e, v, left=v[0], right=0.0)

    if lo == hi:
        return _ev(lo)
    w = (y1c - lo) / (hi - lo)
    return (1.0 - w) * _ev(lo) + w * _ev(hi)


def f_s_linear(eta, d_ref: float):
    """Blending band f_s, FODIBM Eq. 49: 1 at the wall, 0 at eta >= d_ref.

    Args:
        eta:   wall-normal distance field [cells]; may be signed (negative
               = behind the wall). The caller masks non-fluid cells -- this
               function only shapes the band, clipping to [0, 1].
        d_ref: band height / reference height [cells].
    """
    x = (d_ref - eta) / d_ref
    return x.clip(0.0, 1.0)


def tau_model_shear(rho, dudn, fs, d_ref: float, kappa: float = KAPPA):
    """Model shear stress magnitude sigma^m, FODIBM Eq. 54 (scalar part).

        sigma^m = rho (kappa d_ref)^2 (du_xi/deta) |du_xi/deta| f_s

    The SIGN is carried by dudn, so one global-frame evaluation serves both
    walls of a channel (upper wall: dudn < 0 -> sigma^m < 0), and the
    tensor structure (xi_i eta_j + eta_i xi_j) reduces to the single xz
    component in the channel frame.

    ★A/B CONTROL ONLY (patch 26): on cut cells the FD gradient reads the
    un-collapsible cut-layer segment and overproduces sigma^m 2.2~2.8x.
    The default is `tau_model_shear_wall` (Eq. 50).
    """
    c = (kappa * d_ref) ** 2
    return c * rho * dudn * abs(dudn) * fs


def tau_model_shear_wall(tau_w, fs, sgn):
    """Model shear stress sigma^m from the WALL-MODEL tau_w, Eq. 50.

        sigma^m = sgn * tau_w * f_s

    tau_w is the DYNAMIC wall stress of the nearest wall (the surfel facet's
    tau_out already includes rho), mapped to the cell -- so no rho factor
    and no gradient anywhere. sgn encodes the tensor sign (xi_i eta_j +
    eta_i xi_j) in the channel frame: +1 where the wall-normal is +z (lower
    wall), -1 where it is -z (upper wall), with xi = +x (mean-flow frame;
    testbed wiring). sigma^m <= tau_w holds by construction, which is what
    kills the cut-cell overproduction channel of Eq. 54.
    """
    return sgn * tau_w * fs


def moment_injection(sigma_m, tau_bar: float):
    """Post-collision second-moment shift dPi realizing sigma_m.

    Momentum flux = -stress in the sigma = +2 mu S convention (the viscous
    flux is J_visc = -mu dU/dz), and the steady carried/transported moment
    of a per-step post-collision injection is tau_bar * dPi (deviation 1 in
    the module docstring), so

        dPi = -sigma_m / tau_bar.
    """
    return -sigma_m / tau_bar


def hermite2_xz_lanes(W: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-lane weights h_i of the pure-xz Hermite-2 population term.

        h_i = 9 w_i c_ix c_iz        (12 non-zero lanes on D3Q27)

    satisfying  sum_i h_i = 0,  sum_i h_i c_ia = 0 (a = x, y, z),
    sum_i h_i c_ix c_iz = 1, and every OTHER second moment = 0, so adding
    h_i * dPi to the populations shifts the xz second moment by exactly dPi
    and nothing else. Gate s12 [A] checks this at machine precision.
    The 9 is 1/c_s^4 * 1/2 * 2: Hermite normalisation w_i H^(2)_ixz /
    (2 c_s^4) times the symmetric double (xz + zx).
    """
    return 9.0 * W * C[:, 0] * C[:, 2]


def hermite2_lanes(W, C, t_hat, n_hat):
    """Per-lane weights of the general (t n + n t) Hermite-2 term (S8b).

        h_i = 9 w_i (c_i . t_hat)(c_i . n_hat),     t_hat ⊥ n_hat, unit

    The rotation of `hermite2_xz_lanes` to an arbitrary orthonormal
    (shear-direction, wall-normal) pair -- the tensor M_ab = t_a n_b +
    n_a t_b is symmetric AND traceless (t ⊥ n), so the -c_s^2 delta_ab
    part of H^(2) drops and the weight reduces to the product form.
    Exact on D3Q27 by 4th-moment isotropy (sum w c_a c_b c_c c_d =
    c_s^4 (d_ab d_cd + d_ac d_bd + d_ad d_bc)): adding h_i * dPi shifts
    the second moment by exactly dPi * M_ab, zero mass, zero momentum,
    all other second moments 0 -- for EVERY orientation, t = x, n = z
    recovering hermite2_xz_lanes bit-for-math. Gate s13 [T] pins this at
    machine precision on random orthonormal pairs.

    Args:
        W: (Q,) lattice weights; C: (Q, 3) lattice velocities.
        t_hat, n_hat: (M, 3) per-cell unit vectors (namespace-agnostic).

    Returns:
        (Q, M) per-lane weights.
    """
    ct = C @ t_hat.T                      # (Q, M)
    cn = C @ n_hat.T
    return 9.0 * W[:, None] * ct * cn
