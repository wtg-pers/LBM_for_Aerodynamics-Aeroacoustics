"""Kleine (2022) non-iterative vortex smearing correction — Phase 1 core.

arXiv:2206.05448 (Kleine, Hanifi, Henningson). See
patch_notes/kleine_smearing_correction/00_design.md.

Phase 1 = **non-iterative linearized solve** with a PRESCRIBED straight
semi-infinite trailed-vortex wake. For that wake geometry the deficit (missing)
induced velocity is the Lamb-Oseen form already used by our Dağ correction
(`actuator_line._viscous_core_correction`):

    w_i = -(1/4π) Σ_m (dΓ/dr)_m · [exp(-(d_im/ε_m)²) / d_im] · Δr        (deficit)

The improvement over Dağ's single pass is that the implicit Γ–w coupling is
solved in ONE linear system (Kleine Eq. 5.15) instead of fixed-point relaxation:

    [ I − diag(b†) A ] ΔΓ = Γ† − Γⁿ⁻¹ ,   Γⁿ = Γⁿ⁻¹ + ΔΓ

where  A_ik = ∂w_i/∂Γ_k  (linearized deficit influence),
       b†_j = ∂Γ_j/∂u_n   at the linearization point † (needs ∂C_l/∂α, Phase 0).

The exact finite-segment Φ kernel (Eq. 3.20) and the convected free-vortex wake
are Phase 2 (this module keeps the trusted semi-infinite deficit kernel).

All quantities in lattice units (consistent with the ALM rotor). CPU; N small.
"""
from typing import Callable, Tuple

import numpy as np

_INV4PI = 1.0 / (4.0 * np.pi)
_DEG = 180.0 / np.pi

# =============================================================================
# Phase 2 — exact smeared finite-segment kernel (Kleine Eq. 3.20-3.24)
# =============================================================================
# Used by the free-vortex wake (Phase 2). Verified against the integrand of
# Eq. 3.19 (numerical quadrature) and the semi-infinite Lamb-Oseen limit
# Eq. 3.24. NOTE: the Korean summary (docs/papers_kr/...) transcribed Eq. 3.20
# with a spurious (Z/ε) factor in the 2nd term (which diverges); the form below
# is the PDF's correct one.
from scipy.special import erf as _erf  # noqa: E402


def phi_smeared(r, Z, eps):
    """Φ(r,Z), Kleine Eq. 3.20 — smeared finite vortex-segment kernel.

        Φ = (1/r)[ -Z/√(r²+Z²)·erf(√(r²+Z²)/ε) + exp(-r²/ε²)·erf(Z/ε) ]
    """
    s = np.sqrt(r * r + Z * Z)
    return (1.0 / r) * (-Z / s * _erf(s / eps)
                        + np.exp(-(r * r) / (eps * eps)) * _erf(Z / eps))


def phi_ideal(r, Z):
    """Φⁱ(r,Z), Kleine Eq. 3.21 — singular (ε→0) limit: (1/r)(-Z/√(r²+Z²))."""
    s = np.sqrt(r * r + Z * Z)
    return (1.0 / r) * (-Z / s)


def segment_missing_theta(r, Z_plus, Z_minus, Gamma, eps):
    """Missing (deficit) azimuthal velocity of one finite segment, Kleine Eq. 3.23.

        u_m = (Γ/4π)[ (Φⁱ(Z+) - Φⁱ(Z-)) - (Φ(Z+) - Φ(Z-)) ]   (ideal - smeared)

    Z± = z - z_{j±}: relative axial positions of the segment ends from the eval
    point (perpendicular distance r to the filament). Deficit form ⇒ the LBM-
    resolved (smeared) part is subtracted (no double counting).
    """
    ideal = phi_ideal(r, Z_plus) - phi_ideal(r, Z_minus)
    smear = phi_smeared(r, Z_plus, eps) - phi_smeared(r, Z_minus, eps)
    return _INV4PI * Gamma * (ideal - smear)


def segment_induced_velocity(x, p1, p2, Gamma, eps, r_min=1e-9):
    """Deficit-form induced velocity (3-D vector) at point x from a finite
    smeared vortex segment p1→p2 (circulation Γ along p1→p2).

    Generalizes the z-aligned kernel (Eq. 3.20-3.23) to arbitrary segment
    orientation: project x onto the filament line to get the perpendicular
    distance r and the axial extent Z± of the ends; magnitude from
    `segment_missing_theta`; azimuthal direction = L̂ × r̂.

    Returns np.zeros(3) if x lies on the filament line (r < r_min) — the
    self / core term is excluded by construction (control points are offset
    from the trailed filaments).
    """
    L = p2 - p1
    Llen = np.sqrt(L @ L)
    if Llen < 1e-12:
        return np.zeros(3)
    Lhat = L / Llen
    t = (x - p1) @ Lhat                       # foot parameter
    foot = p1 + t * Lhat
    d_vec = x - foot
    r = np.sqrt(d_vec @ d_vec)
    if r < r_min:
        return np.zeros(3)
    r_hat = d_vec / r
    # Axial positions of the ends along L̂ from the perpendicular foot (z_eval=0).
    Z1 = -t                                    # (p1 - foot)·L̂
    Z2 = Llen - t                              # (p2 - foot)·L̂
    um = segment_missing_theta(r, -Z2, -Z1, Gamma, eps)   # z - z_{j±}
    return um * np.cross(Lhat, r_hat)          # azimuthal direction


# =============================================================================
# Phase 2 step 2 — free-vortex wake (Kleine §3.4)
# =============================================================================

class FreeWake:
    """Per-blade convected trailed-vortex wake GEOMETRY.

    Stores newest-first "rings" of node positions — one node per trailed
    filament (= per control point, matching Phase 1's dΓ/dr discretization).
    Each step: shed a new ring at the blade, convect existing rings by the
    sampled velocity. Circulation is NOT stored — it is applied at influence-
    matrix build time from the CURRENT Γ-gradient (steady hover: exact; a mild
    approximation in strongly unsteady flow, cf. Kleine Eq. 5.5 u_mp/u_mc split).
    """

    def __init__(self, n_w: int = 50):
        self.rings: list = []          # newest-first list of (N,3) [lu]
        self.n_w = int(n_w)

    def shed(self, positions: np.ndarray) -> None:
        self.rings.insert(0, np.array(positions, dtype=np.float64))
        if len(self.rings) > self.n_w:
            self.rings.pop()

    def convect(self, vel_fn, dt: float) -> None:
        """Convect every ring by dt·vel_fn(points). vel_fn: (M,3)->(M,3)."""
        for i in range(len(self.rings)):
            self.rings[i] = self.rings[i] + dt * vel_fn(self.rings[i])

    def __len__(self) -> int:
        return len(self.rings)


def _freewake_influence_loop(control_pos, rings, eps, axial_dir, tail_len=1.0e6):
    """Reference (slow, triple-loop) version of `freewake_influence` — kept for
    cross-checking the vectorized form. Same result, O(nc·ne·L) Python calls.

    B[i,m] = (induced velocity at control i from a UNIT-Γ trailed filament m)
    · axial_dir.  Filament m = polyline through rings[a][m] (newest→oldest) plus
    a semi-infinite straight tail from the oldest node.
    """
    nc = control_pos.shape[0]
    ne = rings[0].shape[0]
    L = len(rings)
    ax = np.asarray(axial_dir, dtype=np.float64)
    ax = ax / np.sqrt(ax @ ax)
    B = np.zeros((nc, ne), dtype=np.float64)
    for m in range(ne):
        pts = [rings[a][m] for a in range(L)]
        # tail direction = last (oldest) segment
        tail = None
        if L >= 2:
            d = pts[-1] - pts[-2]
            dn = np.sqrt(d @ d)
            if dn > 1e-12:
                tail = pts[-1] + (tail_len / dn) * d
        em = float(eps[m])
        for i in range(nc):
            x = control_pos[i]
            v = np.zeros(3)
            for a in range(L - 1):
                v += segment_induced_velocity(x, pts[a], pts[a + 1], 1.0, em)
            if tail is not None:
                v += segment_induced_velocity(x, pts[-1], tail, 1.0, em)
            B[i, m] = v @ ax
    return B


def _build_segments(rings, eps, tail_len):
    """Flatten all trailed filaments into segment arrays.

    Returns P1,P2 (ns,3), seg_eps (ns,), seg_edge (ns,). Segments = consecutive
    rings per edge + one semi-infinite tail per edge (oldest segment direction).
    """
    R = np.asarray(rings, dtype=np.float64)          # (L, ne, 3)
    L, ne, _ = R.shape
    P1s, P2s, edges = [], [], []
    if L >= 2:
        P1s.append(R[:-1].reshape(-1, 3))
        P2s.append(R[1:].reshape(-1, 3))
        edges.append(np.tile(np.arange(ne), L - 1))
        d = R[-1] - R[-2]
        dn = np.sqrt((d * d).sum(1))
        dn = np.where(dn > 1e-12, dn, 1.0)
        tail = R[-1] + (tail_len / dn)[:, None] * d
        P1s.append(R[-1]); P2s.append(tail); edges.append(np.arange(ne))
    else:                                            # single ring → tail only undefined
        return (np.zeros((0, 3)), np.zeros((0, 3)),
                np.zeros(0), np.zeros(0, dtype=int))
    P1 = np.concatenate(P1s); P2 = np.concatenate(P2s)
    seg_edge = np.concatenate(edges)
    seg_eps = np.asarray(eps, dtype=np.float64)[seg_edge]
    return P1, P2, seg_eps, seg_edge


def _seg_vz_batch(X, P1, P2, seg_eps, axial, r_min=1e-9):
    """Axial-dir induced velocity at each control X (nc,3) from each unit-Γ
    segment P1→P2 (ns,3). Vectorized → Vz (nc, ns)."""
    Lv = P2 - P1
    Llen = np.sqrt((Lv * Lv).sum(1))
    safe = Llen > 1e-12
    Lhat = Lv / np.where(safe, Llen, 1.0)[:, None]              # (ns,3)
    Xc = X[:, None, :]; P1c = P1[None, :, :]; Lc = Lhat[None, :, :]
    t = ((Xc - P1c) * Lc).sum(2)                                # (nc,ns)
    foot = P1c + t[:, :, None] * Lc
    dvec = Xc - foot
    r = np.sqrt((dvec * dvec).sum(2))                           # (nc,ns)
    rhat = dvec / np.where(r > r_min, r, 1.0)[:, :, None]
    Z1 = -t; Z2 = Llen[None, :] - t
    um = segment_missing_theta(np.where(r > r_min, r, 1.0),
                               -Z2, -Z1, 1.0, seg_eps[None, :])
    cross = np.cross(np.broadcast_to(Lc, dvec.shape), rhat)     # (nc,ns,3)
    V = um[:, :, None] * cross
    bad = (r < r_min) | (~safe[None, :])
    V = np.where(bad[:, :, None], 0.0, V)
    return (V * axial[None, None, :]).sum(2)                    # (nc,ns)


def freewake_influence(control_pos, rings, eps, axial_dir, tail_len=1.0e6):
    """Vectorized B[i,m] = (induced velocity at control i from unit-Γ trailed
    filament m) · axial_dir.  Caller forms A = Δr · (B @ G); straight semi-
    infinite limit == Phase 1 `influence_matrix`. (Cross-checked vs
    `_freewake_influence_loop`.)"""
    ne = rings[0].shape[0]
    ax = np.asarray(axial_dir, dtype=np.float64)
    ax = ax / np.sqrt(ax @ ax)
    P1, P2, seg_eps, seg_edge = _build_segments(rings, eps, tail_len)
    if P1.shape[0] == 0:
        return np.zeros((control_pos.shape[0], ne))
    Vz = _seg_vz_batch(np.asarray(control_pos, dtype=np.float64),
                       P1, P2, seg_eps, ax)                     # (nc, ns)
    S = np.zeros((Vz.shape[1], ne))                             # edge reduction
    S[np.arange(Vz.shape[1]), seg_edge] = 1.0
    return Vz @ S                                               # (nc, ne)


def _gradient_matrix(r: np.ndarray) -> np.ndarray:
    """G such that (G @ Γ) == np.gradient(Γ, r), for non-uniform r.

    Built by applying np.gradient to the unit columns so the discrete operator
    matches the Dağ correction exactly (one-sided at the ends).
    """
    n = len(r)
    G = np.zeros((n, n), dtype=np.float64)
    e = np.zeros(n)
    for k in range(n):
        e[:] = 0.0
        e[k] = 1.0
        G[:, k] = np.gradient(e, r)
    return G


def influence_matrix(r: np.ndarray, eps: np.ndarray, dr: float) -> np.ndarray:
    """A_ik = ∂w_i/∂Γ_k  for the semi-infinite straight trailed wake.

    w = A @ Γ reproduces the Dağ deficit downwash linearly in Γ:
        A = -(1/4π)·Δr · (Kmat @ G),
        Kmat_im = exp(-((r_i-r_m)/ε_m)²)/(r_i-r_m)   (signed, 0 on diagonal),
        G = gradient matrix (dΓ/dr).
    """
    n = len(r)
    d = r[:, None] - r[None, :]                 # (i,m) signed distance  [lu]
    far = np.abs(d) > 1e-9
    safe_d = np.where(far, d, 1.0)
    eps_m = eps[None, :]                          # core at trailed vortex m
    Kmat = np.where(far, np.exp(-(d / eps_m) ** 2) / safe_d, 0.0)
    G = _gradient_matrix(r)
    return (-_INV4PI * dr) * (Kmat @ G)          # (n,n)


def _triangle(u_n: np.ndarray, u_tan: np.ndarray, twist_deg: np.ndarray):
    """(u_rel, phi_deg, alpha_deg) from axial u_n and tangential u_tan."""
    u_rel = np.sqrt(u_n * u_n + u_tan * u_tan)
    phi = np.arctan2(u_n, u_tan)                  # [rad]
    alpha_deg = twist_deg - phi * _DEG
    return u_rel, phi, alpha_deg


def correct_noniterative(
    r: np.ndarray,
    chord: np.ndarray,
    dr: float,
    eps: np.ndarray,
    u_n: np.ndarray,
    u_tan: np.ndarray,
    twist_deg: np.ndarray,
    cl_eval: Callable[[np.ndarray], np.ndarray],
    dcl_dalpha_eval: Callable[[np.ndarray], np.ndarray],
    Gamma_prev: np.ndarray,
    A: np.ndarray = None,
    active: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One non-iterative Kleine correction step (Kleine Eq. 5.4/5.14/5.15).

    Args:
        r, chord, eps: per-marker geometry  [lu].
        dr: marker spacing  [lu].
        u_n: sampled axial (normal) velocity  [lu/lt].
        u_tan: tangential rel. velocity  ωr − u_θ  [lu/lt]  (caller forms it).
        twist_deg: blade twist/pitch per marker  [deg].
        cl_eval(alpha_deg[]) -> C_l[]: bound to Re/Mach/airfoil.
        dcl_dalpha_eval(alpha_deg[]) -> dC_l/dα[]  [1/rad].
        Gamma_prev: Γⁿ⁻¹ per marker  [lu²/lt].
        A: optional precomputed influence matrix (else built here).

    Returns:
        (u_n_corrected, alpha_deg_corrected, Gamma_new, w_corr).
    """
    if A is None:
        A = influence_matrix(r, eps, dr)
    n = len(r)

    # ── Linearization point † (Kleine Eq. 5.1-5.4): use Γⁿ⁻¹ ──
    w_prev = A @ Gamma_prev                                   # missing downwash
    u_n_d = u_n + w_prev
    u_rel_d, _, alpha_d = _triangle(u_n_d, u_tan, twist_deg)
    CL_d = np.asarray(cl_eval(alpha_d), dtype=np.float64)
    dCL_d = np.asarray(dcl_dalpha_eval(alpha_d), dtype=np.float64)   # [1/rad]
    Gamma_dag = 0.5 * chord * u_rel_d * CL_d                  # Γ†

    # ── Sensitivity b† = ∂Γ/∂u_n  at † (Kleine Eq. A5/A6 analogue) ──
    #   Γ = ½ c u_rel C_l(α);  ∂u_rel/∂u_n = u_n/u_rel;
    #   ∂α/∂u_n = -u_tan/u_rel²  →  b = ½ c (u_n C_l - u_tan dC_l/dα)/u_rel
    safe_url = np.where(u_rel_d > 1e-12, u_rel_d, 1.0)
    b = 0.5 * chord * (u_n_d * CL_d - u_tan * dCL_d) / safe_url

    # Inactive markers (root / u_rel→0): zero feedback so their rows are identity
    # (ΔΓ=0) and they don't destabilize the solve.
    if active is not None:
        b = np.where(active, b, 0.0)
        Gamma_dag = np.where(active, Gamma_dag, 0.0)

    # ── Non-iterative linear solve (Kleine Eq. 5.15) ──
    M = np.eye(n) - (b[:, None] * A)                         # I - diag(b) A
    rhs = Gamma_dag - Gamma_prev
    dGamma = np.linalg.solve(M, rhs)
    Gamma_new = Gamma_prev + dGamma

    # ── Apply: corrected downwash → triangle → forces (caller) ──
    w_corr = A @ Gamma_new
    u_n_c = u_n + w_corr
    _, _, alpha_c = _triangle(u_n_c, u_tan, twist_deg)
    return u_n_c, alpha_c, Gamma_new, w_corr


def correct_iterative_reference(
    r, chord, dr, eps, u_n, u_tan, twist_deg, cl_eval, Gamma_prev,
    A: np.ndarray = None, max_iter: int = 200, tol: float = 1e-12,
):
    """Fixed-point reference (for validating the non-iterative solve).

    Iterates  Γ ← ½ c u_rel(Γ) C_l(α(Γ))  to convergence (no relaxation needed
    for these synthetic linear-C_l tests). Returns Gamma_converged.
    """
    if A is None:
        A = influence_matrix(r, eps, dr)
    Gamma = Gamma_prev.copy()
    for _ in range(max_iter):
        w = A @ Gamma
        u_rel, _, alpha = _triangle(u_n + w, u_tan, twist_deg)
        CL = np.asarray(cl_eval(alpha), dtype=np.float64)
        Gamma_new = 0.5 * chord * u_rel * CL
        if np.max(np.abs(Gamma_new - Gamma)) < tol:
            Gamma = Gamma_new
            break
        Gamma = Gamma_new
    return Gamma
