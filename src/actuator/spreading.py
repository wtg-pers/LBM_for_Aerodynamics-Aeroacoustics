"""
Marker-to-Grid Force Spreading for Actuator Line Model

This module distributes the aerodynamic forces computed at marker particles
onto the LBM grid using a smooth Gaussian regularization kernel (Eq. 13).

Physical Concept:
================
The BEM calculation yields a point force F_j^AL at each marker particle j.
These point forces must be converted into a smooth volumetric body force
field F(x) on the Eulerian LBM grid.

Watanabe et al. Eq. 13:

    F(x) = Σ_{j=1}^{N_p}  -F_j^{AL} / (π^{3/2} · ε³) · exp(-(d_j/ε)²)

where:
    N_p     = number of actuator marker particles
    F_j^AL  = aerodynamic force at marker j in global frame  [N or lattice force]
              = (F_n, F_θ·cos(θ), -F_θ·sin(θ))
    d_j     = |x - x_j|  distance from grid point x to marker j  [m or lu]
    ε       = Gaussian filter width = max(c_a/4, 2·Δx)  [m or lu]

The NEGATIVE SIGN represents Newton's third law:
    Blade receives F^AL from fluid → Fluid receives -F^AL from blade

The Gaussian kernel η_ε(d) = exp(-d²/ε²) / (π^{3/2}·ε³) satisfies:
    ∫ η_ε(d) dV = 1     (volume integral over all space)

This ensures the total force applied to the fluid equals the sum of
marker forces: ∫ F(x) dV = -Σ F_j^AL · Δr_j

In Lattice Units:
    Δx = 1, so F(x) has units of [force/volume] = [lattice force / lu³]
    This is exactly the body force F used in Guo forcing.

Support Radius:
    The kernel is truncated at r_cut = n_cut · ε (default n_cut = 3).
    exp(-9) ≈ 1.2×10⁻⁴, ensuring negligible error from truncation.

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026 (Eq. 13)
    - Sørensen & Shen, J. Fluids Eng. 124, 393-399, 2002
    - Martínez-Tossas & Meneveau, J. Fluid Mech. 863, 269-292, 2019

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


# =============================================================================
# §1. Single-Marker Force Spreading
# =============================================================================

def _cylindrical_radius(gx, gy, gz, axis, hub):
    """Perpendicular distance of grid node(s) from the rotor axis line.

    axis: unit rotation axis (3,); hub: a point on the axis (3,) — same
    lattice frame as the marker positions. Works elementwise on arrays.
    """
    px = gx - hub[0]; py = gy - hub[1]; pz = gz - hub[2]
    dot = px * axis[0] + py * axis[1] + pz * axis[2]      # axial coord
    rx = px - dot * axis[0]; ry = py - dot * axis[1]; rz = pz - dot * axis[2]
    return np.sqrt(rx * rx + ry * ry + rz * rz)


def compute_radial_scales(
    domain_shape, marker_positions, marker_epsilon, marker_active,
    axis, hub, r_root, r_tip, n_cut: float = 3.0
) -> 'npt.NDArray':
    """Per-marker renormalisation factor for radial truncation at [r_root, r_tip].

    Merabet & Laurendeau (2021): the source term is radially truncated at the
    geometric blade limits (root/tip) and re-normalised so the marker's total
    force is conserved but deposited only within the blade's radial span. For a
    marker whose Gaussian straddles a limit, the kept (in-span) nodes are scaled
    by S_all/S_kept so Σ(kept η·dV) == Σ(all η·dV) — i.e. the SAME total force
    the untruncated spread would deposit, just relocated inboard.

    Returns scales (N_markers,); 1.0 for markers whose stencil lies fully within
    [r_root, r_tip] (interior markers → no change, path stays bit-identical).
    """
    Nx, Ny, Nz = domain_shape
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    hub = np.asarray(hub, dtype=np.float64)
    n = marker_positions.shape[0]
    scales = np.ones(n, dtype=np.float64)
    if marker_active is None:
        marker_active = np.ones(n, dtype=bool)
    # Only markers whose Gaussian can actually reach a radial limit need the
    # (costly) stencil sum — a marker at cylindrical radius r_m is untruncated
    # (scale ≡ 1) unless r_m + n_cut·ε > r_tip or r_m − n_cut·ε < r_root. This
    # leaves only the few outer/inner markers per blade; the rest stay exactly
    # 1.0. Keeps the per-step host cost ~O(boundary markers), not O(all).
    r_m = _cylindrical_radius(
        marker_positions[:, 0], marker_positions[:, 1], marker_positions[:, 2],
        axis, hub)
    reach = n_cut * np.asarray(marker_epsilon, dtype=np.float64)
    near_limit = (r_m + reach > r_tip) | (r_m - reach < r_root)
    for j in range(n):
        if not marker_active[j] or not near_limit[j]:
            continue
        eps = float(marker_epsilon[j])
        r_cut = n_cut * eps
        p = marker_positions[j]
        ix0 = max(int(np.floor(p[0] - r_cut)), 0)
        ix1 = min(int(np.ceil(p[0] + r_cut)), Nx - 1)
        iy0 = max(int(np.floor(p[1] - r_cut)), 0)
        iy1 = min(int(np.ceil(p[1] + r_cut)), Ny - 1)
        iz0 = max(int(np.floor(p[2] - r_cut)), 0)
        iz1 = min(int(np.ceil(p[2] + r_cut)), Nz - 1)
        if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
            continue
        gx, gy, gz = np.meshgrid(
            np.arange(ix0, ix1 + 1, dtype=np.float64),
            np.arange(iy0, iy1 + 1, dtype=np.float64),
            np.arange(iz0, iz1 + 1, dtype=np.float64), indexing='ij')
        d_sq = (gx - p[0]) ** 2 + (gy - p[1]) ** 2 + (gz - p[2]) ** 2
        sph = d_sq <= r_cut * r_cut
        if not np.any(sph):
            continue
        norm = 1.0 / (np.pi ** 1.5 * eps ** 3)
        eta = np.where(sph, norm * np.exp(-d_sq / (eps * eps)), 0.0)
        S_all = float(eta.sum())
        rn = _cylindrical_radius(gx, gy, gz, axis, hub)
        keep = (rn >= r_root) & (rn <= r_tip)
        S_kept = float(eta[keep].sum())
        scales[j] = S_all / S_kept if S_kept > 1e-30 else 1.0
    return scales


def compute_radial_scales_batch(
    xp, domain_shape, marker_positions, marker_epsilon, marker_active,
    axis, hub, r_root, r_tip, n_cut: float = 3.0
) -> 'npt.NDArray':
    """Vectorised (xp-agnostic) compute_radial_scales — same node selection.

    The reference implementation loops the near-limit markers on the HOST with
    a numpy meshgrid stencil sum per marker; at D40 (~30+ near-limit markers,
    root boxes ~41^3) this cost ~30 ms per ALM sub-step and made the archB
    (radial-truncation) runs ~1.4x slower per coarse step (profiled: spread
    5.1 -> 37.4 ms/call). Here all selected markers are batched into ONE padded
    (M, K, K, K) broadcast on `xp` (cupy -> GPU, where the scales are consumed
    anyway; returns a device array). Box floor/ceil + domain clip + sphere cut
    are identical to the reference; only the summation ORDER differs, so
    results agree to fp64 round-off (gate: alm_radial_scales_gate.py).
    """
    Nx, Ny, Nz = domain_shape
    axis_np = np.asarray(axis, dtype=np.float64)
    axis_np = axis_np / (np.linalg.norm(axis_np) + 1e-30)
    hub_np = np.asarray(hub, dtype=np.float64)
    pos_np = np.asarray(marker_positions, dtype=np.float64)
    eps_np = np.asarray(marker_epsilon, dtype=np.float64)
    n = pos_np.shape[0]
    if marker_active is None:
        marker_active = np.ones(n, dtype=bool)

    # near-limit selection: identical to the reference implementation.
    r_m = _cylindrical_radius(
        pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], axis_np, hub_np)
    reach = n_cut * eps_np
    sel = np.where(np.asarray(marker_active, dtype=bool)
                   & ((r_m + reach > r_tip) | (r_m - reach < r_root)))[0]
    out = xp.ones(n, dtype=xp.float64)
    if sel.size == 0:
        return out

    p = pos_np[sel]                                   # (M, 3)
    eps = eps_np[sel]                                 # (M,)
    rc = n_cut * eps                                  # (M,)
    # Per-marker inclusive box [floor(p-rc), ceil(p+rc)] clipped to the domain
    # (same as the reference); padded to the largest box K.
    lo = np.floor(p - rc[:, None])
    hi = np.ceil(p + rc[:, None])
    for d, N in enumerate((Nx, Ny, Nz)):
        lo[:, d] = np.clip(lo[:, d], 0, N - 1)
        hi[:, d] = np.clip(hi[:, d], 0, N - 1)
    K = int((hi - lo).max()) + 1

    lo_x = xp.asarray(lo); hi_x = xp.asarray(hi)
    p_x = xp.asarray(p); eps_x = xp.asarray(eps); rc_x = xp.asarray(rc)
    ar = xp.arange(K, dtype=xp.float64)               # (K,)

    gx = lo_x[:, 0, None] + ar[None, :]               # (M, K)
    gy = lo_x[:, 1, None] + ar[None, :]
    gz = lo_x[:, 2, None] + ar[None, :]
    mx = (gx <= hi_x[:, 0, None])[:, :, None, None]   # inside per-marker box
    my = (gy <= hi_x[:, 1, None])[:, None, :, None]
    mz = (gz <= hi_x[:, 2, None])[:, None, None, :]

    dx = (gx - p_x[:, 0, None])[:, :, None, None]     # (M, K, 1, 1)
    dy = (gy - p_x[:, 1, None])[:, None, :, None]
    dz = (gz - p_x[:, 2, None])[:, None, None, :]
    d_sq = dx * dx + dy * dy + dz * dz                # (M, K, K, K)
    sph = (mx & my & mz) & (d_sq <= (rc_x ** 2)[:, None, None, None])

    norm = 1.0 / (np.pi ** 1.5) / (eps_x ** 3)        # (M,)
    eta = xp.where(
        sph, norm[:, None, None, None]
        * xp.exp(-d_sq / (eps_x ** 2)[:, None, None, None]), 0.0)
    S_all = eta.sum(axis=(1, 2, 3))                   # (M,)

    ax_x = xp.asarray(axis_np); hub_x = xp.asarray(hub_np)
    hx = (gx - hub_x[0])[:, :, None, None]
    hy = (gy - hub_x[1])[:, None, :, None]
    hz = (gz - hub_x[2])[:, None, None, :]
    dot = hx * ax_x[0] + hy * ax_x[1] + hz * ax_x[2]
    rn = xp.sqrt(xp.maximum(hx * hx + hy * hy + hz * hz - dot * dot, 0.0))
    S_kept = xp.where((rn >= r_root) & (rn <= r_tip), eta, 0.0).sum(axis=(1, 2, 3))

    sc_sel = xp.where((S_kept > 1e-30) & (S_all > 0.0),
                      S_all / xp.maximum(S_kept, 1e-300), 1.0)
    out[xp.asarray(sel)] = sc_sel
    return out


def spread_force_single_marker(
    F_grid: 'npt.NDArray',
    marker_pos: 'npt.NDArray',
    marker_force: 'npt.NDArray',
    epsilon: float,
    n_cut: float = 3.0,
    radial: Optional[tuple] = None
) -> None:
    """Spread a single marker's force onto the grid (in-place accumulation)

    Implements Eq. 13 for one marker particle:
        F(x) += -F_j^{AL} · η_ε(|x - x_j|)

    where η_ε(d) = 1/(π^{3/2}·ε³) · exp(-(d/ε)²)

    The negative sign (Newton's third law) is applied INSIDE this function.

    Args:
        F_grid: Body force field, shape (3, Nx, Ny, Nz)  [lattice force/lu³]
                Modified IN-PLACE by accumulation.
        marker_pos: Marker position (x, y, z)  [lu]
        marker_force: Force at marker F^AL (F_x, F_y, F_z)  [lattice force]
        epsilon: Gaussian filter width ε  [lu]
        n_cut: Cutoff in units of ε  [dimensionless]
    """
    _, Nx, Ny, Nz = F_grid.shape
    eps_sq = epsilon * epsilon                          # [lu²]
    r_cut = n_cut * epsilon                             # [lu]
    r_cut_sq = r_cut * r_cut                            # [lu²]

    # Gaussian normalization: 1 / (π^{3/2} · ε³)  [1/lu³]
    norm = 1.0 / (np.pi ** 1.5 * epsilon ** 3)

    # --- Bounding box of affected grid nodes ---
    ix_min = max(int(np.floor(marker_pos[0] - r_cut)), 0)
    ix_max = min(int(np.ceil(marker_pos[0] + r_cut)), Nx - 1)
    iy_min = max(int(np.floor(marker_pos[1] - r_cut)), 0)
    iy_max = min(int(np.ceil(marker_pos[1] + r_cut)), Ny - 1)
    iz_min = max(int(np.floor(marker_pos[2] - r_cut)), 0)
    iz_max = min(int(np.ceil(marker_pos[2] + r_cut)), Nz - 1)

    # Grid coordinate arrays
    ix = np.arange(ix_min, ix_max + 1, dtype=np.float64)
    iy = np.arange(iy_min, iy_max + 1, dtype=np.float64)
    iz = np.arange(iz_min, iz_max + 1, dtype=np.float64)

    if len(ix) == 0 or len(iy) == 0 or len(iz) == 0:
        return

    # 3D meshgrid
    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing='ij')

    # Squared distance
    dx = gx - marker_pos[0]                 # [lu]
    dy = gy - marker_pos[1]                 # [lu]
    dz = gz - marker_pos[2]                 # [lu]
    d_sq = dx * dx + dy * dy + dz * dz      # [lu²]

    # Spherical cutoff mask
    mask = d_sq <= r_cut_sq

    if not np.any(mask):
        return

    # Gaussian kernel values at valid nodes
    eta = np.zeros_like(d_sq)
    eta[mask] = norm * np.exp(-d_sq[mask] / eps_sq)     # [1/lu³]

    # Radial truncation + renormalisation (Merabet 2021): zero nodes outside
    # the blade's geometric span [r_root, r_tip] and rescale the kept nodes so
    # the marker's total deposited force is unchanged (relocated inboard).
    if radial is not None:
        axis, hub, r_root, r_tip, scale = radial
        rn = _cylindrical_radius(gx, gy, gz, axis, hub)
        keep = (rn >= r_root) & (rn <= r_tip)
        eta = np.where(keep, eta * scale, 0.0)

    # Accumulate: F_grid(x) += -F^AL · η_ε(|x - x_j|)
    # Negative sign = Newton's third law (reaction on fluid)
    slc = (slice(None), slice(ix_min, ix_max+1),
           slice(iy_min, iy_max+1), slice(iz_min, iz_max+1))
    for d in range(3):
        F_grid[d, ix_min:ix_max+1, iy_min:iy_max+1, iz_min:iz_max+1] += (
            -marker_force[d] * eta                       # [lattice force / lu³]
        )


def spread_force_single_marker_aniso(
    F_grid: 'npt.NDArray',
    marker_pos: 'npt.NDArray',
    marker_force: 'npt.NDArray',
    ec: 'npt.NDArray', et: 'npt.NDArray', er: 'npt.NDArray',
    eps_c: float, eps_t: float, eps_r: float,
    n_cut: float = 3.0,
) -> None:
    """Anisotropic Gaussian spread of one marker (Natelson 2026 Eq. 7).

        η(x) = 1/(π^{3/2} ε_c ε_t ε_r) · exp(-(d_c/ε_c)² -(d_t/ε_t)² -(d_r/ε_r)²)

    with d_c/d_t/d_r the components of (x - x_j) along the blade-local chord (ec),
    thickness (et), and radial (er) unit vectors. Ellipsoidal cutoff at n_cut.
    Reduces EXACTLY to the isotropic spread_force_single_marker when
    ec/et/er are orthonormal and ε_c=ε_t=ε_r. Negative sign (Newton's 3rd law)
    applied inside. (Radial truncation is not combined here — see Step A note.)
    """
    _, Nx, Ny, Nz = F_grid.shape
    eps_max = max(eps_c, eps_t, eps_r)                  # [lu]
    r_cut = n_cut * eps_max                             # [lu] (bounding box)

    ix_min = max(int(np.floor(marker_pos[0] - r_cut)), 0)
    ix_max = min(int(np.ceil(marker_pos[0] + r_cut)), Nx - 1)
    iy_min = max(int(np.floor(marker_pos[1] - r_cut)), 0)
    iy_max = min(int(np.ceil(marker_pos[1] + r_cut)), Ny - 1)
    iz_min = max(int(np.floor(marker_pos[2] - r_cut)), 0)
    iz_max = min(int(np.ceil(marker_pos[2] + r_cut)), Nz - 1)
    if ix_max < ix_min or iy_max < iy_min or iz_max < iz_min:
        return

    gx, gy, gz = np.meshgrid(
        np.arange(ix_min, ix_max + 1, dtype=np.float64),
        np.arange(iy_min, iy_max + 1, dtype=np.float64),
        np.arange(iz_min, iz_max + 1, dtype=np.float64), indexing='ij')
    dx = gx - marker_pos[0]; dy = gy - marker_pos[1]; dz = gz - marker_pos[2]

    # Project the node offset onto the blade-local axes
    d_c = dx * ec[0] + dy * ec[1] + dz * ec[2]         # [lu]
    d_t = dx * et[0] + dy * et[1] + dz * et[2]
    d_r = dx * er[0] + dy * er[1] + dz * er[2]
    arg = (d_c / eps_c) ** 2 + (d_t / eps_t) ** 2 + (d_r / eps_r) ** 2

    mask = arg <= (n_cut * n_cut)                       # ellipsoidal cutoff
    if not np.any(mask):
        return
    norm = 1.0 / (np.pi ** 1.5 * eps_c * eps_t * eps_r)  # [1/lu³]
    eta = np.zeros_like(arg)
    eta[mask] = norm * np.exp(-arg[mask])              # [1/lu³]

    for d in range(3):
        F_grid[d, ix_min:ix_max+1, iy_min:iy_max+1, iz_min:iz_max+1] += (
            -marker_force[d] * eta)


# =============================================================================
# §2. Batch Force Spreading (All Markers)
# =============================================================================

def spread_forces_to_grid(
    domain_shape: Tuple[int, int, int],
    marker_positions: 'npt.NDArray',
    marker_forces: 'npt.NDArray',
    marker_epsilon: 'npt.NDArray',
    marker_active: Optional['npt.NDArray'] = None,
    n_cut: float = 3.0,
    F_grid: Optional['npt.NDArray'] = None,
    radial_trunc: Optional[dict] = None,
    aniso: Optional[dict] = None
) -> 'npt.NDArray':
    """Spread all marker forces onto the LBM grid (Eq. 13)

    This is the MAIN entry point for force spreading in the AL model.
    Produces the body force field F(x) that enters the Guo forcing scheme.

    Physical Process (Eq. 13):
        F(x) = Σ_j  -F_j^{AL} / (π^{3/2}·ε_j³) · exp(-(d_j/ε_j)²)

    The output F_grid is used directly as the body force in:
        - Guo forcing: S_i computation
        - Velocity correction: u = (Σξ_i f_i + F/2) / ρ

    Args:
        domain_shape: (Nx, Ny, Nz)  [lu]
        marker_positions: shape (N_markers, 3)  [lu]
        marker_forces: shape (N_markers, 3) — F^AL per marker  [lattice force]
        marker_epsilon: shape (N_markers,) — ε per marker  [lu]
        marker_active: shape (N_markers,) — bool mask (default: all True)
        n_cut: Cutoff in units of ε  [dimensionless]
        F_grid: Optional pre-allocated array, shape (3, Nx, Ny, Nz)
                If provided, forces are ACCUMULATED (not zeroed first).
                Useful for combining AL force with other body forces.

    Returns:
        F_grid: Body force field, shape (3, Nx, Ny, Nz)  [lattice force / lu³]
    """
    Nx, Ny, Nz = domain_shape
    n_markers = marker_positions.shape[0]

    # Allocate or reuse output array
    if F_grid is None:
        F_grid = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)

    # Default: all markers active
    if marker_active is None:
        marker_active = np.ones(n_markers, dtype=bool)

    # ── Anisotropic path (Natelson Eq. 7) — separate loop; radial truncation is
    # not combined in Step A. aniso carries per-marker frames/widths. ──
    if aniso is not None:
        ec = aniso['ec']; et = aniso['et']; er = aniso['er']
        eps_c = aniso['eps_c']; eps_t = aniso['eps_t']; eps_r = aniso['eps_r']
        for j in range(n_markers):
            if not marker_active[j]:
                continue
            if np.sum(marker_forces[j] ** 2) < 1e-60:
                continue
            spread_force_single_marker_aniso(
                F_grid, marker_positions[j], marker_forces[j],
                ec[j], et[j], er[j],
                float(eps_c[j]), float(eps_t[j]), float(eps_r[j]), n_cut=n_cut)
        return F_grid

    # Radial truncation: precompute per-marker renorm scales once (shared with
    # the GPU path via compute_radial_scales for identical results).
    _scales = None
    if radial_trunc is not None:
        _scales = compute_radial_scales(
            domain_shape, marker_positions, marker_epsilon, marker_active,
            radial_trunc['axis'], radial_trunc['hub'],
            radial_trunc['r_root'], radial_trunc['r_tip'], n_cut=n_cut)
        _ax = np.asarray(radial_trunc['axis'], dtype=np.float64)
        _ax = _ax / (np.linalg.norm(_ax) + 1e-30)
        _hub = np.asarray(radial_trunc['hub'], dtype=np.float64)

    # Accumulate force from each active marker
    for j in range(n_markers):
        if not marker_active[j]:
            continue

        # Skip zero-force markers (saves computation)
        force_mag_sq = np.sum(marker_forces[j] ** 2)
        if force_mag_sq < 1e-60:
            continue

        _radial = None
        if _scales is not None:
            _radial = (_ax, _hub, radial_trunc['r_root'],
                       radial_trunc['r_tip'], _scales[j])

        spread_force_single_marker(
            F_grid,
            marker_positions[j],
            marker_forces[j],
            float(marker_epsilon[j]),
            n_cut=n_cut,
            radial=_radial
        )

    return F_grid  # (3, Nx, Ny, Nz)  [lattice force / lu³]


# =============================================================================
# §3. Optimized Batch Spreading (Uniform ε)
# =============================================================================

def spread_forces_uniform_epsilon(
    domain_shape: Tuple[int, int, int],
    marker_positions: 'npt.NDArray',
    marker_forces: 'npt.NDArray',
    epsilon_uniform: float,
    marker_active: Optional['npt.NDArray'] = None,
    n_cut: float = 3.0,
    F_grid: Optional['npt.NDArray'] = None
) -> 'npt.NDArray':
    """Optimized spreading when all markers share the same ε

    Precomputes the Gaussian normalization constant and stencil size.
    ~1.5x faster than general version for N_markers > 100.

    Args:
        domain_shape: (Nx, Ny, Nz)  [lu]
        marker_positions: shape (N_markers, 3)  [lu]
        marker_forces: shape (N_markers, 3) — F^AL  [lattice force]
        epsilon_uniform: Uniform ε for all markers  [lu]
        marker_active: bool mask  (default: all True)
        n_cut: Cutoff  [dimensionless]
        F_grid: Optional pre-allocated output

    Returns:
        F_grid: shape (3, Nx, Ny, Nz)  [lattice force / lu³]
    """
    Nx, Ny, Nz = domain_shape
    n_markers = marker_positions.shape[0]

    if F_grid is None:
        F_grid = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)

    if marker_active is None:
        marker_active = np.ones(n_markers, dtype=bool)

    # Precompute constants (shared for all markers)
    eps_sq = epsilon_uniform ** 2                        # [lu²]
    r_cut = n_cut * epsilon_uniform                      # [lu]
    r_cut_sq = r_cut * r_cut                            # [lu²]
    norm = 1.0 / (np.pi ** 1.5 * epsilon_uniform ** 3)  # [1/lu³]
    half = int(np.ceil(r_cut))

    # Precompute stencil offsets
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    ox, oy, oz = np.meshgrid(offsets, offsets, offsets, indexing='ij')

    for j in range(n_markers):
        if not marker_active[j]:
            continue

        force_mag_sq = np.sum(marker_forces[j] ** 2)
        if force_mag_sq < 1e-60:
            continue

        xj, yj, zj = marker_positions[j]

        # Grid indices of stencil center (nearest node)
        ix0 = int(np.round(xj))
        iy0 = int(np.round(yj))
        iz0 = int(np.round(zj))

        # Absolute grid positions
        gx = ix0 + ox
        gy = iy0 + oy
        gz = iz0 + oz

        # Domain bounds
        valid = (
            (gx >= 0) & (gx < Nx) &
            (gy >= 0) & (gy < Ny) &
            (gz >= 0) & (gz < Nz)
        )

        # Actual distances (fractional marker position)
        dx = gx - xj                           # [lu]
        dy = gy - yj                           # [lu]
        dz = gz - zj                           # [lu]
        d_sq = dx * dx + dy * dy + dz * dz      # [lu²]

        # Combined mask: in domain AND within cutoff sphere
        active_mask = valid & (d_sq <= r_cut_sq)

        if not np.any(active_mask):
            continue

        # Gaussian kernel
        eta = np.zeros_like(d_sq)
        eta[active_mask] = norm * np.exp(-d_sq[active_mask] / eps_sq)

        # Absolute grid indices for accumulation
        gx_int = gx.astype(int)
        gy_int = gy.astype(int)
        gz_int = gz.astype(int)

        # Flatten for scatter accumulation
        idx_x = gx_int[active_mask]
        idx_y = gy_int[active_mask]
        idx_z = gz_int[active_mask]
        eta_flat = eta[active_mask]

        # Accumulate: F_grid(x) += -F^AL_j · η_ε(d)
        for d in range(3):
            np.add.at(
                F_grid[d],
                (idx_x, idx_y, idx_z),
                -marker_forces[j, d] * eta_flat          # [lattice force / lu³]
            )

    return F_grid


# =============================================================================
# §4. Force Conservation Check
# =============================================================================

def check_force_conservation(
    F_grid: 'npt.NDArray',
    marker_forces: 'npt.NDArray',
    marker_active: Optional['npt.NDArray'] = None,
    dx: float = 1.0
) -> dict:
    """Verify that the total grid force matches the total marker force

    Conservation requirement:
        ∫ F(x) dV = -Σ_j F_j^{AL}     (Newton's third law)

    In lattice units (Δx = 1):
        Σ_x F_grid(x) = -Σ_j F_j^{AL}

    Args:
        F_grid: Body force field, shape (3, Nx, Ny, Nz)  [lattice force / lu³]
        marker_forces: shape (N_markers, 3)  [lattice force]
        marker_active: Optional bool mask
        dx: Grid spacing (default 1.0 for lattice units)  [lu]

    Returns:
        dict with:
            - F_grid_total: Total force on grid (3,)
            - F_marker_total: Total marker force (3,)
            - relative_error: |F_grid + F_marker| / |F_marker| per component
            - is_conserved: True if relative error < 1e-3
    """
    # Total force on grid = Σ F_grid · dx³
    dV = dx ** 3                                        # [lu³]
    F_grid_total = np.array([
        np.sum(F_grid[d]) * dV for d in range(3)
    ])

    # Total marker force
    if marker_active is not None:
        active_forces = marker_forces[marker_active]
    else:
        active_forces = marker_forces

    F_marker_total = np.sum(active_forces, axis=0)

    # Conservation: F_grid_total should equal -F_marker_total
    # → F_grid_total + F_marker_total should be ≈ 0
    residual = F_grid_total + F_marker_total
    magnitude = np.abs(F_marker_total) + 1e-30
    rel_error = np.abs(residual) / magnitude

    return {
        'F_grid_total': F_grid_total,
        'F_marker_total': F_marker_total,
        'residual': residual,
        'relative_error': rel_error,
        'is_conserved': bool(np.all(rel_error < 1e-3)),
    }


# =============================================================================
# §5. GPU-Capable Force Spreading (xp-agnostic)
# =============================================================================
#
# Design:
#   - F_grid lives entirely on GPU — never transferred to CPU
#   - Marker data (positions, forces, epsilon) sent to GPU (~KB)
#
# Evolution:
#   P2 (Method A): Python marker loop, each stencil GPU-vectorized
#   P5 (Method B): CuPy RawKernel — ALL markers × ALL stencil nodes
#                  processed in a single CUDA kernel launch using atomicAdd.
#                  Eliminates Python loop entirely.
#
# RawKernel Strategy (P5):
#   Thread grid: (n_active_markers × S³) total threads
#   Each thread:
#     1. Decode (marker_idx, local_x, local_y, local_z) from linear index
#     2. Compute absolute grid position → bounds check
#     3. Compute distance, Gaussian weight, cutoff check
#     4. atomicAdd(-force * eta) to F_grid[d, ix, iy, iz]
#
#   atomicAdd on double is supported on compute capability ≥ 6.0 (Pascal+).
#   For 200 markers × S=13 → 439,400 threads — well within GPU capacity.
#
# =============================================================================

# --- CUDA kernel source (P5) ---
_SPREAD_KERNEL_SRC = r"""
extern "C" __global__
void spread_forces_kernel(
    double* F_grid,             // (3, Nx, Ny, Nz) - output, atomicAdd target
    const double* positions,    // (N_active, 3) - marker positions [lu]
    const double* forces,       // (N_active, 3) - marker forces [lattice force]
    const double* epsilons,     // (N_active,) - per-marker epsilon [lu]
    const int N_active,         // number of active markers
    const int Nx, const int Ny, const int Nz,
    const int S,                // stencil width (2*half+1)
    const int half,             // stencil half-width
    const double n_cut,         // cutoff multiplier [dimensionless]
    const double inv_pi32       // 1 / pi^{3/2} [dimensionless]
)
{
    // Global thread index -> (marker_idx, stencil linear index)
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int S3 = S * S * S;
    int total = N_active * S3;
    if (tid >= total) return;

    int m = tid / S3;           // marker index
    int s = tid % S3;           // stencil linear index

    // Decode stencil (sx, sy, sz) from linear index s
    int sz = s % S;
    int sy = (s / S) % S;
    int sx = s / (S * S);

    // Stencil offsets: [-half, ..., +half]
    int ox = sx - half;
    int oy = sy - half;
    int oz = sz - half;

    // Marker position
    double xj = positions[m * 3 + 0];
    double yj = positions[m * 3 + 1];
    double zj = positions[m * 3 + 2];

    // Nearest grid node
    int ix0 = (int)round(xj);
    int iy0 = (int)round(yj);
    int iz0 = (int)round(zj);

    // Absolute grid position
    int ix = ix0 + ox;
    int iy = iy0 + oy;
    int iz = iz0 + oz;

    // Domain bounds check
    if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz)
        return;

    // Per-marker epsilon and cutoff
    double eps = epsilons[m];
    double eps_sq = eps * eps;
    double r_cut = n_cut * eps;
    double r_cut_sq = r_cut * r_cut;

    // Distance from marker to grid node
    double dx = (double)ix - xj;
    double dy = (double)iy - yj;
    double dz = (double)iz - zj;
    double d_sq = dx*dx + dy*dy + dz*dz;

    // Cutoff check (spherical)
    if (d_sq > r_cut_sq) return;

    // Gaussian kernel: eta = (1 / (pi^{3/2} * eps^3)) * exp(-d_sq / eps_sq)
    double norm = inv_pi32 / (eps * eps * eps);     // [1/lu^3]
    double eta = norm * exp(-d_sq / eps_sq);         // [1/lu^3]

    // Force components (Newton's third law: negative sign)
    double fx = -forces[m * 3 + 0] * eta;  // [lattice force / lu^3]
    double fy = -forces[m * 3 + 1] * eta;
    double fz = -forces[m * 3 + 2] * eta;

    // Flat index into F_grid: F_grid[d, ix, iy, iz]
    // F_grid layout: (3, Nx, Ny, Nz) row-major -> index = d*Nx*Ny*Nz + ix*Ny*Nz + iy*Nz + iz
    int grid_size = Nx * Ny * Nz;
    int flat_idx = ix * Ny * Nz + iy * Nz + iz;

    atomicAdd(&F_grid[0 * grid_size + flat_idx], fx);
    atomicAdd(&F_grid[1 * grid_size + flat_idx], fy);
    atomicAdd(&F_grid[2 * grid_size + flat_idx], fz);
}
"""

# --- CUDA kernel source (radial-truncation variant) ---
# Identical to _SPREAD_KERNEL_SRC but truncates the source term at the blade's
# geometric radial span [r_root, r_tip] (Merabet 2021) and rescales the kept
# nodes by a per-marker factor (computed on the host to match the CPU path so
# the marker's total deposited force is conserved, relocated inboard).
_SPREAD_KERNEL_RADIAL_SRC = r"""
extern "C" __global__
void spread_forces_kernel_radial(
    double* F_grid,
    const double* positions,
    const double* forces,
    const double* epsilons,
    const double* scales,       // (N_active,) per-marker renorm factor
    const double* axis,         // (3,) unit rotation axis
    const double* hub,          // (3,) point on axis (marker frame)
    const double r_root,
    const double r_tip,
    const int N_active,
    const int Nx, const int Ny, const int Nz,
    const int S, const int half,
    const double n_cut, const double inv_pi32)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int S3 = S * S * S;
    if (tid >= N_active * S3) return;
    int m = tid / S3;
    int s = tid % S3;
    int sz = s % S;
    int sy = (s / S) % S;
    int sx = s / (S * S);
    int ox = sx - half, oy = sy - half, oz = sz - half;
    double xj = positions[m*3+0], yj = positions[m*3+1], zj = positions[m*3+2];
    int ix = (int)round(xj) + ox;
    int iy = (int)round(yj) + oy;
    int iz = (int)round(zj) + oz;
    if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz) return;
    double eps = epsilons[m];
    double eps_sq = eps * eps;
    double r_cut = n_cut * eps;
    double dx = (double)ix - xj, dy = (double)iy - yj, dz = (double)iz - zj;
    double d_sq = dx*dx + dy*dy + dz*dz;
    if (d_sq > r_cut * r_cut) return;
    // cylindrical radius of this node from the axis line through hub
    double px = (double)ix - hub[0], py = (double)iy - hub[1], pz = (double)iz - hub[2];
    double dot = px*axis[0] + py*axis[1] + pz*axis[2];
    double qx = px - dot*axis[0], qy = py - dot*axis[1], qz = pz - dot*axis[2];
    double rn = sqrt(qx*qx + qy*qy + qz*qz);
    if (rn < r_root || rn > r_tip) return;      // radial truncation
    double norm = inv_pi32 / (eps * eps * eps);
    double eta = norm * exp(-d_sq / eps_sq) * scales[m];   // renormalised
    double fx = -forces[m*3+0] * eta;
    double fy = -forces[m*3+1] * eta;
    double fz = -forces[m*3+2] * eta;
    int grid_size = Nx * Ny * Nz;
    int flat_idx = ix * Ny * Nz + iy * Nz + iz;
    atomicAdd(&F_grid[0 * grid_size + flat_idx], fx);
    atomicAdd(&F_grid[1 * grid_size + flat_idx], fy);
    atomicAdd(&F_grid[2 * grid_size + flat_idx], fz);
}
"""

# --- CUDA kernel source (anisotropic variant, Natelson 2026 Eq. 7) ---
# Same structure as _SPREAD_KERNEL_SRC but the isotropic Gaussian is replaced by
# an ellipsoidal one: the node offset is projected onto the per-marker blade-local
# axes ec (chord), et (thickness), er (radial), each with its own width. Reduces
# EXACTLY to the isotropic kernel when ec/et/er are orthonormal and the three
# widths are equal. ASCII ONLY (nvrtc breaks on non-ASCII under POSIX locale).
_SPREAD_KERNEL_ANISO_SRC = r"""
extern "C" __global__
void spread_forces_kernel_aniso(
    double* F_grid,
    const double* positions,    // (N_active, 3) [lu]
    const double* forces,       // (N_active, 3) [lattice force]
    const double* ec,           // (N_active, 3) chord unit vector
    const double* et,           // (N_active, 3) thickness unit vector
    const double* er,           // (N_active, 3) radial unit vector
    const double* eps_c,        // (N_active,) chord width [lu]
    const double* eps_t,        // (N_active,) thickness width [lu]
    const double* eps_r,        // (N_active,) radial width [lu]
    const int N_active,
    const int Nx, const int Ny, const int Nz,
    const int S, const int half,
    const double n_cut, const double inv_pi32)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int S3 = S * S * S;
    if (tid >= N_active * S3) return;
    int m = tid / S3;
    int s = tid % S3;
    int sz = s % S;
    int sy = (s / S) % S;
    int sx = s / (S * S);
    int ox = sx - half, oy = sy - half, oz = sz - half;
    double xj = positions[m*3+0], yj = positions[m*3+1], zj = positions[m*3+2];
    int ix = (int)round(xj) + ox;
    int iy = (int)round(yj) + oy;
    int iz = (int)round(zj) + oz;
    if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz) return;
    double dx = (double)ix - xj;
    double dy = (double)iy - yj;
    double dz = (double)iz - zj;
    // Project the node offset onto the blade-local axes
    double dc = dx*ec[m*3+0] + dy*ec[m*3+1] + dz*ec[m*3+2];
    double dt = dx*et[m*3+0] + dy*et[m*3+1] + dz*et[m*3+2];
    double dr = dx*er[m*3+0] + dy*er[m*3+1] + dz*er[m*3+2];
    double epc = eps_c[m], ept = eps_t[m], epr = eps_r[m];
    double ac = dc / epc, at = dt / ept, ar = dr / epr;
    double arg = ac*ac + at*at + ar*ar;
    if (arg > n_cut * n_cut) return;           // ellipsoidal cutoff
    double norm = inv_pi32 / (epc * ept * epr);
    double eta = norm * exp(-arg);
    double fx = -forces[m*3+0] * eta;
    double fy = -forces[m*3+1] * eta;
    double fz = -forces[m*3+2] * eta;
    int grid_size = Nx * Ny * Nz;
    int flat_idx = ix * Ny * Nz + iy * Nz + iz;
    atomicAdd(&F_grid[0 * grid_size + flat_idx], fx);
    atomicAdd(&F_grid[1 * grid_size + flat_idx], fy);
    atomicAdd(&F_grid[2 * grid_size + flat_idx], fz);
}
"""


# Compiled kernel cache (lazy init)
_spread_kernel_cache = {}


def _get_spread_kernel(xp):
    """Get or compile the spreading CUDA kernel (cached)"""
    if 'kernel' not in _spread_kernel_cache:
        _spread_kernel_cache['kernel'] = xp.RawKernel(
            _SPREAD_KERNEL_SRC,
            'spread_forces_kernel'
        )
    return _spread_kernel_cache['kernel']


def _get_spread_kernel_radial(xp):
    """Get or compile the radial-truncation spreading kernel (cached)"""
    if 'kernel_radial' not in _spread_kernel_cache:
        _spread_kernel_cache['kernel_radial'] = xp.RawKernel(
            _SPREAD_KERNEL_RADIAL_SRC,
            'spread_forces_kernel_radial'
        )
    return _spread_kernel_cache['kernel_radial']


def _get_spread_kernel_aniso(xp):
    """Get or compile the anisotropic spreading kernel (cached)"""
    if 'kernel_aniso' not in _spread_kernel_cache:
        _spread_kernel_cache['kernel_aniso'] = xp.RawKernel(
            _SPREAD_KERNEL_ANISO_SRC,
            'spread_forces_kernel_aniso'
        )
    return _spread_kernel_cache['kernel_aniso']


def spread_forces_to_grid_gpu(
    domain_shape: Tuple[int, int, int],
    marker_positions: 'npt.NDArray',
    marker_forces: 'npt.NDArray',
    marker_epsilon: 'npt.NDArray',
    xp,
    marker_active: Optional['npt.NDArray'] = None,
    n_cut: float = 3.0,
    F_grid: Optional['npt.NDArray'] = None,
    radial_trunc: Optional[dict] = None,
    aniso: Optional[dict] = None
) -> 'npt.NDArray':
    """GPU-capable batch force spreading (Eq. 13) — xp dispatch

    When xp is cupy:
      - P5 path (default): CuPy RawKernel with atomicAdd.
        ALL markers processed in a single CUDA kernel launch.
        No Python loop. Maximum GPU parallelism.
      - Fallback: Method A (marker loop) if RawKernel compilation fails.

    When xp is numpy, delegates to existing CPU function (§2).

    The returned F_grid is an xp array (cupy if GPU, numpy if CPU),
    ready for direct use in Guo forcing without any host↔device transfer.

    Physical formula (Watanabe et al. Eq. 13):
        F(x) = Σ_j  -F_j^AL · η_ε(|x - x_j|)
        η_ε(d) = 1/(π^{3/2}·ε³) · exp(-d²/ε²)    [1/lu³]

    Args:
        domain_shape: (Nx, Ny, Nz)  [lu]
        marker_positions: (N_markers, 3) numpy  [lu]
        marker_forces: (N_markers, 3) numpy  [lattice force]
        marker_epsilon: (N_markers,) numpy  [lu]
        xp: Array module — numpy or cupy.
        marker_active: (N_markers,) bool numpy (default: all True)
        n_cut: Cutoff in units of ε  [dimensionless]
        F_grid: Optional pre-allocated xp array, shape (3, Nx, Ny, Nz).
                If provided, forces are ACCUMULATED (not zeroed).
                Must be an xp array matching the backend.

    Returns:
        F_grid: Body force field, shape (3, Nx, Ny, Nz)  [lattice force / lu³]
                Same backend as xp (GPU array if cupy, CPU if numpy).
    """
    # ── CPU fallback: delegate to existing §2 (handles aniso) ──
    if xp.__name__ == 'numpy':
        return spread_forces_to_grid(
            domain_shape, marker_positions, marker_forces,
            marker_epsilon, marker_active=marker_active,
            n_cut=n_cut, F_grid=F_grid, radial_trunc=radial_trunc,
            aniso=aniso
        )

    # ── GPU path ──
    Nx, Ny, Nz = domain_shape
    n_markers = marker_positions.shape[0]

    # Allocate F_grid on GPU if not provided
    if F_grid is None:
        F_grid = xp.zeros((3, Nx, Ny, Nz), dtype=xp.float64)
        # [lattice force / lu³]

    # Default: all markers active
    if marker_active is None:
        marker_active = np.ones(n_markers, dtype=bool)

    # Filter to active, nonzero-force markers (on CPU, cheap)
    force_mag_sq = np.sum(marker_forces ** 2, axis=1)   # (N_markers,)
    active_mask = marker_active & (force_mag_sq > 1e-60)
    n_active = int(np.sum(active_mask))

    if n_active == 0:
        return F_grid

    active_pos = marker_positions[active_mask]        # (N_active, 3)
    active_forces = marker_forces[active_mask]        # (N_active, 3)
    active_eps = marker_epsilon[active_mask]           # (N_active,)

    # ── Anisotropic GPU path (Natelson Eq. 7, Step B) ──
    # radial truncation is not combined with aniso (mirror of the CPU path).
    if aniso is not None:
        a_ec = np.ascontiguousarray(aniso['ec'][active_mask])
        a_et = np.ascontiguousarray(aniso['et'][active_mask])
        a_er = np.ascontiguousarray(aniso['er'][active_mask])
        a_epc = np.ascontiguousarray(aniso['eps_c'][active_mask])
        a_ept = np.ascontiguousarray(aniso['eps_t'][active_mask])
        a_epr = np.ascontiguousarray(aniso['eps_r'][active_mask])
        try:
            _spread_rawkernel_aniso_gpu(
                F_grid, active_pos, active_forces, a_ec, a_et, a_er,
                a_epc, a_ept, a_epr, domain_shape, xp, n_cut)
        except Exception as _e:
            # Correct-but-slower fallback: compute on the host, add to GPU F_grid.
            if not _spread_kernel_cache.get('warned_aniso_fallback'):
                _spread_kernel_cache['warned_aniso_fallback'] = True
                import warnings
                warnings.warn(
                    "spread_forces_kernel_aniso RawKernel failed to compile; using "
                    "the CPU-host fallback (slow). Check _SPREAD_KERNEL_ANISO_SRC "
                    "for non-ASCII / CUDA errors. First error: " + repr(_e),
                    RuntimeWarning)
            F_host = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
            spread_forces_to_grid(
                domain_shape, active_pos, active_forces, active_eps,
                n_cut=n_cut, F_grid=F_host,
                aniso={'ec': a_ec, 'et': a_et, 'er': a_er,
                       'eps_c': a_epc, 'eps_t': a_ept, 'eps_r': a_epr})
            F_grid += xp.asarray(F_host)
        return F_grid

    # Radial truncation: per-marker renorm scales. Batched on the GPU (they are
    # consumed by the GPU kernel anyway) — the host per-marker loop cost ~30 ms
    # per ALM sub-step at D40 and dominated the archB slowdown. Node selection
    # identical to the CPU reference; fp64 sum order differs (round-off only,
    # gate: alm_radial_scales_gate.py).
    radial_gpu = None
    if radial_trunc is not None:
        # Multi-GPU (patch 17 M3): the renormalisation scales must be
        # IDENTICAL on every rank -> compute from GLOBAL geometry when the
        # caller provides it (scale_*); the kernel's per-node radial cut
        # still uses the local-shifted hub in radial_trunc['hub'].
        scales_full = compute_radial_scales_batch(
            xp,
            radial_trunc.get('scale_domain_shape') or domain_shape,
            radial_trunc.get('scale_positions', marker_positions),
            marker_epsilon, active_mask,
            radial_trunc['axis'],
            radial_trunc.get('scale_hub', radial_trunc['hub']),
            radial_trunc['r_root'], radial_trunc['r_tip'], n_cut=n_cut)
        _ax = np.asarray(radial_trunc['axis'], dtype=np.float64)
        _ax = _ax / (np.linalg.norm(_ax) + 1e-30)
        radial_gpu = {
            'scales': scales_full[xp.asarray(active_mask)],
            'axis': _ax,
            'hub': np.asarray(radial_trunc['hub'], dtype=np.float64),
            'r_root': float(radial_trunc['r_root']),
            'r_tip': float(radial_trunc['r_tip']),
        }

    # ── Try RawKernel (P5) ──
    try:
        _spread_rawkernel_gpu(
            F_grid, active_pos, active_forces, active_eps,
            domain_shape, xp, n_cut, radial=radial_gpu
        )
    except Exception as _e:
        # Fallback to Method A (marker loop) if kernel fails. WARN ONCE — a silent
        # fallback here is ~40x slower than the RawKernel and dominated the ALM step
        # (2026-07: non-ASCII in _SPREAD_KERNEL_SRC broke nvrtc on POSIX locale).
        if not _spread_kernel_cache.get('warned_fallback'):
            _spread_kernel_cache['warned_fallback'] = True
            import warnings
            warnings.warn(
                "spread_forces_kernel RawKernel failed to compile; using the ~40x "
                "slower marker-loop fallback (dominates the ALM step). Check "
                "_SPREAD_KERNEL_SRC for non-ASCII / CUDA errors. First error: "
                + repr(_e), RuntimeWarning)
        _spread_method_a_gpu(
            F_grid, active_pos, active_forces, active_eps,
            xp, n_cut
        )

    return F_grid   # (3, Nx, Ny, Nz) GPU array [lattice force / lu³]


# -----------------------------------------------------------------------------
# §5.1 RawKernel Spreading (P5 — Full Parallelism)
# -----------------------------------------------------------------------------

def _spread_rawkernel_gpu(
    F_grid: 'npt.NDArray',
    positions: 'npt.NDArray',
    forces: 'npt.NDArray',
    epsilons: 'npt.NDArray',
    domain_shape: Tuple[int, int, int],
    xp,
    n_cut: float,
    radial: Optional[dict] = None
) -> None:
    """Spread all marker forces using CUDA RawKernel with atomicAdd

    Single kernel launch processes ALL markers × ALL stencil nodes.
    No Python loop. Race conditions resolved by atomicAdd on double.

    Thread layout:
        total_threads = N_active × S³
        thread i → marker (i // S³), stencil node (i % S³)

    Memory: only marker arrays transferred CPU→GPU (~KB).
    F_grid modified in-place on GPU.

    Args:
        F_grid: (3, Nx, Ny, Nz) GPU array, modified IN-PLACE
        positions: (N_active, 3) numpy  [lu]
        forces: (N_active, 3) numpy  [lattice force]
        epsilons: (N_active,) numpy  [lu]
        domain_shape: (Nx, Ny, Nz)  [lu]
        xp: cupy module
        n_cut: Cutoff multiplier  [dimensionless]
    """
    Nx, Ny, Nz = domain_shape
    n_active = positions.shape[0]

    # Stencil size from max epsilon
    eps_max = float(np.max(epsilons))
    r_cut_max = n_cut * eps_max
    half = int(np.ceil(r_cut_max)) + 1       # +1 for floor/ceil matching
    S = 2 * half + 1

    # Transfer marker data to GPU (~KB)
    pos_gpu = xp.asarray(positions.astype(np.float64).ravel(),
                         dtype=xp.float64)         # (N_active*3,)
    force_gpu = xp.asarray(forces.astype(np.float64).ravel(),
                           dtype=xp.float64)       # (N_active*3,)
    eps_gpu = xp.asarray(epsilons.astype(np.float64),
                         dtype=xp.float64)         # (N_active,)

    inv_pi32 = 1.0 / (np.pi ** 1.5)                # [dimensionless]

    # Launch kernel
    total_threads = n_active * S * S * S
    block_size = 256
    grid_size = (total_threads + block_size - 1) // block_size

    if radial is None:
        kernel = _get_spread_kernel(xp)
        kernel(
            (grid_size,), (block_size,),
            (
                F_grid, pos_gpu, force_gpu, eps_gpu,
                np.int32(n_active),
                np.int32(Nx), np.int32(Ny), np.int32(Nz),
                np.int32(S), np.int32(half),
                np.float64(n_cut), np.float64(inv_pi32)
            )
        )
    else:
        scales_gpu = xp.asarray(radial['scales'].astype(np.float64),
                                dtype=xp.float64)
        axis_gpu = xp.asarray(radial['axis'].astype(np.float64), dtype=xp.float64)
        hub_gpu = xp.asarray(radial['hub'].astype(np.float64), dtype=xp.float64)
        kernel = _get_spread_kernel_radial(xp)
        kernel(
            (grid_size,), (block_size,),
            (
                F_grid, pos_gpu, force_gpu, eps_gpu,
                scales_gpu, axis_gpu, hub_gpu,
                np.float64(radial['r_root']), np.float64(radial['r_tip']),
                np.int32(n_active),
                np.int32(Nx), np.int32(Ny), np.int32(Nz),
                np.int32(S), np.int32(half),
                np.float64(n_cut), np.float64(inv_pi32)
            )
        )
    # F_grid modified in-place via atomicAdd


def _spread_rawkernel_aniso_gpu(
    F_grid, positions, forces, ec, et, er, eps_c, eps_t, eps_r,
    domain_shape, xp, n_cut
) -> None:
    """Anisotropic (Natelson Eq. 7) spread via CUDA RawKernel with atomicAdd.

    Mirrors _spread_rawkernel_gpu but with per-marker blade-local axes (ec/et/er)
    and per-axis widths (eps_c/eps_t/eps_r). Stencil sized from the largest width.
    """
    Nx, Ny, Nz = domain_shape
    n_active = positions.shape[0]

    eps_max = float(np.max(np.maximum(np.maximum(eps_c, eps_t), eps_r)))
    half = int(np.ceil(n_cut * eps_max)) + 1
    S = 2 * half + 1

    def _g(a, ncol=None):
        a = np.asarray(a, dtype=np.float64)
        return xp.asarray(a.ravel() if ncol else a, dtype=xp.float64)

    pos_gpu = _g(positions, 3); force_gpu = _g(forces, 3)
    ec_gpu = _g(ec, 3); et_gpu = _g(et, 3); er_gpu = _g(er, 3)
    epc_gpu = _g(eps_c); ept_gpu = _g(eps_t); epr_gpu = _g(eps_r)
    inv_pi32 = 1.0 / (np.pi ** 1.5)

    total_threads = n_active * S * S * S
    block_size = 256
    grid_size = (total_threads + block_size - 1) // block_size
    kernel = _get_spread_kernel_aniso(xp)
    kernel(
        (grid_size,), (block_size,),
        (F_grid, pos_gpu, force_gpu, ec_gpu, et_gpu, er_gpu,
         epc_gpu, ept_gpu, epr_gpu,
         np.int32(n_active), np.int32(Nx), np.int32(Ny), np.int32(Nz),
         np.int32(S), np.int32(half), np.float64(n_cut), np.float64(inv_pi32))
    )
    # F_grid modified in-place via atomicAdd


# -----------------------------------------------------------------------------
# §5.2 Method A Fallback (marker loop, P2 legacy)
# -----------------------------------------------------------------------------

def _spread_method_a_gpu(
    F_grid: 'npt.NDArray',
    positions: 'npt.NDArray',
    forces: 'npt.NDArray',
    epsilons: 'npt.NDArray',
    xp,
    n_cut: float
) -> None:
    """Fallback: spread via Python marker loop (Method A from P2)

    Each marker's stencil is computed as a GPU-vectorized operation.
    Safe (no race conditions), sufficient for N_markers ≤ 200.
    Used only when RawKernel compilation fails.
    """
    for j in range(positions.shape[0]):
        _spread_single_marker_gpu(
            F_grid,
            positions[j],
            forces[j],
            float(epsilons[j]),
            xp,
            n_cut
        )


# -----------------------------------------------------------------------------
# §5.1 Single Marker GPU Spreading (stencil vectorized)
# -----------------------------------------------------------------------------

def _spread_single_marker_gpu(
    F_grid: 'npt.NDArray',
    marker_pos: 'npt.NDArray',
    marker_force: 'npt.NDArray',
    epsilon: float,
    xp,
    n_cut: float
) -> None:
    """Spread one marker's force onto GPU-resident F_grid

    The stencil is a small box [x_j ± r_cut] typically ~13³ = 2197 nodes.
    All operations within this box (meshgrid, distance, Gaussian, accumulation)
    run as vectorized GPU kernels.

    Memory per call: ~17 KB for ε=2, n_cut=3 — negligible.

    Args:
        F_grid: (3, Nx, Ny, Nz) GPU array, modified IN-PLACE  [lattice force / lu³]
        marker_pos: (3,) numpy  [lu]
        marker_force: (3,) numpy  [lattice force]
        epsilon: Gaussian width  [lu]
        xp: cupy module
        n_cut: Cutoff multiplier  [dimensionless]
    """
    _, Nx, Ny, Nz = F_grid.shape
    eps_sq = epsilon * epsilon                            # [lu²]
    r_cut = n_cut * epsilon                               # [lu]
    r_cut_sq = r_cut * r_cut                              # [lu²]

    # Gaussian normalization: 1 / (π^{3/2} · ε³)  [1/lu³]
    norm = 1.0 / (np.pi ** 1.5 * epsilon ** 3)

    # ── Bounding box (matches §1 floor/ceil convention) ──
    xj, yj, zj = float(marker_pos[0]), float(marker_pos[1]), float(marker_pos[2])

    x_min = max(int(np.floor(xj - r_cut)), 0)
    x_max = min(int(np.ceil(xj + r_cut)), Nx - 1)
    y_min = max(int(np.floor(yj - r_cut)), 0)
    y_max = min(int(np.ceil(yj + r_cut)), Ny - 1)
    z_min = max(int(np.floor(zj - r_cut)), 0)
    z_max = min(int(np.ceil(zj + r_cut)), Nz - 1)

    if x_max < x_min or y_max < y_min or z_max < z_min:
        return

    # ── GPU stencil coordinate arrays ──
    ix = xp.arange(x_min, x_max + 1, dtype=xp.float64)   # [lu]
    iy = xp.arange(y_min, y_max + 1, dtype=xp.float64)   # [lu]
    iz = xp.arange(z_min, z_max + 1, dtype=xp.float64)   # [lu]

    gx, gy, gz = xp.meshgrid(ix, iy, iz, indexing='ij')   # each (sx, sy, sz)

    # ── Squared distances ──
    dx = gx - xj                                           # [lu]
    dy = gy - yj                                           # [lu]
    dz = gz - zj                                           # [lu]
    d_sq = dx * dx + dy * dy + dz * dz                     # [lu²]

    # ── Gaussian kernel with spherical cutoff ──
    mask = d_sq <= r_cut_sq
    eta = xp.where(mask, norm * xp.exp(-d_sq / eps_sq), 0.0)   # [1/lu³]

    # ── Accumulate: F_grid(x) += -F^AL · η_ε ──
    # Negative sign = Newton's third law (reaction on fluid)
    for d in range(3):
        F_grid[d, x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] += (
            -float(marker_force[d]) * eta                  # [lattice force / lu³]
        )


# =============================================================================
# §6. GPU vs CPU Spreading Verification Utility
# =============================================================================

def verify_gpu_spreading(
    domain_shape: Tuple[int, int, int],
    marker_positions: 'npt.NDArray',
    marker_forces: 'npt.NDArray',
    marker_epsilon: 'npt.NDArray',
    marker_active: Optional['npt.NDArray'] = None,
    n_cut: float = 3.0,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> dict:
    """Verify GPU spreading against CPU reference

    Runs both CPU (§2) and GPU (§5) paths, comparing F_grid element-wise.
    Also checks force conservation for both paths.

    Args:
        domain_shape: (Nx, Ny, Nz)  [lu]
        marker_positions: (N_markers, 3) numpy  [lu]
        marker_forces: (N_markers, 3) numpy  [lattice force]
        marker_epsilon: (N_markers,) numpy  [lu]
        marker_active: (N_markers,) bool or None
        n_cut: Cutoff multiplier  [dimensionless]
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        dict with:
            - match: bool — True if grids match within tolerance
            - max_abs_error: Maximum absolute difference  [lattice force / lu³]
            - max_rel_error: Maximum relative difference  [dimensionless]
            - cpu_conserved: Force conservation on CPU path
            - gpu_conserved: Force conservation on GPU path
            - F_cpu: CPU result (3, Nx, Ny, Nz) numpy
            - F_gpu: GPU result (3, Nx, Ny, Nz) numpy
    """
    try:
        import cupy as cp
    except ImportError:
        return {'match': None, 'error': 'cupy not available'}

    # CPU reference
    F_cpu = spread_forces_to_grid(
        domain_shape, marker_positions, marker_forces,
        marker_epsilon, marker_active=marker_active,
        n_cut=n_cut
    )

    # GPU path
    F_gpu_dev = spread_forces_to_grid_gpu(
        domain_shape, marker_positions, marker_forces,
        marker_epsilon, xp=cp, marker_active=marker_active,
        n_cut=n_cut
    )
    F_gpu = cp.asnumpy(F_gpu_dev)

    # Element-wise comparison
    abs_err = np.abs(F_cpu - F_gpu)
    max_abs = float(np.max(abs_err))

    nonzero_mask = np.abs(F_cpu) > atol
    denom = np.maximum(np.abs(F_cpu), 1e-30)
    max_rel = float(np.max(abs_err[nonzero_mask] / denom[nonzero_mask])) \
        if np.any(nonzero_mask) else 0.0

    match = np.allclose(F_cpu, F_gpu, rtol=rtol, atol=atol)

    # Force conservation
    cons_cpu = check_force_conservation(
        F_cpu, marker_forces, marker_active
    )
    cons_gpu = check_force_conservation(
        F_gpu, marker_forces, marker_active
    )

    return {
        'match': match,
        'max_abs_error': max_abs,
        'max_rel_error': max_rel,
        'cpu_conserved': cons_cpu['is_conserved'],
        'gpu_conserved': cons_gpu['is_conserved'],
        'cpu_conservation_error': cons_cpu['relative_error'],
        'gpu_conservation_error': cons_gpu['relative_error'],
        'F_cpu': F_cpu,
        'F_gpu': F_gpu,
    }