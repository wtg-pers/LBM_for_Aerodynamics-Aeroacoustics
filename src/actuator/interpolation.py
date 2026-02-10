"""
Grid-to-Marker Velocity Interpolation for Actuator Line Model

This module samples the LBM velocity field at marker particle positions
using Gaussian-weighted interpolation, providing the input velocity for
the BEM force calculation at each actuator point.

Physical Concept:
================
Marker particles do NOT generally coincide with LBM grid nodes. To obtain
the flow velocity at an off-grid marker position x_j, we use the same
Gaussian kernel that appears in the force spreading (Eq. 13):

    η_ε(d) = 1 / (π^{3/2} · ε³) · exp(-(d/ε)²)       [1/m³ or 1/lu³]

The interpolated velocity at marker j is a normalized Gaussian-weighted
average of nearby grid velocities:

    u(x_j) = Σ_x  u(x) · η_ε(|x - x_j|) · Δx³
              ————————————————————————————————————
              Σ_x  η_ε(|x - x_j|) · Δx³

The normalization ensures conservation: a uniform velocity field is
interpolated exactly regardless of marker position.

Support Radius:
    The Gaussian is truncated at r_cut = n_cut · ε (default n_cut = 3).
    At d = 3ε: exp(-9) ≈ 1.2×10⁻⁴, which is negligible.
    This limits the stencil to a small box around each marker.

Performance Notes:
    - Each marker queries only grid nodes within [x_j ± r_cut]
    - The Δx³ factors cancel in the ratio, simplifying computation
    - In lattice units (Δx = 1), the volume element is unity

Coordinate Convention:
    Grid arrays use shape (dim, Nx, Ny, Nz) for velocity.
    Marker positions are (N_markers, 3) in global coordinates.

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026 (Sec. 2.2)
    - Sørensen & Shen, J. Fluids Eng. 124, 393-399, 2002
    - Martínez-Tossas et al., Wind Energy 20, 1083-1098, 2017

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


# =============================================================================
# §1. Gaussian Kernel
# =============================================================================

def gaussian_kernel_3d(
    d_sq: 'npt.NDArray',
    epsilon: float
) -> 'npt.NDArray':
    """Evaluate 3D isotropic Gaussian regularization kernel

    η_ε(d) = 1 / (π^{3/2} · ε³) · exp(-d²/ε²)    [1/length³]

    This is the kernel used in BOTH interpolation and spreading (Eq. 13).
    The same kernel ensures momentum conservation between the two operations.

    Args:
        d_sq: Squared distances |x - x_j|²  [length²]
        epsilon: Gaussian filter width ε     [length]

    Returns:
        Kernel values η_ε(d)  [1/length³]
    """
    eps_sq = epsilon * epsilon                          # [length²]
    norm = 1.0 / (np.pi ** 1.5 * epsilon ** 3)         # [1/length³]
    return norm * np.exp(-d_sq / eps_sq)                # [1/length³]


# =============================================================================
# §2. Single-Marker Velocity Interpolation
# =============================================================================

def interpolate_velocity_at_marker(
    u_field: 'npt.NDArray',
    marker_pos: 'npt.NDArray',
    epsilon: float,
    n_cut: float = 3.0
) -> 'npt.NDArray':
    """Interpolate grid velocity at a single marker position

    Computes the Gaussian-weighted average of grid velocities within
    the support radius of the marker:

        u(x_j) = Σ u(x) · η_ε(|x - x_j|) / Σ η_ε(|x - x_j|)

    In lattice units (Δx = 1), the volume element Δx³ = 1 cancels.

    Args:
        u_field: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        marker_pos: Marker position (x, y, z)  [lattice units]
        epsilon: Gaussian filter width ε       [lattice units]
        n_cut: Cutoff in units of ε (default 3.0)  [dimensionless]

    Returns:
        Interpolated velocity (u_x, u_y, u_z)  [Δx/Δt]
    """
    _, Nx, Ny, Nz = u_field.shape
    r_cut = n_cut * epsilon                             # [lu]

    # --- Bounding box of affected grid nodes ---
    # Clamp to domain [0, N-1]
    ix_min = max(int(np.floor(marker_pos[0] - r_cut)), 0)
    ix_max = min(int(np.ceil(marker_pos[0] + r_cut)), Nx - 1)
    iy_min = max(int(np.floor(marker_pos[1] - r_cut)), 0)
    iy_max = min(int(np.ceil(marker_pos[1] + r_cut)), Ny - 1)
    iz_min = max(int(np.floor(marker_pos[2] - r_cut)), 0)
    iz_max = min(int(np.ceil(marker_pos[2] + r_cut)), Nz - 1)

    # Grid coordinate arrays within the bounding box
    ix = np.arange(ix_min, ix_max + 1)     # [lu]
    iy = np.arange(iy_min, iy_max + 1)     # [lu]
    iz = np.arange(iz_min, iz_max + 1)     # [lu]

    if len(ix) == 0 or len(iy) == 0 or len(iz) == 0:
        return np.zeros(3)

    # 3D meshgrid of local grid positions
    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing='ij')  # each (nx, ny, nz)

    # Squared distance from marker to each grid node
    dx = gx - marker_pos[0]                 # [lu]
    dy = gy - marker_pos[1]                 # [lu]
    dz = gz - marker_pos[2]                 # [lu]
    d_sq = dx * dx + dy * dy + dz * dz      # [lu²]

    # Apply cutoff mask (circular, not box)
    r_cut_sq = r_cut * r_cut
    mask = d_sq <= r_cut_sq

    if not np.any(mask):
        return np.zeros(3)

    # Gaussian kernel weights (normalization cancels, but compute for clarity)
    weights = np.zeros_like(d_sq)
    weights[mask] = np.exp(-d_sq[mask] / (epsilon * epsilon))  # [dimensionless]
    # Note: the 1/(π^{3/2}·ε³) prefactor cancels in numerator/denominator

    # Sum of weights for normalization
    W_sum = np.sum(weights)                  # [dimensionless]

    if W_sum < 1e-30:
        return np.zeros(3)

    # Weighted velocity: u_interp = Σ w · u / Σ w
    u_interp = np.zeros(3, dtype=np.float64)
    for d in range(3):
        u_local = u_field[d, ix_min:ix_max+1, iy_min:iy_max+1, iz_min:iz_max+1]
        u_interp[d] = np.sum(weights * u_local) / W_sum   # [Δx/Δt]

    return u_interp  # [Δx/Δt]


# =============================================================================
# §3. Batch Velocity Interpolation (All Markers)
# =============================================================================

def interpolate_velocity_batch(
    u_field: 'npt.NDArray',
    marker_positions: 'npt.NDArray',
    marker_epsilon: 'npt.NDArray',
    n_cut: float = 3.0
) -> 'npt.NDArray':
    """Interpolate grid velocity at all marker positions

    Loops over markers and calls single-marker interpolation.
    Each marker may have a different Gaussian width ε(r).

    Data Flow in AL Timestep:
        1. Rotor provides marker positions → marker_positions
        2. THIS FUNCTION: grid velocity → marker velocities
        3. Rotor decomposes u_n, u_θ from marker velocities
        4. BEM computes forces from u_rel, α

    Args:
        u_field: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        marker_positions: All marker positions, shape (N_markers, 3)  [lu]
        marker_epsilon: Gaussian width per marker, shape (N_markers,)  [lu]
        n_cut: Cutoff in units of ε  [dimensionless]

    Returns:
        u_markers: Velocity at each marker, shape (N_markers, 3)  [Δx/Δt]
    """
    n_markers = marker_positions.shape[0]
    u_markers = np.zeros((n_markers, 3), dtype=np.float64)

    for j in range(n_markers):
        u_markers[j] = interpolate_velocity_at_marker(
            u_field,
            marker_positions[j],
            float(marker_epsilon[j]),
            n_cut=n_cut
        )

    return u_markers  # (N_markers, 3)  [Δx/Δt]


# =============================================================================
# §4. Optimized Batch Interpolation (Shared Stencil)
# =============================================================================

def interpolate_velocity_batch_fast(
    u_field: 'npt.NDArray',
    marker_positions: 'npt.NDArray',
    epsilon_uniform: float,
    n_cut: float = 3.0
) -> 'npt.NDArray':
    """Fast batch interpolation when all markers share the same ε

    When ε is uniform (common in practice when 2Δx dominates c/4),
    the support radius is identical for all markers, enabling
    precomputation of the relative stencil.

    Performance:
        ~2-5x faster than per-marker loop for N_markers > 100
        by precomputing the stencil offsets.

    Args:
        u_field: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        marker_positions: Marker positions, shape (N_markers, 3)  [lu]
        epsilon_uniform: Uniform Gaussian width  [lu]
        n_cut: Cutoff in units of ε  [dimensionless]

    Returns:
        u_markers: shape (N_markers, 3)  [Δx/Δt]
    """
    _, Nx, Ny, Nz = u_field.shape
    n_markers = marker_positions.shape[0]
    r_cut = n_cut * epsilon_uniform                     # [lu]
    eps_sq = epsilon_uniform * epsilon_uniform           # [lu²]
    r_cut_sq = r_cut * r_cut                            # [lu²]

    # Precompute relative stencil offsets
    half = int(np.ceil(r_cut))
    offsets_1d = np.arange(-half, half + 1)             # [lu]
    # 3D relative offsets
    ox, oy, oz = np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing='ij')
    rel_d_sq = ox**2 + oy**2 + oz**2                    # [lu²]
    stencil_mask = rel_d_sq <= r_cut_sq                 # Boolean

    # Flatten stencil for efficiency
    ox_flat = ox[stencil_mask]
    oy_flat = oy[stencil_mask]
    oz_flat = oz[stencil_mask]
    rel_d_sq_flat = rel_d_sq[stencil_mask]

    # Gaussian weights at stencil positions (relative to center)
    # These are the INTEGER-offset contributions; the fractional offset
    # must be added per marker.
    # Actually, we need per-marker computation because the fractional
    # position changes the d_sq for each stencil point.

    u_markers = np.zeros((n_markers, 3), dtype=np.float64)

    for j in range(n_markers):
        xj, yj, zj = marker_positions[j]

        # Nearest grid node
        ix0 = int(np.round(xj))
        iy0 = int(np.round(yj))
        iz0 = int(np.round(zj))

        # Absolute grid positions of stencil nodes
        gx = ix0 + ox_flat
        gy = iy0 + oy_flat
        gz = iz0 + oz_flat

        # Domain bounds check
        valid = (
            (gx >= 0) & (gx < Nx) &
            (gy >= 0) & (gy < Ny) &
            (gz >= 0) & (gz < Nz)
        )

        if not np.any(valid):
            continue

        gx_v = gx[valid]
        gy_v = gy[valid]
        gz_v = gz[valid]

        # Actual squared distances (accounting for fractional marker position)
        dx = gx_v.astype(np.float64) - xj          # [lu]
        dy = gy_v.astype(np.float64) - yj          # [lu]
        dz = gz_v.astype(np.float64) - zj          # [lu]
        d_sq = dx * dx + dy * dy + dz * dz          # [lu²]

        # Gaussian weights (prefactor cancels in normalization)
        w = np.exp(-d_sq / eps_sq)                   # [dimensionless]
        W_sum = np.sum(w)

        if W_sum < 1e-30:
            continue

        # Weighted velocity
        for d in range(3):
            u_markers[j, d] = np.sum(w * u_field[d, gx_v, gy_v, gz_v]) / W_sum

    return u_markers  # (N_markers, 3)  [Δx/Δt]


# =============================================================================
# §5. Interpolation Diagnostics
# =============================================================================

def compute_interpolation_stencil_info(
    epsilon: float,
    n_cut: float = 3.0
) -> dict:
    """Compute diagnostic information about the interpolation stencil

    Useful for verifying that the stencil size is reasonable and
    for estimating computational cost.

    Args:
        epsilon: Gaussian filter width  [lu]
        n_cut: Cutoff multiplier  [dimensionless]

    Returns:
        dict with:
            - r_cut: Support radius  [lu]
            - stencil_width: Box width (2·r_cut + 1)  [lu]
            - n_nodes_box: Total nodes in bounding box
            - n_nodes_sphere: Approximate nodes in sphere
            - kernel_at_cutoff: η(r_cut) relative to η(0)
    """
    r_cut = n_cut * epsilon
    width = 2 * int(np.ceil(r_cut)) + 1
    n_box = width ** 3
    n_sphere = int(4.0 / 3.0 * np.pi * r_cut ** 3)
    kernel_ratio = np.exp(-n_cut ** 2)

    return {
        'epsilon': epsilon,
        'n_cut': n_cut,
        'r_cut': r_cut,
        'stencil_width': width,
        'n_nodes_box': n_box,
        'n_nodes_sphere': n_sphere,
        'kernel_at_cutoff': kernel_ratio,
    }