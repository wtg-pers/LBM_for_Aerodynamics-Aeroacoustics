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

def spread_force_single_marker(
    F_grid: 'npt.NDArray',
    marker_pos: 'npt.NDArray',
    marker_force: 'npt.NDArray',
    epsilon: float,
    n_cut: float = 3.0
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

    # Accumulate: F_grid(x) += -F^AL · η_ε(|x - x_j|)
    # Negative sign = Newton's third law (reaction on fluid)
    slc = (slice(None), slice(ix_min, ix_max+1),
           slice(iy_min, iy_max+1), slice(iz_min, iz_max+1))
    for d in range(3):
        F_grid[d, ix_min:ix_max+1, iy_min:iy_max+1, iz_min:iz_max+1] += (
            -marker_force[d] * eta                       # [lattice force / lu³]
        )


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
    F_grid: Optional['npt.NDArray'] = None
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

    # Accumulate force from each active marker
    for j in range(n_markers):
        if not marker_active[j]:
            continue

        # Skip zero-force markers (saves computation)
        force_mag_sq = np.sum(marker_forces[j] ** 2)
        if force_mag_sq < 1e-60:
            continue

        spread_force_single_marker(
            F_grid,
            marker_positions[j],
            marker_forces[j],
            float(marker_epsilon[j]),
            n_cut=n_cut
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