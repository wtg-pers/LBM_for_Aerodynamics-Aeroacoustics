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


# =============================================================================
# §6. GPU-Capable Batch Interpolation (xp-agnostic)
# =============================================================================
#
# Design:
#   - u_field lives on GPU (cupy) → interpolation runs on GPU
#   - u_field lives on CPU (numpy) → falls back to existing §3/§4 functions
#   - Return value is ALWAYS numpy (CPU) → feeds into BEM (CPU-only)
#
# Data Transfer:
#   GPU path: only marker data crosses PCIe (~KB, not ~MB)
#     GPU→CPU: u_markers  (N_markers × 3 × 8 bytes ≈ 4.8 KB for 200 markers)
#     CPU→GPU: marker_positions, marker_epsilon (same order)
#
# Memory:
#   N=200, ε=2Δx, n_cut=3 → S=13 → (200,13,13,13)×8B ≈ 34 MB per array
#   Total ~120 MB — well within typical GPU memory
#
# =============================================================================

def interpolate_velocity_batch_gpu(
    u_field: 'npt.NDArray',
    marker_positions: 'npt.NDArray',
    marker_epsilon: 'npt.NDArray',
    xp,
    n_cut: float = 3.0,
    gpu_mem_limit_mb: float = 512.0,
    return_sums: bool = False,
    clip_bounds=None
) -> 'npt.NDArray':
    """GPU-capable batch velocity interpolation at all marker positions

    When xp is cupy, performs interpolation entirely on GPU using the
    u_field already resident in GPU memory. Only marker-sized arrays
    (~KB) cross the PCIe bus, eliminating the ~228 MB round-trip
    that previously bottlenecked every timestep.

    P4 Enhancement: Automatic chunking when estimated GPU memory
    exceeds gpu_mem_limit_mb. Splits markers into chunks that fit
    within the budget, processes each on GPU, and concatenates results.

    When xp is numpy, delegates to existing CPU functions (§3 or §4).

    Physical formula (same as §2):
        u(x_j) = Σ_x u(x)·η_ε(|x - x_j|) / Σ_x η_ε(|x - x_j|)

    where η_ε(d) = (1/π^{3/2}ε³)·exp(-d²/ε²) truncated at d = n_cut·ε.
    The normalization prefactor cancels in the ratio, so only exp(-d²/ε²)
    is computed.

    Args:
        u_field: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
                 Can be cupy array (GPU) or numpy array (CPU).
        marker_positions: All marker positions, shape (N_markers, 3)  [lu]
                          Always numpy (CPU) — from rotor kinematics.
        marker_epsilon: Gaussian width per marker, shape (N_markers,)  [lu]
                        Always numpy (CPU).
        xp: Array module — numpy or cupy.
        n_cut: Cutoff in units of ε  [dimensionless]
        gpu_mem_limit_mb: Maximum GPU memory for interpolation work arrays
                          [MB]. Triggers chunking if exceeded. Default 512.

    Returns:
        u_markers: Velocity at each marker, shape (N_markers, 3)  [Δx/Δt]
                   Always numpy (CPU) — feeds into BEM pipeline.
    """
    if return_sums and xp.__name__ == 'numpy':
        raise NotImplementedError(
            "return_sums (distributed partial sums) is GPU-only")

    # ── CPU fallback: delegate to existing functions ──
    if xp.__name__ == 'numpy':
        # Always use §3 (per-marker loop) as the reference implementation.
        # §4 (batch_fast) uses round-based stencils that can miss edge nodes
        # vs §2's floor/ceil stencils, causing ~1e-5 discrepancies.
        return interpolate_velocity_batch(
            u_field, marker_positions,
            marker_epsilon, n_cut=n_cut
        )

    # ── GPU fast path: one-launch RawKernel (backlog #3) ──
    import os as _os
    if u_field.dtype == xp.float32 and u_field.flags.c_contiguous and \
            _os.environ.get("LBM_ALM_SAMPLE_KERNEL", "1") == "1":
        num, den = _sample_markers_rawkernel(
            u_field, marker_positions, marker_epsilon, xp, n_cut,
            clip_bounds=clip_bounds)
        if return_sums:
            return num, den
        return num / np.maximum(den, 1e-30)[:, None]

    if clip_bounds is not None:
        raise NotImplementedError(
            "clip_bounds requires the RawKernel sampling path "
            "(LBM_ALM_SAMPLE_KERNEL=1, f32 contiguous u_field)")

    # ── GPU path with auto-chunking (P4) ──
    n_markers = marker_positions.shape[0]
    eps_unique = np.unique(marker_epsilon)
    is_uniform = (len(eps_unique) == 1)
    epsilon_for_stencil = float(eps_unique[0]) if is_uniform \
        else float(np.max(marker_epsilon))

    # Estimate memory and determine chunk size
    chunk_size = _compute_chunk_size(
        n_markers, epsilon_for_stencil, n_cut, gpu_mem_limit_mb
    )

    if chunk_size >= n_markers:
        # ── Single batch (no chunking needed) ──
        if is_uniform:
            u_gpu = _interpolate_uniform_eps_gpu(
                u_field, marker_positions,
                float(eps_unique[0]), xp, n_cut, return_sums=return_sums
            )
        else:
            u_gpu = _interpolate_varying_eps_gpu(
                u_field, marker_positions,
                marker_epsilon, xp, n_cut, return_sums=return_sums
            )
        if return_sums:
            return xp.asnumpy(u_gpu[0]), xp.asnumpy(u_gpu[1])
        return xp.asnumpy(u_gpu)

    # ── Chunked processing ──
    # Chunk membership does not change any marker's own weighted sums (each
    # marker reduces over its own stencil box), so return_sums chunks
    # transparently: raw (num, W) per chunk, assembled in marker order.
    u_all = np.zeros((n_markers, 3), dtype=np.float64)   # [Δx/Δt]
    num_all = np.zeros((n_markers, 3), dtype=np.float64) if return_sums \
        else None
    den_all = np.zeros(n_markers, dtype=np.float64) if return_sums else None

    for c_start in range(0, n_markers, chunk_size):
        c_end = min(c_start + chunk_size, n_markers)
        pos_chunk = marker_positions[c_start:c_end]
        eps_chunk = marker_epsilon[c_start:c_end]

        if is_uniform:
            u_chunk_gpu = _interpolate_uniform_eps_gpu(
                u_field, pos_chunk,
                float(eps_unique[0]), xp, n_cut, return_sums=return_sums
            )
        else:
            u_chunk_gpu = _interpolate_varying_eps_gpu(
                u_field, pos_chunk, eps_chunk, xp, n_cut,
                return_sums=return_sums
            )

        if return_sums:
            num_all[c_start:c_end] = xp.asnumpy(u_chunk_gpu[0])
            den_all[c_start:c_end] = xp.asnumpy(u_chunk_gpu[1])
        else:
            u_all[c_start:c_end] = xp.asnumpy(u_chunk_gpu)

    if return_sums:
        return num_all, den_all
    return u_all   # (N_markers, 3)  [Δx/Δt]


def _compute_chunk_size(
    n_markers: int,
    epsilon: float,
    n_cut: float,
    mem_limit_mb: float
) -> int:
    """Compute optimal chunk size to fit within GPU memory budget

    Memory estimation for (N_chunk, S, S, S) tensors:
        - gx, gy, gz: 3 × N × S³ × 4 bytes (int32)
        - gx_c, gy_c, gz_c: 3 × N × S³ × 4 bytes (int32)
        - valid: N × S³ × 1 byte (bool)
        - dx, dy, dz, d_sq: 4 × N × S³ × 8 bytes (float64)
        - weights: N × S³ × 8 bytes (float64)
        - u_local: N × S³ × 8 bytes (float64) × 3 components (sequential)
        Total ≈ N × S³ × (24 + 24 + 1 + 32 + 8 + 8) = N × S³ × 97 bytes
        With overhead: ~N × S³ × 120 bytes

    Args:
        n_markers: Total number of markers
        epsilon: Gaussian width (or max ε for varying)  [lu]
        n_cut: Cutoff multiplier  [dimensionless]
        mem_limit_mb: GPU memory budget  [MB]

    Returns:
        chunk_size: Number of markers per chunk (≥1, ≤n_markers)
    """
    r_cut = n_cut * epsilon
    half = int(np.ceil(r_cut)) + 1
    S = 2 * half + 1                                        # stencil width
    S_cubed = S * S * S                                     # stencil volume

    bytes_per_marker = S_cubed * 120                        # [bytes]
    mem_limit_bytes = mem_limit_mb * 1024 * 1024            # [bytes]

    if bytes_per_marker <= 0:
        return n_markers

    chunk = int(mem_limit_bytes / bytes_per_marker)
    chunk = max(1, min(chunk, n_markers))
    return chunk


# -----------------------------------------------------------------------------
# §6.1 Uniform ε — Fully Vectorized GPU Interpolation
# -----------------------------------------------------------------------------

def _interpolate_uniform_eps_gpu(
    u_field: 'npt.NDArray',
    marker_positions: 'npt.NDArray',
    epsilon: float,
    xp,
    n_cut: float,
    return_sums: bool = False
) -> 'npt.NDArray':
    """Fully vectorized GPU interpolation for uniform Gaussian width

    All markers share the same ε, so the stencil shape (S, S, S) is
    identical. This enables a single batched tensor operation:

        shape: (N_markers, S, S, S)

    for grid indices, distances, weights, and velocities — processed
    in one GPU kernel launch per operation.

    Memory layout:
        gx, gy, gz:  (N_markers, S, S, S) int32   — grid indices
        d_sq:        (N_markers, S, S, S) float64  — squared distances  [lu²]
        weights:     (N_markers, S, S, S) float64  — Gaussian weights   [dimless]
        u_local:     (N_markers, S, S, S) float64  — per-component vel  [Δx/Δt]

    Args:
        u_field: (3, Nx, Ny, Nz) GPU array  [Δx/Δt]
        marker_positions: (N_markers, 3) numpy  [lu]
        epsilon: Uniform Gaussian width  [lu]
        xp: cupy module
        n_cut: Cutoff multiplier  [dimensionless]

    Returns:
        u_markers: (N_markers, 3) GPU array  [Δx/Δt]
    """
    _, Nx, Ny, Nz = u_field.shape
    n_markers = marker_positions.shape[0]

    r_cut = n_cut * epsilon              # [lu]
    eps_sq = epsilon * epsilon            # [lu²]
    r_cut_sq = r_cut * r_cut             # [lu²]
    # +1 ensures we capture all nodes that §2's floor/ceil bounding box
    # would include, regardless of fractional marker position.
    # Extra nodes beyond r_cut get zero weight from spherical cutoff.
    half = int(np.ceil(r_cut)) + 1       # [lu, integer]

    # ── CPU → GPU: marker positions only (~KB) ──
    pos_gpu = xp.asarray(marker_positions, dtype=xp.float64)   # (N_markers, 3) [lu]

    # ── Stencil offsets (shared for all markers) ──
    offsets = xp.arange(-half, half + 1, dtype=xp.float64)     # (S,) [lu]
    S = len(offsets)    # stencil width = 2*half + 1

    # 3D stencil offset grid: (S, S, S) each
    ox, oy, oz = xp.meshgrid(offsets, offsets, offsets, indexing='ij')

    # ── Nearest grid node for each marker ──
    ix0 = xp.round(pos_gpu[:, 0]).astype(xp.int32)   # (N_markers,)
    iy0 = xp.round(pos_gpu[:, 1]).astype(xp.int32)
    iz0 = xp.round(pos_gpu[:, 2]).astype(xp.int32)

    # ── Absolute grid indices: (N_markers, S, S, S) ──
    # Broadcasting: (N,1,1,1) + (1,S,S,S) → (N,S,S,S)
    gx = ix0[:, None, None, None] + ox[None, :, :, :].astype(xp.int32)
    gy = iy0[:, None, None, None] + oy[None, :, :, :].astype(xp.int32)
    gz = iz0[:, None, None, None] + oz[None, :, :, :].astype(xp.int32)

    # ── Domain bounds check ──
    valid = (
        (gx >= 0) & (gx < Nx) &
        (gy >= 0) & (gy < Ny) &
        (gz >= 0) & (gz < Nz)
    )   # (N_markers, S, S, S) bool

    # Clamp for safe indexing (invalid nodes masked to zero weight)
    gx_c = xp.clip(gx, 0, Nx - 1)
    gy_c = xp.clip(gy, 0, Ny - 1)
    gz_c = xp.clip(gz, 0, Nz - 1)

    # ── Squared distances from each marker to its stencil nodes ──
    # pos_gpu[:, d] shape (N,) → (N,1,1,1) for broadcasting
    dx = gx.astype(xp.float64) - pos_gpu[:, 0, None, None, None]   # [lu]
    dy = gy.astype(xp.float64) - pos_gpu[:, 1, None, None, None]   # [lu]
    dz = gz.astype(xp.float64) - pos_gpu[:, 2, None, None, None]   # [lu]
    d_sq = dx * dx + dy * dy + dz * dz                              # [lu²]

    # ── Gaussian weights with cutoff ──
    # η ∝ exp(-d²/ε²), masked by domain bounds and cutoff radius
    weights = xp.where(
        valid & (d_sq <= r_cut_sq),
        xp.exp(-d_sq / eps_sq),    # [dimensionless] (prefactor cancels)
        0.0
    )   # (N_markers, S, S, S)

    # Normalization per marker
    W_raw = xp.sum(weights, axis=(1, 2, 3))        # (N_markers,) [dimless]
    W_sum = xp.maximum(W_raw, 1e-30)               # avoid division by zero

    # ── Weighted velocity for each component ──
    u_markers = xp.zeros((n_markers, 3), dtype=xp.float64)   # [Δx/Δt]

    for d in range(3):
        # Gather grid velocity at stencil nodes: u_field[d, gx, gy, gz]
        u_local = u_field[d, gx_c, gy_c, gz_c]     # (N_markers, S, S, S) [Δx/Δt]
        u_local = u_local * weights                  # weighted [Δx/Δt × dimless]
        u_markers[:, d] = xp.sum(u_local, axis=(1, 2, 3))   # [Δx/Δt]

    if return_sums:
        return u_markers, W_raw       # raw sums (distributed allreduce)
    u_markers /= W_sum[:, None]
    return u_markers   # (N_markers, 3) GPU array [Δx/Δt]


# -----------------------------------------------------------------------------
# §6.2 Varying ε — Padded GPU Interpolation
# -----------------------------------------------------------------------------

_SAMPLE_KERNEL_CACHE = {}

_SAMPLE_KERNEL_SRC = r"""
extern "C" __global__ void alm_sample_markers(
    const float* __restrict__ u,
    const double* __restrict__ pos,
    const double* __restrict__ eps2,
    const double* __restrict__ rc2,
    double* __restrict__ num,
    double* __restrict__ den,
    const int Nx, const int Ny, const int Nz,
    const int half,
    const int cx0, const int cx1,
    const int cy0, const int cy1,
    const int cz0, const int cz1)
{
    const int m = blockIdx.x;
    const int T = blockDim.x;
    const int t = threadIdx.x;
    const int S = 2 * half + 1;
    const long long SC = (long long)S * S * S;
    const double px = pos[3 * m], py = pos[3 * m + 1], pz = pos[3 * m + 2];
    const int ix0 = (int)llrint(px);
    const int iy0 = (int)llrint(py);
    const int iz0 = (int)llrint(pz);
    const double e2 = eps2[m], r2 = rc2[m];
    const long long NN = (long long)Nx * Ny * Nz;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, aw = 0.0;
    for (long long c = t; c < SC; c += T) {
        int oz = (int)(c % S) - half;
        int oy = (int)((c / S) % S) - half;
        int ox = (int)(c / ((long long)S * S)) - half;
        int gx = ix0 + ox;
        int gy = iy0 + oy;
        int gz = iz0 + oz;
        if (gx < cx0 || gx >= cx1 || gy < cy0 || gy >= cy1 ||
            gz < cz0 || gz >= cz1) continue;
        double dxx = (double)gx - px;
        double dyy = (double)gy - py;
        double dzz = (double)gz - pz;
        double d2 = dxx * dxx + dyy * dyy + dzz * dzz;
        if (d2 > r2) continue;
        double w = exp(-d2 / e2);
        long long b = ((long long)gx * Ny + gy) * Nz + gz;
        a0 += w * (double)u[b];
        a1 += w * (double)u[NN + b];
        a2 += w * (double)u[2 * NN + b];
        aw += w;
    }
    __shared__ double sh[4 * 256];
    sh[t] = a0; sh[256 + t] = a1; sh[512 + t] = a2; sh[768 + t] = aw;
    __syncthreads();
    for (int st = T / 2; st > 0; st >>= 1) {
        if (t < st) {
            sh[t] += sh[t + st];
            sh[256 + t] += sh[256 + t + st];
            sh[512 + t] += sh[512 + t + st];
            sh[768 + t] += sh[768 + t + st];
        }
        __syncthreads();
    }
    if (t == 0) {
        num[3 * m] = sh[0];
        num[3 * m + 1] = sh[256];
        num[3 * m + 2] = sh[512];
        den[m] = sh[768];
    }
}
"""


def _sample_markers_rawkernel(u_field, marker_positions, marker_epsilon,
                              xp, n_cut, clip_bounds=None):
    """One-launch marker sampling (patch 17 backlog #3).

    One BLOCK per marker; each thread strides the stencil box with serial
    f64 accumulation and a fixed shared-memory tree reduction ->
    DETERMINISTIC for any shape/run (no atomics). Replaces the chunked
    (N, S^3) intermediate path whose per-chunk CPU index builds + D2H
    syncs dominated the ALM section (~85-90% of step() at D40). Same
    semantics as _interpolate_varying_eps_gpu: box center llrint(pos)
    (numpy round = half-even = llrint), half = ceil(n_cut*eps_max)+1,
    w = exp(-d2/eps_m^2) within the per-marker radius, f64 sums.
    clip_bounds restricts valid cells (distributed OWNED clip).
    """
    import numpy as _np
    _, Nx, Ny, Nz = u_field.shape
    if "k" not in _SAMPLE_KERNEL_CACHE:
        _SAMPLE_KERNEL_CACHE["k"] = xp.RawKernel(
            _SAMPLE_KERNEL_SRC, "alm_sample_markers")
    ker = _SAMPLE_KERNEL_CACHE["k"]
    n = marker_positions.shape[0]
    eps = _np.asarray(marker_epsilon, dtype=_np.float64)
    half = int(_np.ceil(n_cut * float(eps.max()))) + 1
    pos = xp.asarray(marker_positions, dtype=xp.float64).ravel()
    eps2 = xp.asarray(eps * eps)
    rc2 = xp.asarray((n_cut * eps) ** 2)
    num = xp.empty(3 * n, xp.float64)
    den = xp.empty(n, xp.float64)
    cb = clip_bounds or ((0, Nx), (0, Ny), (0, Nz))
    ker((n,), (256,),
        (u_field, pos, eps2, rc2, num, den,
         _np.int32(Nx), _np.int32(Ny), _np.int32(Nz), _np.int32(half),
         _np.int32(max(0, cb[0][0])), _np.int32(min(Nx, cb[0][1])),
         _np.int32(max(0, cb[1][0])), _np.int32(min(Ny, cb[1][1])),
         _np.int32(max(0, cb[2][0])), _np.int32(min(Nz, cb[2][1]))))
    return xp.asnumpy(num).reshape(n, 3), xp.asnumpy(den)


def _interpolate_varying_eps_gpu(
    u_field: 'npt.NDArray',
    marker_positions: 'npt.NDArray',
    marker_epsilon: 'npt.NDArray',
    xp,
    n_cut: float,
    return_sums: bool = False
) -> 'npt.NDArray':
    """GPU interpolation when markers have different Gaussian widths

    Strategy: Use the MAXIMUM ε to define the stencil size, then apply
    per-marker cutoff via masking. Markers with smaller ε simply have
    more zero-weight nodes in their stencil — computationally wasteful
    but fully vectorized.

    This is acceptable when the ε variation is modest (e.g., ε(r) based
    on local chord vs. 2Δx clamping). For extreme variation, chunk-based
    processing would be more memory-efficient.

    Args:
        u_field: (3, Nx, Ny, Nz) GPU array  [Δx/Δt]
        marker_positions: (N_markers, 3) numpy  [lu]
        marker_epsilon: (N_markers,) numpy  [lu]
        xp: cupy module
        n_cut: Cutoff multiplier  [dimensionless]

    Returns:
        u_markers: (N_markers, 3) GPU array  [Δx/Δt]
    """
    _, Nx, Ny, Nz = u_field.shape
    n_markers = marker_positions.shape[0]

    # Use max ε for stencil sizing
    eps_max = float(np.max(marker_epsilon))          # [lu]
    r_cut_max = n_cut * eps_max                      # [lu]
    # +1 to capture all nodes §2's floor/ceil bounding box would include
    half = int(np.ceil(r_cut_max)) + 1               # [lu, integer]

    # ── CPU → GPU: marker data ──
    pos_gpu = xp.asarray(marker_positions, dtype=xp.float64)   # (N,3) [lu]
    eps_gpu = xp.asarray(marker_epsilon, dtype=xp.float64)     # (N,)  [lu]
    eps_sq_gpu = eps_gpu * eps_gpu                              # (N,)  [lu²]
    r_cut_gpu = n_cut * eps_gpu                                 # (N,)  [lu]
    r_cut_sq_gpu = r_cut_gpu * r_cut_gpu                        # (N,)  [lu²]

    # ── Stencil offsets (sized by max ε) ──
    offsets = xp.arange(-half, half + 1, dtype=xp.float64)
    S = len(offsets)
    ox, oy, oz = xp.meshgrid(offsets, offsets, offsets, indexing='ij')

    # ── Grid indices: (N_markers, S, S, S) ──
    ix0 = xp.round(pos_gpu[:, 0]).astype(xp.int32)
    iy0 = xp.round(pos_gpu[:, 1]).astype(xp.int32)
    iz0 = xp.round(pos_gpu[:, 2]).astype(xp.int32)

    gx = ix0[:, None, None, None] + ox[None, :, :, :].astype(xp.int32)
    gy = iy0[:, None, None, None] + oy[None, :, :, :].astype(xp.int32)
    gz = iz0[:, None, None, None] + oz[None, :, :, :].astype(xp.int32)

    valid = (
        (gx >= 0) & (gx < Nx) &
        (gy >= 0) & (gy < Ny) &
        (gz >= 0) & (gz < Nz)
    )

    gx_c = xp.clip(gx, 0, Nx - 1)
    gy_c = xp.clip(gy, 0, Ny - 1)
    gz_c = xp.clip(gz, 0, Nz - 1)

    # ── Distances ──
    dx = gx.astype(xp.float64) - pos_gpu[:, 0, None, None, None]
    dy = gy.astype(xp.float64) - pos_gpu[:, 1, None, None, None]
    dz = gz.astype(xp.float64) - pos_gpu[:, 2, None, None, None]
    d_sq = dx * dx + dy * dy + dz * dz   # (N, S, S, S) [lu²]

    # ── Per-marker Gaussian weights with per-marker cutoff ──
    # eps_sq_gpu, r_cut_sq_gpu: (N,) → (N,1,1,1) for broadcasting
    weights = xp.where(
        valid & (d_sq <= r_cut_sq_gpu[:, None, None, None]),
        xp.exp(-d_sq / eps_sq_gpu[:, None, None, None]),
        0.0
    )

    W_raw = xp.sum(weights, axis=(1, 2, 3))
    W_sum = xp.maximum(W_raw, 1e-30)

    # ── Weighted velocity ──
    u_markers = xp.zeros((n_markers, 3), dtype=xp.float64)

    for d in range(3):
        u_local = u_field[d, gx_c, gy_c, gz_c]
        u_local = u_local * weights
        u_markers[:, d] = xp.sum(u_local, axis=(1, 2, 3))

    if return_sums:
        # Distributed partial sums (multi-GPU, patch 17 M3): raw numerator
        # and UNclamped denominator — the caller allreduces then divides.
        return u_markers, W_raw
    u_markers /= W_sum[:, None]
    return u_markers   # (N_markers, 3) GPU array [Δx/Δt]


# =============================================================================
# §7. GPU vs CPU Verification Utility
# =============================================================================

def verify_gpu_interpolation(
    u_field_np: 'npt.NDArray',
    marker_positions: 'npt.NDArray',
    marker_epsilon: 'npt.NDArray',
    n_cut: float = 3.0,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> dict:
    """Verify GPU interpolation against CPU reference

    Runs both CPU (§3) and GPU (§6) paths, comparing results element-wise.
    This function requires cupy to be importable.

    Args:
        u_field_np: Velocity field (numpy), shape (3, Nx, Ny, Nz)  [Δx/Δt]
        marker_positions: (N_markers, 3) numpy  [lu]
        marker_epsilon: (N_markers,) numpy  [lu]
        n_cut: Cutoff multiplier  [dimensionless]
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        dict with:
            - match: bool — True if all within tolerance
            - max_abs_error: Maximum absolute difference  [Δx/Δt]
            - max_rel_error: Maximum relative difference  [dimensionless]
            - u_cpu: CPU result (N_markers, 3)
            - u_gpu: GPU result (N_markers, 3)
    """
    try:
        import cupy as cp
    except ImportError:
        return {'match': None, 'error': 'cupy not available'}

    # CPU reference
    u_cpu = interpolate_velocity_batch(
        u_field_np, marker_positions, marker_epsilon, n_cut=n_cut
    )

    # GPU path
    u_field_gpu = cp.asarray(u_field_np)
    u_gpu = interpolate_velocity_batch_gpu(
        u_field_gpu, marker_positions, marker_epsilon,
        xp=cp, n_cut=n_cut
    )

    # Compare
    abs_err = np.abs(u_cpu - u_gpu)
    max_abs = float(np.max(abs_err))

    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(u_cpu), 1e-30)
    rel_err = abs_err / denom
    max_rel = float(np.max(rel_err[np.abs(u_cpu) > atol]))  \
        if np.any(np.abs(u_cpu) > atol) else 0.0

    match = np.allclose(u_cpu, u_gpu, rtol=rtol, atol=atol)

    return {
        'match': match,
        'max_abs_error': max_abs,
        'max_rel_error': max_rel,
        'u_cpu': u_cpu,
        'u_gpu': u_gpu,
    }


# =============================================================================
# §7. Alternative Velocity Samplers (A/B study)
# =============================================================================
#
# See patch_notes/almlbm_sampler_ab/ and docs/almlbm_paper_analysis_kr.md §2①.
#
# The baseline sampler (§6 interpolate_velocity_batch_gpu, mode "gaussian") is
# the isotropic ±3ε Gaussian average. These add three opt-in alternatives that
# isolate the tip-inflow contribution of the sampling kernel (analysis (b)):
#   - "point"      : trilinear (8-node) interpolation, no spatial filter   (B-i)
#   - "aniso"      : anisotropic Gaussian, radial width ε_r = factor·ε     (B-ii)
#   - "mask_disk"  : isotropic Gaussian, off-disk nodes (cyl radius > R)
#                    dropped from the weighted average                     (B-iii)
# All are xp-generic (numpy CPU / cupy GPU) and return numpy (BEM is CPU-only).
# Mode "gaussian" is NOT routed here — the caller keeps the bit-identical §6 path.
# =============================================================================


def sample_velocity_alt(
    mode: str,
    u_field: 'npt.NDArray',
    marker_positions: 'npt.NDArray',
    marker_epsilon: 'npt.NDArray',
    xp,
    n_cut: float = 3.0,
    hub: 'npt.NDArray' = None,
    axis: 'npt.NDArray' = None,
    radius: float = None,
    eps_r_factor: float = 0.5,
    ec: 'npt.NDArray' = None,
    et: 'npt.NDArray' = None,
    ring_n: int = 20,
    ring_r_factor: float = 1.0,
) -> 'npt.NDArray':
    """Dispatch to an alternative velocity sampler (B-i / B-ii / B-iii / ring).

    Args:
        mode: "point" | "aniso" | "mask_disk" | "ring"  (NOT "gaussian" — use §6).
        u_field: (3, Nx, Ny, Nz)  [Δx/Δt]  (numpy or cupy)
        marker_positions: (N_markers, 3)  [lu]
        marker_epsilon: (N_markers,)  [lu]
        xp: numpy or cupy
        n_cut: Gaussian cutoff in units of ε  [dimensionless]
        hub: Rotor hub center (3,)  [lu]   — needed for "aniso", "mask_disk"
        axis: Rotor rotation axis (3,)  [unit] — needed for "aniso", "mask_disk"
        radius: Rotor tip radius R  [lu]   — needed for "mask_disk"
        eps_r_factor: Radial width factor ε_r = factor·ε — "aniso" only

    Returns:
        u_markers: (N_markers, 3)  [Δx/Δt]  — always numpy (BEM is CPU).
    """
    if mode == "point":
        u = _sample_trilinear(u_field, marker_positions, xp)
    elif mode == "ring":
        u = _sample_ring(u_field, marker_positions, ec, et, marker_epsilon,
                         xp, ring_n, ring_r_factor)
    elif mode == "aniso":
        u = _sample_aniso(u_field, marker_positions, marker_epsilon, xp,
                          n_cut, hub, axis, eps_r_factor)
    elif mode == "mask_disk":
        u = _sample_mask_disk(u_field, marker_positions, marker_epsilon, xp,
                              n_cut, hub, axis, radius)
    else:
        raise ValueError(
            f"Unknown sampling mode {mode!r} "
            "(expected 'point', 'ring', 'aniso', or 'mask_disk')"
        )
    if xp.__name__ != 'numpy':
        u = xp.asnumpy(u)
    return u  # (N_markers, 3) numpy [Δx/Δt]


def _alt_stencil(u_field, marker_positions, marker_epsilon, xp, n_cut):
    """Build the shared (N, S, S, S) stencil tensors (mirrors §6.2).

    Returns a dict of arrays on xp's device:
        gx_c, gy_c, gz_c : clipped int32 grid indices for safe gather
        dx, dy, dz       : float64 offsets g - marker  [lu]
        d_sq             : squared distance  [lu²]
        valid            : bool — node inside the domain
        pos              : (N,3) marker positions  [lu]
        eps              : (N,)  per-marker ε  [lu]
    """
    _, Nx, Ny, Nz = u_field.shape
    eps_max = float(np.max(marker_epsilon))                 # [lu]
    half = int(np.ceil(n_cut * eps_max)) + 1                # [lu] stencil half-width
    pos = xp.asarray(marker_positions, dtype=xp.float64)    # (N,3) [lu]
    eps = xp.asarray(marker_epsilon, dtype=xp.float64)      # (N,)  [lu]

    offsets = xp.arange(-half, half + 1, dtype=xp.float64)
    ox, oy, oz = xp.meshgrid(offsets, offsets, offsets, indexing='ij')

    ix0 = xp.round(pos[:, 0]).astype(xp.int32)
    iy0 = xp.round(pos[:, 1]).astype(xp.int32)
    iz0 = xp.round(pos[:, 2]).astype(xp.int32)

    gx = ix0[:, None, None, None] + ox[None, :, :, :].astype(xp.int32)
    gy = iy0[:, None, None, None] + oy[None, :, :, :].astype(xp.int32)
    gz = iz0[:, None, None, None] + oz[None, :, :, :].astype(xp.int32)

    valid = (
        (gx >= 0) & (gx < Nx) &
        (gy >= 0) & (gy < Ny) &
        (gz >= 0) & (gz < Nz)
    )
    gx_c = xp.clip(gx, 0, Nx - 1)
    gy_c = xp.clip(gy, 0, Ny - 1)
    gz_c = xp.clip(gz, 0, Nz - 1)

    dx = gx.astype(xp.float64) - pos[:, 0, None, None, None]   # [lu]
    dy = gy.astype(xp.float64) - pos[:, 1, None, None, None]
    dz = gz.astype(xp.float64) - pos[:, 2, None, None, None]
    d_sq = dx * dx + dy * dy + dz * dz                          # [lu²]

    return {'gx_c': gx_c, 'gy_c': gy_c, 'gz_c': gz_c,
            'dx': dx, 'dy': dy, 'dz': dz, 'd_sq': d_sq,
            'valid': valid, 'pos': pos, 'eps': eps}


def _alt_gather(u_field, st, weights, xp):
    """Normalized weighted gather: u_j = Σ w·u / Σ w  per marker."""
    n = weights.shape[0]
    W = xp.maximum(xp.sum(weights, axis=(1, 2, 3)), 1e-30)     # (N,)
    out = xp.zeros((n, 3), dtype=xp.float64)
    for d in range(3):
        u_local = u_field[d, st['gx_c'], st['gy_c'], st['gz_c']]  # (N,S,S,S)
        out[:, d] = xp.sum(u_local * weights, axis=(1, 2, 3)) / W
    return out  # (N,3) [Δx/Δt]


def _sample_trilinear(u_field, marker_positions, xp):
    """B-i: trilinear interpolation from the 8 surrounding nodes (no filter)."""
    _, Nx, Ny, Nz = u_field.shape
    n = marker_positions.shape[0]
    pos = xp.asarray(marker_positions, dtype=xp.float64)
    i0 = xp.floor(pos[:, 0]).astype(xp.int32); fx = pos[:, 0] - i0
    j0 = xp.floor(pos[:, 1]).astype(xp.int32); fy = pos[:, 1] - j0
    k0 = xp.floor(pos[:, 2]).astype(xp.int32); fz = pos[:, 2] - k0

    out = xp.zeros((n, 3), dtype=xp.float64)
    for cx, wx in ((0, 1.0 - fx), (1, fx)):
        ix = xp.clip(i0 + cx, 0, Nx - 1)
        for cy, wy in ((0, 1.0 - fy), (1, fy)):
            iy = xp.clip(j0 + cy, 0, Ny - 1)
            for cz, wz in ((0, 1.0 - fz), (1, fz)):
                iz = xp.clip(k0 + cz, 0, Nz - 1)
                w = wx * wy * wz                              # (N,)
                for d in range(3):
                    out[:, d] += w * u_field[d, ix, iy, iz]
    return out  # (N,3) [Δx/Δt]


def _sample_ring(u_field, marker_positions, ec, et, marker_epsilon, xp,
                 n_sensors=20, r_factor=1.0):
    """Ring inflow sampling (Natelson 2026 / ROAM): average the trilinear velocity
    over ``n_sensors`` points on a circle in the airfoil SECTION plane (spanned by
    the chord ec and thickness et unit vectors) at radius ``r_factor·ε`` around
    each marker. This averages out the marker's own smeared body-force deficit at
    the centre, giving a cleaner estimate of the inflow the blade section sees.

    Args:
        u_field: (3, Nx, Ny, Nz)  [Δx/Δt]
        marker_positions: (N, 3)  [lu]
        ec, et: (N, 3) chord / thickness unit vectors (section plane basis)
        marker_epsilon: (N,)  [lu]
        n_sensors: number of ring points (Natelson: 20)
        r_factor: ring radius = r_factor · ε

    Returns:
        u_markers: (N, 3)  [Δx/Δt]  (xp array; caller moves to host)
    """
    n = marker_positions.shape[0]
    mp = np.asarray(marker_positions, dtype=np.float64)              # (N,3)
    ec = np.asarray(ec, dtype=np.float64)
    et = np.asarray(et, dtype=np.float64)
    R = (r_factor * np.asarray(marker_epsilon, dtype=np.float64))[:, None]  # (N,1)
    acc = xp.zeros((n, 3), dtype=xp.float64)
    for i in range(n_sensors):
        phi = 2.0 * np.pi * i / n_sensors
        # sensor offset lies in the section plane -> zero component along span
        offset = np.cos(phi) * ec + np.sin(phi) * et                # (N,3) unit
        sensor_pos = mp + R * offset                                # (N,3) [lu]
        acc = acc + _sample_trilinear(u_field, sensor_pos, xp)
    return acc / float(n_sensors)                                   # (N,3) [Δx/Δt]


def _sample_aniso(u_field, marker_positions, marker_epsilon, xp,
                  n_cut, hub, axis, eps_r_factor):
    """B-ii: anisotropic Gaussian — radial width ε_r=factor·ε, ε_⊥=ε."""
    st = _alt_stencil(u_field, marker_positions, marker_epsilon, xp, n_cut)
    eps = st['eps']
    rcut_sq = ((n_cut * eps) ** 2)[:, None, None, None]
    eps_sq = (eps ** 2)[:, None, None, None]
    eps_r_sq = ((eps_r_factor * eps) ** 2)[:, None, None, None]

    hub = xp.asarray(hub, dtype=xp.float64)
    axis = xp.asarray(axis, dtype=xp.float64)
    axis = axis / xp.sqrt(xp.sum(axis * axis))

    # Per-marker radial unit vector ê_r = normalize((p-hub)_⊥axis)
    ph = st['pos'] - hub[None, :]                              # (N,3)
    pa = ph[:, 0] * axis[0] + ph[:, 1] * axis[1] + ph[:, 2] * axis[2]
    er = ph - pa[:, None] * axis[None, :]                      # (N,3)
    er = er / xp.maximum(xp.sqrt(xp.sum(er * er, axis=1)), 1e-30)[:, None]

    # Radial / perpendicular split of the offset d
    d_r = (st['dx'] * er[:, 0, None, None, None] +
           st['dy'] * er[:, 1, None, None, None] +
           st['dz'] * er[:, 2, None, None, None])              # (N,S,S,S)
    d_perp_sq = xp.maximum(st['d_sq'] - d_r * d_r, 0.0)
    arg = d_r * d_r / eps_r_sq + d_perp_sq / eps_sq

    weights = xp.where(st['valid'] & (st['d_sq'] <= rcut_sq),
                       xp.exp(-arg), 0.0)
    return _alt_gather(u_field, st, weights, xp)


def _sample_mask_disk(u_field, marker_positions, marker_epsilon, xp,
                      n_cut, hub, axis, radius):
    """B-iii: isotropic Gaussian, off-disk nodes (cyl radius > R) dropped."""
    st = _alt_stencil(u_field, marker_positions, marker_epsilon, xp, n_cut)
    eps = st['eps']
    rcut_sq = ((n_cut * eps) ** 2)[:, None, None, None]
    eps_sq = (eps ** 2)[:, None, None, None]

    hub = xp.asarray(hub, dtype=xp.float64)
    axis = xp.asarray(axis, dtype=xp.float64)
    axis = axis / xp.sqrt(xp.sum(axis * axis))

    # Node global positions g = marker + d; cylindrical radius from rotor axis
    gpx = st['pos'][:, 0, None, None, None] + st['dx'] - hub[0]
    gpy = st['pos'][:, 1, None, None, None] + st['dy'] - hub[1]
    gpz = st['pos'][:, 2, None, None, None] + st['dz'] - hub[2]
    ax_comp = gpx * axis[0] + gpy * axis[1] + gpz * axis[2]    # axial proj
    cyl_sq = xp.maximum((gpx * gpx + gpy * gpy + gpz * gpz) - ax_comp * ax_comp, 0.0)

    weights = xp.where(
        st['valid'] & (st['d_sq'] <= rcut_sq) & (cyl_sq <= radius * radius),
        xp.exp(-st['d_sq'] / eps_sq), 0.0)
    return _alt_gather(u_field, st, weights, xp)