"""
Vortex Identification Criteria for LBM Post-Processing

This module computes vortex identification scalar fields from the
velocity gradient tensor, enabling iso-surface visualization of
coherent vortex structures in ParaView.

Implemented Criteria:
====================

1. Q-criterion (Hunt et al., 1988):
   Q = ½(||Ω||² - ||S||²)

   where:
     S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i)   — strain rate tensor (symmetric)
     Ω_ij = ½(∂u_i/∂x_j - ∂u_j/∂x_i)   — rotation rate tensor (antisymmetric)

   Vortex core identified where Q > 0 (rotation dominates strain).
   ||·||² denotes the Frobenius norm squared: ||A||² = Σ_ij A_ij²

2. Lambda2 (λ₂) criterion (Jeong & Hussain, 1995):
   Eigenvalues of (S² + Ω²): λ₁ ≤ λ₂ ≤ λ₃
   Vortex core identified where λ₂ < 0.

   Physical interpretation: λ₂ < 0 indicates a local pressure minimum
   in a plane, excluding unsteady straining and viscous effects.

3. Vorticity magnitude |ω|:
   ω = ∇ × u
   |ω| = √(ω_x² + ω_y² + ω_z²)

Velocity Gradient Computation:
    ∂u_i/∂x_j is computed via 2nd-order central finite differences:
        ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2·Δx)

    At domain boundaries, 1st-order one-sided differences are used.

Normalization:
    For visualization, it is common to normalize Q by (U_ref/L_ref)²
    or use dimensionless Q* = Q · (D/U_∞)².

Performance:
    - Q-criterion: O(N) — only requires velocity gradients
    - Lambda2: O(N) — requires 3×3 eigenvalue solve per grid point
      (uses characteristic polynomial for analytical solution)
    - All operations are fully vectorized (no Python loops over grid)

References:
    - Hunt, Wray & Moin, CTR-S88, 1988 (Q-criterion)
    - Jeong & Hussain, J. Fluid Mech. 285, 69-94, 1995 (Lambda2)
    - Kolář, Int. J. Heat Fluid Flow 28, 13-18, 2007 (Review)

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Tuple, Dict

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


# =============================================================================
# §1. Velocity Gradient Tensor
# =============================================================================

def compute_velocity_gradient_3d(
    u: 'npt.NDArray',
    dx: float = 1.0,
    solid_mask: Optional['npt.NDArray'] = None
) -> 'npt.NDArray':
    """Compute the 3D velocity gradient tensor ∂u_i/∂x_j

    Uses 2nd-order central differences in the interior and
    1st-order one-sided differences at domain boundaries.

        ∂u_i/∂x_j ≈ (u_i[..., x_j+1, ...] - u_i[..., x_j-1, ...]) / (2·Δx)

    Args:
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        dx: Grid spacing  [lu] (default 1.0 in lattice units)
        solid_mask: Optional solid mask (Nx, Ny, Nz), True = solid.
                    Gradients at/near solid nodes are set to zero.

    Returns:
        grad_u: Velocity gradient tensor, shape (3, 3, Nx, Ny, Nz)  [1/Δt]
                grad_u[i, j] = ∂u_i/∂x_j
    """
    _, Nx, Ny, Nz = u.shape
    grad_u = np.zeros((3, 3, Nx, Ny, Nz), dtype=np.float64)

    inv_2dx = 1.0 / (2.0 * dx)  # [1/lu]

    # --- ∂/∂x (axis=0 of spatial dims, axis=1 of u) ---
    # Central difference in interior
    grad_u[:, 0, 1:-1, :, :] = (u[:, 2:, :, :] - u[:, :-2, :, :]) * inv_2dx
    # One-sided at boundaries
    inv_dx = 1.0 / dx
    grad_u[:, 0, 0, :, :] = (u[:, 1, :, :] - u[:, 0, :, :]) * inv_dx
    grad_u[:, 0, -1, :, :] = (u[:, -1, :, :] - u[:, -2, :, :]) * inv_dx

    # --- ∂/∂y (axis=1 of spatial dims, axis=2 of u) ---
    grad_u[:, 1, :, 1:-1, :] = (u[:, :, 2:, :] - u[:, :, :-2, :]) * inv_2dx
    grad_u[:, 1, :, 0, :] = (u[:, :, 1, :] - u[:, :, 0, :]) * inv_dx
    grad_u[:, 1, :, -1, :] = (u[:, :, -1, :] - u[:, :, -2, :]) * inv_dx

    # --- ∂/∂z (axis=2 of spatial dims, axis=3 of u) ---
    grad_u[:, 2, :, :, 1:-1] = (u[:, :, :, 2:] - u[:, :, :, :-2]) * inv_2dx
    grad_u[:, 2, :, :, 0] = (u[:, :, :, 1] - u[:, :, :, 0]) * inv_dx
    grad_u[:, 2, :, :, -1] = (u[:, :, :, -1] - u[:, :, :, -2]) * inv_dx

    # Zero out gradients at solid nodes
    if solid_mask is not None:
        grad_u[:, :, solid_mask] = 0.0

    return grad_u  # (3, 3, Nx, Ny, Nz)  [1/Δt]


def compute_strain_rotation(
    grad_u: 'npt.NDArray'
) -> Tuple['npt.NDArray', 'npt.NDArray']:
    """Decompose velocity gradient into strain rate and rotation rate tensors

    S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i)   — symmetric part     [1/Δt]
    Ω_ij = ½(∂u_i/∂x_j - ∂u_j/∂x_i)   — antisymmetric part [1/Δt]

    Such that: ∂u_i/∂x_j = S_ij + Ω_ij

    Args:
        grad_u: Velocity gradient, shape (3, 3, Nx, Ny, Nz)  [1/Δt]

    Returns:
        S: Strain rate tensor, shape (3, 3, Nx, Ny, Nz)  [1/Δt]
        Omega: Rotation rate tensor, shape (3, 3, Nx, Ny, Nz)  [1/Δt]
    """
    # grad_u[i,j] = ∂u_i/∂x_j
    # grad_u_T[i,j] = grad_u[j,i] = ∂u_j/∂x_i
    grad_u_T = np.swapaxes(grad_u, 0, 1)  # Transpose first two indices

    S = 0.5 * (grad_u + grad_u_T)          # [1/Δt]
    Omega = 0.5 * (grad_u - grad_u_T)      # [1/Δt]

    return S, Omega


# =============================================================================
# §2. Q-criterion
# =============================================================================

def compute_q_criterion(
    u: 'npt.NDArray',
    dx: float = 1.0,
    solid_mask: Optional['npt.NDArray'] = None
) -> 'npt.NDArray':
    """Compute Q-criterion scalar field (Hunt et al., 1988)

    Q = ½(||Ω||² - ||S||²)   [1/Δt²]

    where ||A||² = Σ_{ij} A_{ij}² (Frobenius norm squared)

    Physical Meaning:
        Q > 0: Rotation dominates strain → VORTEX CORE
        Q < 0: Strain dominates rotation → Shear/extensional flow
        Q = 0: Balance between rotation and strain

    ParaView Visualization:
        Apply "Contour" filter → Select "Q_criterion" → Set iso-value > 0
        Typical iso-values: Q* = Q·(D/U)² ∈ [0.1, 10]

    Args:
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        dx: Grid spacing  [lu]
        solid_mask: Optional solid mask (True = solid)

    Returns:
        Q: Q-criterion field, shape (Nx, Ny, Nz)  [1/Δt²]
    """
    grad_u = compute_velocity_gradient_3d(u, dx, solid_mask)
    S, Omega = compute_strain_rotation(grad_u)

    # ||S||² = Σ_{ij} S_{ij}²  (sum over all 9 components)
    S_norm_sq = np.sum(S * S, axis=(0, 1))          # (Nx, Ny, Nz) [1/Δt²]

    # ||Ω||² = Σ_{ij} Ω_{ij}²
    Omega_norm_sq = np.sum(Omega * Omega, axis=(0, 1))  # (Nx, Ny, Nz) [1/Δt²]

    Q = 0.5 * (Omega_norm_sq - S_norm_sq)           # (Nx, Ny, Nz) [1/Δt²]

    # Zero out solid nodes
    if solid_mask is not None:
        Q[solid_mask] = 0.0

    return Q


# =============================================================================
# §3. Lambda2 Criterion
# =============================================================================

def compute_lambda2(
    u: 'npt.NDArray',
    dx: float = 1.0,
    solid_mask: Optional['npt.NDArray'] = None
) -> 'npt.NDArray':
    """Compute Lambda2 (λ₂) criterion (Jeong & Hussain, 1995)

    Steps:
        1. Compute S and Ω from velocity gradient
        2. Form the tensor M = S² + Ω²
        3. Find eigenvalues of M: λ₁ ≤ λ₂ ≤ λ₃
        4. Vortex core where λ₂ < 0

    Physical Meaning:
        λ₂ < 0 indicates a pressure minimum in a plane due to
        vortical motion, excluding effects of unsteady straining
        and viscous diffusion.

    The 3×3 symmetric eigenvalue problem is solved analytically
    using the characteristic polynomial (Cardano's formula),
    fully vectorized over the entire grid.

    ParaView Visualization:
        Apply "Contour" filter → Select "Lambda2" → Set iso-value < 0
        Typical: λ₂* = λ₂·(D/U)² ∈ [-10, -0.1]

    Args:
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        dx: Grid spacing  [lu]
        solid_mask: Optional solid mask

    Returns:
        lambda2: λ₂ field, shape (Nx, Ny, Nz)  [1/Δt²]
    """
    grad_u = compute_velocity_gradient_3d(u, dx, solid_mask)
    S, Omega = compute_strain_rotation(grad_u)

    # M = S·S + Ω·Ω  (matrix product at each grid point)
    # M[i,j] = Σ_k (S[i,k]*S[k,j] + Ω[i,k]*Ω[k,j])
    # Using einsum for vectorized matrix multiplication
    M = np.einsum('ik...,kj...->ij...', S, S) + \
        np.einsum('ik...,kj...->ij...', Omega, Omega)
    # M: shape (3, 3, Nx, Ny, Nz)  [1/Δt²]

    # M is real symmetric → eigenvalues are real
    # Solve analytically using characteristic polynomial
    lambda2 = _symmetric_3x3_middle_eigenvalue(M)

    if solid_mask is not None:
        lambda2[solid_mask] = 0.0

    return lambda2  # (Nx, Ny, Nz)  [1/Δt²]


def _symmetric_3x3_middle_eigenvalue(
    M: 'npt.NDArray'
) -> 'npt.NDArray':
    """Compute the middle eigenvalue (λ₂) of a 3×3 symmetric tensor field

    Uses the analytical solution via the characteristic polynomial
    of a real symmetric 3×3 matrix (Cardano's formula).

    For a symmetric 3×3 matrix A with eigenvalues λ₁ ≤ λ₂ ≤ λ₃,
    the characteristic polynomial is:
        λ³ - I₁λ² + I₂λ - I₃ = 0

    where:
        I₁ = tr(A)                           — first invariant
        I₂ = ½(tr(A)² - tr(A²))             — second invariant
        I₃ = det(A)                          — third invariant

    The roots are found using the trigonometric method for three real roots.

    Args:
        M: Symmetric tensor field, shape (3, 3, Nx, Ny, Nz)

    Returns:
        lambda2: Middle eigenvalue field, shape (Nx, Ny, Nz)
    """
    # Extract components (M is symmetric: M[i,j] = M[j,i])
    a = M[0, 0]  # M_xx
    b = M[1, 1]  # M_yy
    c = M[2, 2]  # M_zz
    d = M[0, 1]  # M_xy = M_yx
    e = M[0, 2]  # M_xz = M_zx
    f = M[1, 2]  # M_yz = M_zy

    # First invariant: I₁ = trace(M) = a + b + c
    I1 = a + b + c

    # Second invariant: I₂ = ½(tr²(M) - tr(M²))
    # tr(M²) = a² + b² + c² + 2(d² + e² + f²)  for symmetric M
    tr_M2 = a*a + b*b + c*c + 2.0*(d*d + e*e + f*f)
    I2 = 0.5 * (I1*I1 - tr_M2)

    # Third invariant: I₃ = det(M)
    # det = a(bc - f²) - d(dc - ef) + e(df - eb)
    I3 = a*(b*c - f*f) - d*(d*c - e*f) + e*(d*f - e*b)

    # --- Cardano's formula for x³ - I₁x² + I₂x - I₃ = 0 ---
    # Substitute x = t + I₁/3 to get depressed cubic:
    # t³ + pt + q = 0
    p = I2 - I1*I1 / 3.0
    q = (2.0*I1*I1*I1 - 9.0*I1*I2 + 27.0*I3) / 27.0

    # Discriminant for three real roots: p must be ≤ 0 for symmetric M
    # For numerical safety, clamp p
    p = np.minimum(p, 0.0)

    # r = √(-p/3),  cos(3θ) = q / (2r³)
    r = np.sqrt(-p / 3.0 + 1e-30)

    # Clamp argument of arccos to [-1, 1]
    cos_arg = np.clip(q / (2.0 * r*r*r + 1e-30), -1.0, 1.0)
    theta = np.arccos(cos_arg) / 3.0

    # Three eigenvalues (sorted: λ₁ ≤ λ₂ ≤ λ₃)
    shift = I1 / 3.0
    # λ₁ = shift + 2r·cos(θ + 2π/3)  (smallest)
    # λ₂ = shift + 2r·cos(θ + 4π/3)  (middle)
    # λ₃ = shift + 2r·cos(θ)          (largest)

    two_pi_3 = 2.0 * np.pi / 3.0

    lam1 = shift + 2.0 * r * np.cos(theta + 2.0 * two_pi_3)
    lam2 = shift + 2.0 * r * np.cos(theta + two_pi_3)
    lam3 = shift + 2.0 * r * np.cos(theta)

    # Ensure correct ordering: λ₁ ≤ λ₂ ≤ λ₃
    # The trigonometric solution may not always give ordered results
    # due to numerical precision, so we sort
    eigenvalues = np.stack([lam1, lam2, lam3], axis=0)  # (3, Nx, Ny, Nz)
    eigenvalues.sort(axis=0)

    return eigenvalues[1]  # λ₂ = middle eigenvalue


# =============================================================================
# §4. Vorticity
# =============================================================================

def compute_vorticity_3d(
    u: 'npt.NDArray',
    dx: float = 1.0,
    solid_mask: Optional['npt.NDArray'] = None
) -> 'npt.NDArray':
    """Compute vorticity vector field ω = ∇ × u

    Components:
        ω_x = ∂u_z/∂y - ∂u_y/∂z
        ω_y = ∂u_x/∂z - ∂u_z/∂x
        ω_z = ∂u_y/∂x - ∂u_x/∂y

    Args:
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        dx: Grid spacing  [lu]
        solid_mask: Optional solid mask

    Returns:
        omega: Vorticity vector, shape (3, Nx, Ny, Nz)  [1/Δt]
    """
    grad_u = compute_velocity_gradient_3d(u, dx, solid_mask)
    # grad_u[i, j] = ∂u_i/∂x_j

    omega = np.zeros_like(u, dtype=np.float64)
    omega[0] = grad_u[2, 1] - grad_u[1, 2]  # ω_x = ∂u_z/∂y - ∂u_y/∂z
    omega[1] = grad_u[0, 2] - grad_u[2, 0]  # ω_y = ∂u_x/∂z - ∂u_z/∂x
    omega[2] = grad_u[1, 0] - grad_u[0, 1]  # ω_z = ∂u_y/∂x - ∂u_x/∂y

    return omega  # (3, Nx, Ny, Nz)  [1/Δt]


def compute_vorticity_magnitude(
    u: 'npt.NDArray',
    dx: float = 1.0,
    solid_mask: Optional['npt.NDArray'] = None
) -> 'npt.NDArray':
    """Compute vorticity magnitude |ω| = |∇ × u|

    Args:
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        dx: Grid spacing  [lu]
        solid_mask: Optional solid mask

    Returns:
        |ω|: Vorticity magnitude, shape (Nx, Ny, Nz)  [1/Δt]
    """
    omega = compute_vorticity_3d(u, dx, solid_mask)
    return np.sqrt(np.sum(omega * omega, axis=0))


# =============================================================================
# §5. Convenience: Compute All Criteria at Once
# =============================================================================

def compute_all_vortex_criteria(
    u: 'npt.NDArray',
    dx: float = 1.0,
    solid_mask: Optional['npt.NDArray'] = None,
    compute_lambda2: bool = True
) -> Dict[str, 'npt.NDArray']:
    """Compute all vortex identification criteria in a single pass

    Shares the velocity gradient computation across all criteria,
    saving ~50% computation compared to calling each independently.

    Args:
        u: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
        dx: Grid spacing  [lu]
        solid_mask: Optional solid mask
        compute_lambda2: If True, also compute λ₂ (more expensive)

    Returns:
        dict with:
            'Q_criterion': shape (Nx, Ny, Nz)  [1/Δt²]
            'vorticity': shape (3, Nx, Ny, Nz)  [1/Δt]
            'vorticity_magnitude': shape (Nx, Ny, Nz)  [1/Δt]
            'lambda2': shape (Nx, Ny, Nz)  [1/Δt²]  (if requested)
    """
    # Shared velocity gradient (computed once)
    grad_u = compute_velocity_gradient_3d(u, dx, solid_mask)
    S, Omega = compute_strain_rotation(grad_u)

    # Q-criterion
    S_norm_sq = np.sum(S * S, axis=(0, 1))
    Omega_norm_sq = np.sum(Omega * Omega, axis=(0, 1))
    Q = 0.5 * (Omega_norm_sq - S_norm_sq)

    # Vorticity
    omega = np.zeros_like(u, dtype=np.float64)
    omega[0] = grad_u[2, 1] - grad_u[1, 2]
    omega[1] = grad_u[0, 2] - grad_u[2, 0]
    omega[2] = grad_u[1, 0] - grad_u[0, 1]
    omega_mag = np.sqrt(np.sum(omega * omega, axis=0))

    result = {
        'Q_criterion': Q,                     # [1/Δt²]
        'vorticity': omega,                    # [1/Δt]
        'vorticity_magnitude': omega_mag,      # [1/Δt]
    }

    # Lambda2 (optional, more expensive)
    if compute_lambda2:
        M = np.einsum('ik...,kj...->ij...', S, S) + \
            np.einsum('ik...,kj...->ij...', Omega, Omega)
        lam2 = _symmetric_3x3_middle_eigenvalue(M)
        if solid_mask is not None:
            lam2[solid_mask] = 0.0
        result['lambda2'] = lam2               # [1/Δt²]

    if solid_mask is not None:
        Q[solid_mask] = 0.0
        omega[:, solid_mask] = 0.0

    return result


# =============================================================================
# §6. Normalization Utilities
# =============================================================================

def normalize_q_criterion(
    Q: 'npt.NDArray',
    u_ref: float,
    L_ref: float,
    dx: float = 1.0
) -> 'npt.NDArray':
    """Normalize Q-criterion to dimensionless Q*

    Q* = Q · (L_ref / U_ref)²

    Useful for comparing across different grid resolutions.

    Args:
        Q: Raw Q-criterion  [1/Δt²]
        u_ref: Reference velocity  [Δx/Δt]
        L_ref: Reference length  [lu]
        dx: Grid spacing  [lu]

    Returns:
        Q*: Dimensionless Q-criterion  [dimensionless]
    """
    return Q * (L_ref / (u_ref + 1e-30)) ** 2


def normalize_lambda2(
    lam2: 'npt.NDArray',
    u_ref: float,
    L_ref: float
) -> 'npt.NDArray':
    """Normalize Lambda2 to dimensionless λ₂*

    λ₂* = λ₂ · (L_ref / U_ref)²

    Args:
        lam2: Raw Lambda2  [1/Δt²]
        u_ref: Reference velocity  [Δx/Δt]
        L_ref: Reference length  [lu]

    Returns:
        λ₂*: Dimensionless Lambda2  [dimensionless]
    """
    return lam2 * (L_ref / (u_ref + 1e-30)) ** 2