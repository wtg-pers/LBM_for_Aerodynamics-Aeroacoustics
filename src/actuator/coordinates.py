"""
Generalized Rotor Coordinate System for Actuator Line Model

This module provides a coordinate system abstraction that supports
arbitrary rotation axes and inflow directions, enabling both HAWT
(horizontal-axis) and VAWT (vertical-axis) wind turbine simulations.

Physical Concept:
=================
A rotor rotates about an arbitrary axis through a hub center. The
coordinate system is FULLY defined by:

    1. Hub center:        Origin of the rotor coordinate system  [m or lu]
    2. Rotation axis:     Unit vector n̂ defining the spin axis
    3. Inflow direction:  Unit vector defining wind approach direction
    4. Reference axis:    Unit vector ê_ref for θ=0 blade direction
    5. Perpendicular:     ê_perp = n̂ × ê_ref (completes right-hand system)

Wake and Thrust Direction Convention:
=====================================
    
    HAWT Example (X-axis rotation, wind from +X):
    
        Wind (Inflow)         Rotor Plane            Wake
        ═══════════════►      ┃  ┃  ┃  ┃       ═ ═ ═ ═ ═ ═ ═ ►
            U_∞               ┃  ┃  ┃  ┃          U_wake < U_∞
                              ┃  ┃  ┃  ┃
                                  │
                              rotation axis (n̂)
        
        Thrust ON BLADE (F_n > 0):  ◄───────  (opposes wind)
        Body force ON FLUID:        ───────►  (-F^AL, Eq. 13, creates wake)
    
    Key Relations:
        - inflow_direction:  Direction wind comes FROM (toward rotor)
        - wake_direction:    Same as inflow_direction (downstream of rotor)
        - F_n > 0:          Force on blade in +n̂ direction
        - Body force:       -F^AL (Eq. 13), decelerates fluid → wake
    
    IMPORTANT: For HAWT, inflow_direction should be PARALLEL to rotation axis.
               The sign determines which side is "upstream":
               - inflow = +n̂:  Wind from +n̂ side, wake on -n̂ side  
               - inflow = -n̂:  Wind from -n̂ side, wake on +n̂ side

Rotation Matrix:
================
The transformation from local rotor coordinates (r, θ) to global (x, y, z)
is performed via the rotation matrix R:

    R = [ê_ref | ê_perp | n̂]    (3×3 orthonormal matrix)
    
    x_global = hub_center + R @ [r·cos(θ), r·sin(θ), 0]ᵀ

This makes coordinate transformations explicit and efficient.

Marker Position Formula (generalized):
======================================
For a marker at radial distance r and azimuth angle θ:

    x_marker = hub + r·cos(θ)·ê_ref + r·sin(θ)·ê_perp     [m or lu]
             = hub + R @ [r·cos(θ), r·sin(θ), 0]ᵀ

Velocity Decomposition:
=======================
    u_n = u · n̂                                           [m/s]
    u_θ = u · ê_θ(θ)                                      [m/s]
    
    where ê_θ(θ) = -sin(θ)·ê_ref + cos(θ)·ê_perp
    (tangent to rotation, perpendicular to radial direction)
    
    Note: Watanabe Eq. 5 uses u·ê_r (radial direction) which coincides
    with the mathematical tangent only for HAWT_X. The tangent vector
    used here generalizes correctly to arbitrary rotation axes.

Force Projection:
=================
    F^AL = F_n·n̂ + F_θ·ê_θ(θ)                            [N or lu_force]
    
    where ê_θ(θ) = -sin(θ)·ê_ref + cos(θ)·ê_perp
    (standard tangent vector ∂x/∂θ, perpendicular to radial direction)
    
    This choice ensures azimuth-independent torque:
        Q = Σ (r_j × F^AL_j) · n̂ = Σ F_θ,j · r_j
    which is physically required for uniform inflow.

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026 (Sec. 2.2)
    - Sørensen & Shen, J. Fluids Eng. 124, 393-399, 2002

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Tuple, Optional, Union, Dict
from dataclasses import dataclass, field

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


# =============================================================================
# §1. Axis Label Utility
# =============================================================================

def _axis_label(n_axis: np.ndarray) -> str:
    """Generate a human-readable label from the rotation axis vector
    
    Standard Cartesian axes get short labels; others show the vector.
    
    Args:
        n_axis: Normalized rotation axis  [dimensionless]
    
    Returns:
        Label string, e.g. "+x-axis", "-z-axis", "(0.500, 0.000, 0.866)"
    """
    _NAMED = {
        (1, 0, 0): "x-axis",
        (0, 1, 0): "y-axis",
        (0, 0, 1): "z-axis",
    }
    for key, label in _NAMED.items():
        ref = np.array(key, dtype=np.float64)
        if np.allclose(n_axis, ref, atol=1e-10):
            return f"+{label}"
        if np.allclose(n_axis, -ref, atol=1e-10):
            return f"-{label}"
    return f"({n_axis[0]:.4f}, {n_axis[1]:.4f}, {n_axis[2]:.4f})"


# =============================================================================
# §2. Rotor Coordinate System Class
# =============================================================================

class RotorCoordinateSystem:
    """Generalized coordinate system for arbitrary rotor orientation
    
    Encapsulates all coordinate transformations needed for the AL model:
        - Marker position computation
        - Velocity decomposition (global → local)
        - Force projection (local → global)
        - Wake/Thrust direction tracking
    
    The coordinate system is defined by:
        n_axis:           Rotation axis (normal to rotor plane)
        e_ref:            Reference direction for θ=0
        e_perp:           Perpendicular direction = n̂ × ê_ref
        inflow_direction: Wind approach direction
        rotation_matrix:  R = [e_ref | e_perp | n_axis] for efficient transforms
    
    Attributes:
        hub_center: Rotor center position  [m or lu]
        n_axis: Rotation axis unit vector  [dimensionless]
        e_ref: Reference axis unit vector  [dimensionless]
        e_perp: Perpendicular axis unit vector  [dimensionless]
        inflow_direction: Wind approach direction  [dimensionless]
        rotation_matrix: 3x3 orthonormal transformation matrix
        axis_label: Human-readable label for the rotation axis
    
    Example:
        >>> coord = RotorCoordinateSystem.from_axis_vector(
        ...     hub_center=(3.66, 1.341, 0.817),
        ...     rotation_axis=[1, 0, 0]
        ... )
        >>> pos = coord.marker_position(r=0.4, theta=np.pi/4)
        >>> print(coord.wake_direction)  # [1, 0, 0] (downstream)
    """
    
    def __init__(
        self,
        hub_center: Union[Tuple[float, float, float], np.ndarray],
        rotation_axis: Union[Tuple[float, float, float], np.ndarray],
        reference_axis: Union[Tuple[float, float, float], np.ndarray],
        inflow_direction: Union[Tuple[float, float, float], np.ndarray],
    ) -> None:
        """Initialize coordinate system from axis vectors
        
        The reference_axis will be orthogonalized against the rotation_axis
        using Gram-Schmidt to ensure orthonormality.
        
        Args:
            hub_center: Rotor center position (x, y, z)  [m or lu]
            rotation_axis: Rotation axis direction (will be normalized)
            reference_axis: θ=0 blade direction (will be orthogonalized)
            inflow_direction: Wind approach direction (will be normalized)
        
        Raises:
            ValueError: If axes are parallel or zero-length
        """
        # Convert to numpy arrays
        self.hub_center = np.asarray(hub_center, dtype=np.float64)
        n = np.asarray(rotation_axis, dtype=np.float64)
        e_ref = np.asarray(reference_axis, dtype=np.float64)
        inflow = np.asarray(inflow_direction, dtype=np.float64)
        
        # Validate rotation axis
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            raise ValueError("rotation_axis cannot be zero vector")
        
        # Validate reference axis
        e_ref_norm = np.linalg.norm(e_ref)
        if e_ref_norm < 1e-12:
            raise ValueError("reference_axis cannot be zero vector")
        
        # Validate inflow direction
        inflow_norm = np.linalg.norm(inflow)
        if inflow_norm < 1e-12:
            raise ValueError("inflow_direction cannot be zero vector")
        
        # Normalize rotation axis
        self.n_axis = n / n_norm
        
        # Normalize inflow direction
        self.inflow_direction = inflow / inflow_norm
        
        # Gram-Schmidt: orthogonalize e_ref against n_axis
        # e_ref_ortho = e_ref - (e_ref · n̂) · n̂
        e_ref_proj = np.dot(e_ref, self.n_axis) * self.n_axis
        e_ref_ortho = e_ref - e_ref_proj
        
        ortho_norm = np.linalg.norm(e_ref_ortho)
        if ortho_norm < 1e-12:
            raise ValueError(
                "reference_axis is parallel to rotation_axis; "
                "they must be linearly independent"
            )
        
        self.e_ref = e_ref_ortho / ortho_norm
        
        # Complete right-hand system: ê_perp = n̂ × ê_ref
        self.e_perp = np.cross(self.n_axis, self.e_ref)
        
        # Build rotation matrix: R = [e_ref | e_perp | n_axis]
        # This transforms from local (rotor plane) to global coordinates
        self.rotation_matrix = np.column_stack([
            self.e_ref,    # Column 0: local x → e_ref direction
            self.e_perp,   # Column 1: local y → e_perp direction  
            self.n_axis    # Column 2: local z → n_axis direction
        ])
        
        self.axis_label: str = _axis_label(self.n_axis)  # e.g. "+z-axis"
        
        # Validate orthonormality
        if not self._check_orthonormality():
            raise RuntimeError("Failed to construct orthonormal coordinate system")
    
    def _check_orthonormality(self, tol: float = 1e-10) -> bool:
        """Internal orthonormality check"""
        # Check R^T R = I
        should_be_identity = self.rotation_matrix.T @ self.rotation_matrix
        return np.allclose(should_be_identity, np.eye(3), atol=tol)
    
    # -----------------------------------------------------------------
    # §2.1 Factory Method — from_axis_vector (Primary Interface)
    # -----------------------------------------------------------------
    
    @classmethod
    def from_axis_vector(
        cls,
        hub_center: Union[Tuple[float, float, float], np.ndarray],
        rotation_axis: Union[Tuple[float, float, float], np.ndarray],
        inflow_direction: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        reference_axis: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
    ) -> 'RotorCoordinateSystem':
        """Create coordinate system from a rotation axis vector
        
        This is the primary (and only) factory method.
        Supply the rotation axis as a 3-component vector; the remaining
        axes are auto-derived unless explicitly provided.
        
        Auto-Derivation Logic:
            1. inflow_direction:
               - If provided → use as-is (allows VAWT / oblique inflow)
               - If omitted  → inflow = rotation_axis (standard HAWT)
            
            2. reference_axis (θ=0 blade direction):
               - If provided → orthogonalized against n̂ via Gram-Schmidt
               - If omitted  → auto-generated perpendicular to n̂
                 Algorithm: pick the global axis LEAST aligned with n̂,
                 then Gram-Schmidt.  This guarantees a well-conditioned
                 ê_ref for any rotation axis, including tilted ones.
        
        Config Usage:
            "rotation_axis": [1, 0, 0]        # HAWT along X
            "rotation_axis": [0, 0, 1]        # HAWT along Z
            "rotation_axis": [1, 0, 0.1]      # tilted rotor
        
        Args:
            hub_center:       Rotor center position        [m or lu]
            rotation_axis:    Rotation axis direction       [will be normalized]
            inflow_direction: Wind approach direction       [optional, default = rotation_axis]
            reference_axis:   θ=0 blade direction           [optional, auto-derived if omitted]
        
        Returns:
            Configured RotorCoordinateSystem
        
        Raises:
            ValueError: If rotation_axis is zero-length
        
        Example:
            >>> coord = RotorCoordinateSystem.from_axis_vector(
            ...     hub_center=(5, 3, 3), rotation_axis=[1, 0, 0]
            ... )
            >>> print(coord.axis_label)   # +x-axis
            
            >>> # Tilted rotor (5 deg pitch)
            >>> coord = RotorCoordinateSystem.from_axis_vector(
            ...     hub_center=(5, 3, 3),
            ...     rotation_axis=[np.cos(np.radians(5)), 0, np.sin(np.radians(5))]
            ... )
        """
        # --- Input validation: must be a numeric sequence, not a string ---
        if isinstance(rotation_axis, str):
            raise TypeError(
                f"rotation_axis must be a 3D vector (list/tuple/ndarray), "
                f"got string '{rotation_axis}'. "
                f"Example: rotation_axis=[1, 0, 0]"
            )
        
        n = np.asarray(rotation_axis, dtype=np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            raise ValueError("rotation_axis cannot be zero vector")
        n_hat = n / n_norm  # [dimensionless]
        
        # --- Inflow direction ---
        if inflow_direction is not None:
            inflow = np.asarray(inflow_direction, dtype=np.float64)
        else:
            inflow = n_hat.copy()  # HAWT default: inflow parallel to rotation axis
        
        # --- Reference axis (auto-derive if not provided) ---
        if reference_axis is not None:
            e_ref_candidate = np.asarray(reference_axis, dtype=np.float64)
        else:
            # Pick the global axis LEAST aligned with n̂
            # This avoids the Gram-Schmidt singularity when n̂ is parallel to candidate
            candidates = np.eye(3)  # [x̂, ŷ, ẑ]
            dots = np.abs(candidates @ n_hat)  # |cos(angle)| with each axis
            best_idx = np.argmin(dots)  # axis most perpendicular to n̂
            e_ref_candidate = candidates[best_idx]
        
        return cls(
            hub_center=hub_center,
            rotation_axis=n_hat,
            reference_axis=e_ref_candidate,
            inflow_direction=inflow,
        )
    
    # -----------------------------------------------------------------
    # §2.2 Wake and Thrust Direction Properties
    # -----------------------------------------------------------------
    
    @property
    def wake_direction(self) -> np.ndarray:
        """Direction in which wake propagates (downstream)
        
        For HAWT: Same as inflow_direction (wind carries wake downstream)
        For VAWT: Wake spreads in the inflow direction
        
        Returns:
            Unit vector in wake propagation direction
        """
        return self.inflow_direction.copy()
    
    @property
    def upstream_direction(self) -> np.ndarray:
        """Direction toward upstream (opposite of wake)
        
        Returns:
            Unit vector pointing upstream
        """
        return -self.inflow_direction
    
    def is_hawt_configuration(self) -> bool:
        """Check if this is a HAWT configuration
        
        HAWT: inflow_direction is parallel to rotation_axis
        VAWT: inflow_direction is perpendicular to rotation_axis
        
        Returns:
            True if HAWT (inflow parallel to rotation axis)
        """
        dot = abs(np.dot(self.inflow_direction, self.n_axis))
        return dot > 0.9  # Allow small misalignment
    
    def get_thrust_sign_convention(self) -> str:
        """Explain thrust sign convention for this configuration
        
        Returns:
            Human-readable description of thrust convention
        """
        inflow_dot_n = np.dot(self.inflow_direction, self.n_axis)
        
        if abs(inflow_dot_n) > 0.9:
            # HAWT configuration
            if inflow_dot_n > 0:
                return (
                    "HAWT configuration: Wind from +n_axis direction.\n"
                    "  F_n > 0: Blade feels force in +n_axis (thrust opposes wind)\n"
                    "  F_n < 0: Blade feels force in -n_axis (unusual)\n"
                    "  Body force on fluid: -F_n·n̂ (Eq. 13)"
                )
            else:
                return (
                    "HAWT configuration: Wind from -n_axis direction.\n"
                    "  F_n > 0: Blade feels force in +n_axis (same as wind)\n"
                    "  F_n < 0: Blade feels force in -n_axis (thrust opposes wind)\n"
                    "  Body force on fluid: -F_n·n̂ (Eq. 13)"
                )
        else:
            return (
                "VAWT configuration: Wind perpendicular to rotation axis.\n"
                "  Thrust varies with azimuth angle.\n"
                "  F_n: Axial force component along rotation axis."
            )
    
    # -----------------------------------------------------------------
    # §2.3 Core Transformations
    # -----------------------------------------------------------------
    
    def marker_position(
        self,
        r: Union[float, np.ndarray],
        theta: float
    ) -> np.ndarray:
        """Compute marker position(s) in global coordinates
        
        Physical Formula (using rotation matrix):
            x_local = [r·cos(θ), r·sin(θ), 0]ᵀ   (in rotor plane)
            x_global = hub_center + R @ x_local
        
        Expanded:
            x_global = hub + r·cos(θ)·ê_ref + r·sin(θ)·ê_perp
        
        For HAWT_X_AXIS (Watanabe convention), this gives:
            x = x_hub
            y = y_hub + r·cos(θ)
            z = z_hub + r·sin(θ)
        
        Args:
            r: Radial distance(s) from hub  [m or lu]
               Can be scalar or array of shape (N,)
            theta: Azimuth angle  [radians]
        
        Returns:
            position: Global (x, y, z) coordinates
                      Shape (3,) if r is scalar, (N, 3) if r is array
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        r = np.asarray(r)
        is_scalar = (r.ndim == 0)
        
        if is_scalar:
            # Single marker: use rotation matrix directly
            x_local = np.array([float(r) * cos_t, float(r) * sin_t, 0.0])
            return self.hub_center + self.rotation_matrix @ x_local
        else:
            # Multiple markers: vectorized
            # x_local shape: (N, 3)
            x_local = np.column_stack([
                r * cos_t,
                r * sin_t,
                np.zeros_like(r)
            ])
            # R @ x_local.T gives (3, N), transpose to (N, 3)
            return self.hub_center + (self.rotation_matrix @ x_local.T).T
    
    def local_to_global(self, x_local: np.ndarray) -> np.ndarray:
        """Transform local rotor coordinates to global frame
        
        Args:
            x_local: Local coordinates, shape (3,) or (N, 3)
                     [x_local, y_local, z_local] where z_local is along n_axis
        
        Returns:
            x_global: Global coordinates, same shape as input
        """
        x_local = np.asarray(x_local)
        if x_local.ndim == 1:
            return self.hub_center + self.rotation_matrix @ x_local
        else:
            return self.hub_center + (self.rotation_matrix @ x_local.T).T
    
    def global_to_local(self, x_global: np.ndarray) -> np.ndarray:
        """Transform global coordinates to local rotor frame
        
        Args:
            x_global: Global coordinates, shape (3,) or (N, 3)
        
        Returns:
            x_local: Local coordinates, same shape as input
        """
        x_global = np.asarray(x_global)
        x_rel = x_global - self.hub_center
        if x_rel.ndim == 1:
            return self.rotation_matrix.T @ x_rel
        else:
            return (self.rotation_matrix.T @ x_rel.T).T
    
    def radial_vector(self, theta: float) -> np.ndarray:
        """Compute radial unit vector at azimuth θ
        
        Physical Definition:
            ê_r(θ) = cos(θ)·ê_ref + sin(θ)·ê_perp
        
        This points outward from hub toward blade tip.
        
        Note: This is NOT the vector used in velocity decomposition or
        force projection. Those use tangent_vector() instead, which
        ensures azimuth-independent torque conservation.
        
        The radial vector is perpendicular to the tangent vector:
            ê_r(θ) ⊥ ê_θ(θ) ⊥ n̂
        
        Args:
            theta: Azimuth angle  [radians]
        
        Returns:
            e_r: Radial unit vector, shape (3,)
        """
        return np.cos(theta) * self.e_ref + np.sin(theta) * self.e_perp
    
    def tangent_vector(self, theta: float) -> np.ndarray:
        """Compute mathematical tangent vector (direction of increasing θ)
        
        This is the standard ∂x/∂θ direction:
            ê_θ(θ) = -sin(θ)·ê_ref + cos(θ)·ê_perp
        
        This vector is used in:
            - decompose_velocity(): u_θ = u · ê_θ(θ)
            - project_force_to_global(): F^AL = F_n·n̂ + F_θ·ê_θ(θ)
        
        The tangent vector ensures azimuth-independent torque:
            Q = Σ (r_j × F^AL_j) · n̂ = Σ F_θ,j · r_j   (for all θ)
        
        Note: Watanabe Eq. 5 uses the radial vector instead, which breaks
        torque conservation for non-X-axis rotors. Our implementation
        intentionally uses the mathematical tangent for generality.
        
        For HAWT_X_AXIS (ê_ref=ŷ, ê_perp=ẑ):
            ê_θ(θ) = (0, -sin(θ), cos(θ))
        
        For HAWT_Z_AXIS (ê_ref=x̂, ê_perp=ŷ):
            ê_θ(θ) = (-sin(θ), cos(θ), 0)
        
        Args:
            theta: Azimuth angle  [radians]
        
        Returns:
            e_theta: Tangent unit vector, shape (3,)  [dimensionless]
        """
        return -np.sin(theta) * self.e_ref + np.cos(theta) * self.e_perp
    
    def decompose_velocity(
        self,
        u_global: np.ndarray,
        theta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose global velocity into normal and tangential components
        
        Velocity Decomposition:
            u_n = u · n̂                           (axial / normal to rotor plane)
            u_θ = u · ê_θ(θ)                      (tangential / rotation direction)
        
        where ê_θ(θ) = -sin(θ)·ê_ref + cos(θ)·ê_perp
        (mathematical tangent, direction of increasing θ)
        
        For HAWT_X_AXIS (n̂=x̂, ê_ref=ŷ, ê_perp=ẑ):
            u_n = u_x
            u_θ = -u_y·sin(θ) + u_z·cos(θ)
        
        For HAWT_Z_AXIS (n̂=ẑ, ê_ref=x̂, ê_perp=ŷ):
            u_n = u_z
            u_θ = -u_x·sin(θ) + u_y·cos(θ)
        
        Convention Difference from Watanabe et al.:
            Watanabe Eq. 5 uses the RADIAL projection:
                u_θ^W = u · ê_r(θ) = u_y·cos(θ) + u_z·sin(θ)  (HAWT_X)
            Our code uses the TANGENT projection:
                u_θ   = u · ê_θ(θ) = -u_y·sin(θ) + u_z·cos(θ) (HAWT_X)
            
            The tangent projection is used because it:
            (1) Correctly represents the velocity in the blade motion direction
            (2) Ensures azimuth-independent torque (torque conservation)
            (3) Generalizes to arbitrary rotation axes
            
            For pure axial flow (u ∥ n̂), both give u_θ = 0, so the
            BEM velocity triangle is unaffected. Differences appear only
            in induced in-plane velocities, which are small.
        
        Physical Interpretation:
            u_n:  Velocity through the rotor plane (creates thrust)  [m/s]
            u_θ:  Velocity in tangential direction at current blade position
                  (combined with ω·r to compute relative velocity)  [m/s]
        
        Args:
            u_global: Velocity vector(s) in global frame
                      Shape (3,) for single point or (N, 3) for multiple
            theta: Azimuth angle  [radians]
        
        Returns:
            (u_n, u_theta): Normal and tangential components
                            Shape matches input (scalar or (N,))
        """
        # Rotational direction vector (Watanabe Eq. 5 convention)
        e_rot = self.tangent_vector(theta)
        
        u_global = np.asarray(u_global)
        
        if u_global.ndim == 1:
            # Single velocity vector
            u_n = np.dot(u_global, self.n_axis)
            u_theta = np.dot(u_global, e_rot)
        else:
            # Multiple velocity vectors: shape (N, 3)
            u_n = u_global @ self.n_axis           # (N,)
            u_theta = u_global @ e_rot             # (N,)
        
        return u_n, u_theta

    def project_force_to_global(
        self,
        F_n: Union[float, np.ndarray],
        F_theta: Union[float, np.ndarray],
        theta: float,
        rotation_sign: float = 1.0,
        thrust_axis: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Project local forces to global coordinate frame

        Force Projection:
            F^AL = F_n·t̂ + rotation_sign · F_θ·ê_θ(θ)

        where t̂ = thrust_axis (default: n_axis = rotation axis).

        The thrust_axis allows decoupling the thrust direction from
        the rotation axis, analogous to Star-CCM+'s Virtual Disk
        disk-direction vector.

        Sign Convention:
            F_n > 0:  Force on blade in +t̂ direction (thrust)
            F_θ > 0:  Force on blade in +ê_θ direction (drives rotation)

            The BODY FORCE on fluid (Eq. 13) is -F^AL:
                F_body = -F_n·t̂ - rotation_sign·F_θ·ê_θ(θ)

        Args:
            F_n: Normal (axial) force(s)  [N or lattice force]
            F_theta: Tangential force(s)  [N or lattice force]
            theta: Azimuth angle  [radians]
            rotation_sign: sign(ω), +1.0 or -1.0  [-]
            thrust_axis: Unit vector for F_n projection  [dimensionless]
                         Default: self.n_axis (rotation axis)

        Returns:
            F_global: Force ON BLADE in global frame
        """
        e_tan = self.tangent_vector(theta)
        t_axis = self.n_axis if thrust_axis is None else thrust_axis

        F_n = np.asarray(F_n)
        F_theta = np.asarray(F_theta)

        F_theta_global = rotation_sign * F_theta

        is_scalar = (F_n.ndim == 0)

        if is_scalar:
            return float(F_n) * t_axis + float(F_theta_global) * e_tan
        else:
            return (F_n[:, None] * t_axis[None, :] +
                    F_theta_global[:, None] * e_tan[None, :])
    
    # -----------------------------------------------------------------
    # §2.4 Unit Conversion
    # -----------------------------------------------------------------
    @classmethod
    def _from_validated_axes(
        cls,
        hub_center: np.ndarray,
        n_axis: np.ndarray,
        e_ref: np.ndarray,
        e_perp: np.ndarray,
        inflow_direction: np.ndarray,
        rotation_matrix: np.ndarray,
        axis_label: str
    ) -> 'RotorCoordinateSystem':
        """Internal constructor bypassing Gram-Schmidt for already-validated axes
        
        Used by to_lattice_units() where only hub_center changes and
        all axis vectors are already orthonormalized.
        
        Args:
            hub_center: Rotor center position  [m or lu]
            n_axis, e_ref, e_perp: Orthonormal axis vectors  [dimensionless]
            inflow_direction: Wind approach direction  [dimensionless]
            rotation_matrix: Pre-built 3x3 rotation matrix
            axis_label: Human-readable label for the rotation axis
        
        Returns:
            RotorCoordinateSystem with pre-validated axes
        """
        obj = object.__new__(cls)
        obj.hub_center = np.asarray(hub_center, dtype=np.float64)
        obj.n_axis = n_axis
        obj.e_ref = e_ref
        obj.e_perp = e_perp
        obj.inflow_direction = inflow_direction
        obj.rotation_matrix = rotation_matrix
        obj.axis_label = axis_label
        return obj
    
    def to_lattice_units(self, length_scale: float) -> 'RotorCoordinateSystem':
        """Create a new coordinate system with hub center in lattice units
        
        Conversion: L_lu = L_phys / length_scale
        
        Note: Only the hub_center has units; the axis vectors and
        rotation matrix are dimensionless and remain unchanged.
        Bypasses Gram-Schmidt since axes are already validated.
        
        Args:
            length_scale: Physical size of one lattice cell  [m/lu]
        
        Returns:
            New RotorCoordinateSystem in lattice units
        """
        return RotorCoordinateSystem._from_validated_axes(
            hub_center=self.hub_center / length_scale,
            n_axis=self.n_axis,
            e_ref=self.e_ref,
            e_perp=self.e_perp,
            inflow_direction=self.inflow_direction,
            rotation_matrix=self.rotation_matrix,
            axis_label=self.axis_label
        )
    
    # -----------------------------------------------------------------
    # §2.6 Utility Methods
    # -----------------------------------------------------------------
    
    def get_info(self) -> str:
        """Human-readable summary of the coordinate system"""
        is_hawt = self.is_hawt_configuration()
        config_type = "HAWT" if is_hawt else "VAWT"
        
        lines = [
            "RotorCoordinateSystem",
            "=" * 60,
            f"  Rotation axis:    {self.axis_label}",
            f"  Configuration:    {config_type}",
            f"  Hub center:       ({self.hub_center[0]:.4f}, "
            f"{self.hub_center[1]:.4f}, {self.hub_center[2]:.4f})",
            "",
            "  Axis Vectors:",
            f"    n̂ (rotation):   ({self.n_axis[0]:.4f}, "
            f"{self.n_axis[1]:.4f}, {self.n_axis[2]:.4f})",
            f"    ê_ref (θ=0):    ({self.e_ref[0]:.4f}, "
            f"{self.e_ref[1]:.4f}, {self.e_ref[2]:.4f})",
            f"    ê_perp:         ({self.e_perp[0]:.4f}, "
            f"{self.e_perp[1]:.4f}, {self.e_perp[2]:.4f})",
            "",
            "  Flow Directions:",
            f"    Inflow:         ({self.inflow_direction[0]:.4f}, "
            f"{self.inflow_direction[1]:.4f}, {self.inflow_direction[2]:.4f})",
            f"    Wake:           ({self.wake_direction[0]:.4f}, "
            f"{self.wake_direction[1]:.4f}, {self.wake_direction[2]:.4f})",
            "",
            "  Rotation Matrix R = [e_ref | e_perp | n_axis]:",
            f"    [{self.rotation_matrix[0,0]:7.4f} {self.rotation_matrix[0,1]:7.4f} {self.rotation_matrix[0,2]:7.4f}]",
            f"    [{self.rotation_matrix[1,0]:7.4f} {self.rotation_matrix[1,1]:7.4f} {self.rotation_matrix[1,2]:7.4f}]",
            f"    [{self.rotation_matrix[2,0]:7.4f} {self.rotation_matrix[2,1]:7.4f} {self.rotation_matrix[2,2]:7.4f}]",
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (f"RotorCoordinateSystem(axis={self.axis_label}, "
                f"hub={tuple(np.round(self.hub_center, 4))}, "
                f"inflow={tuple(np.round(self.inflow_direction, 4))})")
    
    def verify_orthonormality(self, tol: float = 1e-10) -> bool:
        """Verify that the coordinate system is orthonormal
        
        Checks:
            R^T R = I (rotation matrix is orthonormal)
            |n̂| = |ê_ref| = |ê_perp| = 1
        
        Args:
            tol: Tolerance for numerical checks
        
        Returns:
            True if orthonormal, False otherwise
        """
        return self._check_orthonormality(tol)


# =============================================================================
# §3. Convenience Factory Function
# =============================================================================

def create_coordinate_system(
    hub_center: Union[Tuple[float, float, float], np.ndarray],
    rotation_axis: Union[Tuple[float, float, float], list, np.ndarray],
    reference_axis: Optional[Tuple[float, float, float]] = None,
    inflow_direction: Optional[Tuple[float, float, float]] = None
) -> RotorCoordinateSystem:
    """Create a rotor coordinate system from a rotation axis vector
    
    The rotation axis is specified as a 3-component vector (e.g. [1, 0, 0]).
    Reference axis and inflow direction are auto-derived if not provided.
    
    Auto-Derivation:
        - inflow_direction: defaults to rotation_axis (standard HAWT)
        - reference_axis:   auto-generated perpendicular to rotation_axis
    
    Args:
        hub_center:       Rotor center position  [m or lu]
        rotation_axis:    3D direction vector, e.g. [1, 0, 0] or [0, 0, 1]
        reference_axis:   θ=0 blade direction [optional, auto-derived]
        inflow_direction: Wind approach direction [optional, default = rotation_axis]
    
    Returns:
        Configured RotorCoordinateSystem
    
    Example:
        >>> coord = create_coordinate_system(hub, rotation_axis=[1, 0, 0])
        >>> coord = create_coordinate_system(hub, rotation_axis=[0, 0, 1])
        >>> coord = create_coordinate_system(hub, rotation_axis=[1, 0, 0.1])  # tilted
        
        >>> # VAWT: rotation perpendicular to inflow
        >>> coord = create_coordinate_system(
        ...     hub, rotation_axis=[0, 0, 1], inflow_direction=[1, 0, 0]
        ... )
    """
    return RotorCoordinateSystem.from_axis_vector(
        hub_center=hub_center,
        rotation_axis=rotation_axis,
        inflow_direction=inflow_direction,
        reference_axis=reference_axis,
    )