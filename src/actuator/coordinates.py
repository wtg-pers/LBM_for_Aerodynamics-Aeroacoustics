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

Velocity Decomposition (Watanabe Eq. 5):
========================================
    u_n = u · n̂                                           [m/s]
    u_θ = u · ê_rot(θ)                                    [m/s]
    
    where ê_rot(θ) = cos(θ)·ê_ref + sin(θ)·ê_perp
    (This is the RADIAL direction, matching Watanabe's convention)

Force Projection (Watanabe convention):
=======================================
    F^AL = F_n·n̂ + F_θ·ê_tan(θ)                          [N or lu_force]
    
    where ê_tan(θ) = cos(θ)·ê_ref - sin(θ)·ê_perp
    (Watanabe's tangent convention, 90° from mathematical tangent)

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026 (Sec. 2.2)
    - Sørensen & Shen, J. Fluids Eng. 124, 393-399, 2002

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Tuple, Optional, Union, Dict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


# =============================================================================
# §1. Preset Rotation Axis Configurations
# =============================================================================

class RotorAxisPreset(Enum):
    """Predefined rotor axis configurations
    
    Each preset defines (rotation_axis, reference_axis, inflow_direction).
    
    Physical Convention:
        - rotation_axis:    Normal to rotor plane (spin axis)
        - reference_axis:   Direction of blade at θ=0
        - inflow_direction: Direction wind approaches FROM
    """
    # Standard HAWT: rotation about X-axis (streamwise)
    # Wind from +X, θ=0 → blade in +Y, rotation CCW from upstream (+X view)
    # Wake forms in +X direction (downstream)
    HAWT_X_AXIS = "hawt_x"
    
    # HAWT with Z-axis rotation (e.g., vertical tower, top-down simulation)
    # Wind from +Z, θ=0 → blade in +X, rotation CCW from above (+Z view)
    HAWT_Z_AXIS = "hawt_z"
    
    # Vertical-axis wind turbine (Darrieus/H-rotor)
    # Rotation about Z (vertical), wind from +X (horizontal)
    # Note: For VAWT, inflow is PERPENDICULAR to rotation axis
    VAWT = "vawt"
    
    # Custom: user provides axes directly
    CUSTOM = "custom"


# Preset axis definitions: (rotation_axis, reference_axis, inflow_direction)
_PRESET_AXES: Dict[RotorAxisPreset, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    RotorAxisPreset.HAWT_X_AXIS: (
        np.array([1.0, 0.0, 0.0]),  # n̂: rotation about X
        np.array([0.0, 1.0, 0.0]),  # ê_ref: θ=0 in +Y
        np.array([1.0, 0.0, 0.0]),  # inflow: wind from +X
    ),
    RotorAxisPreset.HAWT_Z_AXIS: (
        np.array([0.0, 0.0, 1.0]),  # n̂: rotation about Z
        np.array([1.0, 0.0, 0.0]),  # ê_ref: θ=0 in +X
        np.array([0.0, 0.0, 1.0]),  # inflow: wind from +Z
    ),
    RotorAxisPreset.VAWT: (
        np.array([0.0, 0.0, 1.0]),  # n̂: rotation about Z (vertical)
        np.array([1.0, 0.0, 0.0]),  # ê_ref: θ=0 in +X
        np.array([1.0, 0.0, 0.0]),  # inflow: wind from +X (horizontal)
    ),
}


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
        rotation_matrix: 3×3 orthonormal transformation matrix
        preset: Configuration preset used (for serialization)
    
    Example:
        >>> # Standard HAWT (X-axis rotation, wind from +X)
        >>> coord = RotorCoordinateSystem.from_preset(
        ...     RotorAxisPreset.HAWT_X_AXIS,
        ...     hub_center=(3.66, 1.341, 0.817)
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
        preset: RotorAxisPreset = RotorAxisPreset.CUSTOM
    ) -> None:
        """Initialize coordinate system from axis vectors
        
        The reference_axis will be orthogonalized against the rotation_axis
        using Gram-Schmidt to ensure orthonormality.
        
        Args:
            hub_center: Rotor center position (x, y, z)  [m or lu]
            rotation_axis: Rotation axis direction (will be normalized)
            reference_axis: θ=0 blade direction (will be orthogonalized)
            inflow_direction: Wind approach direction (will be normalized)
            preset: Configuration preset identifier
        
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
        
        self.preset = preset
        
        # Validate orthonormality
        if not self._check_orthonormality():
            raise RuntimeError("Failed to construct orthonormal coordinate system")
    
    def _check_orthonormality(self, tol: float = 1e-10) -> bool:
        """Internal orthonormality check"""
        # Check R^T R = I
        should_be_identity = self.rotation_matrix.T @ self.rotation_matrix
        return np.allclose(should_be_identity, np.eye(3), atol=tol)
    
    # -----------------------------------------------------------------
    # §2.1 Factory Methods
    # -----------------------------------------------------------------
    
    @classmethod
    def from_preset(
        cls,
        preset: RotorAxisPreset,
        hub_center: Union[Tuple[float, float, float], np.ndarray]
    ) -> 'RotorCoordinateSystem':
        """Create coordinate system from a preset configuration
        
        Available Presets:
            HAWT_X_AXIS: Standard horizontal-axis (Watanabe et al.)
                         Wind from +X, wake in +X, rotation about X
            HAWT_Z_AXIS: Horizontal-axis with Z rotation
                         Wind from +Z, wake in +Z, rotation about Z
            VAWT:        Vertical-axis turbine
                         Wind from +X, rotation about Z (vertical)
        
        Args:
            preset: Preset configuration
            hub_center: Rotor center position  [m or lu]
        
        Returns:
            Configured RotorCoordinateSystem
        
        Raises:
            ValueError: If preset is CUSTOM (use constructor instead)
        """
        if preset == RotorAxisPreset.CUSTOM:
            raise ValueError(
                "CUSTOM preset requires explicit axes; "
                "use the constructor directly"
            )
        
        rotation_axis, reference_axis, inflow_dir = _PRESET_AXES[preset]
        return cls(
            hub_center=hub_center,
            rotation_axis=rotation_axis,
            reference_axis=reference_axis,
            inflow_direction=inflow_dir,
            preset=preset
        )
    
    @classmethod
    def hawt_standard(
        cls,
        hub_center: Union[Tuple[float, float, float], np.ndarray]
    ) -> 'RotorCoordinateSystem':
        """Create standard HAWT coordinate system (X-axis rotation)
        
        This matches the Watanabe et al. (2026) convention:
            - Flow in +X direction (wind from +X)
            - Rotor plane in Y-Z
            - θ=0: blade in +Y direction
            - CCW rotation when viewed from upstream (+X)
            - Wake forms in +X direction (downstream)
        
        Args:
            hub_center: Rotor center position  [m or lu]
        
        Returns:
            HAWT_X_AXIS coordinate system
        """
        return cls.from_preset(RotorAxisPreset.HAWT_X_AXIS, hub_center)
    
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
    
    def tangent_vector(self, theta: float) -> np.ndarray:
        """Compute tangential unit vector at azimuth θ (Watanabe convention)
        
        Physical Definition (Watanabe et al.):
            The tangent vector used in force projection follows the convention
            from the AL force projection formula:
            
                F^AL = (F_n, F_θ·cos(θ), -F_θ·sin(θ))
            
            This implies the tangent vector is:
                ê_tan(θ) = cos(θ)·ê_ref - sin(θ)·ê_perp
            
            (This is 90° ahead of the mathematical tangent ∂x/∂θ)
        
        Convention at specific angles (HAWT_X_AXIS):
            θ = 0:    ê_tan = (0, 1, 0) = +ŷ  (blade chord direction)
            θ = π/2:  ê_tan = (0, 0, -1) = -ẑ
            θ = π:    ê_tan = (0, -1, 0) = -ŷ
        
        Args:
            theta: Azimuth angle  [radians]
        
        Returns:
            e_tan: Tangential unit vector (Watanabe convention), shape (3,)
        """
        return np.cos(theta) * self.e_ref - np.sin(theta) * self.e_perp
    
    def radial_vector(self, theta: float) -> np.ndarray:
        """Compute radial unit vector at azimuth θ
        
        Physical Definition:
            ê_r(θ) = cos(θ)·ê_ref + sin(θ)·ê_perp
        
        This points outward from hub toward blade tip.
        Same as ê_rot used in velocity decomposition.
        
        Args:
            theta: Azimuth angle  [radians]
        
        Returns:
            e_r: Radial unit vector, shape (3,)
        """
        return np.cos(theta) * self.e_ref + np.sin(theta) * self.e_perp
    
    def math_tangent_vector(self, theta: float) -> np.ndarray:
        """Compute mathematical tangent vector (direction of increasing θ)
        
        This is the standard ∂x/∂θ direction:
            ê_θ^math = -sin(θ)·ê_ref + cos(θ)·ê_perp
        
        Note: NOT used in Watanabe's AL model force projection.
        Provided for reference and debugging.
        
        Args:
            theta: Azimuth angle  [radians]
        
        Returns:
            e_theta_math: Mathematical tangent, shape (3,)
        """
        return -np.sin(theta) * self.e_ref + np.cos(theta) * self.e_perp
    
    def decompose_velocity(
        self,
        u_global: np.ndarray,
        theta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose global velocity into normal and tangential components
        
        Watanabe et al. Eq. 5 Convention:
            u_n = u · n̂                           (axial / streamwise)
            u_θ = u · ê_rot(θ)                    (rotational direction)
        
        where ê_rot(θ) = cos(θ)·ê_ref + sin(θ)·ê_perp (radial direction)
        
        For HAWT_X_AXIS, this reduces to:
            u_n = u_x
            u_θ = u_y·cos(θ) + u_z·sin(θ)     [Watanabe Eq. 5]
        
        Physical Interpretation:
            u_n:  Velocity component through rotor plane (creates thrust)
            u_θ:  Velocity component in the radial direction of current blade position
                  (used with ω·r to compute relative velocity for lift/drag)
        
        Args:
            u_global: Velocity vector(s) in global frame
                      Shape (3,) for single point or (N, 3) for multiple
            theta: Azimuth angle  [radians]
        
        Returns:
            (u_n, u_theta): Normal and tangential components
                            Shape matches input (scalar or (N,))
        """
        # Rotational direction vector (Watanabe Eq. 5 convention)
        e_rot = self.radial_vector(theta)
        
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
        theta: float
    ) -> np.ndarray:
        """Project local forces to global coordinate frame
        
        Watanabe Convention:
            F^AL = F_n·n̂ + F_θ·ê_tan(θ)
                 = F_n·n̂ + F_θ·(cos(θ)·ê_ref - sin(θ)·ê_perp)
        
        For HAWT_X_AXIS, this gives:
            F^AL = (F_n, F_θ·cos(θ), -F_θ·sin(θ))   [Watanabe convention]
        
        Sign Convention (IMPORTANT):
            F_n > 0:  Force on blade in +n̂ direction
            F_θ > 0:  Force on blade in +ê_tan(θ) direction
            
            The BODY FORCE on fluid (Eq. 13) is -F^AL:
                F_body = -F^AL = -F_n·n̂ - F_θ·ê_tan(θ)
            
            This means positive thrust (F_n > 0) decelerates the flow,
            creating the wake deficit.
        
        Args:
            F_n: Normal (axial) force(s)  [N or lattice force]
                 Scalar or shape (N,)
            F_theta: Tangential force(s)  [N or lattice force]
                     Scalar or shape (N,)
            theta: Azimuth angle  [radians]
        
        Returns:
            F_global: Force ON BLADE in global frame
                      Shape (3,) if inputs are scalar, (N, 3) if arrays
        """
        e_tan = self.tangent_vector(theta)
        
        F_n = np.asarray(F_n)
        F_theta = np.asarray(F_theta)
        
        is_scalar = (F_n.ndim == 0)
        
        if is_scalar:
            # Single force: return shape (3,)
            return float(F_n) * self.n_axis + float(F_theta) * e_tan
        else:
            # Multiple forces: return shape (N, 3)
            return (F_n[:, None] * self.n_axis[None, :] +
                    F_theta[:, None] * e_tan[None, :])
    
    # -----------------------------------------------------------------
    # §2.4 Batch Operations (for vectorized marker handling)
    # -----------------------------------------------------------------
    
    def all_marker_positions(
        self,
        r_array: np.ndarray,
        theta: float
    ) -> np.ndarray:
        """Compute positions for all markers at once
        
        Optimized batch version of marker_position() for array inputs.
        
        Args:
            r_array: Radial positions, shape (N,)  [m or lu]
            theta: Azimuth angle  [radians]
        
        Returns:
            positions: Shape (N, 3) global coordinates
        """
        return self.marker_position(r_array, theta)
    
    def decompose_velocity_batch(
        self,
        u_global: np.ndarray,
        theta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose velocities for all markers at once
        
        Args:
            u_global: Velocities at markers, shape (N, 3)  [m/s or lu/lt]
            theta: Azimuth angle  [radians]
        
        Returns:
            (u_n, u_theta): Each shape (N,)
        """
        return self.decompose_velocity(u_global, theta)
    
    def project_forces_batch(
        self,
        F_n: np.ndarray,
        F_theta: np.ndarray,
        theta: float
    ) -> np.ndarray:
        """Project forces for all markers at once
        
        Args:
            F_n: Normal forces, shape (N,)  [N or lattice force]
            F_theta: Tangential forces, shape (N,)
            theta: Azimuth angle  [radians]
        
        Returns:
            F_global: Shape (N, 3) global forces ON BLADE
        """
        return self.project_force_to_global(F_n, F_theta, theta)
    
    # -----------------------------------------------------------------
    # §2.5 Unit Conversion
    # -----------------------------------------------------------------
    
    def to_lattice_units(self, length_scale: float) -> 'RotorCoordinateSystem':
        """Create a new coordinate system with hub center in lattice units
        
        Conversion: L_lu = L_phys / length_scale
        
        Note: Only the hub_center has units; the axis vectors and
        rotation matrix are dimensionless and remain unchanged.
        
        Args:
            length_scale: Physical size of one lattice cell  [m/lu]
        
        Returns:
            New RotorCoordinateSystem in lattice units
        """
        return RotorCoordinateSystem(
            hub_center=self.hub_center / length_scale,
            rotation_axis=self.n_axis,
            reference_axis=self.e_ref,
            inflow_direction=self.inflow_direction,
            preset=self.preset
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
            f"  Preset:           {self.preset.value}",
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
        return (f"RotorCoordinateSystem(preset={self.preset.value}, "
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
# §3. Convenience Functions
# =============================================================================

def create_coordinate_system(
    hub_center: Union[Tuple[float, float, float], np.ndarray],
    preset: str = "hawt_x",
    rotation_axis: Optional[Tuple[float, float, float]] = None,
    reference_axis: Optional[Tuple[float, float, float]] = None,
    inflow_direction: Optional[Tuple[float, float, float]] = None
) -> RotorCoordinateSystem:
    """Factory function to create a coordinate system
    
    Provides a simple interface for common use cases.
    
    Args:
        hub_center: Rotor center position  [m or lu]
        preset: Preset name ("hawt_x", "hawt_z", "vawt", "custom")
        rotation_axis: Custom rotation axis (required if preset="custom")
        reference_axis: Custom reference axis (required if preset="custom")
        inflow_direction: Custom inflow direction (required if preset="custom")
    
    Returns:
        Configured RotorCoordinateSystem
    
    Example:
        >>> # Standard HAWT (wind from +X)
        >>> coord = create_coordinate_system(
        ...     hub_center=(3.66, 1.341, 0.817),
        ...     preset="hawt_x"
        ... )
        
        >>> # Custom tilted rotor
        >>> coord = create_coordinate_system(
        ...     hub_center=(0, 0, 0),
        ...     preset="custom",
        ...     rotation_axis=(1, 0, 0.1),    # slightly tilted
        ...     reference_axis=(0, 1, 0),
        ...     inflow_direction=(1, 0, 0)    # wind still from +X
        ... )
    """
    preset_map = {
        "hawt_x": RotorAxisPreset.HAWT_X_AXIS,
        "hawt_z": RotorAxisPreset.HAWT_Z_AXIS,
        "vawt": RotorAxisPreset.VAWT,
        "custom": RotorAxisPreset.CUSTOM,
    }
    
    preset_enum = preset_map.get(preset.lower())
    if preset_enum is None:
        available = ", ".join(preset_map.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    if preset_enum == RotorAxisPreset.CUSTOM:
        if rotation_axis is None or reference_axis is None or inflow_direction is None:
            raise ValueError(
                "Custom preset requires rotation_axis, reference_axis, "
                "and inflow_direction"
            )
        return RotorCoordinateSystem(
            hub_center=hub_center,
            rotation_axis=rotation_axis,
            reference_axis=reference_axis,
            inflow_direction=inflow_direction,
            preset=preset_enum
        )
    else:
        return RotorCoordinateSystem.from_preset(preset_enum, hub_center)