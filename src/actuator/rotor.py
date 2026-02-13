"""
Multi-Blade Rotor Kinematics for Actuator Line Model

This module defines the Rotor class that manages multiple blades rotating
about a common hub center. The rotor is the central kinematic object in
the Actuator Line (AL) model, responsible for:

    1. Distributing N_b identical blades at equal azimuthal spacing
    2. Advancing the azimuth angles each timestep: θ_k(t+Δt) = θ_k(t) + ω·Δt
    3. Computing all marker positions/forces across all blades simultaneously
    4. Converting between physical and lattice units

Physical Model:
===============
A wind turbine rotor with N_b blades. Each blade is represented by a line
of marker particles (see blade.py). The rotor rotates at angular velocity
ω about the hub center.

    Azimuth angle for blade k at time t:
        θ_k(t) = θ_0 + ω·t + 2πk/N_b     [rad]

    Angular velocity from operating conditions:
        ω = TSR · U_∞ / R                 [rad/s]

Coordinate System Support:
==========================
This module now supports arbitrary rotation axes via RotorCoordinateSystem.

    Presets:
        - "hawt_x":  Standard HAWT, rotation about X-axis (Watanabe et al.)
        - "hawt_z":  HAWT with vertical tower, rotation about Z-axis
        - "vawt":    Vertical-axis wind turbine
        - "custom":  User-defined rotation axis

    The coordinate system is automatically propagated to all blades.

Data Flow in AL Simulation:
============================
    Each timestep:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. rotor.advance(dt)           → update all θ_k        │
    │ 2. rotor.get_all_marker_positions() → (N_total, 3)     │
    │ 3. interpolation: grid → markers  (velocity sampling)  │
    │ 4. BEM: compute F_L, F_D at each marker                │
    │ 5. rotor.project_all_forces() → global force array      │
    │ 6. spreading: markers → grid   (Gaussian smearing)     │
    │ 7. Guo forcing: apply body force to LBM                │
    └─────────────────────────────────────────────────────────┘

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026 (Sec. 2.2, 3.2)
    - Sørensen & Shen, J. Fluids Eng. 124, 393-399, 2002
    - Krogstad & Eriksen, Renewable Energy 50, 325-333, 2013

Author: LBM Development Team
Date: 2026-02
"""

import copy
from typing import (
    TYPE_CHECKING, Optional, List, Tuple, Dict, Union
)

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from .blade import Blade, BladeSection
from .coordinates import (
    RotorCoordinateSystem, 
    RotorAxisPreset,
    create_coordinate_system
)


# =============================================================================
# §1. Rotor Class
# =============================================================================

class Rotor:
    """Multi-blade rotor for the Actuator Line model

    Manages N_b identical blades uniformly distributed in azimuth,
    rotating at angular velocity ω about a hub center.

    Physical Process:
    =================
    1. User creates a prototype Blade (geometry definition)
    2. Rotor creates a RotorCoordinateSystem and assigns it to all blades
    3. Rotor replicates blade N_b times with azimuthal offsets 2πk/N_b
    4. Each timestep: advance θ_k, compute marker positions, aggregate forces

    Coordinate System:
    ==================
    The rotor manages a RotorCoordinateSystem that defines:
        - Rotation axis (n̂)
        - Hub center position
        - Inflow direction (for wake tracking)
        - Reference axis for θ=0

    All blades share this coordinate system, ensuring consistent
    position and force transformations.

    Attributes:
        n_blades:       Number of blades                   [dimensionless]
        blades:         List of Blade instances
        coord_system:   RotorCoordinateSystem for transformations
        radius:         Rotor radius R = r_tip             [m or lu]
        omega:          Angular velocity                   [rad/s or rad/lu_time]
        theta:          Array of current azimuth angles    [rad]
        time:           Current simulation time            [s or lu_time]

    Example:
        >>> rotor = Rotor.from_ntnu_bt1(tsr=6.0, u_inf=10.0)
        >>> print(rotor.coord_system.preset)  # hawt_x
        >>> rotor.advance(dt=8.38e-6)
        >>> positions = rotor.get_all_marker_positions()
    """

    def __init__(
        self,
        blade_prototype: Blade,
        n_blades: int,
        hub_center: Tuple[float, float, float],
        omega: float,
        theta_0: float = 0.0,
        rotation_axis: str = "hawt_x",
        inflow_direction: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """Initialize rotor from a prototype blade

        Creates n_blades copies of the prototype, each offset by 2π/n_blades
        in azimuth angle. Also creates and assigns a coordinate system.

        Physical Setup:
            θ_k(0) = θ_0 + 2πk/N_b   for k = 0, 1, ..., N_b-1

        Args:
            blade_prototype: Template blade (must have markers generated)
            n_blades: Number of blades  [dimensionless]
            hub_center: Rotor center position (x, y, z)  [m or lu]
            omega: Angular velocity  [rad/s or rad/lu_time]
                   Positive = CCW when viewed from upstream
            theta_0: Initial azimuth of blade 0  [rad]
                     Default 0.0 = blade 0 in reference direction
            rotation_axis: Rotation axis preset or "custom"
                           Options: "hawt_x", "hawt_z", "vawt", "custom"
            inflow_direction: Wind approach direction (required if rotation_axis="custom")
                              For presets, this is automatically determined.

        Raises:
            ValueError: If n_blades < 1 or blade has no markers
        """
        if n_blades < 1:
            raise ValueError(f"n_blades must be >= 1, got {n_blades}")

        if blade_prototype.n_markers == 0:
            raise ValueError(
                "Blade prototype must have markers generated. "
                "Call blade.generate_markers() before creating Rotor."
            )

        self.n_blades: int = n_blades                         # [dimensionless]
        self.omega: float = omega                             # [rad/s or rad/lu_time]
        self.radius: float = blade_prototype.r_tip            # [m or lu]

        # --- Create coordinate system ---
        if rotation_axis == "custom":
            if inflow_direction is None:
                raise ValueError(
                    "inflow_direction is required when rotation_axis='custom'"
                )
            # For custom, we need more parameters - use a default reference axis
            # User should use the explicit constructor for full custom control
            raise ValueError(
                "For custom rotation axis, create RotorCoordinateSystem manually "
                "and use Rotor.from_coordinate_system() factory method."
            )
        
        self._coord_system = create_coordinate_system(
            hub_center=hub_center,
            preset=rotation_axis
        )

        # --- Create blade copies with azimuthal offset ---
        delta_theta = 2.0 * np.pi / n_blades  # [rad] blade spacing
        self.theta: np.ndarray = np.array([
            theta_0 + k * delta_theta for k in range(n_blades)
        ], dtype=np.float64)  # [rad]

        # Deep-copy blade for each blade slot and assign coordinate system
        self.blades: List[Blade] = []
        for _ in range(n_blades):
            blade_copy = copy.deepcopy(blade_prototype)
            blade_copy.coord_system = self._coord_system
            self.blades.append(blade_copy)

        # --- Derived quantities ---
        self.markers_per_blade: int = blade_prototype.n_markers  # [dimensionless]
        self.total_markers: int = n_blades * self.markers_per_blade  # [dimensionless]

        # --- Time tracking ---
        self.time: float = 0.0      # [s or lu_time]
        self.n_revolutions: float = 0.0  # [dimensionless] accumulated revolutions

    # -----------------------------------------------------------------
    # §1.0 Coordinate System Access
    # -----------------------------------------------------------------

    @property
    def coord_system(self) -> RotorCoordinateSystem:
        """Get the rotor coordinate system"""
        return self._coord_system

    @property
    def hub_center(self) -> Tuple[float, float, float]:
        """Get hub center position (for backward compatibility)
        
        Returns:
            (x, y, z) hub center position
        """
        return tuple(self._coord_system.hub_center)

    @property
    def rotation_axis(self) -> np.ndarray:
        """Get rotation axis unit vector"""
        return self._coord_system.n_axis

    @property
    def inflow_direction(self) -> np.ndarray:
        """Get inflow direction unit vector"""
        return self._coord_system.inflow_direction

    @property
    def wake_direction(self) -> np.ndarray:
        """Get wake propagation direction"""
        return self._coord_system.wake_direction

    # -----------------------------------------------------------------
    # §1.1 Time Advancement
    # -----------------------------------------------------------------

    def advance(self, dt: float) -> None:
        """Advance rotor rotation by one timestep

        Physical Update:
            θ_k(t + Δt) = θ_k(t) + ω · Δt    [rad]

        The azimuth angles are wrapped to [0, 2π) for numerical cleanliness,
        while total revolutions are tracked separately.

        Args:
            dt: Timestep  [s or lu_time]
        """
        d_theta = self.omega * dt  # [rad] angular increment per step

        self.theta += d_theta      # [rad]
        self.time += dt            # [s or lu_time]

        # Track total revolutions (unwrapped)
        self.n_revolutions += d_theta / (2.0 * np.pi)  # [dimensionless]

        # Wrap azimuth to [0, 2π) for numerical cleanliness
        self.theta = self.theta % (2.0 * np.pi)  # [rad]

    def set_azimuth(self, theta_0: float) -> None:
        """Reset azimuth angles to a specific configuration

        Useful for:
            - Restarting from a checkpoint
            - Setting up specific blade configurations for testing
            - Phase-locking analysis

        The azimuthal spacing 2π/N_b is preserved.

        Args:
            theta_0: New azimuth of blade 0  [rad]
        """
        delta_theta = 2.0 * np.pi / self.n_blades  # [rad]
        self.theta = np.array([
            theta_0 + k * delta_theta for k in range(self.n_blades)
        ], dtype=np.float64)  # [rad]

    # -----------------------------------------------------------------
    # §1.2 Marker Position Queries
    # -----------------------------------------------------------------

    def get_blade_marker_positions(self, blade_idx: int) -> np.ndarray:
        """Get marker positions for a single blade

        Args:
            blade_idx: Blade index (0 to N_b-1)

        Returns:
            positions: shape (n_markers, 3) — global (x, y, z)
                       [m or lu]
        """
        # Blade has coord_system, so no hub_center needed
        return self.blades[blade_idx].get_marker_positions(
            theta=self.theta[blade_idx]
        )  # [m or lu]

    def get_all_marker_positions(self) -> np.ndarray:
        """Get marker positions for ALL blades, concatenated

        Returns:
            positions: shape (N_b * n_markers, 3) — global (x, y, z)
                       Ordered as [blade_0_markers, blade_1_markers, ...]
                       [m or lu]

        Note:
            This is the primary interface for the velocity interpolation
            step in the AL model. The actuator_line controller samples
            the flow velocity at these positions.
        """
        all_pos = np.zeros((self.total_markers, 3), dtype=np.float64)

        for k in range(self.n_blades):
            i_start = k * self.markers_per_blade
            i_end = i_start + self.markers_per_blade
            all_pos[i_start:i_end, :] = self.blades[k].get_marker_positions(
                theta=self.theta[k]
            )  # [m or lu]

        return all_pos  # [m or lu]

    # -----------------------------------------------------------------
    # §1.3 Force Projection
    # -----------------------------------------------------------------

    def project_blade_forces(
        self,
        blade_idx: int,
        F_n: np.ndarray,
        F_theta: np.ndarray
    ) -> np.ndarray:
        """Project forces on a single blade to global frame

        Uses the blade's coordinate system for projection.

        Args:
            blade_idx: Blade index
            F_n:     Normal (axial) forces, shape (n_markers,)  [N or lu_force]
            F_theta: Tangential (rotational) forces, shape (n_markers,) [N or lu_force]

        Returns:
            F_global: shape (n_markers, 3) — (F_x, F_y, F_z)
                      [N or lu_force]
        """
        return self.blades[blade_idx].project_forces_to_global(
            F_n=F_n,
            F_theta=F_theta,
            theta=self.theta[blade_idx]
        )  # [N or lu_force]

    def project_all_forces(
        self,
        F_n_all: np.ndarray,
        F_theta_all: np.ndarray
    ) -> np.ndarray:
        """Project forces on ALL blades to global frame

        Args:
            F_n_all:     Normal forces, shape (N_b * n_markers,)  [N or lu_force]
            F_theta_all: Tangential forces, shape (N_b * n_markers,) [N or lu_force]
                         Ordered as [blade_0_markers, blade_1_markers, ...]

        Returns:
            F_global: shape (N_b * n_markers, 3) — (F_x, F_y, F_z)
                      [N or lu_force]
        """
        F_global = np.zeros((self.total_markers, 3), dtype=np.float64)

        for k in range(self.n_blades):
            i_start = k * self.markers_per_blade
            i_end = i_start + self.markers_per_blade

            F_global[i_start:i_end, :] = self.blades[k].project_forces_to_global(
                F_n=F_n_all[i_start:i_end],
                F_theta=F_theta_all[i_start:i_end],
                theta=self.theta[k]
            )  # [N or lu_force]

        return F_global  # [N or lu_force]

    # -----------------------------------------------------------------
    # §1.4 Velocity Decomposition Helpers
    # -----------------------------------------------------------------

    def get_blade_unit_vectors(
        self,
        blade_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get local coordinate unit vectors for a single blade

        Returns (ê_n, ê_θ, ê_r) at the current azimuth of the blade.

        Args:
            blade_idx: Blade index

        Returns:
            (e_n, e_theta, e_r): Each shape (3,)  [dimensionless]
        """
        return self.blades[blade_idx].get_marker_unit_vectors(
            theta=self.theta[blade_idx]
        )

    def decompose_velocity(
        self,
        blade_idx: int,
        u_global: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose global velocity into normal and tangential components

        Uses the coordinate system's decomposition method.

        Args:
            blade_idx: Blade index
            u_global: Global velocity vectors, shape (n_markers, 3)  [m/s or lu/lt]

        Returns:
            (u_n, u_theta): Each shape (n_markers,)  [m/s or lu/lt]
        """
        return self._coord_system.decompose_velocity(
            u_global=u_global,
            theta=self.theta[blade_idx]
        )

    def compute_relative_velocity(
        self,
        blade_idx: int,
        u_global: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute relative velocity components for BEM calculation

        From Watanabe et al. Eq. 5-8:
            u_rel = √(u_n² + (ω·r - u_θ)²)       [Eq. 6]
            φ = atan2(u_n, ω·r - u_θ)            [Eq. 7]
            α = φ - γ                             [Eq. 8]

        Args:
            blade_idx: Blade index
            u_global: Global velocity at markers, shape (n_markers, 3)  [m/s or lu/lt]

        Returns:
            (u_rel, phi_deg, alpha_deg):
                u_rel:     Relative velocity magnitude, shape (n_markers,)  [m/s or lu/lt]
                phi_deg:   Flow angle, shape (n_markers,)                   [degrees]
                alpha_deg: Angle of attack, shape (n_markers,)              [degrees]
        """
        blade = self.blades[blade_idx]

        # Decompose velocity using coordinate system
        u_n, u_theta = self._coord_system.decompose_velocity(
            u_global=u_global,
            theta=self.theta[blade_idx]
        )

        # Blade rotation velocity (Eq. 6 setup)
        omega_r = self.omega * blade.marker_r  # [m/s or lu/lt]

        # Tangential velocity relative to blade (Eq. 6)
        u_tangential_rel = omega_r - u_theta   # [m/s or lu/lt]

        # Relative velocity magnitude (Eq. 6)
        u_rel = np.sqrt(u_n**2 + u_tangential_rel**2)  # [m/s or lu/lt]

        # Flow angle (Eq. 7)  [rad]
        phi_rad = np.arctan2(u_n, u_tangential_rel)     # [rad]
        phi_deg = np.degrees(phi_rad)                    # [degrees]

        # Angle of attack (Eq. 8)  [degrees]
        alpha_deg = phi_deg - blade.marker_twist         # [degrees]

        return u_rel, phi_deg, alpha_deg

    # -----------------------------------------------------------------
    # §1.5 Marker Property Access (Aggregated)
    # -----------------------------------------------------------------

    def get_all_marker_chords(self) -> np.ndarray:
        """Get chord lengths for all markers across all blades

        Returns:
            chords: shape (N_b * n_markers,)  [m or lu]
        """
        return np.tile(self.blades[0].marker_chord, self.n_blades)  # [m or lu]

    def get_all_marker_dr(self) -> float:
        """Get marker spacing (uniform for all markers)

        Returns:
            dr: Marker spacing  [m or lu]
        """
        return self.blades[0].marker_dr  # [m or lu]

    def get_all_marker_epsilon(self) -> np.ndarray:
        """Get Gaussian filter widths for all markers across all blades

        Returns:
            epsilon: shape (N_b * n_markers,)  [m or lu]
        """
        return np.tile(self.blades[0].marker_epsilon, self.n_blades)  # [m or lu]

    def get_all_marker_active(self) -> np.ndarray:
        """Get active flags for all markers across all blades

        Returns:
            active: shape (N_b * n_markers,)  [bool]
        """
        return np.tile(self.blades[0].marker_active, self.n_blades)

    def get_all_marker_airfoils(self) -> List[str]:
        """Get airfoil identifiers for all markers across all blades

        Returns:
            airfoils: List of length (N_b * n_markers)
        """
        return self.blades[0].marker_airfoil * self.n_blades

    def get_all_marker_twists(self) -> np.ndarray:
        """Get twist angles for all markers across all blades

        Returns:
            twists: shape (N_b * n_markers,)  [degrees]
        """
        return np.tile(self.blades[0].marker_twist, self.n_blades)  # [degrees]

    def get_all_marker_radii(self) -> np.ndarray:
        """Get radial positions for all markers across all blades

        Returns:
            radii: shape (N_b * n_markers,)  [m or lu]
        """
        return np.tile(self.blades[0].marker_r, self.n_blades)  # [m or lu]

    # -----------------------------------------------------------------
    # §1.6 Aerodynamic Output Computation
    # -----------------------------------------------------------------

    def compute_torque_thrust(
        self,
        F_n_all: np.ndarray,
        F_theta_all: np.ndarray
    ) -> Tuple[float, float]:
        """Compute total rotor torque and thrust from marker forces

        Physical Definitions:
            Thrust T = Σ F_n,j           [N or lu_force]    (along rotation axis)
            Torque Q = Σ F_θ,j · r_j     [N·m or lu_torque] (about rotation axis)

        Only active markers contribute to these sums.

        Args:
            F_n_all:     Normal forces, shape (N_b * n_markers,) [N or lu_force]
            F_theta_all: Tangential forces, shape (N_b * n_markers,) [N or lu_force]

        Returns:
            (thrust, torque): Total rotor thrust and torque
        """
        active = self.get_all_marker_active()  # [bool]
        radii = self.get_all_marker_radii()    # [m or lu]

        thrust = np.sum(F_n_all[active])                    # [N or lu_force]
        torque = np.sum(F_theta_all[active] * radii[active])  # [N·m or lu_torque]

        return thrust, torque

    def compute_coefficients(
        self,
        F_n_all: np.ndarray,
        F_theta_all: np.ndarray,
        rho: float,
        u_inf: float
    ) -> Tuple[float, float]:
        """Compute thrust and power coefficients

        Definitions:
            C_T = T / (0.5 · ρ · U_∞² · A)       [dimensionless]
            C_P = P / (0.5 · ρ · U_∞³ · A)       [dimensionless]
                = ω · Q / (0.5 · ρ · U_∞³ · A)

        Args:
            F_n_all:     Normal forces, shape (N_b * n_markers,) [N or lu_force]
            F_theta_all: Tangential forces, shape (N_b * n_markers,) [N or lu_force]
            rho:   Fluid density       [kg/m³ or lu_density]
            u_inf: Freestream velocity [m/s or lu/lt]

        Returns:
            (C_T, C_P): Thrust and power coefficients [dimensionless]
        """
        thrust, torque = self.compute_torque_thrust(F_n_all, F_theta_all)
        power = self.omega * torque  # [W or lu_power]

        # Rotor swept area
        A = np.pi * self.radius**2   # [m² or lu²]

        # Dynamic pressure × area
        q_A = 0.5 * rho * u_inf**2 * A  # [N or lu_force]

        # Coefficients
        C_T = thrust / q_A if q_A > 0 else 0.0  # [dimensionless]
        C_P = power / (q_A * u_inf) if q_A > 0 else 0.0  # [dimensionless]

        return C_T, C_P
    
    def compute_power(
        self,
        F_theta_all: np.ndarray
    ) -> float:
        """Compute total rotor power from tangential forces

        Physical Definition:
            P = ω · Q = ω · Σ F_θ,j · r_j   [W or lu_power]

        Only active markers contribute to the power calculation.

        Args:
            F_theta_all: Tangential forces, shape (N_b * n_markers,) [N or lu_force]

        Returns:
            power: Total rotor power  [W or lu_power]
        """
        active = self.get_all_marker_active()  # [bool]
        radii = self.get_all_marker_radii()    # [m or lu]

        torque = np.sum(F_theta_all[active] * radii[active])  # [N·m or lu_torque]
        power = self.omega * torque                            # [W or lu_power]

        return power

    def compute_ct_cp(
        self,
        F_n_all: np.ndarray,
        F_theta_all: np.ndarray,
        rho: float,
        u_inf: float
    ) -> Tuple[float, float]:
        """Compute thrust and power coefficients (alias for compute_coefficients)

        This is an alias for compute_coefficients() to match the interface
        expected by ActuatorLineModel.get_rotor_performance().

        Definitions:
            C_T = T / (0.5 · ρ · U_∞² · A)       [dimensionless]
            C_P = P / (0.5 · ρ · U_∞³ · A)       [dimensionless]

        Args:
            F_n_all:     Normal forces, shape (N_b * n_markers,) [N or lu_force]
            F_theta_all: Tangential forces, shape (N_b * n_markers,) [N or lu_force]
            rho:   Fluid density       [kg/m³ or lu_density]
            u_inf: Freestream velocity [m/s or lu/lt]

        Returns:
            (C_T, C_P): Thrust and power coefficients [dimensionless]
        """
        return self.compute_coefficients(F_n_all, F_theta_all, rho, u_inf)


    # -----------------------------------------------------------------
    # §1.7 Information & Diagnostics
    # -----------------------------------------------------------------

    def get_info(self) -> str:
        """Human-readable rotor summary"""
        omega_rpm = self.omega * 60.0 / (2.0 * np.pi)  # [RPM]
        period = (2.0 * np.pi / self.omega) if self.omega > 0 else float('inf')
        hub = self.hub_center

        lines = [
            "Rotor Summary",
            "=" * 60,
            f"  Blades:           {self.n_blades}",
            f"  Radius:           {self.radius:.4f}",
            f"  Hub center:       ({hub[0]:.3f}, {hub[1]:.3f}, {hub[2]:.3f})",
            f"  Rotation axis:    {self._coord_system.preset.value}",
            f"  ω:                {self.omega:.4f} rad/s  ({omega_rpm:.2f} RPM)",
            f"  Period:           {period:.6f} s",
            f"  Markers/blade:    {self.markers_per_blade}",
            f"  Total markers:    {self.total_markers}",
            f"  Current θ:        {np.degrees(self.theta)} deg",
            f"  Time:             {self.time:.6f} s",
            f"  Revolutions:      {self.n_revolutions:.4f}",
            "",
            "  --- Coordinate System ---",
            f"  n̂ (rotation):     {self._coord_system.n_axis}",
            f"  Inflow:           {self._coord_system.inflow_direction}",
            f"  Wake:             {self._coord_system.wake_direction}",
        ]

        # Add blade info (from blade 0 since all identical)
        lines.append("")
        lines.append("  --- Blade Geometry (shared) ---")
        lines.append(f"  Span:             {self.blades[0].span:.4f}")
        lines.append(f"  Marker spacing:   {self.blades[0].marker_dr:.5f}")
        lines.append(
            f"  Chord range:      [{self.blades[0].marker_chord.min():.4f}, "
            f"{self.blades[0].marker_chord.max():.4f}]"
        )
        lines.append(
            f"  Active markers:   {np.sum(self.blades[0].marker_active)}"
            f"/{self.markers_per_blade}"
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Rotor(n_blades={self.n_blades}, R={self.radius:.3f}, "
            f"ω={self.omega:.2f} rad/s, axis={self._coord_system.preset.value}, "
            f"markers={self.total_markers})"
        )

    # =================================================================
    # §2. Factory Methods — Preset Rotor Configurations
    # =================================================================

    @classmethod
    def from_coordinate_system(
        cls,
        blade_prototype: Blade,
        n_blades: int,
        coord_system: RotorCoordinateSystem,
        omega: float,
        theta_0: float = 0.0
    ) -> 'Rotor':
        """Create rotor from an explicit coordinate system

        This is the most flexible factory method, allowing full control
        over the coordinate system configuration.

        Args:
            blade_prototype: Template blade (must have markers generated)
            n_blades: Number of blades
            coord_system: Pre-configured RotorCoordinateSystem
            omega: Angular velocity  [rad/s]
            theta_0: Initial azimuth of blade 0  [rad]

        Returns:
            Rotor instance with the given coordinate system
        """
        if blade_prototype.n_markers == 0:
            raise ValueError(
                "Blade prototype must have markers generated."
            )

        # Create instance without calling __init__
        rotor = object.__new__(cls)
        
        rotor.n_blades = n_blades
        rotor.omega = omega
        rotor.radius = blade_prototype.r_tip
        rotor._coord_system = coord_system

        # Azimuth angles
        delta_theta = 2.0 * np.pi / n_blades
        rotor.theta = np.array([
            theta_0 + k * delta_theta for k in range(n_blades)
        ], dtype=np.float64)

        # Create blade copies
        rotor.blades = []
        for _ in range(n_blades):
            blade_copy = copy.deepcopy(blade_prototype)
            blade_copy.coord_system = coord_system
            rotor.blades.append(blade_copy)

        rotor.markers_per_blade = blade_prototype.n_markers
        rotor.total_markers = n_blades * blade_prototype.n_markers
        rotor.time = 0.0
        rotor.n_revolutions = 0.0

        return rotor

    @classmethod
    def from_ntnu_bt1(
        cls,
        tsr: float = 6.0,
        u_inf: float = 10.0,
        hub_center: Tuple[float, float, float] = (3.66, 1.341, 0.817),
        dx: Optional[float] = None,
        resolution: int = 160,
        theta_0: float = 0.0,
        rotation_axis: str = "hawt_x"
    ) -> 'Rotor':
        """Create the NTNU Blind Test 1 rotor (3 × NREL S826 blades)

        Reference: Watanabe et al., Comp. & Fluids 305, 106901, 2026
                   Krogstad & Eriksen, Renewable Energy 50, 325-333, 2013

        Rotor Specifications:
            - Diameter: D = 0.894 m
            - 3 blades, NREL S826 airfoil
            - Hub radius: 0.045 m (90 mm nacelle)
            - Design TSR = 6, U_∞ = 10 m/s

        Args:
            tsr: Tip speed ratio                         [dimensionless]
            u_inf: Freestream velocity                   [m/s]
            hub_center: Hub position in global coords    [m]
            dx: Lattice spacing (if None, computed from resolution)  [m]
            resolution: D/Δx grid cells per diameter     [dimensionless]
            theta_0: Initial azimuth of blade 0          [rad]
            rotation_axis: Rotation axis preset          [default: "hawt_x"]

        Returns:
            Rotor instance with 3 blades and markers generated
        """
        # NTNU BT1 parameters
        D = 0.894          # [m] rotor diameter
        R = D / 2.0        # [m] rotor radius
        n_blades = 3       # [dimensionless]

        # Angular velocity from TSR
        omega = tsr * u_inf / R  # [rad/s]

        # Grid spacing
        if dx is None:
            dx = D / resolution  # [m]
        dr = dx / 2.0            # [m] marker spacing (Sec. 3.2)

        # Create prototype blade
        blade = Blade.from_ntnu_bt1()
        blade.generate_markers(dr=dr)
        blade.set_lattice_spacing(dx=dx)

        return cls(
            blade_prototype=blade,
            n_blades=n_blades,
            hub_center=hub_center,
            omega=omega,
            theta_0=theta_0,
            rotation_axis=rotation_axis
        )

    @classmethod
    def from_simple(
        cls,
        n_blades: int = 3,
        radius: float = 0.447,
        hub_radius: float = 0.045,
        chord: float = 0.04,
        twist_root: float = 10.0,
        twist_tip: float = 0.0,
        omega: float = 100.0,
        hub_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dr: float = 0.005,
        dx: Optional[float] = None,
        theta_0: float = 0.0,
        rotation_axis: str = "hawt_x"
    ) -> 'Rotor':
        """Create a simple constant-chord rotor for testing

        Args:
            n_blades: Number of blades  [dimensionless]
            radius: Rotor radius (r_tip) [m or lu]
            hub_radius: Hub (inner) radius [m or lu]
            chord: Constant chord length [m or lu]
            twist_root: Twist at hub [degrees]
            twist_tip: Twist at tip [degrees]
            omega: Angular velocity [rad/s or rad/lu_time]
            hub_center: Hub center position [m or lu]
            dr: Marker spacing [m or lu]
            dx: Lattice spacing (for ε calculation) [m or lu]
            theta_0: Initial azimuth [rad]
            rotation_axis: Rotation axis preset [default: "hawt_x"]

        Returns:
            Rotor instance
        """
        blade = Blade.from_simple_blade(
            r_hub=hub_radius,
            r_tip=radius,
            chord=chord,
            twist_root=twist_root,
            twist_tip=twist_tip,
            airfoil='flat_plate'
        )
        blade.generate_markers(dr=dr)
        if dx is not None:
            blade.set_lattice_spacing(dx=dx)

        return cls(
            blade_prototype=blade,
            n_blades=n_blades,
            hub_center=hub_center,
            omega=omega,
            theta_0=theta_0,
            rotation_axis=rotation_axis
        )

    @classmethod
    def from_config(cls, config: Dict) -> 'Rotor':
        """Create rotor from a configuration dictionary

        Expected Config Format:
            {
                "n_blades": 3,
                "hub_center": [3.66, 1.341, 0.817],
                "omega": 134.23,          # or use tsr + u_inf
                "tsr": 6.0,               # alternative: compute ω
                "u_inf": 10.0,            # required if tsr is given
                "theta_0": 0.0,
                "rotation_axis": "hawt_x",  # NEW: rotation axis preset
                "blade": {
                    "preset": "ntnu_bt1"  # or explicit sections
                },
                "grid": {
                    "dx": 0.00559,        # or use resolution
                    "resolution": 160     # D/Δx
                }
            }

        Args:
            config: Rotor configuration dictionary

        Returns:
            Rotor instance
        """
        # Check for NTNU BT1 preset (shortcut)
        if config.get('preset') in ('ntnu_bt1', 'NTNU_BT1'):
            return cls.from_ntnu_bt1(
                tsr=config.get('tsr', 6.0),
                u_inf=config.get('u_inf', 10.0),
                hub_center=tuple(config.get('hub_center', (3.66, 1.341, 0.817))),
                resolution=config.get('grid', {}).get('resolution', 160),
                dx=config.get('grid', {}).get('dx', None),
                theta_0=config.get('theta_0', 0.0),
                rotation_axis=config.get('rotation_axis', 'hawt_x'),
            )

        # --- General construction ---
        # Blade
        blade_cfg = config.get('blade', {})
        blade = Blade.from_config(blade_cfg)

        # Grid
        grid_cfg = config.get('grid', {})
        dx = grid_cfg.get('dx', None)
        resolution = grid_cfg.get('resolution', 160)

        if dx is None:
            # Estimate dx from blade radius and resolution
            D = 2.0 * blade.r_tip
            dx = D / resolution

        dr = dx / 2.0  # Δr = Δx/2
        blade.generate_markers(dr=dr)
        blade.set_lattice_spacing(dx=dx)

        # Hub center
        hub_center = tuple(config.get('hub_center', (0.0, 0.0, 0.0)))

        # Angular velocity
        omega = config.get('omega', None)
        if omega is None:
            tsr = config.get('tsr', 6.0)
            u_inf = config.get('u_inf', 10.0)
            R = blade.r_tip
            omega = tsr * u_inf / R

        # Rotation axis
        rotation_axis = config.get('rotation_axis', 'hawt_x')

        return cls(
            blade_prototype=blade,
            n_blades=config.get('n_blades', 3),
            hub_center=hub_center,
            omega=omega,
            theta_0=config.get('theta_0', 0.0),
            rotation_axis=rotation_axis,
        )

    # =================================================================
    # §3. Unit Conversion Support
    # =================================================================

    def to_lattice_units(
        self,
        length_scale: float,
        time_scale: float
    ) -> 'Rotor':
        """Create a new Rotor with all quantities in lattice units

        Conversion Rules:
            Length:  L_lu = L_phys / length_scale        [lu]
            Time:    T_lu = T_phys / time_scale          [lu_time]
            Velocity: U_lu = U_phys · time_scale / length_scale  [lu/lt]
            ω_lu = ω_phys · time_scale                   [rad/lu_time]

        The coordinate system is also converted to lattice units.

        Args:
            length_scale: Physical size of one lattice cell  [m/lu]
            time_scale: Physical size of one timestep        [s/lu_time]

        Returns:
            New Rotor instance in lattice units
        """
        # Convert blade prototype to lattice units
        blade_lu = self.blades[0].to_lattice_units(length_scale)

        # Regenerate markers in lattice units
        dr_lu = self.blades[0].marker_dr / length_scale  # [lu]
        blade_lu.generate_markers(dr=dr_lu)
        blade_lu.set_lattice_spacing(dx=1.0)  # Δx = 1 in lattice units

        # Convert coordinate system to lattice units
        coord_lu = self._coord_system.to_lattice_units(length_scale)

        # Convert angular velocity: ω_lu = ω_phys · Δt
        omega_lu = self.omega * time_scale  # [rad/lu_time]

        # Use the coordinate system factory
        rotor_lu = Rotor.from_coordinate_system(
            blade_prototype=blade_lu,
            n_blades=self.n_blades,
            coord_system=coord_lu,
            omega=omega_lu,
            theta_0=self.theta[0]
        )

        return rotor_lu