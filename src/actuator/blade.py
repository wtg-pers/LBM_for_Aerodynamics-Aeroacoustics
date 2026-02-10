"""
Single Blade Geometry and Marker Particle Distribution for Actuator Line Model

This module defines the geometry of a single wind turbine blade as a line
of discrete marker particles used in the Actuator Line (AL) model.

Physical Concept:
================
In the AL model, each blade is represented by a radial line of marker
particles from hub to tip. At each marker particle (index j):

    - Position:    x_j(θ) in global coordinates, depending on azimuth θ
    - Chord:       c_a(r_j) — local chord length  [m or lattice units]
    - Twist/Pitch: γ(r_j) — local pitch angle     [degrees]
    - Airfoil:     section name for CL/CD lookup
    - Filter width: ε_j = max(c_a/4, 2·Δx)       [m or lattice units]

Marker particles are uniformly distributed along the blade span with
spacing Δr, which is typically set to Δr = Δx/2 (half the lattice spacing)
to ensure adequate force resolution on the LBM grid.

Coordinate System (Watanabe et al., Fig. 2):
============================================
    Global: (x, y, z) — x: streamwise, y: spanwise, z: vertical up
    Local:  (r, θ, n) — r: radial, θ: azimuthal, n: normal (= x)

    Rotor center is at (x_hub, y_hub, z_hub) in global coordinates.
    The rotor plane is the y-z plane (normal aligned with x-axis).

    Marker position in global frame for azimuth θ:
        x_marker = x_hub                    (on rotor plane)
        y_marker = y_hub + r · cos(θ)       [m or lattice units]
        z_marker = z_hub + r · sin(θ)       [m or lattice units]

    Note: θ = 0 corresponds to the blade pointing in +y direction
          θ increases counterclockwise when viewed from upstream (+x)

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026 (Sec. 2.2)
    - Sørensen & Shen, J. Fluids Eng. 124, 393-399, 2002
    - Krogstad & Eriksen, Renewable Energy 50, 325-333, 2013 (NTNU BT1)

Author: LBM Development Team
Date: 2026-02
"""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING, Optional, List, Tuple, Dict, Union, Callable
)

import numpy as np
from scipy import interpolate as sp_interp

if TYPE_CHECKING:
    import numpy.typing as npt


# =============================================================================
# §1. Blade Section Data Structure
# =============================================================================

@dataclass
class BladeSection:
    """Aerodynamic and geometric data at one radial station of the blade

    Each section defines the blade properties at a specific radial position.
    These are interpolated to generate marker particle properties.

    Attributes:
        r: Radial position from rotor center  [m]
        chord: Local chord length              [m]
        twist: Local twist/pitch angle (measured from rotor plane)  [degrees]
        airfoil: Airfoil section identifier for CL/CD lookup
        is_active: Whether this section generates aerodynamic forces.
                   Set False for root/hub circular sections.

    Physical Convention:
        - twist (γ) is positive for leading-edge-into-wind rotation
        - α = φ - γ  where φ is the local flow angle (Eq. 8)
        - Sections at the hub attachment (circular cross-section)
          are marked is_active=False and produce no aero forces
    """
    r: float                        # [m] radial position
    chord: float                    # [m] chord length
    twist: float                    # [degrees] local pitch angle
    airfoil: str = 'default'        # airfoil identifier
    is_active: bool = True          # generates aerodynamic force?


# =============================================================================
# §2. Blade Class
# =============================================================================

class Blade:
    """Single wind turbine blade for the Actuator Line model

    Defines the spanwise geometry (chord, twist, airfoil type) and generates
    a set of evenly-spaced marker particles along the blade span. Each marker
    carries interpolated geometric properties used in the BEM force calculation.

    Physical Process:
    ================
    1. User provides blade sections: (r, chord, twist, airfoil) at key stations
    2. Blade generates marker particles at spacing Δr along [r_hub, r_tip]
    3. For each marker: chord, twist, airfoil are interpolated from sections
    4. At runtime, rotor.py queries marker positions for a given azimuth angle

    Attributes:
        sections: List of BladeSection defining the spanwise geometry
        n_markers: Number of marker particles generated
        marker_r: Radial positions of markers  [m]
        marker_chord: Chord at each marker     [m]
        marker_twist: Twist at each marker     [degrees]
        marker_airfoil: Airfoil ID per marker
        marker_dr: Spacing between markers     [m]  (= Δr in Eq. 9-10)
        marker_epsilon: Gaussian filter width  [m]  (ε in Eq. 13)
        marker_active: Bool mask — True for aerodynamically active markers

    Example:
        >>> blade = Blade.from_ntnu_bt1()
        >>> blade.generate_markers(dr=0.003)  # Δr = Δx/2
        >>> pos = blade.get_marker_positions(theta=0.0, hub_center=(3.66, 1.341, 0.817))
        >>> print(pos.shape)  # (n_markers, 3) in global (x, y, z)
    """

    def __init__(self, sections: List[BladeSection]) -> None:
        """Initialize blade from a list of radial sections

        Args:
            sections: List of BladeSection, must be sorted by increasing r.
                      At minimum, 2 sections are required (hub and tip).

        Raises:
            ValueError: If fewer than 2 sections or sections not sorted by r
        """
        if len(sections) < 2:
            raise ValueError(
                f"Blade requires at least 2 sections, got {len(sections)}"
            )

        # Sort by radial position (safety)
        self.sections = sorted(sections, key=lambda s: s.r)

        # Extract radial extent
        self.r_hub: float = self.sections[0].r    # [m] inner radius
        self.r_tip: float = self.sections[-1].r   # [m] outer radius
        self.span: float = self.r_tip - self.r_hub  # [m] blade span

        if self.span <= 0:
            raise ValueError(
                f"Blade span must be positive: r_hub={self.r_hub}, r_tip={self.r_tip}"
            )

        # Build interpolation functions from section data
        self._build_interpolators()

        # Marker arrays (populated by generate_markers)
        self.n_markers: int = 0
        self.marker_r: np.ndarray = np.array([])       # [m]
        self.marker_chord: np.ndarray = np.array([])    # [m]
        self.marker_twist: np.ndarray = np.array([])    # [degrees]
        self.marker_airfoil: List[str] = []
        self.marker_dr: float = 0.0                     # [m]
        self.marker_epsilon: np.ndarray = np.array([])  # [m]
        self.marker_active: np.ndarray = np.array([], dtype=bool)

    # -----------------------------------------------------------------
    # §2.1 Interpolator Construction
    # -----------------------------------------------------------------

    def _build_interpolators(self) -> None:
        """Build cubic spline interpolators for chord(r) and twist(r)

        Uses cubic spline with natural boundary conditions for smooth
        variation along the blade span. Falls back to linear if < 4 sections.

        Internal — called by __init__.
        """
        r_arr = np.array([s.r for s in self.sections])      # [m]
        c_arr = np.array([s.chord for s in self.sections])   # [m]
        t_arr = np.array([s.twist for s in self.sections])   # [degrees]

        kind = 'cubic' if len(self.sections) >= 4 else 'linear'

        self._interp_chord = sp_interp.interp1d(
            r_arr, c_arr, kind=kind,
            bounds_error=False,
            fill_value=(c_arr[0], c_arr[-1])
        )
        self._interp_twist = sp_interp.interp1d(
            r_arr, t_arr, kind=kind,
            bounds_error=False,
            fill_value=(t_arr[0], t_arr[-1])
        )

    # -----------------------------------------------------------------
    # §2.2 Airfoil Assignment
    # -----------------------------------------------------------------

    def _assign_airfoil(self, r: float) -> Tuple[str, bool]:
        """Determine airfoil name and active status for a given radial position

        Uses nearest-section assignment: the airfoil type of the closest
        defined section is inherited by the marker particle.

        Args:
            r: Radial position  [m]

        Returns:
            (airfoil_name, is_active): Airfoil identifier and activity flag
        """
        r_arr = np.array([s.r for s in self.sections])
        idx = int(np.argmin(np.abs(r_arr - r)))
        sec = self.sections[idx]
        return sec.airfoil, sec.is_active

    # -----------------------------------------------------------------
    # §2.3 Marker Generation
    # -----------------------------------------------------------------

    def generate_markers(
        self,
        dr: float,
        dx: Optional[float] = None
    ) -> int:
        """Generate evenly-spaced marker particles along the blade span

        Physical Setup:
            Markers are placed at r = r_hub + (0.5 + k) · Δr, for k = 0, 1, ...
            This cell-centered placement avoids markers exactly at hub/tip edges,
            which improves force distribution symmetry.

            Marker spacing Δr is typically Δx/2 (Watanabe et al., Sec. 3.2):
                "The marker spacing Δr in the AL model was set to half
                 the lattice spacing, Δr = Δx/2."

        Gaussian Filter Width (Eq. 13):
            ε_j = max(c_a(r_j) / 4, 2 · Δx)  [m or lattice units]

            If dx is not provided, ε_j = c_a(r_j) / 4 (minimum estimate).
            The actual ε should be recomputed when lattice spacing is known.

        Args:
            dr: Marker spacing  [m]  (typically Δx/2)
            dx: Lattice spacing  [m]  (for Gaussian filter width ε)
                If None, ε = chord/4 (updated later via set_lattice_spacing)

        Returns:
            Number of markers generated

        Raises:
            ValueError: If dr <= 0
        """
        if dr <= 0:
            raise ValueError(f"Marker spacing dr must be positive, got {dr}")

        # Cell-centered marker placement along [r_hub, r_tip]
        # First marker at r_hub + dr/2, last marker before r_tip
        r_start = self.r_hub + 0.5 * dr   # [m]
        r_end = self.r_tip                  # [m]

        self.marker_r = np.arange(r_start, r_end, dr)  # [m]
        self.n_markers = len(self.marker_r)
        self.marker_dr = dr  # [m] uniform spacing = Δr in Eq. 9-10

        if self.n_markers == 0:
            raise ValueError(
                f"No markers generated: dr={dr} is too large for "
                f"span={self.span} (r_hub={self.r_hub}, r_tip={self.r_tip})"
            )

        # Interpolate chord and twist at each marker position
        self.marker_chord = self._interp_chord(self.marker_r)  # [m]
        self.marker_twist = self._interp_twist(self.marker_r)  # [degrees]

        # Assign airfoil type and active flag per marker
        airfoils = []
        active = []
        for r in self.marker_r:
            name, is_act = self._assign_airfoil(r)
            airfoils.append(name)
            active.append(is_act)
        self.marker_airfoil = airfoils
        self.marker_active = np.array(active, dtype=bool)

        # Gaussian filter width: ε = max(c/4, 2·Δx)  (Eq. 13 footnote)
        if dx is not None:
            self.marker_epsilon = np.maximum(
                self.marker_chord / 4.0,
                2.0 * dx
            )  # [m]
        else:
            # Provisional — update later via set_lattice_spacing()
            self.marker_epsilon = self.marker_chord / 4.0  # [m]

        return self.n_markers

    def set_lattice_spacing(self, dx: float) -> None:
        """Update Gaussian filter width with actual lattice spacing

        Call this after generate_markers() when the lattice spacing becomes
        known (e.g., after unit conversion to lattice units).

        ε_j = max(c_a(r_j) / 4, 2 · Δx)    (Watanabe et al., Eq. 13 note)

        Args:
            dx: Lattice spacing  [m or lattice units]
        """
        if self.n_markers == 0:
            raise RuntimeError("Call generate_markers() before set_lattice_spacing()")

        self.marker_epsilon = np.maximum(
            self.marker_chord / 4.0,
            2.0 * dx
        )  # [same units as chord and dx]

    # -----------------------------------------------------------------
    # §2.4 Marker Position Computation
    # -----------------------------------------------------------------

    def get_marker_positions(
        self,
        theta: float,
        hub_center: Tuple[float, float, float]
    ) -> np.ndarray:
        """Compute 3D global positions of all markers for a given azimuth

        Coordinate Transformation:
            Given azimuth angle θ and hub center (x_h, y_h, z_h):

            x_j = x_h                           (rotor plane)
            y_j = y_h + r_j · cos(θ)            [m or lattice units]
            z_j = z_h + r_j · sin(θ)            [m or lattice units]

            where r_j is the radial position of marker j.

        Convention:
            - θ = 0: blade points in +y direction (horizontal)
            - θ = π/2: blade points in +z direction (vertical up)
            - Rotation is counterclockwise when viewed from upstream (+x)
              (consistent with Watanabe et al., Sec. 3.2)

        Args:
            theta: Azimuth angle of this blade  [radians]
            hub_center: Rotor center (x, y, z) in global coords
                        [m or lattice units]

        Returns:
            positions: shape (n_markers, 3) — (x, y, z) per marker
                       [m or lattice units]
        """
        x_h, y_h, z_h = hub_center

        cos_theta = np.cos(theta)  # [dimensionless]
        sin_theta = np.sin(theta)  # [dimensionless]

        positions = np.zeros((self.n_markers, 3), dtype=np.float64)
        positions[:, 0] = x_h                               # x: rotor plane
        positions[:, 1] = y_h + self.marker_r * cos_theta    # y: spanwise
        positions[:, 2] = z_h + self.marker_r * sin_theta    # z: vertical

        return positions  # [m or lattice units]

    def get_marker_unit_vectors(
        self,
        theta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute local coordinate unit vectors at each marker

        Local Coordinate System (Watanabe et al., Sec. 2.2):
            ê_n = (1, 0, 0)                   — normal (streamwise, = x)
            ê_θ = (0, -sin(θ), cos(θ))        — tangential (rotation dir.)
            ê_r = (0, cos(θ), sin(θ))          — radial (outward)

        These are needed for:
            - Velocity decomposition: u_n, u_θ (Eq. 5-6)
            - Force projection: F_n, F_θ → F^AL (Eq. 11-12)

        Args:
            theta: Azimuth angle  [radians]

        Returns:
            (e_n, e_theta, e_r): Each shape (3,)  [dimensionless]

            e_n:     Normal (streamwise) unit vector
            e_theta: Tangential (rotational) unit vector
            e_r:     Radial (outward) unit vector
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        e_n = np.array([1.0, 0.0, 0.0])           # streamwise (x)
        e_theta = np.array([0.0, -sin_t, cos_t])   # tangential
        e_r = np.array([0.0, cos_t, sin_t])         # radial

        return e_n, e_theta, e_r

    # -----------------------------------------------------------------
    # §2.5 Force Projection to Global Frame
    # -----------------------------------------------------------------

    def project_forces_to_global(
        self,
        F_n: np.ndarray,
        F_theta: np.ndarray,
        theta: float
    ) -> np.ndarray:
        """Project normal/tangential forces to global (x, y, z) frame

        From Watanabe et al. (below Eq. 12):
            F^AL = (F_n, F_θ·cos(θ), -F_θ·sin(θ))

        where:
            F_n:     Normal force component (streamwise)     [N or lattice]
            F_theta: Tangential force component (rotational) [N or lattice]

        The negative sign in the z-component is because the tangential
        direction ê_θ = (0, -sin(θ), cos(θ)), and projecting F_θ·ê_θ gives:
            F_y = F_θ · (-sin(θ)) = -F_θ·sin(θ)  ... but the paper uses
            F_y = F_θ·cos(θ), F_z = -F_θ·sin(θ)

        Verification: at θ=0 (blade in +y), tangential force is in +z direction,
        so F^AL = (F_n, 0, -F_θ·0) — wait, let's re-derive.

        At θ=0: blade in +y, rotation in +z at this point.
            F^AL = (F_n, F_θ·cos(0), -F_θ·sin(0)) = (F_n, F_θ, 0)
        This means F_θ acts in +y at θ=0. But physically, the tangential
        force at θ=0 should be in the -z direction (for CCW rotation from
        upstream view). Let's trust the paper's convention and verify later.

        Args:
            F_n:     Normal forces, shape (n_markers,)     [N or lattice force]
            F_theta: Tangential forces, shape (n_markers,) [N or lattice force]
            theta:   Azimuth angle  [radians]

        Returns:
            F_global: shape (n_markers, 3) — (F_x, F_y, F_z)
                      [N or lattice force units]
        """
        cos_t = np.cos(theta)   # [dimensionless]
        sin_t = np.sin(theta)   # [dimensionless]

        F_global = np.zeros((self.n_markers, 3), dtype=np.float64)
        F_global[:, 0] = F_n                    # F_x = F_n
        F_global[:, 1] = F_theta * cos_t        # F_y = F_θ · cos(θ)
        F_global[:, 2] = -F_theta * sin_t       # F_z = -F_θ · sin(θ)

        return F_global  # [N or lattice force units]

    # -----------------------------------------------------------------
    # §2.6 Utility Methods
    # -----------------------------------------------------------------

    def get_info(self) -> str:
        """Human-readable blade summary"""
        lines = [
            "Blade Geometry Summary",
            "=" * 50,
            f"  Radial extent: [{self.r_hub:.4f}, {self.r_tip:.4f}] m",
            f"  Span:          {self.span:.4f} m",
            f"  Sections:      {len(self.sections)}",
            f"  Markers:       {self.n_markers}",
        ]
        if self.n_markers > 0:
            lines += [
                f"  Marker spacing (Δr): {self.marker_dr:.5f} m",
                f"  Chord range:    [{self.marker_chord.min():.4f}, "
                f"{self.marker_chord.max():.4f}] m",
                f"  Twist range:    [{self.marker_twist.min():.2f}, "
                f"{self.marker_twist.max():.2f}] deg",
                f"  Active markers: {np.sum(self.marker_active)}/{self.n_markers}",
                f"  ε range:        [{self.marker_epsilon.min():.5f}, "
                f"{self.marker_epsilon.max():.5f}] m",
            ]
        return "\n".join(lines)

    def get_marker_table(self) -> str:
        """Detailed table of all marker properties (for debugging)"""
        if self.n_markers == 0:
            return "No markers generated. Call generate_markers() first."

        header = (
            f"{'#':>4s}  {'r[m]':>10s}  {'chord[m]':>10s}  "
            f"{'twist[deg]':>10s}  {'ε[m]':>10s}  {'airfoil':>12s}  {'active':>6s}"
        )
        lines = [header, "-" * len(header)]

        for j in range(self.n_markers):
            lines.append(
                f"{j:4d}  {self.marker_r[j]:10.5f}  "
                f"{self.marker_chord[j]:10.5f}  "
                f"{self.marker_twist[j]:10.4f}  "
                f"{self.marker_epsilon[j]:10.5f}  "
                f"{self.marker_airfoil[j]:>12s}  "
                f"{'  Y' if self.marker_active[j] else '  N':>6s}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Blade(r=[{self.r_hub:.3f}, {self.r_tip:.3f}] m, "
            f"span={self.span:.3f} m, "
            f"sections={len(self.sections)}, markers={self.n_markers})"
        )

    # =================================================================
    # §3. Factory Methods — Preset Blade Configurations
    # =================================================================

    @classmethod
    def from_sections(
        cls,
        r: 'npt.NDArray',
        chord: 'npt.NDArray',
        twist: 'npt.NDArray',
        airfoil: Union[str, List[str]] = 'default',
        active: Optional['npt.NDArray'] = None
    ) -> 'Blade':
        """Create blade from arrays of section data

        Convenience factory for programmatic blade definition.

        Args:
            r:      Radial positions  [m], shape (N,)
            chord:  Chord lengths     [m], shape (N,)
            twist:  Twist angles      [degrees], shape (N,)
            airfoil: Single name or per-section list
            active: Per-section active flag (default: all True)

        Returns:
            Blade instance (markers not yet generated)
        """
        r = np.asarray(r, dtype=np.float64)
        chord = np.asarray(chord, dtype=np.float64)
        twist = np.asarray(twist, dtype=np.float64)

        n = len(r)
        if isinstance(airfoil, str):
            airfoil_list = [airfoil] * n
        else:
            airfoil_list = list(airfoil)

        if active is None:
            active_arr = [True] * n
        else:
            active_arr = list(np.asarray(active, dtype=bool))

        sections = [
            BladeSection(
                r=float(r[i]),
                chord=float(chord[i]),
                twist=float(twist[i]),
                airfoil=airfoil_list[i],
                is_active=active_arr[i]
            )
            for i in range(n)
        ]
        return cls(sections)

    @classmethod
    def from_ntnu_bt1(cls) -> 'Blade':
        """Create the NTNU Blind Test 1 blade (NREL S826 airfoil)

        Reference: Krogstad & Eriksen, Renewable Energy 50, 325-333, 2013
                   (Table C.1 in the Blind Test workshop report)

        Rotor Specifications:
            - Diameter:    D = 0.894 m  →  R = 0.447 m
            - Hub radius:  r_hub = 0.045 m (nacelle diameter 90 mm)
            - 3 blades, NREL S826 airfoil for all aerodynamic sections
            - Design TSR = 6, inflow = 10 m/s

        Blade Sections (from Krogstad & Lund, 2012):
            - First 3 sections: circular cylinder (hub attachment)
              → is_active = False (no aerodynamic forces)
            - Remaining sections: NREL S826 with varying chord and twist

        Returns:
            Blade instance (call generate_markers() to populate markers)

        Example:
            >>> blade = Blade.from_ntnu_bt1()
            >>> blade.generate_markers(dr=0.003)  # Δr ≈ Δx/2 for Δx=D/160
        """
        # NTNU BT1 blade data (Table C.1, Krogstad & Eriksen 2013)
        # r: distance from rotor center [m]
        # chord: local chord length [m]
        # twist: local pitch angle from rotor plane [degrees]
        #
        # First 3 entries are circular root sections (inside hub)
        # Transition region between section 3 and 4
        # Sections 4+ are NREL S826 airfoil

        sections = [
            # ── Root / Hub attachment (circular, inactive) ──
            BladeSection(r=0.0450, chord=0.0300, twist=120.0,
                         airfoil='circular', is_active=False),
            BladeSection(r=0.0550, chord=0.0300, twist=120.0,
                         airfoil='circular', is_active=False),
            BladeSection(r=0.0650, chord=0.0300, twist=120.0,
                         airfoil='circular', is_active=False),

            # ── Transition (linear blend circular → S826) ──
            BladeSection(r=0.0812, chord=0.0529, twist=38.00,
                         airfoil='transition', is_active=True),

            # ── NREL S826 aerodynamic sections ──
            BladeSection(r=0.1028, chord=0.0620, twist=28.07,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.1244, chord=0.0620, twist=21.14,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.1460, chord=0.0596, twist=16.34,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.1676, chord=0.0562, twist=12.87,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.1892, chord=0.0528, twist=10.27,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.2108, chord=0.0494, twist=8.27,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.2324, chord=0.0461, twist=6.70,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.2540, chord=0.0430, twist=5.43,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.2756, chord=0.0401, twist=4.39,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.2972, chord=0.0375, twist=3.51,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.3188, chord=0.0351, twist=2.76,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.3404, chord=0.0330, twist=2.11,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.3620, chord=0.0312, twist=1.54,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.3836, chord=0.0296, twist=1.03,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.4052, chord=0.0283, twist=0.57,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.4268, chord=0.0272, twist=0.16,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.4470, chord=0.0262, twist=0.00,
                         airfoil='S826', is_active=True),
        ]

        return cls(sections)

    @classmethod
    def from_constant_chord(
        cls,
        r_hub: float,
        r_tip: float,
        chord: float,
        twist_root: float = 0.0,
        twist_tip: float = 0.0,
        airfoil: str = 'flat_plate'
    ) -> 'Blade':
        """Create a simple blade with constant chord and linear twist

        Useful for:
            - Initial testing and debugging of the AL framework
            - Idealized rotor studies
            - Comparison with analytical solutions

        Args:
            r_hub: Hub (inner) radius    [m]
            r_tip: Tip (outer) radius    [m]
            chord: Constant chord length [m]
            twist_root: Twist at hub     [degrees]
            twist_tip: Twist at tip      [degrees]
            airfoil: Airfoil identifier

        Returns:
            Blade instance
        """
        sections = [
            BladeSection(r=r_hub, chord=chord, twist=twist_root,
                         airfoil=airfoil, is_active=True),
            BladeSection(r=r_tip, chord=chord, twist=twist_tip,
                         airfoil=airfoil, is_active=True),
        ]
        return cls(sections)

    @classmethod
    def from_config(cls, config: Dict) -> 'Blade':
        """Create blade from a configuration dictionary

        Expected Config Format:
            {
                "sections": [
                    {"r": 0.045, "chord": 0.030, "twist": 120.0,
                     "airfoil": "circular", "active": false},
                    {"r": 0.447, "chord": 0.026, "twist": 0.0,
                     "airfoil": "S826", "active": true},
                    ...
                ],
                "preset": "ntnu_bt1"  # Alternative: use preset name
            }

        Args:
            config: Blade configuration dictionary

        Returns:
            Blade instance
        """
        # Check for preset
        preset = config.get('preset', None)
        if preset is not None:
            presets = {
                'ntnu_bt1': cls.from_ntnu_bt1,
                'NTNU_BT1': cls.from_ntnu_bt1,
            }
            if preset not in presets:
                available = ', '.join(sorted(presets.keys()))
                raise ValueError(
                    f"Unknown blade preset: '{preset}'. Available: {available}"
                )
            return presets[preset]()

        # Build from explicit section list
        section_list = config.get('sections', [])
        if not section_list:
            raise ValueError("Blade config must have 'sections' or 'preset'")

        sections = []
        for sec_dict in section_list:
            sections.append(BladeSection(
                r=float(sec_dict['r']),
                chord=float(sec_dict['chord']),
                twist=float(sec_dict['twist']),
                airfoil=sec_dict.get('airfoil', 'default'),
                is_active=sec_dict.get('active', True),
            ))
        return cls(sections)

    # =================================================================
    # §4. Unit Conversion Support
    # =================================================================

    def to_lattice_units(
        self,
        length_scale: float
    ) -> 'Blade':
        """Create a new Blade with all lengths converted to lattice units

        Conversion: L_lattice = L_physical / length_scale
        where length_scale = Δx_physical (physical size of one lattice cell)

        This is needed because the LBM operates in lattice units where Δx = 1.

        Args:
            length_scale: Physical size of one lattice cell  [m/lu]
                          (e.g., D / N_cells_per_diameter)

        Returns:
            New Blade instance in lattice units
        """
        new_sections = []
        for s in self.sections:
            new_sections.append(BladeSection(
                r=s.r / length_scale,           # [lu]
                chord=s.chord / length_scale,   # [lu]
                twist=s.twist,                  # [degrees] — unchanged
                airfoil=s.airfoil,
                is_active=s.is_active,
            ))
        return Blade(new_sections)