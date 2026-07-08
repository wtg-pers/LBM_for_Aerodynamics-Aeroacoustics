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

Coordinate System:
==================
This module now supports arbitrary rotation axes via RotorCoordinateSystem.

    Legacy mode (coord_system=None):
        - Hardcoded X-axis rotation (Watanabe et al. convention)
        - Rotor plane is Y-Z plane
        - θ=0: blade in +Y direction

    New mode (coord_system set):
        - Arbitrary rotation axis support
        - Uses RotorCoordinateSystem for all transformations
        - Supports HAWT, VAWT, and custom configurations

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
    from src.actuator.coordinates import RotorCoordinateSystem


# =============================================================================
# §0. Marker radial-distribution helpers
# =============================================================================
def _cosine_map(xi: np.ndarray, side: str) -> np.ndarray:
    """Map ξ∈[0,1] → [0,1], monotonic, clustering points toward an edge.

    side="both": dense at both ends, "tip": dense at ξ=1, "root": dense at ξ=0.
    All three satisfy f(0)=0, f(1)=1 (so the span endpoints are preserved).
    """
    if side == "both":
        return 0.5 * (1.0 - np.cos(np.pi * xi))
    if side == "tip":
        return np.sin(0.5 * np.pi * xi)              # dense near ξ=1
    if side == "root":
        return 1.0 - np.cos(0.5 * np.pi * xi)        # dense near ξ=0
    raise ValueError(f"unknown cosine_side {side!r} (expected 'both','tip','root')")


def _trapezoid_widths(r: np.ndarray) -> np.ndarray:
    """Trapezoidal control-volume widths for nodes r (endpoints get half-cells).

    Δr_0 = (r_1-r_0)/2,  Δr_{N-1} = (r_{N-1}-r_{N-2})/2,
    Δr_j = (r_{j+1}-r_{j-1})/2 (interior).  Σ Δr = r_{N-1} - r_0 (= span).
    """
    r = np.asarray(r, dtype=np.float64)
    dr = np.empty_like(r)
    dr[1:-1] = 0.5 * (r[2:] - r[:-2])
    dr[0] = 0.5 * (r[1] - r[0])
    dr[-1] = 0.5 * (r[-1] - r[-2])
    return dr


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
    sweep: float = 0.0              # [degrees] LE sweep angle Λ (0 = unswept)


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

    Coordinate System Support:
    ==========================
    The blade can operate in two modes:
    
    1. New mode (recommended): Set coord_system via Rotor
       - Supports arbitrary rotation axes
       - Uses RotorCoordinateSystem for all transformations
    
    2. Legacy mode: coord_system = None
       - Hardcoded X-axis rotation
       - Requires hub_center parameter in get_marker_positions()
       - Issues DeprecationWarning

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
        >>> blade.generate_markers(dr=0.003)
        >>> # New way: use coordinate system
        >>> from src.actuator.coordinates import RotorCoordinateSystem
        >>> coord = RotorCoordinateSystem.hawt_standard((3.66, 1.341, 0.817))
        >>> blade.set_coordinate_system(coord)
        >>> pos = blade.get_marker_positions(theta=0.0)
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
        
        # Root-cut radius: first active section's radial position  [m]
        # Determines where marker particles actually begin.
        # Physical meaning: hub/nacelle region produces no aero forces
        active_sections = [s for s in self.sections if s.is_active]
        if active_sections:
            self.r_root_cut: float = active_sections[0].r   # [m]
        else:
            self.r_root_cut: float = self.r_hub             # [m] fallback

        # Build interpolation functions from section data
        self._build_interpolators()

        # Marker arrays (populated by generate_markers)
        self.n_markers: int = 0
        self.marker_r: np.ndarray = np.array([])       # [m]
        self.marker_chord: np.ndarray = np.array([])    # [m]
        self.marker_twist: np.ndarray = np.array([])    # [degrees]
        self.marker_sweep: np.ndarray = np.array([])    # [degrees] LE sweep Λ
        # Geometric tip sweep: cumulative LE tangential displacement Δs(r) [m]
        # (Step 1, patch_notes/alm_tip_sweep_refine). When `sweep_geometric` is
        # True, markers are physically offset by (sweep_sign·Δs)·ê_θ each step so
        # sampling/spreading act at the swept-back location. Default OFF →
        # positions are bit-identical to the straight-line marker line.
        self.marker_sweep_offset: np.ndarray = np.array([])   # [m] Δs along ê_θ
        self.sweep_geometric: bool = False              # apply geometric offset?
        self._sweep_sign: float = -1.0                  # -sign(ω): swept-back = -motion
        # Sharp tip sweep (method 2): straight swept LE beyond a break r/R at a
        # CONSTANT angle. When sweep_break_rR is set, Λ(r) is a STEP (not the
        # section-interpolated ramp) and Δs(r) is exactly linear → matches the TM
        # tip back-set. None → legacy section-interpolated sweep.
        self.sweep_break_rR: Optional[float] = None     # r/R where sweep starts
        self.sweep_tip_deg: float = 0.0                 # constant Λ beyond break [deg]
        self.marker_airfoil: List[str] = []
        self.marker_dr: float = 0.0                     # [m]
        self.marker_epsilon: np.ndarray = np.array([])  # [m]
        self.marker_active: np.ndarray = np.array([], dtype=bool)
        self._marker_distribution: str = "uniform"      # set by generate_markers
        self._cosine_side: str = "both"

        # Gaussian projection-width (ε) taper controls.
        # "default" reproduces ε = max(chord/4, 2·Δx) exactly; "tip_taper"
        # narrows ε toward the tip (Diaz 2023 §2.1.4) to reduce tip-vortex
        # over-smearing. Set by Rotor.from_config; see set_lattice_spacing().
        self.epsilon_mode: str = "default"      # "default" | "tip_taper"
        self.epsilon_tip_factor: float = 1.0    # tip ε target = max(factor·2·Δx, 2·Δx)
        self.epsilon_taper_start: float = 0.7   # r/R where taper begins

        # Coordinate system (set by Rotor or explicitly)
        # If None, falls back to legacy hardcoded X-axis rotation
        self._coord_system: Optional['RotorCoordinateSystem'] = None

    # -----------------------------------------------------------------
    # §2.1 Interpolator Construction
    # -----------------------------------------------------------------

    def _build_interpolators(self) -> None:
        """Build interpolators for chord(r) and twist(r)

        Uses linear interpolation for 2 sections, cubic spline for 3+ sections.
        """
        r_arr = np.array([s.r for s in self.sections])
        chord_arr = np.array([s.chord for s in self.sections])
        twist_arr = np.array([s.twist for s in self.sections])
        sweep_arr = np.array([s.sweep for s in self.sections])

        # Choose interpolation method based on number of sections
        if len(self.sections) <= 2:
            # Linear interpolation for 2 sections
            kind = 'linear'
        else:
            # Cubic spline for 3+ sections
            kind = 'cubic'

        self._chord_interp: Callable[[float], float] = sp_interp.interp1d(
            r_arr, chord_arr, kind=kind, fill_value='extrapolate'
        )
        self._twist_interp: Callable[[float], float] = sp_interp.interp1d(
            r_arr, twist_arr, kind=kind, fill_value='extrapolate'
        )
        # Sweep Λ(r): always LINEAR — the LE sweep is a piecewise-linear ramp
        # (0 inboard, swept tip), and cubic would overshoot the break.
        self._sweep_interp: Callable[[float], float] = sp_interp.interp1d(
            r_arr, sweep_arr, kind='linear', fill_value='extrapolate'
        )

        # Build airfoil lookup: find nearest section for airfoil ID
        self._section_r = r_arr
        self._section_airfoil = [s.airfoil for s in self.sections]
        self._section_active = np.array([s.is_active for s in self.sections])

    def _get_airfoil_at_r(self, r: float) -> str:
        """Get airfoil identifier at radial position r (nearest-neighbor)"""
        idx = np.argmin(np.abs(self._section_r - r))
        return self._section_airfoil[idx]

    def _get_active_at_r(self, r: float) -> bool:
        """Get active flag at radial position r (nearest-neighbor)"""
        idx = np.argmin(np.abs(self._section_r - r))
        return bool(self._section_active[idx])

    # -----------------------------------------------------------------
    # §2.2 Marker Generation
    # -----------------------------------------------------------------

    def generate_markers(
            self,
            n_radial: int,
            skip_inactive_root: bool = True,
            distribution: str = "uniform",
            cosine_side: str = "both",
        ) -> None:
        """Generate evenly-spaced marker particles along the blade span

        Markers are placed at:
            r_j = r_start + (j + 0.5) · Δr    for j = 0, 1, ..., N-1
            where Δr = effective_span / n_radial

        The user explicitly specifies how many radial elements to use.
        This decouples blade element resolution from LBM grid resolution
        (RESOLUTION = D/Δx), which controls a completely different aspect
        of the simulation.

        Guideline:
            Watanabe et al. recommend Δr ≈ Δx/2 as a starting point.
            With RESOLUTION=40 (R=20 cells) and active span ≈ 0.8R,
            that gives n_radial ≈ 0.8×20×2 = 32. But the user is free
            to choose any value based on their accuracy/cost trade-off.

        The root region (r < r_root_cut) belongs to the hub/nacelle and
        produces no aerodynamic forces. Placing markers there wastes
        computation and creates visual artifacts.

        Args:
            n_radial: Number of markers along active blade span  [-]
            skip_inactive_root: If True, start markers from first active
                                section. If False, start from r_hub.
                                [default: True]

        Raises:
            ValueError: If n_radial < 1 or no active span
        """
        if n_radial < 1:
            raise ValueError(f"n_radial must be >= 1, got {n_radial}")

        # --- Determine starting radius ---
        if skip_inactive_root and hasattr(self, 'r_root_cut'):
            r_start = self.r_root_cut     # [m] first active section
        else:
            r_start = self.r_hub          # [m] legacy: from absolute hub

        # Effective span for marker placement  [m]
        effective_span = self.r_tip - r_start  # [m]

        if effective_span <= 0:
            raise ValueError(
                f"No active blade span: r_start={r_start:.6f}, "
                f"r_tip={self.r_tip:.6f}. Check blade section definitions."
            )

        n = n_radial
        self.n_markers = n
        r_tip = self.r_tip
        # Remember the distribution so unit conversions (to_lattice_units) that
        # regenerate markers preserve it instead of silently reverting to uniform.
        self._marker_distribution = distribution
        self._cosine_side = cosine_side

        # --- Radial marker placement (3 distributions; Σ marker_dr = span) ---
        if distribution == "uniform":
            # Cell-centred, equal spacing.  marker_dr stays a SCALAR so the
            # downstream correction path is byte-identical to all prior runs.
            dr = effective_span / n                          # [m]
            self.marker_dr = dr
            self.marker_r = r_start + (np.arange(n) + 0.5) * dr   # [m]
        elif distribution == "cosine":
            # Cell-centred clustering toward root/tip/both.  Cells are defined by
            # edges mapped through the cosine, so Σ(cell width) = span exactly;
            # markers sit at cell centres.  Reduces to "uniform" for a linear map.
            xi_e = np.arange(n + 1) / n                      # N+1 edges in [0,1]
            r_edges = r_start + effective_span * _cosine_map(xi_e, cosine_side)
            self.marker_r = 0.5 * (r_edges[:-1] + r_edges[1:])    # cell centres [m]
            self.marker_dr = np.diff(r_edges)                # (n,) per-cell Δr [m]
        elif distribution == "endpoint":
            # Nodes placed AT the endpoints (root-cut and tip) with trapezoidal
            # control volumes (half-width at the ends) → Σ Δr = span.
            self.marker_r = np.linspace(r_start, r_tip, n)   # [m] endpoints incl.
            self.marker_dr = _trapezoid_widths(self.marker_r)    # (n,) Δr [m]
        elif distribution == "cosine_endpoint":
            # HYBRID of cosine ⊕ endpoint: cosine-CLUSTERED NODES that still land
            # exactly on the endpoints (root-cut & tip, since _cosine_map(0)=0,
            # (1)=1), with trapezoidal control volumes.  → tip/root clustering AND
            # closure markers at r/R=root-cut & 1.0.  cosine_side picks the side(s).
            xi = np.arange(n) / (n - 1)                      # N nodes in [0,1] incl. ends
            self.marker_r = r_start + effective_span * _cosine_map(xi, cosine_side)
            self.marker_dr = _trapezoid_widths(self.marker_r)    # (n,) Δr [m]
        else:
            raise ValueError(
                f"unknown marker distribution {distribution!r} (expected "
                "'uniform', 'cosine', 'endpoint', or 'cosine_endpoint')")

        # Interpolate properties at each marker
        self.marker_chord = self._chord_interp(self.marker_r)    # [m]
        self.marker_twist = self._twist_interp(self.marker_r)    # [degrees]
        # LE sweep Λ(r) and the geometric back-set Δs(r) of the swept quarter-
        # chord line from the straight pitch axis (0 inboard; used by the aero
        # cosΛ and, when sweep_geometric is on, by the marker offset).
        if self.sweep_break_rR is not None:
            # Sharp sweep (method 2): straight swept LE at CONSTANT Λ beyond the
            # break → Λ is a STEP and Δs is EXACTLY linear (tan Λ·(r − r_break)),
            # independent of marker/section spacing. Matches the TM tip back-set.
            rR = self.marker_r / self.r_tip
            self.marker_sweep = np.where(
                rR > self.sweep_break_rR, self.sweep_tip_deg, 0.0)   # [deg] step
            r_break = self.sweep_break_rR * self.r_tip               # [m]
            self.marker_sweep_offset = (
                np.tan(np.radians(self.sweep_tip_deg))
                * np.maximum(self.marker_r - r_break, 0.0))          # [m] exact
        else:
            # Legacy: section-interpolated Λ + discretised Δs = Σ tan(Λ)·dr.
            self.marker_sweep = self._sweep_interp(self.marker_r)    # [deg] Λ
            dr_arr = (self.marker_dr if np.ndim(self.marker_dr) > 0
                      else np.full(n, self.marker_dr))       # (n,) per-marker Δr [m]
            self.marker_sweep_offset = np.cumsum(
                np.tan(np.radians(self.marker_sweep)) * dr_arr)   # [m]

        # Airfoil and active flag (nearest-neighbor lookup)
        self.marker_airfoil = [self._get_airfoil_at_r(r) for r in self.marker_r]
        self.marker_active = np.array(
            [self._get_active_at_r(r) for r in self.marker_r], dtype=bool
        )

        # Gaussian filter width (placeholder, set later by set_lattice_spacing)
        self.marker_epsilon = np.zeros(n)  # [m], set later

    def set_lattice_spacing(self, dx: float) -> None:
        """Set the Gaussian filter width based on lattice spacing

        The baseline regularization parameter ε for each marker is:

            ε_j = max(c_a(r_j) / 4, 2 · Δx)    (Watanabe et al., Eq. 13 note)

        When ``self.epsilon_mode == "tip_taper"`` the baseline is linearly
        blended toward a narrower tip value beyond ``epsilon_taper_start``
        (Diaz 2023 §2.1.4) to reduce tip-vortex over-smearing:

            t   = clip((r/R - taper_start)/(1 - taper_start), 0, 1)
            ε_j = (1 - t)·ε_base + t·max(epsilon_tip_factor·2·Δx, 2·Δx)

        The ``"default"`` branch is byte-identical to the original formula so
        existing results stay bit-reproducible.

        Args:
            dx: Lattice spacing  [m or lattice units]
        """
        if self.n_markers == 0:
            raise RuntimeError("Call generate_markers() before set_lattice_spacing()")

        eps_base = np.maximum(
            self.marker_chord / 4.0,
            2.0 * dx
        )  # [same units as chord and dx]

        if self.epsilon_mode == "tip_taper":
            r_norm = self.marker_r / self.r_tip                      # [dimensionless]
            taper_start = self.epsilon_taper_start
            t = np.clip(
                (r_norm - taper_start) / (1.0 - taper_start),
                0.0, 1.0,
            )                                                        # [dimensionless]
            eps_tip = max(self.epsilon_tip_factor * 2.0 * dx, 2.0 * dx)  # ≥ floor
            self.marker_epsilon = (1.0 - t) * eps_base + t * eps_tip
        else:
            self.marker_epsilon = eps_base

    # -----------------------------------------------------------------
    # §2.3 Coordinate System Integration
    # -----------------------------------------------------------------

    @property
    def coord_system(self) -> Optional['RotorCoordinateSystem']:
        """Get the coordinate system for this blade
        
        Returns:
            RotorCoordinateSystem if set, None otherwise
        """
        return self._coord_system
    
    @coord_system.setter
    def coord_system(self, value: Optional['RotorCoordinateSystem']) -> None:
        """Set the coordinate system for this blade
        
        This is typically called by the Rotor when the blade is attached.
        
        Args:
            value: RotorCoordinateSystem instance or None
        """
        self._coord_system = value
    
    def set_coordinate_system(
        self, 
        coord_system: 'RotorCoordinateSystem'
    ) -> None:
        """Explicitly set the coordinate system
        
        Args:
            coord_system: RotorCoordinateSystem instance
        """
        self._coord_system = coord_system

    def has_coordinate_system(self) -> bool:
        """Check if a coordinate system is set
        
        Returns:
            True if coord_system is set, False otherwise
        """
        return self._coord_system is not None

    # -----------------------------------------------------------------
    # §2.4 Marker Position Computation
    # -----------------------------------------------------------------

    def get_marker_positions(
        self,
        theta: float,
    ) -> np.ndarray:
        """Compute 3D global positions of all markers for a given azimuth

        Coordinate Transformation:
            Uses RotorCoordinateSystem.marker_position() for arbitrary axes.
            Hub center is stored in the coordinate system.

        Args:
            theta: Azimuth angle of this blade  [radians]

        Returns:
            positions: shape (n_markers, 3) — (x, y, z) per marker
                       [m or lattice units]

        Raises:
            RuntimeError: If coord_system has not been set.
        """
        if self._coord_system is None:
            raise RuntimeError(
                "Blade.coord_system is not set. "
                "Use Rotor to create blades with a coordinate system."
            )
        pos = self._coord_system.marker_position(self.marker_r, theta)  # (n,3)
        # Geometric tip sweep (Step 1): shift each marker back along the blade
        # motion direction by Δs·ê_θ (swept-back = -sign(ω)·ê_θ). ê_θ co-rotates
        # with θ so the offset tracks the blade. Disabled (default) → unchanged.
        if self.sweep_geometric and self.marker_sweep_offset.size:
            e_theta = self._coord_system.tangent_vector(theta)          # (3,)
            pos = pos + (self._sweep_sign
                         * self.marker_sweep_offset)[:, None] * e_theta[None, :]
        return pos

    def get_marker_unit_vectors(
        self,
        theta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute local coordinate unit vectors at each marker

        Local Coordinate System:
            ê_n = coord_system.n_axis              (rotation axis / normal)
            ê_θ = coord_system.tangent_vector(θ)   (standard tangent, ∂x/∂θ direction)
            ê_r = coord_system.radial_vector(θ)    (radial / outward)

        These are needed for:
            - Velocity decomposition: u_n, u_θ (Eq. 5-6)
            - Force projection: F_n, F_θ → F^AL (Eq. 11-12)

        Args:
            theta: Azimuth angle  [radians]

        Returns:
            (e_n, e_theta, e_r): Each shape (3,)  [dimensionless]

            e_n:     Normal (axial) unit vector
            e_theta: Tangential (Watanabe convention) unit vector
            e_r:     Radial (outward) unit vector

        Raises:
            RuntimeError: If coord_system has not been set.
        """
        if self._coord_system is None:
            raise RuntimeError(
                "Blade.coord_system is not set. "
                "Use Rotor to create blades with a coordinate system."
            )
        e_n = self._coord_system.n_axis.copy()
        e_theta = self._coord_system.tangent_vector(theta)
        e_r = self._coord_system.radial_vector(theta)
        return e_n, e_theta, e_r

    # -----------------------------------------------------------------
    # §2.5 Force Projection to Global Frame
    # -----------------------------------------------------------------

    def project_forces_to_global(
        self,
        F_n: np.ndarray,
        F_theta: np.ndarray,
        theta: float,
        rotation_sign: float = 1.0,
        thrust_axis: 'Optional[np.ndarray]' = None,
    ) -> np.ndarray:
        """Project normal/tangential forces to global (x, y, z) frame

        Args:
            F_n:     Normal forces, shape (n_markers,)     [N or lattice force]
            F_theta: Tangential forces, shape (n_markers,) [N or lattice force]
            theta:   Azimuth angle  [radians]
            rotation_sign: sign(ω), +1.0 or -1.0  [-]
            thrust_axis: Unit vector for thrust projection  [dimensionless]
                         Default: rotation axis (n_axis)

        Returns:
            F_global: shape (n_markers, 3) — (F_x, F_y, F_z)
        """
        if self._coord_system is None:
            raise RuntimeError(
                "Blade.coord_system is not set. "
                "Use Rotor to create blades with a coordinate system."
            )
        return self._coord_system.project_force_to_global(
            F_n, F_theta, theta,
            rotation_sign=rotation_sign,
            thrust_axis=thrust_axis,
        )

    # -----------------------------------------------------------------
    # §2.6 Utility Methods
    # -----------------------------------------------------------------

    def get_info(self) -> str:
        """Human-readable blade summary"""
        coord_info = (f"{self._coord_system.axis_label}"
                      if self._coord_system is not None else "Not set")
        
        lines = [
            "Blade Geometry Summary",
            "=" * 50,
            f"  Radial extent: [{self.r_hub:.4f}, {self.r_tip:.4f}] m",
            f"  Span:          {self.span:.4f} m",
            f"  Sections:      {len(self.sections)}",
            f"  Markers:       {self.n_markers}",
            f"  Coord system:  {coord_info}",
        ]
        if self.n_markers > 0:
            lines += [
                f"  Marker spacing (Δr): {np.mean(self.marker_dr):.5f} m"
                + ("" if np.ndim(self.marker_dr) == 0 else " (mean)"),
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
        coord_str = "None"
        if self._coord_system is not None:
            coord_str = self._coord_system.axis_label
        return (
            f"Blade(r=[{self.r_hub:.3f}, {self.r_tip:.3f}] m, "
            f"span={self.span:.3f} m, "
            f"sections={len(self.sections)}, markers={self.n_markers}, "
            f"coord={coord_str})"
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
            r: Radial positions  [m]
            chord: Chord lengths [m]
            twist: Twist angles  [degrees]
            airfoil: Single name or list of names
            active: Bool array for active sections (default: all True)

        Returns:
            Blade instance
        """
        n = len(r)
        if isinstance(airfoil, str):
            airfoil = [airfoil] * n
        if active is None:
            active = np.ones(n, dtype=bool)

        sections = []
        for i in range(n):
            sections.append(BladeSection(
                r=float(r[i]),
                chord=float(chord[i]),
                twist=float(twist[i]),
                airfoil=airfoil[i],
                is_active=bool(active[i]),
            ))
        return cls(sections)

    @classmethod
    def from_ntnu_bt1(cls) -> 'Blade':
        """Create blade with NTNU Blind Test 1 geometry

        The NTNU BT1 turbine (Krogstad & Eriksen, 2013) uses NREL S826
        airfoil sections with the following geometry:

            Rotor diameter:  0.894 m
            Hub radius:      0.045 m (cylindrical section, inactive)
            Number of blades: 3
            Optimal TSR:     6.0

        The blade sections are from Table 1 of Krogstad & Eriksen (2013),
        with twist angles corrected per their Table 2 errata.

        Returns:
            Blade instance with NTNU BT1 geometry
        """
        sections = [
            # Hub attachment (circular, inactive)
            BladeSection(r=0.045, chord=0.030, twist=0.0,
                         airfoil='circular', is_active=False),
            # Transition region
            BladeSection(r=0.065, chord=0.060, twist=38.0,
                         airfoil='S826', is_active=True),
            # Main blade body (S826 airfoil)
            BladeSection(r=0.100, chord=0.069, twist=28.0,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.140, chord=0.065, twist=19.0,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.180, chord=0.058, twist=13.0,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.220, chord=0.052, twist=9.0,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.260, chord=0.046, twist=6.0,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.300, chord=0.041, twist=4.0,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.340, chord=0.037, twist=2.5,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.380, chord=0.033, twist=1.5,
                         airfoil='S826', is_active=True),
            BladeSection(r=0.420, chord=0.029, twist=0.5,
                         airfoil='S826', is_active=True),
            # Tip
            BladeSection(r=0.447, chord=0.026, twist=0.0,
                         airfoil='S826', is_active=True),
        ]
        return cls(sections)

    @classmethod
    def from_simple_blade(
        cls,
        r_hub: float,
        r_tip: float,
        chord: float,
        twist_root: float = 0.0,
        twist_tip: float = 0.0,
        airfoil: str = 'flat_plate'
    ) -> 'Blade':
        """Create a simple constant-chord blade for testing

        Useful for:
            - Initial debugging of the AL framework
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
                    {"r": 0.045, "chord": 0.030, "twist": 0.0,
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
                sweep=float(sec_dict.get('sweep', 0.0)),
            ))
        blade = cls(sections)
        # Sharp tip sweep (method 2): constant Λ beyond a break r/R, overriding
        # the section-interpolated ramp. Set before markers are generated.
        if config.get('sweep_break_rR') is not None:
            blade.sweep_break_rR = float(config['sweep_break_rR'])
            blade.sweep_tip_deg = float(config.get('sweep_tip_deg', 0.0))
        return blade

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

        Note: The coordinate system is NOT copied. The Rotor should set
        the converted coordinate system on the new blade separately.

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
                sweep=s.sweep,                  # [degrees] — unchanged
            ))
        
        new_blade = Blade(new_sections)

        # Carry the ε-taper controls onto the lattice-unit blade so that the
        # final set_lattice_spacing(dx=1.0) in Rotor.to_lattice_units honors
        # the requested mode (a fresh Blade() otherwise resets them to default).
        new_blade.epsilon_mode = self.epsilon_mode
        new_blade.epsilon_tip_factor = self.epsilon_tip_factor
        new_blade.epsilon_taper_start = self.epsilon_taper_start
        # Sharp-sweep params are r/R- and degree-based (unit-independent) — carry
        # them so the LU generate_markers below rebuilds the step Λ and exact Δs.
        new_blade.sweep_break_rR = self.sweep_break_rR
        new_blade.sweep_tip_deg = self.sweep_tip_deg

        # Copy marker generation state if markers were generated.
        # Preserve the radial distribution (uniform/cosine/endpoint) — otherwise
        # the lattice-unit blade silently reverts to uniform (bug: fine-level ALM
        # goes through here, so non-uniform distributions were being lost).
        if self.n_markers > 0:
            new_blade.generate_markers(n_radial=self.n_markers,
                                       distribution=self._marker_distribution,
                                       cosine_side=self._cosine_side)
            if self.marker_epsilon.any():
                # Recalculate epsilon in new units
                new_blade.marker_epsilon = self.marker_epsilon / length_scale
        
        # Note: coord_system is NOT copied - Rotor handles this
        
        return new_blade