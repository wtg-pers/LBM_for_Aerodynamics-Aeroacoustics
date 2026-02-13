"""
Actuator Line Model — Main Controller

This module orchestrates the complete Actuator Line (AL) timestep within
the LBM framework, connecting the rotor kinematics, velocity interpolation,
BEM force calculation, and Gaussian force spreading.

Physical Process per AL Timestep:
=================================

    ┌─────────────────────────────────────────────────────────────────┐
    │  LBM Time Loop: each timestep t                                │
    │                                                                 │
    │  1. ADVANCE ROTOR         θ_k(t+Δt) = θ_k(t) + ω·Δt          │
    │         ↓                                                       │
    │  2. GET MARKER POSITIONS  x_j = f(θ_k, r_j, hub)    [lu]      │
    │         ↓                                                       │
    │  3. INTERPOLATE VELOCITY  u(x_j) = Σ w·u(x) / Σ w  [Δx/Δt]  │
    │         ↓                                                       │
    │  4. DECOMPOSE VELOCITY    u_n, u_θ from global u     [Δx/Δt]  │
    │         ↓                                                       │
    │  5. BEM TRIANGLE          u_rel, φ, α at each marker           │
    │         ↓                                                       │
    │  6. AIRFOIL LOOKUP        CL(α, Re), CD(α, Re)                │
    │         ↓                                                       │
    │  7. COMPUTE FORCES        F_L, F_D → F_n, F_θ  (Eq. 9-12)    │
    │         ↓                                                       │
    │  8. PROJECT TO GLOBAL     F^AL = (F_n, F_θ·cosθ, -F_θ·sinθ)  │
    │         ↓                                                       │
    │  9. GAUSSIAN SPREADING    F(x) = Σ -F^AL·η_ε(d) (Eq. 13)     │
    │         ↓                                                       │
    │  10. → F(x) enters Guo forcing in LBM collision               │
    └─────────────────────────────────────────────────────────────────┘

Unit System:
    ALL computations inside ActuatorLineModel are in LATTICE UNITS.
    The Rotor (with blades) must be converted to lattice units before
    being passed to this controller.

    Lattice unit conventions:
        Δx = 1 [lu], Δt = 1 [lt]
        ρ₀ = 1 [lattice density]
        ν = cs²·(τ - 1/2)  [lu²/lt]
        Force: F [lattice force] = ρ·a·Δx³ with a in [lu/lt²]

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026 (Sec. 2.2)
    - Sørensen & Shen, J. Fluids Eng. 124, 393-399, 2002

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Dict, Tuple, Callable, List
from dataclasses import dataclass, field

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from .rotor import Rotor
from .interpolation import (
    interpolate_velocity_batch,
    interpolate_velocity_batch_fast,
)
from .spreading import (
    spread_forces_to_grid,
    spread_forces_uniform_epsilon,
    check_force_conservation,
)


# =============================================================================
# §1. BEM Force Calculator
# =============================================================================

@dataclass
class BEMResult:
    """Result of BEM force calculation at all markers

    All arrays have shape (N_total_markers,) where N_total = N_blades × N_markers.

    Physical quantities at each marker:
        u_rel:  Relative wind speed          [Δx/Δt]
        phi:    Local flow angle             [degrees]
        alpha:  Angle of attack              [degrees]
        Re:     Chord Reynolds number        [dimensionless]
        CL:     Lift coefficient             [dimensionless]
        CD:     Drag coefficient             [dimensionless]
        F_L:    Lift force per marker        [lattice force]
        F_D:    Drag force per marker        [lattice force]
        F_n:    Normal (streamwise) force    [lattice force]
        F_theta: Tangential (rotational)     [lattice force]
    """
    u_rel: np.ndarray       # [Δx/Δt]
    phi: np.ndarray         # [degrees]
    alpha: np.ndarray       # [degrees]
    Re: np.ndarray          # [dimensionless]
    CL: np.ndarray          # [dimensionless]
    CD: np.ndarray          # [dimensionless]
    F_L: np.ndarray         # [lattice force]
    F_D: np.ndarray         # [lattice force]
    F_n: np.ndarray         # [lattice force]
    F_theta: np.ndarray     # [lattice force]


# =============================================================================
# §2. Actuator Line Model Controller
# =============================================================================

class ActuatorLineModel:
    """Main controller for the Actuator Line method in LBM

    Coordinates the complete AL pipeline: rotor kinematics → velocity
    interpolation → BEM forces → Gaussian spreading → body force field.

    All internal computations use LATTICE UNITS.

    Attributes:
        rotor: Rotor instance (in lattice units)
        nu: Kinematic viscosity [lu²/lt]
        rho_ref: Reference density [dimensionless, typically 1.0]
        polar_query: Callable(alpha_deg, Re) → (CL, CD)
        domain_shape: (Nx, Ny, Nz) grid dimensions
        n_cut: Gaussian cutoff in units of ε
        dt_phys: Physical timestep [s] (for output conversion)
        dx_phys: Physical grid spacing [m] (for output conversion)

    Example:
        >>> from src.actuator import ActuatorLineModel, Rotor
        >>> from src.actuator.airfoil_data import create_nrel_s826_database
        >>>
        >>> # Setup
        >>> rotor_phys = Rotor.from_ntnu_bt1(tsr=6, u_inf=10)
        >>> rotor_lu = rotor_phys.to_lattice_units(dx_phys, dt_phys)
        >>> db = create_nrel_s826_database()
        >>> query = db.to_grid_format_query()
        >>>
        >>> al = ActuatorLineModel(
        ...     rotor=rotor_lu,
        ...     nu=nu_lattice,
        ...     domain_shape=(Nx, Ny, Nz),
        ...     polar_query=query
        ... )
        >>>
        >>> # In time loop:
        >>> F_body = al.step(u_field, dt=1.0)
    """

    def __init__(
        self,
        rotor: 'Rotor',
        nu: float,
        domain_shape: Tuple[int, int, int],
        polar_query: Callable[..., Tuple[float, float]],  # 변경: ... 으로 유연하게
        rho_ref: float = 1.0,
        n_cut: float = 3.0,
        dx_phys: float = 1.0,
        dt_phys: float = 1.0,
    ) -> None:
        '''Initialize the Actuator Line model

        Args:
            rotor: Rotor object in LATTICE UNITS
            nu: Kinematic viscosity  [lu²/lt]
            domain_shape: (Nx, Ny, Nz)  [lu]
            polar_query: Airfoil polar query function
                         - Single airfoil: query(alpha_deg, Re) → (CL, CD)
                         - Multi airfoil:  query(alpha_deg, Re, airfoil_name) → (CL, CD)
            rho_ref: Reference density [dimensionless]
            n_cut: Gaussian cutoff in units of ε
            dx_phys: Physical grid spacing [m/lu]
            dt_phys: Physical timestep [s/lt]
        '''
        self.rotor = rotor
        self.nu = nu                    # [lu²/lt]
        self.rho_ref = rho_ref          # [dimensionless]
        self.domain_shape = domain_shape
        self.polar_query = polar_query
        self.n_cut = n_cut
        self.dx_phys = dx_phys          # [m/lu]
        self.dt_phys = dt_phys          # [s/lt]

        # Pre-allocate body force array
        Nx, Ny, Nz = domain_shape
        self._F_grid = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)

        # Diagnostics storage (updated each step)
        self._last_bem_result: Optional[BEMResult] = None
        self._last_positions: Optional[np.ndarray] = None
        self._last_forces_global: Optional[np.ndarray] = None
        self._step_count: int = 0

        # Detect multi-airfoil support
        # Test if polar_query accepts 3 arguments (alpha, Re, airfoil_name)
        self._multi_airfoil = False
        try:
            import inspect
            sig = inspect.signature(polar_query)
            n_params = len([p for p in sig.parameters.values() 
                           if p.default == inspect.Parameter.empty])
            # If 3 required params or has 'airfoil' param, it's multi-airfoil
            self._multi_airfoil = n_params >= 3 or 'airfoil_name' in sig.parameters
        except (ValueError, TypeError):
            # If signature inspection fails, assume single airfoil
            self._multi_airfoil = False

    # -----------------------------------------------------------------
    # §2.1 Main Time Step
    # -----------------------------------------------------------------

    def step(
        self,
        u_field: 'npt.NDArray',
        dt: float = 1.0,
        external_F: Optional['npt.NDArray'] = None
    ) -> 'npt.NDArray':
        """Execute one complete AL timestep

        This is the PRIMARY interface called from the LBM time loop.

        Physical Steps (see module docstring for data flow):
            1. Advance rotor azimuth
            2. Sample velocity at marker positions
            3. Compute BEM forces
            4. Spread forces to grid

        Args:
            u_field: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
            dt: Timestep (default 1.0 in lattice units)  [lt]
            external_F: Additional body force to combine with AL force
                        shape (3, Nx, Ny, Nz)  [lattice force / lu³]

        Returns:
            F_grid: Total body force field, shape (3, Nx, Ny, Nz)
                    [lattice force / lu³]
        """
        # --- Step 1: Advance rotor ---
        self.rotor.advance(dt)

        # --- Step 2: Get marker positions ---
        positions = self.rotor.get_all_marker_positions()   # (N_total, 3) [lu]
        self._last_positions = positions

        # --- Step 3: Interpolate velocity at markers ---
        epsilon_all = self.rotor.get_all_marker_epsilon()    # (N_total,) [lu]
        active_all = self.rotor.get_all_marker_active()      # (N_total,) bool

        u_markers = interpolate_velocity_batch(
            u_field, positions, epsilon_all, n_cut=self.n_cut
        )  # (N_total, 3) [Δx/Δt]

        # --- Step 4-7: BEM force calculation ---
        bem_result = self._compute_bem_forces(u_markers)
        self._last_bem_result = bem_result

        # --- Step 8: Project to global frame ---
        F_global = self.rotor.project_all_forces(
            bem_result.F_n, bem_result.F_theta
        )  # (N_total, 3) [lattice force]
        self._last_forces_global = F_global

        # --- Step 9: Gaussian spreading ---
        self._F_grid[:] = 0.0  # Reset
        if external_F is not None:
            self._F_grid[:] = external_F  # Start from external force

        spread_forces_to_grid(
            self.domain_shape,
            positions,
            F_global,
            epsilon_all,
            marker_active=active_all,
            n_cut=self.n_cut,
            F_grid=self._F_grid
        )

        self._step_count += 1
        return self._F_grid

    # -----------------------------------------------------------------
    # §2.2 BEM Force Computation
    # -----------------------------------------------------------------

    def _compute_bem_forces_modified(self, u_markers: 'npt.NDArray') -> 'BEMResult':
        """Compute BEM aerodynamic forces at all markers (MODIFIED for multi-airfoil)

        Physical Process (Watanabe et al. Eq. 5-12):
            1. Decompose velocity: u_n (normal), u_θ (tangential)  [Eq. 5]
            2. Relative velocity:  u_rel = √(u_n² + (ωr - u_θ)²)   [Eq. 6]
            3. Flow angle:         φ = atan2(u_n, ωr - u_θ)        [Eq. 7]
            4. Angle of attack:    α = φ - γ                        [Eq. 8]
            5. Reynolds number:    Re = u_rel · c_a / ν
            6. Airfoil lookup:     CL(Re, α, [airfoil]), CD(Re, α, [airfoil])
            7. Lift/drag forces:   F_L = ½·ρ·u_rel²·c·Δr·CL        [Eq. 9]
                                F_D = ½·ρ·u_rel²·c·Δr·CD        [Eq. 10]
            8. Project to blade:   F_n, F_θ                        [Eq. 11-12]

        Args:
            u_markers: Interpolated velocity at markers, shape (N_total, 3) [Δx/Δt]

        Returns:
            BEMResult with all computed quantities
        """
        # --- Setup ---
        n_total = u_markers.shape[0]  # [dimensionless]
        chord_all = self.rotor.get_all_marker_chords()     # [lu]
        active_all = self.rotor.get_all_marker_active()    # [bool]
        dr = self.rotor.get_all_marker_dr()                # [lu]

        # Get airfoil names for multi-airfoil mode
        if self._multi_airfoil:
            airfoil_all = self.rotor.get_all_marker_airfoils()  # [str list]

        # Arrays to fill
        u_rel = np.zeros(n_total)       # [Δx/Δt]
        phi = np.zeros(n_total)         # [degrees]
        alpha = np.zeros(n_total)       # [degrees]
        Re = np.zeros(n_total)          # [dimensionless]
        CL = np.zeros(n_total)          # [dimensionless]
        CD = np.zeros(n_total)          # [dimensionless]

        # --- Per-blade processing ---
        n_per = self.rotor.markers_per_blade  # [dimensionless]
        
        for b_idx, blade in enumerate(self.rotor.blades):
            start = b_idx * n_per
            end = start + n_per
            theta_b = self.rotor.theta[b_idx]  # [rad]

            # Velocity at this blade's markers
            u_blade = u_markers[start:end]  # (n_per, 3) [Δx/Δt]

            # Compute relative velocity and angles
            u_rel_b, phi_b, alpha_b = self.rotor.compute_relative_velocity(
                blade, u_blade, theta_b
            )  # [Δx/Δt], [degrees], [degrees]

            u_rel[start:end] = u_rel_b
            phi[start:end] = phi_b
            alpha[start:end] = alpha_b

            # Reynolds number: Re = u_rel · chord / ν
            Re[start:end] = u_rel_b * chord_all[start:end] / self.nu  # [dimensionless]

        # ═══════════════════════════════════════════════════════════════════
        # Airfoil lookup (MODIFIED for multi-airfoil support)
        # ═══════════════════════════════════════════════════════════════════
        for i in range(n_total):
            if active_all[i]:
                if self._multi_airfoil:
                    # Multi-airfoil mode: pass airfoil name
                    CL[i], CD[i] = self.polar_query(alpha[i], Re[i], airfoil_all[i])
                else:
                    # Single airfoil mode: original 2-argument call
                    CL[i], CD[i] = self.polar_query(alpha[i], Re[i])

        # ═══════════════════════════════════════════════════════════════════
        # Force computation (unchanged from original)
        # ═══════════════════════════════════════════════════════════════════
        # Lift and drag (Watanabe Eq. 9-10)
        # F = 0.5 · ρ · u_rel² · c · Δr · C
        q_dyn = 0.5 * self.rho_ref * u_rel**2   # [dimensionless] dynamic pressure
        F_L = q_dyn * chord_all * dr * CL       # [lattice force]
        F_D = q_dyn * chord_all * dr * CD       # [lattice force]

        # Project to normal/tangential (Watanabe Eq. 11-12)
        phi_rad = np.radians(phi)                # [radians]
        F_n = F_L * np.cos(phi_rad) + F_D * np.sin(phi_rad)      # [lattice force]
        F_theta = F_L * np.sin(phi_rad) - F_D * np.cos(phi_rad)  # [lattice force]

        # Zero out inactive markers
        F_L[~active_all] = 0.0
        F_D[~active_all] = 0.0
        F_n[~active_all] = 0.0
        F_theta[~active_all] = 0.0

        return BEMResult(
            u_rel=u_rel,
            phi=phi,
            alpha=alpha,
            Re=Re,
            CL=CL,
            CD=CD,
            F_L=F_L,
            F_D=F_D,
            F_n=F_n,
            F_theta=F_theta,
        )

    # -----------------------------------------------------------------
    # §2.3 Performance Outputs
    # -----------------------------------------------------------------

    def get_rotor_performance(self) -> dict:
        """Get latest rotor performance coefficients

        Returns:
            dict with:
                - thrust [lattice force]
                - torque [lattice force · lu]
                - power  [lattice power]
                - C_T    [dimensionless]
                - C_P    [dimensionless]
                - theta  [degrees] per blade
                - time   [lt]
                - revolutions [dimensionless]
        """
        if self._last_bem_result is None:
            return {'error': 'No step executed yet'}

        bem = self._last_bem_result
        rotor = self.rotor

        thrust, torque = rotor.compute_torque_thrust(
            bem.F_n, bem.F_theta
        )
        power = rotor.compute_power(bem.F_theta)

        # For C_T, C_P we need u_inf in lattice units
        # Estimate from mean normal velocity at markers
        active = rotor.get_all_marker_active()
        u_n_active = bem.u_rel[active]
        u_inf_est = np.mean(u_n_active) if len(u_n_active) > 0 else 1.0

        C_T, C_P = rotor.compute_ct_cp(
            bem.F_n, bem.F_theta,
            rho=self.rho_ref,
            u_inf=u_inf_est
        )

        return {
            'thrust': thrust,
            'torque': torque,
            'power': power,
            'C_T': C_T,
            'C_P': C_P,
            'theta_deg': np.degrees(rotor.theta).tolist(),
            'time': rotor.time,
            'revolutions': rotor.n_revolutions,
            'step': self._step_count,
        }

    def get_blade_diagnostics(self, blade_idx: int = 0) -> dict:
        """Get detailed diagnostics for a single blade

        Useful for debugging and validation — provides the full
        BEM triangle at each marker along the blade span.

        Args:
            blade_idx: Which blade (0 to N_blades-1)

        Returns:
            dict with arrays indexed by marker position along the span
        """
        if self._last_bem_result is None:
            return {'error': 'No step executed yet'}

        bem = self._last_bem_result
        n_m = self.rotor.markers_per_blade
        s = blade_idx * n_m
        e = s + n_m
        blade = self.rotor.blades[blade_idx]

        return {
            'r': blade.marker_r.copy(),                  # [lu]
            'chord': blade.marker_chord.copy(),          # [lu]
            'twist': blade.marker_twist.copy(),          # [degrees]
            'active': blade.marker_active.copy(),
            'u_rel': bem.u_rel[s:e].copy(),              # [Δx/Δt]
            'phi': bem.phi[s:e].copy(),                  # [degrees]
            'alpha': bem.alpha[s:e].copy(),              # [degrees]
            'Re': bem.Re[s:e].copy(),                    # [dimensionless]
            'CL': bem.CL[s:e].copy(),
            'CD': bem.CD[s:e].copy(),
            'F_L': bem.F_L[s:e].copy(),                  # [lattice force]
            'F_D': bem.F_D[s:e].copy(),
            'F_n': bem.F_n[s:e].copy(),
            'F_theta': bem.F_theta[s:e].copy(),
        }

    def check_conservation(self) -> dict:
        """Check force conservation between markers and grid"""
        if self._last_forces_global is None:
            return {'error': 'No step executed yet'}

        return check_force_conservation(
            self._F_grid,
            self._last_forces_global,
            marker_active=self.rotor.get_all_marker_active()
        )

    # -----------------------------------------------------------------
    # §2.4 Output Conversion (Lattice → Physical)
    # -----------------------------------------------------------------

    def convert_to_physical(self, performance: dict) -> dict:
        """Convert lattice-unit performance to physical units

        Conversion factors:
            Force:  F_phys = F_lu · ρ₀ · (Δx/Δt)² · Δx²    [N]
            Torque: Q_phys = Q_lu · ρ₀ · (Δx/Δt)² · Δx³    [N·m]
            Power:  P_phys = P_lu · ρ₀ · (Δx/Δt)³ · Δx²    [W]

        Args:
            performance: dict from get_rotor_performance()

        Returns:
            dict with physical-unit values appended (suffix '_phys')
        """
        dx = self.dx_phys                               # [m/lu]
        dt = self.dt_phys                               # [s/lt]
        rho0 = self.rho_ref                             # [dimensionless]

        # Physical density [kg/m³] (user must set this externally)
        rho_phys = 1.205                                # [kg/m³] (air at 20°C)

        u_ref = dx / dt                                 # [m/s per lu/lt]
        force_scale = rho_phys * u_ref ** 2 * dx ** 2   # [N per lattice force]
        torque_scale = force_scale * dx                  # [N·m]
        power_scale = rho_phys * u_ref ** 3 * dx ** 2   # [W]

        result = dict(performance)
        if 'thrust' in performance:
            result['thrust_phys'] = performance['thrust'] * force_scale
            result['torque_phys'] = performance['torque'] * torque_scale
            result['power_phys'] = performance['power'] * power_scale
        return result

    # -----------------------------------------------------------------
    # §2.5 Info & Repr
    # -----------------------------------------------------------------

    def get_info(self) -> str:
        """Human-readable summary of AL model configuration"""
        lines = [
            "Actuator Line Model Summary",
            "=" * 60,
            f"  Domain:      {self.domain_shape}",
            f"  ν (lattice): {self.nu:.6e}",
            f"  ρ_ref:       {self.rho_ref}",
            f"  n_cut:       {self.n_cut}",
            f"  Steps done:  {self._step_count}",
            "",
            self.rotor.get_info(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ActuatorLineModel(domain={self.domain_shape}, "
            f"blades={self.rotor.n_blades}, "
            f"markers={self.rotor.total_markers}, "
            f"steps={self._step_count})"
        )


# =============================================================================
# §3. Factory Function
# =============================================================================

def create_actuator_line_from_config(
    config: dict,
    domain_shape: Tuple[int, int, int],
    nu_lattice: float,
    polar_query: Callable,
    dx_phys: float = 1.0,
    dt_phys: float = 1.0,
) -> ActuatorLineModel:
    """Create an ActuatorLineModel from a configuration dictionary

    Expected Config Format:
        {
            "rotor": {
                "preset": "ntnu_bt1",
                "tsr": 6.0,
                "u_inf": 10.0,
                "hub_center": [3.66, 1.341, 0.817],
                "resolution": 160
            },
            "gaussian_cutoff": 3.0,
            "rho_ref": 1.0
        }

    Args:
        config: AL model configuration
        domain_shape: (Nx, Ny, Nz)
        nu_lattice: Kinematic viscosity [lu²/lt]
        polar_query: Airfoil data query function
        dx_phys: Physical grid spacing [m/lu]
        dt_phys: Physical timestep [s/lt]

    Returns:
        Configured ActuatorLineModel instance
    """
    rotor_cfg = config.get('rotor', {})

    # Create rotor (in physical units first)
    rotor_phys = Rotor.from_config(rotor_cfg)

    # Convert to lattice units
    rotor_lu = rotor_phys.to_lattice_units(
        length_scale=dx_phys,
        time_scale=dt_phys
    )

    return ActuatorLineModel(
        rotor=rotor_lu,
        nu=nu_lattice,
        domain_shape=domain_shape,
        polar_query=polar_query,
        rho_ref=config.get('rho_ref', 1.0),
        n_cut=config.get('gaussian_cutoff', 3.0),
        dx_phys=dx_phys,
        dt_phys=dt_phys,
    )