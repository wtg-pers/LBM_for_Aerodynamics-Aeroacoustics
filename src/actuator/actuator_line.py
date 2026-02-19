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
    │  1. ADVANCE ROTOR         θ_k(t+Δt) = θ_k(t) + ω·Δt           │
    │         ↓                                                       │
    │  2. GET MARKER POSITIONS  x_j = f(θ_k, r_j, hub)    [lu]       │
    │         ↓                                                       │
    │  3. INTERPOLATE VELOCITY  u(x_j) = Σ w·u(x) / Σ w   [Δx/Δt]   │
    │         ↓                                                       │
    │  4. DECOMPOSE VELOCITY    u_n, u_θ from global u     [Δx/Δt]   │
    │         ↓                                                       │
    │  5. BEM TRIANGLE          u_rel, φ, α at each marker           │
    │         ↓                                                       │
    │  6. AIRFOIL LOOKUP        CL(α, Re), CD(α, Re)                 │
    │         ↓                                                       │
    │  7. COMPUTE FORCES        F_L, F_D → F_n, F_θ  (Eq. 9-12)     │
    │         ↓                                                       │
    │  8. PROJECT TO GLOBAL     F^AL = (F_n, F_θ·cosθ, -F_θ·sinθ)   │
    │         ↓                                                       │
    │  9. GAUSSIAN SPREADING    F(x) = Σ -F^AL·η_ε(d) (Eq. 13)      │
    │         ↓                                                       │
    │  10. → F(x) enters Guo forcing in LBM collision                │
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
import inspect

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
        u_n:    Axial (normal) velocity         [Δx/Δt]   (Phase 0: C_T/C_P 보정용)
        u_theta: Tangential velocity            [Δx/Δt]   (Phase 0: 진단용)
        u_rel:  Relative wind speed             [Δx/Δt]
        phi:    Local flow angle                [degrees]
        alpha:  Angle of attack                 [degrees]
        Re:     Chord Reynolds number           [dimensionless]
        CL:     Lift coefficient                [dimensionless]
        CD:     Drag coefficient                [dimensionless]
        F_L:    Lift force per marker           [lattice force]
        F_D:    Drag force per marker           [lattice force]
        F_n:    Normal (streamwise) force       [lattice force]
        F_theta: Tangential (rotational)        [lattice force]
    """
    u_n: np.ndarray             # [Δx/Δt]  — axial velocity (Watanabe Eq. 5)
    u_theta: np.ndarray         # [Δx/Δt]  — tangential velocity (Watanabe Eq. 5)
    u_rel: np.ndarray           # [Δx/Δt]
    phi: np.ndarray             # [degrees]
    alpha: np.ndarray           # [degrees]
    Re: np.ndarray              # [dimensionless]
    CL: np.ndarray              # [dimensionless]
    CD: np.ndarray              # [dimensionless]
    F_L: np.ndarray             # [lattice force]
    F_D: np.ndarray             # [lattice force]
    F_n: np.ndarray             # [lattice force]
    F_theta: np.ndarray         # [lattice force]


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
                     Or Callable(alpha_deg, Re, airfoil_name) for multi-airfoil
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
        >>> query = db.to_query()
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
        polar_query: Callable[..., Tuple[float, float]],
        rho_ref: float = 1.0,
        n_cut: float = 3.0,
        dx_phys: float = 1.0,
        dt_phys: float = 1.0,
        u_inf_lu: Optional[float] = None,
        coeff_mode: str = 'auto',
    ) -> None:
        """Initialize the Actuator Line model

        Args:
            rotor: Rotor object in LATTICE UNITS
            nu: Kinematic viscosity  [lu²/lt]
            domain_shape: (Nx, Ny, Nz)  [lu]
            polar_query: Callable(alpha_deg, Re) → (CL, CD)
                         Or Callable(alpha_deg, Re, airfoil_name) for multi-airfoil
            rho_ref: Reference density (default 1.0)  [dimensionless]
            n_cut: Gaussian cutoff  [dimensionless]
            dx_phys: Physical grid spacing  [m/lu] (for diagnostics)
            dt_phys: Physical timestep  [s/lt] (for diagnostics)
            u_inf_lu: Freestream velocity in lattice units  [Δx/Δt] or None
                      If provided, used directly for C_T/C_P calculation.
                      If None, fallback to mean(|u_n|) from BEM
                      (note: u_n ≈ u_∞(1-a), underestimates u_∞ by ~30% at Betz limit)
            coeff_mode: Performance coefficient convention
                        'wind_turbine' | 'rotorcraft' | 'auto'
        """
        self.rotor = rotor
        self.nu = nu                    # [lu²/lt]
        self.rho_ref = rho_ref          # [dimensionless]
        self.domain_shape = domain_shape
        self.polar_query = polar_query
        self.n_cut = n_cut
        self.dx_phys = dx_phys          # [m/lu]
        self.dt_phys = dt_phys          # [s/lt]
        self.u_inf_lu = u_inf_lu        # [Δx/Δt] or None
        self.coeff_mode = coeff_mode    # 'wind_turbine' | 'rotorcraft' | 'auto'

        # Pre-allocate body force array
        Nx, Ny, Nz = domain_shape
        self._F_grid = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)

        # Diagnostics storage (updated each step)
        self._last_bem_result: Optional[BEMResult] = None
        self._last_positions: Optional[np.ndarray] = None
        self._last_forces_global: Optional[np.ndarray] = None
        self._step_count: int = 0

        # ═══════════════════════════════════════════════════════════════════
        # NEW: Detect multi-airfoil support
        # ═══════════════════════════════════════════════════════════════════
        self._multi_airfoil = False
        try:
            sig = inspect.signature(polar_query)
            param_names = list(sig.parameters.keys())
            # Check if has 'airfoil_name' parameter or has 3+ parameters
            self._multi_airfoil = ('airfoil_name' in param_names or 
                                   len(param_names) >= 3)
        except (ValueError, TypeError):
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

    def _compute_bem_forces(
        self,
        u_markers: 'npt.NDArray'
    ) -> BEMResult:
        """Compute BEM aerodynamic forces at all markers

        Physical Process (Watanabe et al. Eq. 5-12):
            1. Decompose velocity: u_n (normal), u_θ (tangential)  [Eq. 5]
            2. Relative velocity:  u_rel = √(u_n² + (ωr - u_θ)²)   [Eq. 6]
            3. Flow angle:         φ = atan2(u_n, ωr - u_θ)        [Eq. 7]
            4. Angle of attack:    α = φ - γ                        [Eq. 8]
            5. Reynolds number:    Re = u_rel · c_a / ν
            6. Airfoil lookup:     CL(Re, α), CD(Re, α)
            7. Lift/drag forces:   F_L = ½·ρ·u_rel²·c·Δr·CL        [Eq. 9]
                                   F_D = ½·ρ·u_rel²·c·Δr·CD        [Eq. 10]
            8. Normal/tangential:  F_n = FL·cosφ + FD·sinφ          [Eq. 11]
                                   F_θ = FL·sinφ - FD·cosφ          [Eq. 12]

        Args:
            u_markers: Velocity at markers, shape (N_total, 3)  [Δx/Δt]

        Returns:
            BEMResult with all intermediate and final quantities
        """
        rotor = self.rotor
        n_total = rotor.total_markers
        n_per_blade = rotor.markers_per_blade

        # Pre-allocate output arrays
        u_n_all = np.zeros(n_total, dtype=np.float64)         # [Δx/Δt]  axial
        u_theta_all = np.zeros(n_total, dtype=np.float64)     # [Δx/Δt]  tangential
        u_rel_all = np.zeros(n_total, dtype=np.float64)       # [Δx/Δt]
        phi_all = np.zeros(n_total, dtype=np.float64)         # [degrees]
        alpha_all = np.zeros(n_total, dtype=np.float64)       # [degrees]
        Re_all = np.zeros(n_total, dtype=np.float64)        # [dimensionless]
        CL_all = np.zeros(n_total, dtype=np.float64)        # [dimensionless]
        CD_all = np.zeros(n_total, dtype=np.float64)        # [dimensionless]
        F_L_all = np.zeros(n_total, dtype=np.float64)       # [lattice force]
        F_D_all = np.zeros(n_total, dtype=np.float64)       # [lattice force]
        F_n_all = np.zeros(n_total, dtype=np.float64)       # [lattice force]
        F_theta_all = np.zeros(n_total, dtype=np.float64)   # [lattice force]

        # Process each blade
        for k in range(rotor.n_blades):
            idx_start = k * n_per_blade
            idx_end = idx_start + n_per_blade

            # Blade-specific velocity extraction
            u_blade = u_markers[idx_start:idx_end]          # (n_markers, 3)

            # --- Velocity decomposition (Eq. 5-8) ---
            u_rel, phi_deg, alpha_deg, u_n, u_theta = rotor.compute_relative_velocity(
                blade_idx=k, u_global=u_blade
            )
            # u_rel: [Δx/Δt], phi_deg: [degrees], alpha_deg: [degrees]
            # u_n: [Δx/Δt] axial, u_theta: [Δx/Δt] tangential

            u_n_all[idx_start:idx_end] = u_n
            u_theta_all[idx_start:idx_end] = u_theta
            u_rel_all[idx_start:idx_end] = u_rel
            phi_all[idx_start:idx_end] = phi_deg
            alpha_all[idx_start:idx_end] = alpha_deg

            # --- Blade geometry ---
            blade = rotor.blades[k]
            chord = blade.marker_chord       # [lu]
            dr = blade.marker_dr             # [lu]
            active = blade.marker_active     # bool

            # --- Reynolds number ---
            # Re = u_rel · c_a / ν   [dimensionless]
            Re = u_rel * chord / (self.nu + 1e-30)
            Re_all[idx_start:idx_end] = Re

            # --- Airfoil polar lookup ---
            CL = np.zeros(n_per_blade, dtype=np.float64)
            CD = np.zeros(n_per_blade, dtype=np.float64)

            for j in range(n_per_blade):
                if not active[j]:
                    continue
                if u_rel[j] < 1e-10:        # No flow → no force
                    continue

                # ═══════════════════════════════════════════════════════════
                # MODIFIED: Support multi-airfoil polar query
                # ═══════════════════════════════════════════════════════════
                if self._multi_airfoil:
                    # Multi-airfoil mode: pass airfoil name
                    airfoil_name = blade.marker_airfoil[j]
                    cl_j, cd_j = self.polar_query(
                        float(alpha_deg[j]),      # [degrees]
                        float(Re[j]),             # [dimensionless]
                        airfoil_name              # [string]
                    )
                else:
                    # Single airfoil mode: original 2-argument call
                    cl_j, cd_j = self.polar_query(
                        float(alpha_deg[j]),      # [degrees]
                        float(Re[j])              # [dimensionless]
                    )
                CL[j] = cl_j
                CD[j] = cd_j

            CL_all[idx_start:idx_end] = CL
            CD_all[idx_start:idx_end] = CD

            # --- Lift and drag forces (Eq. 9-10) ---
            # F_L = 0.5 · ρ · u_rel² · c_a · Δr · CL   [lattice force]
            # F_D = 0.5 · ρ · u_rel² · c_a · Δr · CD   [lattice force]
            q = 0.5 * self.rho_ref * u_rel ** 2          # [lattice pressure]
            F_L = q * chord * dr * CL                     # [lattice force]
            F_D = q * chord * dr * CD                     # [lattice force]

            # Inactive markers produce zero force
            F_L[~active] = 0.0
            F_D[~active] = 0.0

            F_L_all[idx_start:idx_end] = F_L
            F_D_all[idx_start:idx_end] = F_D

            # --- Project to normal/tangential (Eq. 11-12) ---
            # F_n = F_L·cos(φ) + F_D·sin(φ)     [lattice force]
            # F_θ = F_L·sin(φ) - F_D·cos(φ)     [lattice force]
            phi_rad = np.radians(phi_deg)                # [radians]
            cos_phi = np.cos(phi_rad)                    # [dimensionless]
            sin_phi = np.sin(phi_rad)                    # [dimensionless]

            F_n = F_L * cos_phi + F_D * sin_phi          # [lattice force]
            F_theta = F_L * sin_phi - F_D * cos_phi      # [lattice force]

            F_n_all[idx_start:idx_end] = F_n
            F_theta_all[idx_start:idx_end] = F_theta

        return BEMResult(
            u_n=u_n_all,
            u_theta=u_theta_all,
            u_rel=u_rel_all,
            phi=phi_all,
            alpha=alpha_all,
            Re=Re_all,
            CL=CL_all,
            CD=CD_all,
            F_L=F_L_all,
            F_D=F_D_all,
            F_n=F_n_all,
            F_theta=F_theta_all,
        )

    # -----------------------------------------------------------------
    # §2.3 Performance Outputs
    # -----------------------------------------------------------------

    def get_rotor_performance(self) -> dict:
        """Get latest rotor performance coefficients

        u_inf estimation strategy (priority order):
            1. self.u_inf_lu from config (most accurate)
            2. mean(|u_n|) from BEM axial velocity (fallback)
               Note: u_n ≈ u_∞(1-a), underestimates u_∞ by ~30% at Betz limit

        Returns:
            dict with:
                - thrust      [lattice force]
                - torque      [lattice force · lu]
                - power       [lattice power]
                - C_T         [dimensionless]
                - C_P         [dimensionless]
                - coeff_mode  'wind_turbine' or 'rotorcraft' (actually used)
                - u_inf_used  [Δx/Δt] u_inf value used for coefficients
                - FM          [dimensionless] Figure of Merit (hover)
                - theta_deg   [degrees] per blade
                - time        [lt]
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

        # --- u_inf estimation (BUG FIX: was using u_rel, now u_n) ---
        # Priority: config value > BEM axial velocity fallback
        if self.u_inf_lu is not None:
            u_inf_used = self.u_inf_lu                              # [Δx/Δt] config
        else:
            active = rotor.get_all_marker_active()
            u_n_active = bem.u_n[active]                            # [Δx/Δt] axial only
            u_inf_used = float(np.mean(np.abs(u_n_active))) \
                if len(u_n_active) > 0 else 0.0                    # [Δx/Δt]

        # --- C_T, C_P (dual mode) ---
        C_T, C_P, actual_mode = rotor.compute_ct_cp(
            bem.F_n, bem.F_theta,
            rho=self.rho_ref,
            u_inf=u_inf_used,
            mode=self.coeff_mode
        )

        # --- Figure of Merit (always rotorcraft convention) ---
        FM = rotor.compute_figure_of_merit(
            bem.F_n, bem.F_theta, rho=self.rho_ref
        )

        return {
            'thrust': thrust,
            'torque': torque,
            'power': power,
            'C_T': C_T,
            'C_P': C_P,
            'coeff_mode': actual_mode,
            'u_inf_used': u_inf_used,
            'FM': FM,
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
            blade_idx: Which blade to query (default: 0)

        Returns:
            dict with arrays (r/R, alpha, phi, CL, CD, F_n, F_theta, ...)
        """
        if self._last_bem_result is None:
            return {'error': 'No step executed yet'}

        bem = self._last_bem_result
        rotor = self.rotor
        blade = rotor.blades[blade_idx]
        n_per = rotor.markers_per_blade

        idx_s = blade_idx * n_per
        idx_e = idx_s + n_per

        r = blade.marker_r                      # [lu]
        r_norm = r / rotor.radius               # [dimensionless]

        return {
            'blade_idx': blade_idx,
            'r': r,                             # [lu]
            'r_R': r_norm,                      # [dimensionless]
            'chord': blade.marker_chord,        # [lu]
            'twist': blade.marker_twist,        # [degrees]
            'active': blade.marker_active,
            'u_rel': bem.u_rel[idx_s:idx_e],    # [Δx/Δt]
            'phi': bem.phi[idx_s:idx_e],        # [degrees]
            'alpha': bem.alpha[idx_s:idx_e],    # [degrees]
            'Re': bem.Re[idx_s:idx_e],          # [dimensionless]
            'CL': bem.CL[idx_s:idx_e],          # [dimensionless]
            'CD': bem.CD[idx_s:idx_e],          # [dimensionless]
            'F_L': bem.F_L[idx_s:idx_e],        # [lattice force]
            'F_D': bem.F_D[idx_s:idx_e],        # [lattice force]
            'F_n': bem.F_n[idx_s:idx_e],        # [lattice force]
            'F_theta': bem.F_theta[idx_s:idx_e],# [lattice force]
        }

    # -----------------------------------------------------------------
    # §2.4 Unit Conversion Helpers
    # -----------------------------------------------------------------

    def to_physical_units(
        self,
        performance: dict,
        rho_phys: float = 1.225,
    ) -> dict:
        """Convert performance dict to physical units

        Uses the stored dx_phys and dt_phys to convert forces,
        torques, and power from lattice to SI units.

        Dimensional derivation (ρ₀ = 1 in lattice):
            F_phys = F_lu × ρ_phys × dx⁴ / dt²     [N]
            Q_phys = Q_lu × ρ_phys × dx⁵ / dt²     [N·m]
            P_phys = P_lu × ρ_phys × dx⁵ / dt³     [W]

        P = F·v → P_phys = F_lu·(ρ·dx⁴/dt²) × (dx/dt) = ρ·dx⁵/dt³ × P_lu

        Args:
            performance: Output from get_rotor_performance()
            rho_phys: Physical air density  [kg/m³]

        Returns:
            dict with added '_phys' keys for thrust, torque, power
        """
        dx = self.dx_phys   # [m/lu]
        dt = self.dt_phys   # [s/lt]

        force_scale  = rho_phys * dx**4 / dt**2     # [N / lu_force]
        torque_scale = rho_phys * dx**5 / dt**2     # [N·m / lu_torque]
        power_scale  = rho_phys * dx**5 / dt**3     # [W / lu_power]

        result = dict(performance)
        if 'thrust' in performance:
            result['thrust_phys'] = performance['thrust'] * force_scale   # [N]
            result['torque_phys'] = performance['torque'] * torque_scale  # [N·m]
            result['power_phys'] = performance['power'] * power_scale     # [W]
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
            f"  Multi-airfoil: {self._multi_airfoil}",
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
# §2.5  Multi-Rotor Manager
# =============================================================================

class MultiRotorManager:
    """Manages multiple ActuatorLineModel instances on a shared domain

    Each rotor operates independently (BEM, Gaussian spreading),
    and their body forces are linearly superposed onto the shared grid.

    Wake interaction is captured IMPLICITLY: rotor k's wake modifies
    the velocity field u(x), which rotor k+1 samples at its markers.

    Attributes:
        models: List of ActuatorLineModel, one per rotor
        names: Human-readable labels (e.g., 'rotor_0', 'upwind')
        domain_shape: (Nx, Ny, Nz)  [lu]
        n_rotors: Number of rotors  [dimensionless]

    Example:
        >>> mgr = MultiRotorManager(domain_shape=(200, 100, 100))
        >>> mgr.add_model(al_model_0, name='upwind')
        >>> mgr.add_model(al_model_1, name='downwind')
        >>>
        >>> # In time loop:
        >>> F_total = mgr.step(u_field, dt=1.0)
    """

    def __init__(
        self,
        domain_shape: Tuple[int, int, int],
    ) -> None:
        """Initialize MultiRotorManager

        Args:
            domain_shape: (Nx, Ny, Nz) shared grid  [lu]
        """
        self.domain_shape = domain_shape    # [lu]
        self.models: List['ActuatorLineModel'] = []
        self.names: List[str] = []

        # Pre-allocate accumulated body force array
        Nx, Ny, Nz = domain_shape
        self._F_total = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
        # [lattice force / lu³]

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def add_model(
        self,
        model: 'ActuatorLineModel',
        name: Optional[str] = None,
    ) -> None:
        """Register an ActuatorLineModel

        Args:
            model: Configured AL model (already in lattice units)
            name: Human-readable label (auto-generated if None)
        """
        if model.domain_shape != self.domain_shape:
            raise ValueError(
                f"Domain shape mismatch: manager has {self.domain_shape}, "
                f"but model '{name}' has {model.domain_shape}"
            )
        if name is None:
            name = f"rotor_{len(self.models)}"
        self.models.append(model)
        self.names.append(name)

    @property
    def n_rotors(self) -> int:
        """Number of registered rotors  [dimensionless]"""
        return len(self.models)

    # -----------------------------------------------------------------
    # §2.5.1  Main Time Step (matches ActuatorLineModel.step interface)
    # -----------------------------------------------------------------

    def step(
        self,
        u_field: 'npt.NDArray',
        dt: float = 1.0,
        external_F: Optional['npt.NDArray'] = None,
    ) -> 'npt.NDArray':
        """Execute one AL timestep for ALL rotors

        Physical Process:
            F_total(x) = Σ_k F_k(x)

        Each ALM_k independently:
            1. Advances its rotor azimuth
            2. Samples u(x) at its markers
            3. Computes BEM forces
            4. Spreads forces via Gaussian kernel

        All F_k are accumulated into a single body force field.

        Args:
            u_field: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
            dt: Timestep  [lt]
            external_F: Additional body force (e.g., from sponge layer)
                        shape (3, Nx, Ny, Nz)  [lattice force / lu³]

        Returns:
            F_total: Combined body force field
                     shape (3, Nx, Ny, Nz)  [lattice force / lu³]
        """
        # Reset accumulator
        self._F_total[:] = 0.0

        if external_F is not None:
            self._F_total[:] = external_F   # [lattice force / lu³]

        # Each rotor spreads its own forces; first rotor writes into
        # fresh _F_grid, subsequent rotors accumulate additively.
        for i, model in enumerate(self.models):
            # Each model.step() returns its own F_grid (internal array)
            F_k = model.step(u_field, dt=dt, external_F=None)
            # [lattice force / lu³]
            self._F_total += F_k    # Linear superposition

        return self._F_total

    # -----------------------------------------------------------------
    # §2.5.2  Performance Queries
    # -----------------------------------------------------------------

    def get_rotor_performance(self, rotor_idx: Optional[int] = None) -> dict:
        """Get performance coefficients for one or all rotors

        Args:
            rotor_idx: If None, returns dict of all rotors.
                       If int, returns single rotor's performance.

        Returns:
            Single rotor: same dict as ActuatorLineModel.get_rotor_performance()
            All rotors: {'rotor_0': {...}, 'rotor_1': {...}, ...}
        """
        if rotor_idx is not None:
            return self.models[rotor_idx].get_rotor_performance()

        results = {}
        for i, (model, name) in enumerate(zip(self.models, self.names)):
            results[name] = model.get_rotor_performance()
        return results

    def get_all_marker_positions(self) -> Dict[str, np.ndarray]:
        """Get marker positions for all rotors (for VTK output)

        Returns:
            {name: positions_array} for each rotor
        """
        return {
            name: model._last_positions
            for name, model in zip(self.names, self.models)
            if model._last_positions is not None
        }

    # -----------------------------------------------------------------
    # §2.5.3  Proxy Properties (backward compatibility)
    # -----------------------------------------------------------------

    @property
    def rotor(self) -> 'Rotor':
        """Primary rotor (first registered) — for backward compatibility"""
        if not self.models:
            raise RuntimeError("No rotors registered")
        return self.models[0].rotor

    @property
    def coeff_mode(self) -> str:
        """Primary rotor's coefficient mode"""
        return self.models[0].coeff_mode if self.models else 'auto'

    @property
    def u_inf_lu(self) -> Optional[float]:
        """Primary rotor's u_inf_lu  [Δx/Δt]"""
        return self.models[0].u_inf_lu if self.models else None

    @property
    def _last_positions(self) -> Optional[np.ndarray]:
        """Concatenated positions from all rotors (for VTK)"""
        positions = []
        for model in self.models:
            if model._last_positions is not None:
                positions.append(model._last_positions)
        return np.vstack(positions) if positions else None

    @property
    def _last_bem_result(self):
        """Primary rotor's BEM result (for backward compatibility)"""
        return self.models[0]._last_bem_result if self.models else None

    @property
    def _last_forces_global(self) -> Optional[np.ndarray]:
        """Concatenated forces from all rotors (for VTK)"""
        forces = []
        for model in self.models:
            if model._last_forces_global is not None:
                forces.append(model._last_forces_global)
        return np.vstack(forces) if forces else None

    # -----------------------------------------------------------------
    # §2.5.4  Info
    # -----------------------------------------------------------------

    def get_info(self) -> str:
        """Human-readable summary"""
        lines = [
            f"MultiRotorManager: {self.n_rotors} rotor(s)",
            "=" * 60,
        ]
        for i, (model, name) in enumerate(zip(self.models, self.names)):
            lines.append(f"\n  [{i}] {name}:")
            lines.append(f"      Hub: {model.rotor.hub_center}")
            lines.append(f"      R={model.rotor.radius:.1f} lu, "
                         f"ω={model.rotor.omega:.6f} rad/lt")
            lines.append(f"      Blades: {model.rotor.n_blades}, "
                         f"Markers: {model.rotor.total_markers}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MultiRotorManager(n_rotors={self.n_rotors}, "
            f"domain={self.domain_shape})"
        )


# =============================================================================
# §3. Factory Functions (UPDATED)
# =============================================================================

def create_actuator_line_from_config(
    config: dict,
    domain_shape: Tuple[int, int, int],
    nu_lattice: float,
    polar_query: Callable,
    dx_phys: float = 1.0,
    dt_phys: float = 1.0,
    u_inf_lu: Optional[float] = None,
    coeff_mode: str = 'auto',
) -> 'ActuatorLineModel':
    """Create a SINGLE ActuatorLineModel from config  (unchanged API)

    Expected Config Format:
        {
            "rotor": { ... },
            "gaussian_cutoff": 3.0,
            "rho_ref": 1.0,
            "coeff_mode": "auto"
        }
    """
    from src.actuator.rotor import Rotor  # local import to avoid circular

    rotor_cfg = config.get('rotor', {})

    if 'grid' not in rotor_cfg:
        rotor_cfg['grid'] = {}
    if 'dx' not in rotor_cfg['grid']:
        rotor_cfg['grid']['dx'] = dx_phys   # [m/lu]

    rotor_phys = Rotor.from_config(rotor_cfg)
    rotor_lu = rotor_phys.to_lattice_units(
        length_scale=dx_phys, time_scale=dt_phys
    )

    resolved_mode = coeff_mode if coeff_mode != 'auto' \
        else config.get('coeff_mode', 'auto')

    return ActuatorLineModel(
        rotor=rotor_lu,
        nu=nu_lattice,
        domain_shape=domain_shape,
        polar_query=polar_query,
        rho_ref=config.get('rho_ref', 1.0),
        n_cut=config.get('gaussian_cutoff', 3.0),
        dx_phys=dx_phys,
        dt_phys=dt_phys,
        u_inf_lu=u_inf_lu,
        coeff_mode=resolved_mode,
    )


def create_multi_rotor_from_config(
    config: dict,
    domain_shape: Tuple[int, int, int],
    nu_lattice: float,
    polar_query: Callable,
    dx_phys: float = 1.0,
    dt_phys: float = 1.0,
    u_inf_lu: Optional[float] = None,
    coeff_mode: str = 'auto',
) -> MultiRotorManager:
    """Create a MultiRotorManager from config containing 'rotors' list

    Supports two config formats:
    
    Format A (single rotor — backward compatible):
        {
            "rotor": { ... },
            "gaussian_cutoff": 3.0,
            ...
        }
    
    Format B (multi rotor):
        {
            "rotors": [
                {"name": "upwind",   "rotor": { ... }},
                {"name": "downwind", "rotor": { ... }},
            ],
            "gaussian_cutoff": 3.0,  # shared defaults
            "rho_ref": 1.0,
            "coeff_mode": "auto",
        }

    Each rotor entry can override shared defaults:
        {
            "name": "rotor_0",
            "rotor": { ... },                # Rotor geometry config
            "gaussian_cutoff": 4.0,          # override shared default
            "rho_ref": 1.0,
            "coeff_mode": "wind_turbine",    # override
            "u_inf_lu": 0.05,               # per-rotor u_inf override
        }

    Args:
        config: AL config dict (with 'rotors' list or single 'rotor')
        domain_shape: (Nx, Ny, Nz)  [lu]
        nu_lattice: Kinematic viscosity  [lu²/lt]
        polar_query: Airfoil data query function
        dx_phys: Physical grid spacing  [m/lu]
        dt_phys: Physical timestep  [s/lt]
        u_inf_lu: Freestream velocity  [Δx/Δt] (default for all rotors)
        coeff_mode: Default coefficient mode

    Returns:
        MultiRotorManager with all rotors registered
    """
    manager = MultiRotorManager(domain_shape=domain_shape)

    # ── Detect format ──
    if 'rotors' in config:
        # Format B: multi-rotor list
        rotors_list = config['rotors']
    elif 'rotor' in config:
        # Format A: single rotor → wrap in list for uniform handling
        rotors_list = [{"rotor": config['rotor']}]
    else:
        raise ValueError(
            "actuator_line config must contain either 'rotor' (single) "
            "or 'rotors' (list) key"
        )

    # ── Shared defaults ──
    shared_defaults = {
        'gaussian_cutoff': config.get('gaussian_cutoff', 3.0),
        'rho_ref': config.get('rho_ref', 1.0),
        'coeff_mode': config.get('coeff_mode', coeff_mode),
    }

    # ── Create each rotor ──
    for i, rotor_entry in enumerate(rotors_list):
        name = rotor_entry.get('name', f'rotor_{i}')

        # Build per-rotor config by merging shared defaults + overrides
        single_config = {
            'rotor': rotor_entry.get('rotor', rotor_entry),
            'gaussian_cutoff': rotor_entry.get(
                'gaussian_cutoff', shared_defaults['gaussian_cutoff']
            ),
            'rho_ref': rotor_entry.get(
                'rho_ref', shared_defaults['rho_ref']
            ),
            'coeff_mode': rotor_entry.get(
                'coeff_mode', shared_defaults['coeff_mode']
            ),
        }

        # Per-rotor u_inf override
        rotor_u_inf = rotor_entry.get('u_inf_lu', u_inf_lu)

        al_model = create_actuator_line_from_config(
            config=single_config,
            domain_shape=domain_shape,
            nu_lattice=nu_lattice,
            polar_query=polar_query,
            dx_phys=dx_phys,
            dt_phys=dt_phys,
            u_inf_lu=rotor_u_inf,
            coeff_mode=single_config['coeff_mode'],
        )

        manager.add_model(al_model, name=name)

    return manager


# =============================================================================
# §3. Factory Function [single factory old version]
# =============================================================================

# def create_actuator_line_from_config(
#     config: dict,
#     domain_shape: Tuple[int, int, int],
#     nu_lattice: float,
#     polar_query: Callable,
#     dx_phys: float = 1.0,
#     dt_phys: float = 1.0,
#     u_inf_lu: Optional[float] = None,
#     coeff_mode: str = 'auto',
# ) -> ActuatorLineModel:
#     """Create an ActuatorLineModel from a configuration dictionary

#     Expected Config Format:
#         {
#             "rotor": { ... },
#             "gaussian_cutoff": 3.0,
#             "rho_ref": 1.0,
#             "coeff_mode": "auto"        # 'wind_turbine' | 'rotorcraft' | 'auto'
#         }

#     Args:
#         config: AL model configuration
#         domain_shape: (Nx, Ny, Nz)
#         nu_lattice: Kinematic viscosity [lu²/lt]
#         polar_query: Airfoil data query function
#         dx_phys: Physical grid spacing [m/lu]
#         dt_phys: Physical timestep [s/lt]
#         u_inf_lu: Freestream velocity [Δx/Δt] or None (BEM fallback)
#         coeff_mode: Override for coefficient mode.
#                     If not provided, reads from config['coeff_mode'].

#     Returns:
#         Configured ActuatorLineModel instance
#     """
#     rotor_cfg = config.get('rotor', {})

#     if 'grid' not in rotor_cfg:
#         rotor_cfg['grid'] = {}
#     if 'dx' not in rotor_cfg['grid']:
#         rotor_cfg['grid']['dx'] = dx_phys          # [m/lu]

#     # Create rotor (in physical units first)
#     rotor_phys = Rotor.from_config(rotor_cfg)

#     # Convert to lattice units
#     rotor_lu = rotor_phys.to_lattice_units(
#         length_scale=dx_phys,
#         time_scale=dt_phys
#     )

#     # Resolve coeff_mode: explicit argument > config key > default
#     resolved_mode = coeff_mode if coeff_mode != 'auto' \
#         else config.get('coeff_mode', 'auto')

#     return ActuatorLineModel(
#         rotor=rotor_lu,
#         nu=nu_lattice,
#         domain_shape=domain_shape,
#         polar_query=polar_query,
#         rho_ref=config.get('rho_ref', 1.0),
#         n_cut=config.get('gaussian_cutoff', 3.0),
#         dx_phys=dx_phys,
#         dt_phys=dt_phys,
#         u_inf_lu=u_inf_lu,
#         coeff_mode=resolved_mode,
#     )