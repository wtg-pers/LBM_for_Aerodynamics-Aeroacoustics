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
    interpolate_velocity_batch_gpu,
    sample_velocity_alt,
)
from .spreading import (
    spread_forces_to_grid,
    spread_forces_uniform_epsilon,
    spread_forces_to_grid_gpu,
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
    # Smearing-correction diagnostics (None unless eps_correction is active)
    w_corr: np.ndarray = None           # [Δx/Δt]  added downwash per marker
    alpha_uncorrected: np.ndarray = None  # [degrees] alpha before correction


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
        xp=None,
        sound_speed: Optional[float] = None,
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
            xp: Array module — numpy or cupy (default: numpy).
                When cupy, interpolation and spreading run on GPU.
                u_field passed to step() should be an xp array.
        """
        self.rotor = rotor
        self.nu = nu                    # [lu²/lt]
        self.rho_ref = rho_ref          # [dimensionless]
        self.domain_shape = domain_shape
        self.polar_query = polar_query
        self.n_cut = n_cut
        self.dx_phys = dx_phys          # [m/lu]
        self.dt_phys = dt_phys          # [s/lt]
        self.a_phys = sound_speed       # [m/s] free-stream sound speed or None
        self.u_inf_lu = u_inf_lu        # [Δx/Δt] or None
        self.coeff_mode = coeff_mode    # 'wind_turbine' | 'rotorcraft' | 'auto'
        self.xp = xp if xp is not None else np   # array backend

        # Pre-allocate body force array (on GPU if xp is cupy)
        Nx, Ny, Nz = domain_shape
        self._F_grid = self.xp.zeros((3, Nx, Ny, Nz), dtype=self.xp.float64)
        # [lattice force / lu³]

        # Diagnostics storage (updated each step)
        self._last_bem_result: Optional[BEMResult] = None
        self._last_positions: Optional[np.ndarray] = None
        self._last_forces_global: Optional[np.ndarray] = None
        self._step_count: int = 0
        self.ramp_steps: int = 0
        self._ramp_factor: float = 1.0

        # ── Prandtl tip/root loss correction ──
        self.prandtl_loss: bool = False
        self._prandtl_tip: bool = True
        self._prandtl_root: bool = True
        # Effective-radius ε offset for Prandtl: R_tip_eff = R_tip - ε_tip.
        # True (default) reproduces the existing non-standard behavior; False
        # uses the textbook R_tip_eff = R_tip (matches BEMT / standard Prandtl,
        # and decouples the tip-loss from the ε taper).
        self._prandtl_eps_offset: bool = True

        # Smearing (viscous-core) correction — recovers the tip induced-velocity
        # deficit of the finite-ε ALM (Dağ & Sørensen 2020). Default OFF →
        # bit-identical. See _viscous_core_correction / docs/alm_tip_overprediction_record_kr.md.
        self._eps_corr: bool = False
        self._eps_corr_target: str = "inviscid"   # "inviscid" | "opt"
        self._eps_opt_factor: float = 0.25        # ε_opt = factor·chord (target="opt")
        self._eps_corr_relax: float = 1.0         # under-relaxation on w_corr
        # Correction method: "dag" (single-pass, current) | "kleine" (Kleine 2022
        # non-iterative linear solve, Phase 1 — patch_notes/kleine_smearing_correction/).
        self._eps_corr_method: str = "dag"
        self._kleine_A: dict = {}                 # per-blade influence matrix cache
        self._kleine_gamma_prev: dict = {}        # per-blade Γⁿ⁻¹ (warm-start persist)
        # Kleine wake model: "straight" (Phase 1 semi-infinite, default) |
        # "free" (Phase 2 convected free-vortex wake → captures tip rollup).
        self._kleine_wake_mode: str = "straight"
        self._kleine_wake_nw: int = 50            # free-wake length [timesteps]
        self._kleine_wake: dict = {}              # per-blade FreeWake (Phase 2)
        # Phase 2 perf (patch_notes/kleine_freewake_perf/):
        #   A) throttle the free-wake influence-matrix rebuild — the wake convects
        #      slowly, so a cached A is a good approximation between rebuilds.
        #      _kleine_rebuild_every == 1 (default) → rebuild every step (exact,
        #      bit-identical to the original Phase 2). >1 → cheaper approximation.
        #   C) cache the dΓ/dr gradient matrix (depends only on the fixed r).
        self._kleine_A_free: dict = {}            # per-blade free-wake A cache (A)
        self._kleine_G: dict = {}                 # per-blade dΓ/dr matrix cache (C)
        self._kleine_rebuild_every: int = 1       # rebuild free-wake A every N steps
        self._kleine_wake_steps: int = 0          # free-wake step counter (cadence)

        # Velocity sampler mode (A/B study — patch_notes/almlbm_sampler_ab/).
        # "gaussian" (default) → bit-identical §6 path. Alternatives isolate the
        # tip-inflow contribution of the ±3ε sampling kernel:
        #   "point" (B-i), "aniso" (B-ii), "mask_disk" (B-iii).
        self._sampling_mode: str = "gaussian"
        self._sampling_eps_r_factor: float = 0.5  # B-ii: ε_r = factor·ε (radial)

        # ═══════════════════════════════════════════════════════════════════
        # NEW: Detect multi-airfoil support
        # ═══════════════════════════════════════════════════════════════════
        self._multi_airfoil = False
        self._polar_wants_mach = False
        try:
            sig = inspect.signature(polar_query)
            param_names = list(sig.parameters.keys())
            # Name-based detection (robust to extra optional args like 'mach').
            self._multi_airfoil = ('airfoil_name' in param_names)
            # Mach-pass: tapered/variable-chord rotors expose a 'mach' arg so
            # the section Mach can be passed directly (the Re→M constant-chord
            # trick breaks under taper).  When absent, behavior is unchanged.
            self._polar_wants_mach = ('mach' in param_names)
        except (ValueError, TypeError):
            self._multi_airfoil = False
            self._polar_wants_mach = False

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
        u_field can be a GPU (cupy) or CPU (numpy) array — the method
        dispatches interpolation and spreading to the appropriate backend
        via self.xp.

        Data flow (GPU path):
            GPU(u) → interpolate(GPU) → CPU(u_markers, ~KB)
            → BEM(CPU) → CPU(F_markers, ~KB)
            → spread(GPU) → GPU(F_grid)
            No full-domain GPU↔CPU transfer needed.

        Physical Steps (see module docstring for data flow):
            1. Advance rotor azimuth
            2. Sample velocity at marker positions
            3. Compute BEM forces
            4. Spread forces to grid

        Args:
            u_field: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
                     Can be xp array (GPU or CPU).
            dt: Timestep (default 1.0 in lattice units)  [lt]
            external_F: Additional body force to combine with AL force
                        shape (3, Nx, Ny, Nz)  [lattice force / lu³]
                        Must be same backend as self.xp.

        Returns:
            F_grid: Total body force field, shape (3, Nx, Ny, Nz)
                    [lattice force / lu³]
                    Same backend as self.xp (GPU if cupy).
        """
        xp = self.xp

        # --- Step 1: Advance rotor (CPU, trivial) ---
        self.rotor.advance(dt)

        # --- Step 2: Get marker positions (CPU, ~200 markers) ---
        positions = self.rotor.get_all_marker_positions()   # (N_total, 3) [lu]
        self._last_positions = positions

        # --- Step 3: Interpolate velocity at markers ---
        # GPU path: u_field stays on GPU, only u_markers (~KB) returns to CPU
        epsilon_all = self.rotor.get_all_marker_epsilon()    # (N_total,) [lu]
        active_all = self.rotor.get_all_marker_active()      # (N_total,) bool

        if self._sampling_mode == "gaussian":
            u_markers = interpolate_velocity_batch_gpu(
                u_field, positions, epsilon_all,
                xp=xp, n_cut=self.n_cut
            )  # (N_total, 3) [Δx/Δt] — always numpy (CPU)
        else:
            # A/B alternatives (B-i/B-ii/B-iii) — patch_notes/almlbm_sampler_ab/
            u_markers = sample_velocity_alt(
                self._sampling_mode, u_field, positions, epsilon_all,
                xp=xp, n_cut=self.n_cut,
                hub=np.asarray(self.rotor.hub_center, dtype=np.float64),
                axis=np.asarray(self.rotor.rotation_axis, dtype=np.float64),
                radius=float(self.rotor.radius),
                eps_r_factor=self._sampling_eps_r_factor,
            )  # (N_total, 3) [Δx/Δt] — always numpy (CPU)

        # --- Phase 2 free-wake: convect + shed (before BEM uses it) ---
        if (self._eps_corr and self._eps_corr_method == "kleine"
                and self._kleine_wake_mode == "free"):
            self._convect_and_shed_wake(u_field, positions, dt, xp)

        # --- Step 4-7: BEM force calculation (CPU, small arrays) ---
        bem_result = self._compute_bem_forces(u_markers)
        self._last_bem_result = bem_result

        # --- Step 8: Project to global frame (CPU, small arrays) ---
        F_global = self.rotor.project_all_forces(
            bem_result.F_n, bem_result.F_theta
        )  # (N_total, 3) [lattice force] — numpy (CPU)
        self._last_forces_global = F_global

        # --- Step 9: Gaussian spreading ---
        # GPU path: F_grid allocated and filled on GPU, no CPU transfer
        self._F_grid[:] = 0.0  # Reset (works for both numpy and cupy)
        if external_F is not None:
            self._F_grid[:] = external_F  # Start from external force

        spread_forces_to_grid_gpu(
            self.domain_shape,
            positions,
            F_global,
            epsilon_all,
            xp=xp,
            marker_active=active_all,
            n_cut=self.n_cut,
            F_grid=self._F_grid
        )

        self._step_count += 1

        # Force ramp-up
        if self.ramp_steps > 0 and self._step_count <= self.ramp_steps:
            self._ramp_factor = float(self._step_count) / float(self.ramp_steps)
            self._F_grid *= self._ramp_factor
        else:
            self._ramp_factor = 1.0

        return self._F_grid

    # -----------------------------------------------------------------
    # §2.2a Prandtl Tip/Root Loss Correction
    # -----------------------------------------------------------------

    def _compute_prandtl_factor(
        self,
        r: np.ndarray,
        phi_deg: np.ndarray,
    ) -> np.ndarray:
        """Prandtl combined tip/root loss factor.

        F_pr = F_tip × F_root  ∈ [0, 1]

        where:
            f_tip  = B·(R_tip,eff  - r) / (2·r·sin|φ|)
            f_root = B·(r - R_root,eff) / (2·r·sin|φ|)
            F      = (2/π)·arccos(exp(-f))

        Effective radii account for Gaussian force spreading:
            R_tip,eff  = R_tip  - ε_tip   (tip marker epsilon)
            R_root,eff = R_root + ε_root  (root marker epsilon)

        Args:
            r:       Marker radial positions  [lu], shape (n_markers,)
            phi_deg: Flow angle per marker    [degrees], shape (n_markers,)

        Returns:
            F_pr: Combined correction factor, shape (n_markers,)
        """
        rotor = self.rotor
        blade = rotor.blades[0]
        B = rotor.n_blades

        R_tip = rotor.radius                          # [lu]
        R_root = blade.marker_r[0] - blade.marker_dr / 2  # [lu] start of active span

        if self._prandtl_eps_offset:
            # Non-standard (legacy default): shrink effective radii by the
            # local ε to account for Gaussian force spreading.
            eps_tip = blade.marker_epsilon[-1]             # [lu] tip marker
            eps_root = blade.marker_epsilon[0]             # [lu] root marker
            R_tip_eff = R_tip - eps_tip
            R_root_eff = R_root + eps_root
        else:
            # Standard Prandtl (matches BEMT): effective radius = actual radius.
            # Decoupled from the ε taper → smooth roll-off, no hard-zero band.
            R_tip_eff = R_tip
            R_root_eff = R_root

        sin_phi = np.abs(np.sin(np.radians(phi_deg)))
        sin_phi = np.maximum(sin_phi, 1e-4)            # avoid division by zero

        F_pr = np.ones_like(r, dtype=np.float64)

        if self._prandtl_tip:
            f_tip = B * np.maximum(R_tip_eff - r, 0.0) / (2.0 * r * sin_phi)
            F_tip = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), -1.0, 1.0))
            F_pr *= F_tip

        if self._prandtl_root:
            f_root = B * np.maximum(r - R_root_eff, 0.0) / (2.0 * r * sin_phi)
            F_root = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_root), -1.0, 1.0))
            F_pr *= F_root

        return F_pr

    def _lookup_cl_cd(self, alpha_deg, Re, u_rel, Mach, blade, n):
        """Per-marker CL/CD polar lookup (multi-airfoil and/or Mach-pass aware).

        Extracted from the BEM loop so the smearing correction can re-query at
        the corrected angle of attack.  Behavior is identical to the original
        inline loop (active & u_rel≥1e-10 markers only; others remain 0).
        """
        active = blade.marker_active
        CL = np.zeros(n, dtype=np.float64)
        CD = np.zeros(n, dtype=np.float64)
        for j in range(n):
            if not active[j] or u_rel[j] < 1e-10:
                continue
            a_deg = float(alpha_deg[j])
            re_j = float(Re[j])
            if self._multi_airfoil:
                name = blade.marker_airfoil[j]
                if Mach is not None:
                    CL[j], CD[j] = self.polar_query(a_deg, re_j, name,
                                                    mach=float(Mach[j]))
                else:
                    CL[j], CD[j] = self.polar_query(a_deg, re_j, name)
            else:
                if Mach is not None:
                    CL[j], CD[j] = self.polar_query(a_deg, re_j,
                                                    mach=float(Mach[j]))
                else:
                    CL[j], CD[j] = self.polar_query(a_deg, re_j)
        return CL, CD

    def _viscous_core_correction(self, r, eps, Gamma, chord, dr):
        """Missing tip induced velocity from Gaussian-smeared trailed vortices.

        Implements the viscous-core de-induction (Dağ & Sørensen 2020): a
        finite-ε ALM sheds trailed vortices with a Lamb-Oseen core = ε, which
        under-induce the downwash (worst at the tip → over-loading).  This adds
        back the missing part along the span.

            Γ(i)   = ½ c_i u_rel_i CL_i                (bound circulation, set by caller)
            Γw(m)  = Γ_inboard − Γ_outboard           (trailed vortex at panel edge)
            w(i)   = −(1/4π) Σ_m Γw(m)/d_im · K(d_im)  (added downwash)
            K = exp(−(d/ε)²)                           target="inviscid"
              = exp(−(d/ε)²) − exp(−(d/ε_opt)²)        target="opt"  (ε_opt=factor·c)

        Trailed vortices sit at panel EDGES (offset ±dr/2 from the cell-centred
        markers) → distances are finite; the bound (self) vortex is excluded by
        construction.  All lattice units; w_corr [Δx/Δt] is ADDED to u_n.
        The sign (−1/4π) gives a downwash (positive u_n) at the tip, verified by
        the C_T / tip-φ A/B.
        """
        N = len(r)
        # Trailed vorticity = dΓ/dr at the control points (markers). np.gradient
        # uses one-sided differences at the ends → captures the strong tip
        # gradient without the edge-panel endpoint artifact.
        gradG = np.gradient(Gamma, r)               # dΓ/dr [Δx/Δt]
        use_opt = (self._eps_corr_target == "opt")
        eps_opt = np.maximum(self._eps_opt_factor * chord, 1e-9)  # [lu]
        inv4pi = 1.0 / (4.0 * np.pi)
        w = np.zeros(N)                             # [Δx/Δt]
        for i in range(N):
            d = r[i] - r                            # control→control [lu]; d[i]=0
            far = np.abs(d) > 1e-9                   # exclude self-term (i=j)
            safe_d = np.where(far, d, 1.0)           # avoid 0-division at i=j
            # Regularized lifting-line kernel (→ 0 at d=0, no singularity):
            #   inviscid target: exp(−(d/ε)²)/d, self-term (i=j) excluded
            #   opt target:      [exp(−(d/ε)²) − exp(−(d/ε_opt)²)]/d  (regular)
            if use_opt:
                K = np.exp(-(d / eps) ** 2) - np.exp(-(d / eps_opt) ** 2)
            else:
                K = np.exp(-(d / eps) ** 2)
            kernel = np.where(far, K / safe_d, 0.0)
            # w = −(1/4π) ∫ (dΓ/dr)·kernel dr   → added downwash (sign by A/B)
            w[i] = -inv4pi * np.sum(gradG * kernel) * dr
        return w

    def _kleine_w_corr(self, k, blade, u_n, u_theta, cos_sweep, chord, dr,
                       CL, active, n):
        """Total deficit downwash via Kleine (2022) non-iterative solve (Phase 1).

        Returns w_corr [Δx/Δt] (same role as the Dağ single-pass: u_n += w_corr).
        Uses the SAME deficit kernel as Dağ (influence_matrix == _viscous_core_
        correction(inviscid)) but solves the implicit Γ–w coupling in one linear
        system (Kleine Eq. 5.15) instead of a single pass. Re/Mach are frozen at
        the linearization point †; ∂C_l/∂α from polar_slope. Falls back to the
        Dağ single pass on a singular / non-finite solve.
        See patch_notes/kleine_smearing_correction/.
        """
        from .smearing_correction import (
            influence_matrix, correct_noniterative, freewake_influence,
            _gradient_matrix)
        from .polar_slope import lift_curve_slope_batch
        rotor = self.rotor
        r = blade.marker_r
        eps = blade.marker_epsilon
        twist = blade.marker_twist

        # Influence matrix A_ik = ∂w_i/∂Γ_k.
        wake = self._kleine_wake.get(k)
        if self._kleine_wake_mode == "free" and wake is not None and len(wake) >= 2:
            # Phase 2: build from the convected free-wake geometry. B·axial →
            # A = Δr·(B @ G). The rebuild (freewake_influence) is the per-step
            # bottleneck, so throttle it: rebuild every `_kleine_rebuild_every`
            # steps (default 1 = every step, exact), else reuse the cached A —
            # the wake convects slowly so this is a good approximation. A blade
            # whose cache is empty/stale (cold start) always rebuilds.
            A = self._kleine_A_free.get(k)
            need_rebuild = (
                A is None or A.shape[0] != n
                or (self._kleine_wake_steps % self._kleine_rebuild_every == 0))
            if need_rebuild:
                ctrl3d = self._last_positions[k * n:(k + 1) * n]
                # Axial (downwash) projection direction. The sign tracks the
                # rotation sense so the free-wake downwash matches the rotation-
                # invariant Phase 1 / Dağ de-induction (verified vs straight).
                axis = -np.sign(rotor.omega) * np.asarray(
                    rotor.rotation_axis, dtype=np.float64)
                B = freewake_influence(ctrl3d, wake.rings, eps, axis)
                G = self._kleine_G.get(k)                  # (C) cache dΓ/dr
                if G is None or G.shape[0] != n:
                    G = _gradient_matrix(r)
                    self._kleine_G[k] = G
                A = dr * (B @ G)
                self._kleine_A_free[k] = A
        else:
            # Phase 1: straight semi-infinite (cached; also the free-wake
            # cold-start fallback while < 2 rings).
            A = self._kleine_A.get(k)
            if A is None or A.shape[0] != n:
                A = influence_matrix(r, eps, dr)
                self._kleine_A[k] = A

        # Tangential rel. velocity (matches rotor.recompute_velocity_triangle).
        rsign = np.sign(rotor.omega)
        u_tan = np.abs(rotor.omega) * r - rsign * u_theta

        # Γⁿ⁻¹ (persist for warm-start); cold start = uncorrected Γ.
        u_aero0 = np.sqrt(u_n ** 2 + u_tan ** 2) * cos_sweep
        Gprev = self._kleine_gamma_prev.get(k)
        if Gprev is None or len(Gprev) != n:
            Gprev = np.where(active, 0.5 * chord * u_aero0 * CL, 0.0)

        # Freeze Re/Mach at the linearization point † (u_n† = u_n + A Γⁿ⁻¹).
        u_aero_d = np.sqrt((u_n + A @ Gprev) ** 2 + u_tan ** 2) * cos_sweep
        Re_d = u_aero_d * chord / (self.nu + 1e-30)
        if self._polar_wants_mach and self.a_phys:
            Mach_d = u_aero_d * (self.dx_phys / self.dt_phys) / self.a_phys
        else:
            Mach_d = None
        multi = self._multi_airfoil
        names = blade.marker_airfoil if multi else None

        def cl_eval(a_deg):
            cl, _ = self._lookup_cl_cd(a_deg, Re_d, u_aero_d, Mach_d, blade, n)
            return cl

        def dcl_eval(a_deg):
            return lift_curve_slope_batch(
                self.polar_query, a_deg, Re_d, active,
                multi_airfoil=multi, marker_airfoil=names,
                mach=Mach_d, delta_deg=1.0)

        try:
            _, _, Gnew, w = correct_noniterative(
                r, chord, dr, eps, u_n, u_tan, twist, cl_eval, dcl_eval,
                Gprev, A=A, active=active)
            # Safety net: a non-finite, or physically-unreasonable (downwash
            # exceeding ~half the tangential scale) solve signals an
            # ill-conditioned/unstable fixed point → fall back to Dağ this step
            # (do NOT persist the bad Γ; keep the previous Γ for warm-start).
            scale = float(np.max(np.abs(u_tan))) + 1e-30
            if (not np.all(np.isfinite(w))) or (float(np.max(np.abs(w))) > 0.5 * scale):
                raise np.linalg.LinAlgError("Kleine correction out of bounds")
            self._kleine_gamma_prev[k] = Gnew
            return w
        except np.linalg.LinAlgError:
            # Fallback: Dağ single pass (keeps the run alive on a bad solve).
            Gamma = np.where(active, 0.5 * chord * u_aero0 * CL, 0.0)
            return self._eps_corr_relax * self._viscous_core_correction(
                r, eps, Gamma, chord, dr)

    def _convect_and_shed_wake(self, u_field, positions, dt, xp):
        """Phase 2 free-vortex wake update (call once per step, before BEM).

        Convect every existing wake ring by the sampled (un-corrected) CFD
        velocity — Kleine §3.4 uses the CFD, not the corrected, velocity — via
        trilinear point sampling, then shed a new ring at the current marker
        positions. Per-blade state in self._kleine_wake.
        """
        from .smearing_correction import FreeWake
        from .interpolation import _sample_trilinear
        rotor = self.rotor
        npb = rotor.markers_per_blade
        is_np = (xp.__name__ == 'numpy')
        self._kleine_wake_steps += 1               # drives the A-rebuild cadence

        for k in range(rotor.n_blades):
            if self._kleine_wake.get(k) is None:
                self._kleine_wake[k] = FreeWake(self._kleine_wake_nw)

        # (B) Convect: gather EVERY ring of EVERY blade into ONE point list and
        # sample the CFD velocity in a single trilinear call (one kernel launch
        # + one device→host copy), instead of one tiny call per ring. The per-
        # point gather is independent, so this is bit-identical to per-ring
        # sampling. Kleine §3.4 convects by the un-corrected CFD velocity.
        index = []                                 # (blade_k, ring_i, n_points)
        chunks = []
        for k in range(rotor.n_blades):
            rings = self._kleine_wake[k].rings
            for i in range(len(rings)):
                index.append((k, i, rings[i].shape[0]))
                chunks.append(rings[i])
        if chunks:
            stacked = np.concatenate(chunks, axis=0)           # (M,3) [lu]
            v = _sample_trilinear(u_field, stacked, xp)
            v = v if is_np else xp.asnumpy(v)                  # single D2H
            off = 0
            for (k, i, m) in index:
                self._kleine_wake[k].rings[i] = (
                    self._kleine_wake[k].rings[i] + dt * v[off:off + m])
                off += m

        # Shed a new ring at the current marker positions (after convection).
        for k in range(rotor.n_blades):
            self._kleine_wake[k].shed(positions[k * npb:(k + 1) * npb])

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
        w_corr_all = np.zeros(n_total, dtype=np.float64)    # [Δx/Δt] smearing corr
        alpha_pre_all = np.zeros(n_total, dtype=np.float64)  # [deg] pre-correction α

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

            # --- Swept-tip correction (simple sweep / cross-flow principle) ---
            # A section swept by Λ feels only the velocity normal to its leading
            # edge, V_n = u_rel·cos Λ.  Using V_n for the polar Mach/Re and the
            # dynamic pressure reduces effective tip Mach (drag-rise relief) and
            # load on swept markers.  Λ=0 (unswept) → cos=1 → byte-identical.
            if blade.marker_sweep.size:
                cos_sweep = np.cos(np.radians(blade.marker_sweep))
            else:
                cos_sweep = 1.0
            u_aero = u_rel * cos_sweep       # [Δx/Δt] velocity normal to LE

            # --- Reynolds number ---
            # Re = V_n · c_a / ν   [dimensionless]
            Re = u_aero * chord / (self.nu + 1e-30)
            Re_all[idx_start:idx_end] = Re

            # --- Section Mach (Mach-pass) ---
            # M = V_n · (dx_phys/dt_phys) / a   — physical relative Mach (sweep-
            # corrected).  dx_phys/dt_phys is the level-correct velocity scale
            # (level-invariant under convective scaling).  Computed ONLY when the
            # polar declares a 'mach' arg (tapered/variable-chord rotors); for
            # constant-chord polars Mach is None and the calls below are
            # byte-identical to the original Reynolds-only lookup.
            if self._polar_wants_mach and self.a_phys:
                Mach = u_aero * (self.dx_phys / self.dt_phys) / self.a_phys
            else:
                Mach = None

            # --- Airfoil polar lookup (α unchanged; V_n drives Mach/Re) ---
            CL, CD = self._lookup_cl_cd(alpha_deg, Re, u_aero, Mach,
                                        blade, n_per_blade)

            # --- Smearing (viscous-core) correction (optional; default off) ---
            # Recover the finite-ε tip induced-velocity deficit (Dağ & Sørensen
            # 2020): add the missing downwash, recompute the velocity triangle,
            # and re-query the polars at the corrected angle of attack. The LBM
            # timestep loop provides the closure (single pass, no inner loop).
            if self._eps_corr and np.any(active):
                alpha_pre_all[idx_start:idx_end] = alpha_deg
                if self._eps_corr_method == "kleine":
                    # Kleine 2022 non-iterative linear solve (Phase 1).
                    w_corr = self._kleine_w_corr(
                        k, blade, u_n, u_theta, cos_sweep, chord, dr,
                        CL, active, n_per_blade)
                else:
                    # Dağ single-pass (default).
                    Gamma = np.where(active, 0.5 * chord * u_aero * CL, 0.0)  # [lu²/lt]
                    w_corr = self._eps_corr_relax * self._viscous_core_correction(
                        blade.marker_r, blade.marker_epsilon, Gamma, chord, dr)
                u_n = u_n + np.where(active, w_corr, 0.0)        # added downwash
                u_rel, phi_deg, alpha_deg = \
                    rotor.recompute_velocity_triangle(k, u_n, u_theta)
                u_aero = u_rel * cos_sweep                       # sweep-corrected
                Re = u_aero * chord / (self.nu + 1e-30)
                if self._polar_wants_mach and self.a_phys:
                    Mach = u_aero * (self.dx_phys / self.dt_phys) / self.a_phys
                CL, CD = self._lookup_cl_cd(alpha_deg, Re, u_aero, Mach,
                                            blade, n_per_blade)
                # overwrite pre-correction kinematics in the output arrays
                u_n_all[idx_start:idx_end] = u_n
                u_rel_all[idx_start:idx_end] = u_rel
                phi_all[idx_start:idx_end] = phi_deg
                alpha_all[idx_start:idx_end] = alpha_deg
                Re_all[idx_start:idx_end] = Re
                w_corr_all[idx_start:idx_end] = np.where(active, w_corr, 0.0)

            CL_all[idx_start:idx_end] = CL
            CD_all[idx_start:idx_end] = CD

            # --- Lift and drag forces (Eq. 9-10) ---
            # F_L = 0.5 · ρ · V_n² · c_a · Δr · CL   [lattice force]
            # F_D = 0.5 · ρ · V_n² · c_a · Δr · CD   [lattice force]
            # Dynamic pressure uses the LE-normal velocity (sweep): q=½ρ(u_rel cosΛ)².
            q = 0.5 * self.rho_ref * u_aero ** 2         # [lattice pressure]
            F_L = q * chord * dr * CL                     # [lattice force]
            F_D = q * chord * dr * CD                     # [lattice force]

            # Inactive markers produce zero force
            F_L[~active] = 0.0
            F_D[~active] = 0.0

            # --- Project to normal/tangential (Leishman / FAST convention) ---
            # F_n = F_L·cos(φ) - F_D·sin(φ)     [lattice force]
            # F_θ = F_L·sin(φ) + F_D·cos(φ)     [lattice force]
            phi_rad = np.radians(phi_deg)                # [radians]
            cos_phi = np.cos(phi_rad)                    # [dimensionless]
            sin_phi = np.sin(phi_rad)                    # [dimensionless]

            F_n = F_L * cos_phi - F_D * sin_phi          # [lattice force]
            F_theta = F_L * sin_phi + F_D * cos_phi      # [lattice force]

            # --- Prandtl tip/root loss correction ---
            if self.prandtl_loss:
                F_pr = self._compute_prandtl_factor(blade.marker_r, phi_deg)
                F_n *= F_pr
                F_theta *= F_pr
                F_L *= F_pr
                F_D *= F_pr

            F_L_all[idx_start:idx_end] = F_L
            F_D_all[idx_start:idx_end] = F_D
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
            w_corr=w_corr_all,
            alpha_uncorrected=alpha_pre_all,
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
        C_T, C_P, actual_mode = rotor.compute_coefficients(
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
            'epsilon': blade.marker_epsilon,    # [lu] Gaussian projection width
            'twist': blade.marker_twist,        # [degrees]
            'active': blade.marker_active,
            'u_n': bem.u_n[idx_s:idx_e],        # [Δx/Δt] axial (induced) velocity
            'u_theta': bem.u_theta[idx_s:idx_e],# [Δx/Δt] tangential velocity
            'u_rel': bem.u_rel[idx_s:idx_e],    # [Δx/Δt] effective velocity
            'phi': bem.phi[idx_s:idx_e],        # [degrees] induced angle
            'alpha': bem.alpha[idx_s:idx_e],    # [degrees] effective AoA
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
        xp=None,
    ) -> None:
        """Initialize MultiRotorManager

        Args:
            domain_shape: (Nx, Ny, Nz) shared grid  [lu]
            xp: Array module — numpy or cupy (default: numpy).
                Controls allocation backend for _F_total accumulator.
        """
        self.domain_shape = domain_shape    # [lu]
        self.xp = xp if xp is not None else np   # array backend
        self.models: List['ActuatorLineModel'] = []
        self.names: List[str] = []

        # Pre-allocate accumulated body force array (on GPU if xp is cupy)
        Nx, Ny, Nz = domain_shape
        self._F_total = self.xp.zeros((3, Nx, Ny, Nz), dtype=self.xp.float64)
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
            2. Samples u(x) at its markers (GPU if xp is cupy)
            3. Computes BEM forces (always CPU)
            4. Spreads forces via Gaussian kernel (GPU if xp is cupy)

        All F_k are accumulated into a single body force field.
        When xp is cupy, all arrays remain on GPU throughout.

        Args:
            u_field: Velocity field, shape (3, Nx, Ny, Nz)  [Δx/Δt]
                     Can be xp array (GPU or CPU).
            dt: Timestep  [lt]
            external_F: Additional body force (e.g., from sponge layer)
                        shape (3, Nx, Ny, Nz)  [lattice force / lu³]
                        Must be same backend as self.xp.

        Returns:
            F_total: Combined body force field
                     shape (3, Nx, Ny, Nz)  [lattice force / lu³]
                     Same backend as self.xp (GPU if cupy).
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
    xp=None,
    sound_speed: Optional[float] = None,
) -> 'ActuatorLineModel':
    """Create a SINGLE ActuatorLineModel from config  (unchanged API)

    Expected Config Format:
        {
            "rotor": { ... },
            "gaussian_cutoff": 3.0,
            "rho_ref": 1.0,
            "coeff_mode": "auto"
        }

    Args:
        ...
        xp: Array module — numpy or cupy (default: numpy).
            Passed to ActuatorLineModel for GPU dispatch.
    """
    from src.actuator.rotor import Rotor  # local import to avoid circular

    rotor_cfg = config.get('rotor', {})

    if 'grid' not in rotor_cfg:
        rotor_cfg['grid'] = {}
    if 'dx' not in rotor_cfg['grid']:
        rotor_cfg['grid']['dx'] = dx_phys   # [m/lu]

    # ε-taper controls are specified at the actuator_line level; forward them
    # into rotor_cfg so Rotor.from_config sees them (a per-rotor override placed
    # directly under 'rotor' takes precedence).
    for _eps_key in ('epsilon_mode', 'epsilon_tip_factor', 'epsilon_taper_start'):
        if _eps_key in config and _eps_key not in rotor_cfg:
            rotor_cfg[_eps_key] = config[_eps_key]

    rotor_phys = Rotor.from_config(rotor_cfg)
    rotor_lu = rotor_phys.to_lattice_units(
        length_scale=dx_phys, time_scale=dt_phys
    )

    resolved_mode = coeff_mode if coeff_mode != 'auto' \
        else config.get('coeff_mode', 'auto')

    model = ActuatorLineModel(
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
        xp=xp,
        sound_speed=sound_speed,
    )

    # Prandtl tip/root loss
    prandtl = config.get('prandtl_loss', False)
    if isinstance(prandtl, dict):
        model.prandtl_loss = prandtl.get('enabled', True)
        model._prandtl_tip = prandtl.get('tip', True)
        model._prandtl_root = prandtl.get('root', True)
        # eps_offset True (default) = legacy R_tip_eff = R - ε_tip;
        # False = standard R_tip_eff = R_tip (BEMT-consistent, ε-decoupled).
        model._prandtl_eps_offset = prandtl.get('eps_offset', True)
    else:
        model.prandtl_loss = bool(prandtl)

    # Smearing (viscous-core) correction (bool-or-dict; default off → inert).
    ec = config.get('eps_correction', False)
    if isinstance(ec, dict):
        model._eps_corr = ec.get('enabled', True)
        model._eps_corr_target = ec.get('target', 'inviscid')
        model._eps_opt_factor = ec.get('eps_opt_factor', 0.25)
        model._eps_corr_relax = ec.get('relax', 1.0)
        model._eps_corr_method = ec.get('method', 'dag')  # "dag" | "kleine"
        model._kleine_wake_mode = ec.get('wake', 'straight')  # "straight" | "free"
        model._kleine_wake_nw = int(ec.get('n_w', 50))
        # Phase 2 perf: rebuild the free-wake influence matrix every N steps
        # (1 = every step, exact/bit-identical; >1 = cheaper approximation).
        model._kleine_rebuild_every = max(1, int(ec.get('rebuild_every', 1)))
    else:
        model._eps_corr = bool(ec)

    # Velocity sampler mode (A/B study — patch_notes/almlbm_sampler_ab/).
    # dict: {"mode": ..., "eps_r_factor": ...}; or shorthand string "point".
    # Absent / "gaussian" → bit-identical baseline (§6 path).
    samp = config.get('sampling', None)
    if isinstance(samp, dict):
        model._sampling_mode = samp.get('mode', 'gaussian')
        model._sampling_eps_r_factor = samp.get('eps_r_factor', 0.5)
    elif isinstance(samp, str):
        model._sampling_mode = samp

    return model


def create_multi_rotor_from_config(
    config: dict,
    domain_shape: Tuple[int, int, int],
    nu_lattice: float,
    polar_query: Callable,
    dx_phys: float = 1.0,
    dt_phys: float = 1.0,
    u_inf_lu: Optional[float] = None,
    coeff_mode: str = 'auto',
    xp=None,
    sound_speed: Optional[float] = None,
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
    manager = MultiRotorManager(domain_shape=domain_shape, xp=xp)

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
        'epsilon_mode': config.get('epsilon_mode', 'default'),
        'epsilon_tip_factor': config.get('epsilon_tip_factor', 1.0),
        'epsilon_taper_start': config.get('epsilon_taper_start', 0.7),
        'sampling': config.get('sampling', None),
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
            'epsilon_mode': rotor_entry.get(
                'epsilon_mode', shared_defaults['epsilon_mode']
            ),
            'epsilon_tip_factor': rotor_entry.get(
                'epsilon_tip_factor', shared_defaults['epsilon_tip_factor']
            ),
            'epsilon_taper_start': rotor_entry.get(
                'epsilon_taper_start', shared_defaults['epsilon_taper_start']
            ),
            'sampling': rotor_entry.get('sampling', shared_defaults['sampling']),
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
            xp=xp,
            sound_speed=sound_speed,
        )

        manager.add_model(al_model, name=name)

    return manager