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
import os

import numpy as np

try:
    import cupy as cp                      # optional GPU backend (BEM residency)
except Exception:                          # pragma: no cover — CPU-only env
    cp = None

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
        # Batched polar lookup caches (see _lookup_cl_cd): per-blade marker→airfoil
        # grouping and per-name resolved query functions. Both are static over a
        # run; caching removes the per-marker Python dispatch that dominated the
        # ALM BEM cost (~21 ms/sub-step → sub-ms on the HVAB C81 decks).
        self._polar_groups: Dict[int, Tuple[object, list]] = {}
        self._polar_qf_cache: Dict[str, object] = {}
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
        # Spanwise Γ smoothing passes before the trailed-vorticity correction. The
        # edge kernel gives w ∝ −Γ″ in the near field, so it AMPLIFIES marker-to-
        # marker Γ noise into a sawtooth (odd-even) instability when the markers are
        # as dense as ε (ε/Δr≈2 here). 0 = off (paper-literal, unstable on dense
        # ALM markers); 1-3 = light [0.25,0.5,0.25] passes break the sawtooth.
        self._eps_corr_smooth: int = 0
        # Correction method: "dag" (single-pass, current) | "kleine" (Kleine 2022
        # non-iterative linear solve, Phase 1 — patch_notes/kleine_smearing_correction/).
        self._eps_corr_method: str = "dag"
        # Dağ wake geometry: "straight" (Eq.17 semi-infinite trailed filaments, the
        # translating-WING kernel) | "prescribed_helix" (Dağ §3.2 ROTOR wake: a
        # prescribed helix per trailed vortex, pitch = local flow angle, `dag_helix_revs`
        # revolutions at `dag_helix_azimuth_deg` steps, Eq.22 vector Biot-Savart).
        self._dag_wake: str = "straight"
        self._dag_helix_revs: float = 2.0             # Dağ §3.2: 2 revolutions
        self._dag_helix_azimuth_deg: float = 2.0      # Dağ §3.2: 2° azimuthal steps
        # Pitch guard for hover: Dağ's pitch = local flow angle assumes wind-turbine
        # positive through-flow (φ>0 everywhere). In hover the tip-most marker samples
        # its own tip-vortex UPWASH (u_n<0) and the root sits in hub recirculation, so
        # the paper-literal pitch makes those filaments climb UPSTREAM of the disk —
        # unphysical (a shed filament is transported downstream by the mean wake).
        # "auto" (default) replaces non-positive edge pitches with the blade's mean
        # positive inflow; a float is an explicit floor w [Δx/Δt]; "off" = paper-literal.
        self._dag_helix_pitch_floor: object = "auto"
        # Endpoint Γ=0 closure (option 3): append virtual Γ=0 trailed-vortex nodes
        # at the root-cut and tip so the deficit model captures the full root/tip
        # shed (net trailed circulation → 0). Default off (byte-identical). Intended
        # ONLY for cell-centred distributions whose markers do NOT reach the
        # endpoints (uniform/cosine); the 'endpoint' distribution already has real
        # markers there, so closure must stay off for it (would double-place).
        self._eps_endpoint_closure: bool = False
        self._kleine_A: dict = {}                 # per-blade influence matrix cache
        self._kleine_gamma_prev: dict = {}        # per-blade Γⁿ⁻¹ (warm-start persist)
        self._kleine_w_prev: dict = {}            # per-blade w_corrⁿ⁻¹ (pure-Kleine freeze)
        self._kleine_fallback_count: int = 0      # stiff/non-finite Kleine solves (diagnostics)
        self._dag_w_prev: dict = {}               # per-blade w_corrⁿ⁻¹ (Dağ relaxation)
        self._dag_helix_rings: dict = {}          # per-blade prescribed-helix nodes (VTK export)
        self._dag_helix_M: dict = {}              # per-blade cached (B@E) operator (fast mode)
        self._dag_helix_steps: int = -1           # helix rebuild-cadence step counter
        # Kleine wake model: "straight" (Phase 1 semi-infinite, default) |
        # "free" (Phase 2 convected free-vortex wake → captures tip rollup).
        self._kleine_wake_mode: str = "straight"
        # Free-wake discretization (Dağ & Sørensen 2020 §: 2 revolutions of wake,
        # shed at 2° azimuthal intervals). n_w = explicit ring-count override
        # (None → computed from wake_revs & the 2° interval); shed one ring every
        # `wake_azimuth_deg` of rotation (NOT every timestep) to match Dağ.
        self._kleine_wake_nw = None               # ring-count override (None=auto)
        self._kleine_wake_revs: float = 2.0       # wake length [revolutions] (Dağ: 2)
        self._kleine_wake_azimuth_deg: float = 2.0  # shed interval [deg] (Dağ: 2°)
        # Spanwise markers that SHED the free wake (Q2). "all"=every marker
        # (default, byte-identical); "tip"=outermost active marker only (single
        # trailing tip filament — the intended minimal tip-vortex model); int N=
        # outermost N active; float f∈(0,1)=markers with r/R≥f; list=explicit idx.
        self._kleine_wake_markers = "all"
        self._kleine_shed_idx = None              # cached shed-marker indices
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
        # Ring sampling (Natelson ②): N sensors on a circle in the section plane,
        # radius ring_r_factor·ε. Only used when _sampling_mode == "ring".
        self._sampling_ring_n: int = 20
        self._sampling_ring_r_factor: float = 1.0

        # Swept-tip aero mode (patch_notes/alm_tip_sweep_refine, Step 2).
        # False (default) = legacy aero-only: u_aero = u_rel·cosΛ (cos²Λ dynamic
        # pressure), α unchanged → byte-identical to all existing sweep runs.
        # True = resolve the velocity into the plane normal to the swept LE
        # (only the in-plane tangential component × cosΛ; axial u_n is already
        # ⊥ the rotor-plane LE), giving V_n, φ_n, α_n. See _le_normal_resolve.
        self._sweep_alpha_normal: bool = False

        # Anisotropic Gaussian force projection (Natelson 2026 Eq. 7; ① Step A).
        # None → isotropic (default, byte-identical). dict{'c','t','r'} → per-axis
        # width factors relative to the isotropic marker ε (chord/thickness/radial).
        self._aniso: Optional[dict] = None

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

        # --- Optional per-phase profiler (env ALM_PROFILE) — GPU-synced wall clock
        # per phase, running mean printed every 100 calls. Inert unless enabled. ---
        _pf = getattr(self, '_alm_pf', None)
        if _pf is None:
            import os as _os
            # ALM_PROFILE_BEM adds a freewake/solve/polar/other breakdown of the
            # bem phase (P1 measurement: locate the free-wake bottleneck). Implies
            # ALM_PROFILE so the outer phase timing + print are active.
            self._pf_bem = ('ALM_PROFILE_BEM' in _os.environ)
            _pf = self._alm_pf = ('ALM_PROFILE' in _os.environ) or self._pf_bem
            self._pf_acc = {'sample': 0.0, 'bem': 0.0, 'spread': 0.0, 'n': 0,
                            'freewake': 0.0, 'solve': 0.0, 'polar': 0.0}
            self._pf_shapes = None
        if _pf:
            import time as _tm

            def _sync():
                if getattr(xp, 'cuda', None) is not None:
                    xp.cuda.Stream.null.synchronize()
            _sync(); _t = [_tm.perf_counter()]

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
            # A/B alternatives (point/aniso/mask_disk/ring) — patch_notes/*.
            # "ring" (Natelson) needs the per-marker section-plane frame (ê_c, ê_t).
            _ec = _et = None
            if self._sampling_mode == "ring":
                _ec, _et, _ = self.rotor.get_all_marker_aero_frame()
            u_markers = sample_velocity_alt(
                self._sampling_mode, u_field, positions, epsilon_all,
                xp=xp, n_cut=self.n_cut,
                hub=np.asarray(self.rotor.hub_center, dtype=np.float64),
                axis=np.asarray(self.rotor.rotation_axis, dtype=np.float64),
                radius=float(self.rotor.radius),
                eps_r_factor=self._sampling_eps_r_factor,
                ec=_ec, et=_et,
                ring_n=self._sampling_ring_n,
                ring_r_factor=self._sampling_ring_r_factor,
            )  # (N_total, 3) [Δx/Δt] — always numpy (CPU)

        # --- Phase 2 free-wake: convect + shed (before BEM uses it) ---
        if (self._eps_corr and self._eps_corr_method == "kleine"
                and self._kleine_wake_mode == "free"):
            self._convect_and_shed_wake(u_field, positions, dt, xp)

        # --- Step 4-7: BEM force calculation (CPU, small arrays) ---
        if _pf: _sync(); _t.append(_tm.perf_counter())   # ← sample+convect done

        bem_result = self._compute_bem_forces(u_markers)
        self._last_bem_result = bem_result

        # --- Step 8: Project to global frame (CPU, small arrays) ---
        F_global = self.rotor.project_all_forces(
            bem_result.F_n, bem_result.F_theta
        )  # (N_total, 3) [lattice force] — numpy (CPU)
        self._last_forces_global = F_global
        if _pf: _sync(); _t.append(_tm.perf_counter())   # ← BEM + project done

        # --- Step 9: Gaussian spreading ---
        # GPU path: F_grid allocated and filled on GPU, no CPU transfer
        self._F_grid[:] = 0.0  # Reset (works for both numpy and cupy)
        if external_F is not None:
            self._F_grid[:] = external_F  # Start from external force

        _radial_trunc = None
        if getattr(self, '_radial_trunc', False):
            blade0 = self.rotor.blades[0]
            _r_root = float(blade0.marker_r[0] - blade0.marker_dr / 2)
            _radial_trunc = {
                'axis': np.asarray(self.rotor.rotation_axis, dtype=np.float64),
                'hub': np.asarray(self.rotor.hub_center, dtype=np.float64),
                'r_root': _r_root,
                'r_tip': float(self.rotor.radius),
            }

        # Anisotropic Gaussian (Natelson Eq. 7): per-marker blade-local frame and
        # per-axis widths (factor × isotropic ε). None → isotropic path.
        _aniso = None
        if self._aniso is not None:
            ec, et, er = self.rotor.get_all_marker_aero_frame()
            fc = self._aniso['c']; ft = self._aniso['t']; fr = self._aniso['r']
            _aniso = {
                'ec': ec, 'et': et, 'er': er,
                'eps_c': fc * epsilon_all, 'eps_t': ft * epsilon_all,
                'eps_r': fr * epsilon_all,
            }

        spread_forces_to_grid_gpu(
            self.domain_shape,
            positions,
            F_global,
            epsilon_all,
            xp=xp,
            marker_active=active_all,
            n_cut=self.n_cut,
            F_grid=self._F_grid,
            radial_trunc=_radial_trunc,
            aniso=_aniso
        )

        if _pf:
            _sync(); _t.append(_tm.perf_counter())
            a = self._pf_acc
            a['sample'] += _t[1] - _t[0]
            a['bem'] += _t[2] - _t[1]
            a['spread'] += _t[3] - _t[2]
            a['n'] += 1
            if a['n'] % 100 == 0:
                n = a['n']
                print(f"[ALM prof] n={n}  sample={a['sample']/n*1e3:6.1f}  "
                      f"bem={a['bem']/n*1e3:6.1f}  spread={a['spread']/n*1e3:6.1f}  "
                      f"ms/call (GPU-synced, per fine sub-step)")
                if self._pf_bem:
                    other = a['bem'] - a['freewake'] - a['solve'] - a['polar']
                    print(f"[ALM prof.bem] freewake={a['freewake']/n*1e3:6.1f}  "
                          f"solve={a['solve']/n*1e3:6.1f}  "
                          f"polar={a['polar']/n*1e3:6.1f}  "
                          f"other={other/n*1e3:6.1f} ms/call  [{self._pf_shapes}]")

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
        the corrected angle of attack.  Numerics are identical to the original
        per-marker loop (active & u_rel≥1e-10 markers only; others remain 0).

        Fast path: markers are grouped by airfoil (static → cached per blade)
        and each group is looked up in ONE vectorized call when the resolved
        query function is batch-capable (`supports_batch`, e.g. the C81 decks —
        _C81Table broadcasts arrays through a single RegularGridInterpolator
        call, bit-identical to scalar queries).  This removes the ~2×192
        Python polar_query dispatches per fine sub-step that dominated the ALM
        BEM cost (profiled 21 ms/call on HVAB).  Scalar-only query functions
        (CSV / NeuralFoil databases) keep the original per-marker loop.
        """
        active = blade.marker_active
        CL = np.zeros(n, dtype=np.float64)
        CD = np.zeros(n, dtype=np.float64)
        valid = np.asarray(active, dtype=bool) & (np.asarray(u_rel) >= 1e-10)
        if not valid.any():
            return CL, CD

        # --- resolve marker→airfoil groups and query functions (cached) -----
        # ALM_POLAR_BATCH=0 forces the original per-marker dispatch (A/B tool).
        groups = None                                   # list of (qf, idx array)
        batch_on = os.environ.get('ALM_POLAR_BATCH', '1') != '0'
        if not batch_on:
            pass
        elif self._multi_airfoil:
            mgr = getattr(self.polar_query, 'manager', None)
            if mgr is not None:
                key = id(blade)
                cached = self._polar_groups.get(key)
                names = blade.marker_airfoil
                if cached is not None and cached[0] is names:
                    groups = cached[1]
                else:
                    arr = np.asarray(names, dtype=object)
                    groups = []
                    for nm in dict.fromkeys(arr.tolist()):  # unique, stable order
                        qf = self._polar_qf_cache.get(nm)
                        if qf is None:
                            qf = mgr.get_query(nm)
                            self._polar_qf_cache[nm] = qf
                        groups.append((qf, np.flatnonzero(arr == nm)))
                    self._polar_groups[key] = (names, groups)
        else:
            groups = [(self.polar_query, np.arange(n))]

        if groups is not None:
            slow = []                                   # scalar-only remainders
            for qf, gidx in groups:
                idx = gidx[valid[gidx]]
                if idx.size == 0:
                    continue
                if not getattr(qf, 'supports_batch', False):
                    slow.append((qf, idx))
                    continue
                pass_mach = Mach is not None and (not self._multi_airfoil
                                                  or getattr(qf, 'wants_mach', False))
                if pass_mach:
                    cl, cd = qf(alpha_deg[idx], Re[idx], mach=Mach[idx])
                else:
                    cl, cd = qf(alpha_deg[idx], Re[idx])
                CL[idx] = cl
                CD[idx] = cd
            for qf, idx in slow:                        # unchanged scalar path
                pass_mach = Mach is not None and (not self._multi_airfoil
                                                  or getattr(qf, 'wants_mach', False))
                for j in idx:
                    if pass_mach:
                        CL[j], CD[j] = qf(float(alpha_deg[j]), float(Re[j]),
                                          mach=float(Mach[j]))
                    else:
                        CL[j], CD[j] = qf(float(alpha_deg[j]), float(Re[j]))
            return CL, CD

        # --- legacy path (batch disabled, or multi without manager handle) --
        # Original per-marker dispatch, byte-for-byte.
        for j in range(n):
            if not valid[j]:
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
        """Missing induced velocity from the Gaussian-smeared trailed wake.

        Discrete trailed-vortex de-induction (Dağ & Sørensen 2020, Eq. 17-18): a
        finite-ε ALM sheds trailed vortices with a Lamb-Oseen core = ε that
        under-induce the downwash (worst at the tip → over-loading).  This adds
        back the deficit between the ideal (point-vortex) and the ε-smeared
        induction; the caller re-queries the polars at the corrected α.

            Γ(j)   = ½ c_j u_rel_j CL_j                bound circulation (caller, Eq.19)
            Γw(j)  = Γ(j) − Γ(j−1)                     wake vortex on panel EDGE j (Eq.18)
            w(i)   = +(1/4π) Σ_{j=1}^{N+1} Γw(j)/d_ij · K(d_ij)          (Eq.17)
            d_ij   = r_i − r_edge_j                    signed distance (Eq.20)
            K = exp(−(d/ε)²)                           target="inviscid"
              = exp(−(d/ε)²) − exp(−(d/ε_opt)²)        target="opt" (ε_opt=factor·c)

        The N markers bound N+1 panel EDGES (root, interior midpoints, tip); the
        trailed vortices live on those edges with strength = the jump in bound
        circulation, Γ padded with 0 outside the blade.  So the OUTERMOST edge
        sheds the full tip vortex (Γw = −Γ_tip) and the innermost the root
        vortex → Σ Γw = 0 (circulation conserved).  The pre-2026-07 code took
        dΓ/dr at the markers instead, which silently DROPPED these two end
        vortices (net ΣΓw ≠ 0) and made the tip correction ~10-30× too weak — the
        reason Dağ had almost no effect on the light grid.  Sign +(1/4π) is the
        standard lifting-line downwash: the caller adds w to u_n (Eq.21), which
        REDUCES the over-predicted tip α.  All lattice units; w_corr [Δx/Δt].
        See patch_notes/alm_dag_edge_fix/.  (The old marker/­endpoint_closure
        paths are superseded — the edge form always closes the ends.)
        """
        N = len(r)
        inv4pi = 1.0 / (4.0 * np.pi)
        use_opt = (self._eps_corr_target == "opt")

        # --- N+1 trailed wake vortices on the panel EDGES (Dağ Eq. 17-18) -------
        drv = np.full(N, float(dr)) if np.ndim(dr) == 0 else np.asarray(dr, float)
        edges = np.empty(N + 1)
        edges[1:-1] = 0.5 * (r[:-1] + r[1:])            # interior midpoints
        edges[0] = r[0] - 0.5 * drv[0]                  # root closure edge
        edges[-1] = r[-1] + 0.5 * drv[-1]              # tip  closure edge
        Gw = np.diff(np.concatenate(([0.0], Gamma, [0.0])))   # Γ(j)−Γ(j−1), len N+1
        eps_e = np.empty(N + 1)                          # ε on the edges
        eps_e[1:-1] = 0.5 * (eps[:-1] + eps[1:])
        eps_e[0] = eps[0]; eps_e[-1] = eps[-1]
        if use_opt:
            eps_opt = np.maximum(self._eps_opt_factor * chord, 1e-9)  # [lu]
            eps_opt_e = np.empty(N + 1)
            eps_opt_e[1:-1] = 0.5 * (eps_opt[:-1] + eps_opt[1:])
            eps_opt_e[0] = eps_opt[0]; eps_opt_e[-1] = eps_opt[-1]

        # --- Biot-Savart deficit sum at each marker (Eq. 17), +1/(4π) ----------
        w = np.zeros(N)                                 # [Δx/Δt]
        for i in range(N):
            d = r[i] - edges                            # signed distance (Eq. 20)
            far = np.abs(d) > 1e-9                       # skip a coincident edge
            safe_d = np.where(far, d, 1.0)
            if use_opt:
                K = np.exp(-(d / eps_e) ** 2) - np.exp(-(d / eps_opt_e) ** 2)
            else:
                K = np.exp(-(d / eps_e) ** 2)
            kernel = np.where(far, K / safe_d, 0.0)
            w[i] = inv4pi * np.sum(Gw * kernel)         # +1/(4π): downwash at the tip
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
            edge_operator)
        from .polar_slope import lift_curve_slope_batch, cubic_slope_4pt
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
                # Axial DOWNWASH projection direction = OPPOSITE the thrust
                # (−thrust_axis). The earlier −sign(ω)·rotation_axis is the THRUST
                # side when thrust = −rotation_axis (e.g. HVAB: thrust=[−1,0,0]),
                # which FLIPS the correction sign (projects the induced velocity onto
                # the thrust direction) — the same root cause fixed in
                # _dag_prescribed_helix_wcorr, confirmed from the helix wake VTP.
                # Only the straight-limit was ever validated (test_kleine_freewake_
                # edge), not this projection on the real rotor. Fall back to the old
                # form if thrust_axis is unset. See patch_notes/alm_dag_edge_fix/.
                _td = getattr(rotor, 'thrust_axis', None)
                if _td is not None:
                    axis = -np.asarray(_td, dtype=np.float64)
                    axis = axis / np.sqrt(axis @ axis)
                else:
                    axis = -np.sign(rotor.omega) * np.asarray(
                        rotor.rotation_axis, dtype=np.float64)
                # Q2: free wake may be shed from a subset of markers (e.g. tip
                # only). The wake rings then hold len(shed_idx) points, so the
                # ε and trailed-vorticity operator must be restricted to match.
                # Trailed vortices on the N+1 panel EDGES (Dağ Eq.17-18): E maps
                # Γ → edge circulations incl. the tip vortex −Γ_tip; wake.rings
                # hold the shed-edge positions (see _convect_and_shed_wake).
                # tip-only shedding keeps just the tip closure edge. A = B_edge @ E.
                E, _, eps_edge = edge_operator(r, eps, dr)          # (n+1,n),(n+1,)
                e_idx = self._shed_edge_idx(blade)                  # edge subset|None
                E_used = E if e_idx is None else E[e_idx, :]        # (src, n)
                eps_edge_src = eps_edge if e_idx is None else eps_edge[e_idx]
                _pb = getattr(self, '_pf_bem', False)
                if _pb:
                    import time as _tm
                    _t0 = _tm.perf_counter()
                    if self._pf_shapes is None:
                        _L = len(wake.rings); _ne = wake.rings[0].shape[0]
                        self._pf_shapes = (
                            f"nb={rotor.n_blades} nc={len(ctrl3d)} ne={_ne} "
                            f"L={_L} ns_full={(_L) * _ne}")
                # ── Biot-Savart influence build (freewake #1 BEM cost, ~87 ms).
                # GPU-resident path (P2/S2): upload the wake geometry (tiny) and
                # run the xp-agnostic freewake_influence on-device, D2H only B
                # (nc×ne). Numerically == CPU (verified 5.6e-16). cp.asnumpy
                # blocks so the freewake timer captures the GPU compute. Full
                # ring residency (no per-call H2D) is S4. ALM_FREEWAKE_GPU=0 →
                # CPU reference (A/B & non-CUDA fallback). ──
                if cp is not None and os.environ.get(
                        'ALM_FREEWAKE_GPU', '1') != '0':
                    B = cp.asnumpy(freewake_influence(
                        cp.asarray(ctrl3d),
                        [cp.asarray(rg) for rg in wake.rings],
                        cp.asarray(eps_edge_src), cp.asarray(axis)))
                else:
                    B = freewake_influence(ctrl3d, wake.rings, eps_edge_src, axis)
                A = B @ E_used                                      # (n, n)
                self._kleine_A_free[k] = A
                if _pb:
                    self._pf_acc['freewake'] += _tm.perf_counter() - _t0
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
            # Vectorized lift-curve slope (Kleine §6.1 cubic ∂C_l/∂α). The 4
            # cubic-fit offsets (±1.5δ, ±0.5δ) are looked up through cl_eval's
            # grouped/Mach-aware BATCH path and fit with a SINGLE multi-RHS
            # np.polyfit, replacing lift_curve_slope_batch's per-marker loop
            # (4 scalar polar dispatches + np.polyfit PER marker → 768 scalar
            # lookups + 192 polyfits/sub-step for nb=4·n=48). That per-marker
            # loop was the bulk of the Kleine 'solve' cost (profiled ~99 ms/
            # sub-step; _lookup_cl_cd already batched the same C81 queries and
            # documents them bit-identical to scalar → numerically equivalent).
            # ALM_DCL_BATCH=0 restores the scalar reference (A/B & bit check).
            if os.environ.get('ALM_DCL_BATCH', '1') == '0':
                return lift_curve_slope_batch(
                    self.polar_query, a_deg, Re_d, active,
                    multi_airfoil=multi, marker_airfoil=names,
                    mach=Mach_d, delta_deg=1.0)
            offs = (-1.5, -0.5, 0.5, 1.5)                    # × δ=1.0 deg
            cls_list = [cl_eval(a_deg + o) for o in offs]    # 4 batched C_l
            _xp = cp if (cp is not None and isinstance(cls_list[0], cp.ndarray)) \
                else np
            cls = _xp.stack(cls_list)                        # (4, n)
            if _xp is np:
                # CPU reference: batched cubic polyfit (bit-identical, patch 04).
                xs = np.array(offs) * (np.pi / 180.0)        # [rad], x=0 at α
                slope = np.polyfit(xs, cls, 3)[2]            # p'(0)=c [1/rad]
            else:
                # GPU-resident path: cupy has no polyfit → closed-form Fornberg
                # derivative of the SAME cubic (== polyfit, <1e-12).
                slope = cubic_slope_4pt(cls, delta_deg=1.0, xp=_xp)
            return _xp.where(active, slope, 0.0)

        _pb2 = getattr(self, '_pf_bem', False)
        if _pb2:
            import time as _tm2
            _ts = _tm2.perf_counter()
        try:
            _, _, Gnew, w = correct_noniterative(
                r, chord, dr, eps, u_n, u_tan, twist, cl_eval, dcl_eval,
                Gprev, A=A, active=active)
            if _pb2:
                self._pf_acc['solve'] += _tm2.perf_counter() - _ts
            # Safety net: a non-finite, or physically-unreasonable (downwash
            # exceeding ~half the tangential scale) solve signals an
            # ill-conditioned/unstable fixed point → fall back to Dağ this step
            # (do NOT persist the bad Γ; keep the previous Γ for warm-start).
            scale = float(np.max(np.abs(u_tan))) + 1e-30
            if (not np.all(np.isfinite(w))) or (float(np.max(np.abs(w))) > 0.5 * scale):
                raise np.linalg.LinAlgError("Kleine correction out of bounds")
            # Same w ∝ −Γ″ sawtooth risk as Dağ (shared edge operator): lightly
            # smooth both the persisted warm-start Γ and the applied correction so
            # the odd-even mode cannot feed back through Gⁿ⁻¹ (opt-in, 0 = literal).
            if self._eps_corr_smooth > 0:
                Gnew = self._smooth_active(Gnew, active, self._eps_corr_smooth)
                w = self._smooth_active(w, active, self._eps_corr_smooth)
            self._kleine_gamma_prev[k] = Gnew
            self._kleine_w_prev[k] = w
            return w
        except np.linalg.LinAlgError:
            # PURE Kleine: on a stiff / non-finite solve do NOT invoke Dağ — that
            # would mix two independent models and break variable control. Freeze
            # the previous Kleine correction (0 on the first step) and count the
            # event so the fallback rate is observable (self._kleine_fallback_count).
            self._kleine_fallback_count += 1
            w_prev = self._kleine_w_prev.get(k)
            return w_prev if w_prev is not None and len(w_prev) == n else np.zeros(n)

    def _shed_idx(self, blade):
        """Resolve the spanwise markers that shed the free wake (Q2).

        Returns marker indices (into a blade's marker array), or None for 'all'
        (every marker — byte-identical legacy behaviour). 'tip'=outermost active
        marker; int N=outermost N active; float f=markers with r/R≥f; list=idx.
        Markers are identical across blades, so the result is shared.
        """
        spec = self._kleine_wake_markers
        if spec is None or spec == "all":
            return None
        active = np.where(blade.marker_active)[0]
        if spec == "tip":
            return active[-1:]
        if isinstance(spec, bool):
            return None
        if isinstance(spec, (int, np.integer)):
            return active[-int(spec):]
        if isinstance(spec, float):
            rR = np.asarray(blade.marker_r, float) / float(blade.r_tip)
            return np.where(blade.marker_active & (rR >= spec))[0]
        return np.asarray(spec, dtype=int)

    def _shed_edge_idx(self, blade):
        """Edge indices (into the N+1 panel edges) that shed the free wake.

        None = all N+1 edges. 'tip' = ONLY the tip closure edge (index N, the
        tip vortex −Γ_tip). int K = K outermost edges; float f = edges with
        r_edge/R ≥ f; list = explicit edge indices. Mirrors `_shed_idx` but on
        the edge grid used by the edge-based trailed-vorticity operator E.
        """
        from .smearing_correction import edge_operator
        spec = self._kleine_wake_markers
        n = len(blade.marker_r)
        if spec is None or spec == "all" or isinstance(spec, bool):
            return None
        if spec == "tip":
            return np.array([n])                      # tip closure edge (index N)
        if isinstance(spec, (int, np.integer)):
            return np.arange(n + 1 - int(spec), n + 1)
        if isinstance(spec, float):
            _, r_edge, _ = edge_operator(
                np.asarray(blade.marker_r, float), np.ones(n), 1.0)
            return np.where(r_edge / float(blade.r_tip) >= spec)[0]
        return np.asarray(spec, dtype=int)

    @staticmethod
    def _edge_positions_3d(pos):
        """N marker positions (N,3) → N+1 panel-edge positions (N+1,3):
        interior midpoints + extrapolated root/tip closure edges (the trailed
        vortices are shed from these edges, matching E)."""
        pos = np.asarray(pos, dtype=np.float64)
        ed = np.empty((pos.shape[0] + 1, pos.shape[1]))
        ed[1:-1] = 0.5 * (pos[:-1] + pos[1:])         # interior midpoints
        ed[0] = 1.5 * pos[0] - 0.5 * pos[1]           # pos[0] − ½(pos[1]−pos[0])
        ed[-1] = 1.5 * pos[-1] - 0.5 * pos[-2]        # tip closure edge
        return ed

    def _kleine_wake_params(self):
        """(shed_every, n_w) for the Dağ-style free wake.

        shed_every = timesteps per `wake_azimuth_deg` of rotation → one ring is
        shed every 2° (Dağ), NOT every timestep. n_w = rings kept = `wake_revs`
        worth of 2°-steps (Dağ 2 rev @ 2° = 360 rings), unless n_w was given
        explicitly (model-selection override, e.g. the tip-only n_w=2 case).
        """
        spr = 2.0 * np.pi / (abs(self.rotor.omega) + 1e-30)      # steps per rev
        shed_every = max(1, int(round(spr * self._kleine_wake_azimuth_deg / 360.0)))
        if self._kleine_wake_nw is not None:
            nw = max(2, int(self._kleine_wake_nw))
        else:
            nw = max(2, int(round(self._kleine_wake_revs
                                  * 360.0 / self._kleine_wake_azimuth_deg)))
        return shed_every, nw

    def _dag_prescribed_helix_wcorr(self, k, blade, Gamma, u_n, u_theta, dr):
        """Dağ §3.2 ROTOR correction: prescribed HELICAL trailed wake (Eq. 22).

        Each trailed vortex (panel edge) forms a helix of constant radius: pitch =
        the local flow angle φ (Dağ: "helical pitch … equal to the relative flow
        angle at the position where the trailing vortex is released"), spanning
        `_dag_helix_revs` revolutions in `_dag_helix_azimuth_deg` steps. The deficit
        induced velocity is the same edge-based Biot-Savart as the free wake
        (A = B_helix @ E), only the wake GEOMETRY is prescribed (analytic helix)
        instead of convected. Returns w_corr [Δx/Δt] (axial de-induction), same role
        as `_viscous_core_correction`. See patch_notes/alm_dag_edge_fix/.

        Fast mode (rebuild_every>1): the (B@E) operator is rebuilt only every N steps
        and reused in between — the helix geometry/pitch drifts slowly (same rationale
        as the free-wake influence-matrix throttle), while Γ is always current.
        """
        # Rebuild-cadence check (advance the counter once per step, at blade 0).
        rebuild_every = max(1, int(self._kleine_rebuild_every))
        if k == 0:
            self._dag_helix_steps += 1
        M = self._dag_helix_M.get(k)
        if not (M is None or rebuild_every == 1
                or self._dag_helix_steps % rebuild_every == 0):
            return M @ np.asarray(Gamma, dtype=np.float64)      # reuse cached operator

        from .smearing_correction import edge_operator, freewake_influence
        rotor = self.rotor
        r = np.asarray(blade.marker_r, dtype=np.float64)
        eps = np.asarray(blade.marker_epsilon, dtype=np.float64)
        n = len(r)
        E, r_edge, eps_edge = edge_operator(r, eps, dr)         # (n+1,n),(n+1,),(n+1,)

        ctrl3d = np.asarray(self._last_positions[k * n:(k + 1) * n], float)  # (n,3)
        edge3d = self._edge_positions_3d(ctrl3d)               # (n+1,3)
        hub = np.asarray(rotor.hub_center, dtype=np.float64)
        axis = np.asarray(rotor.rotation_axis, dtype=np.float64)
        axis = axis / np.sqrt(axis @ axis)
        omega = float(rotor.omega)
        rsign = np.sign(omega) if omega != 0 else 1.0

        # Local flow angle φ per edge = helical pitch (Dağ). u_tan = |ω|r − sign(ω)u_θ.
        u_tan = np.abs(omega) * r - rsign * u_theta
        phi = np.arctan2(u_n, u_tan)                            # (n,) [rad]
        phi_e = np.empty(n + 1)
        phi_e[1:-1] = 0.5 * (phi[:-1] + phi[1:]); phi_e[0] = phi[0]; phi_e[-1] = phi[-1]
        tanphi_e = np.tan(np.clip(phi_e, -1.4, 1.4))           # bound near ±90°

        # --- Hover pitch guard (see __init__._dag_helix_pitch_floor) ------------
        # Where the local pitch is non-positive (tip self-induced upwash, root
        # recirculation) the prescribed filament would climb upstream of the disk.
        # Replace ONLY those edges' pitch so the filament convects downstream at
        # the floor velocity w_floor; edges with valid (positive) pitch keep the
        # paper-literal value. Axial descent per unit ψ is r_e·tanφ_e = w/|ω|, so
        # the floor pitch at edge e is tanφ_floor = w_floor/(|ω|·r_e).
        pf = self._dag_helix_pitch_floor
        if pf is not None and pf != "off" and pf is not False:
            if isinstance(pf, (int, float)) and not isinstance(pf, bool):
                w_floor = float(pf)                            # explicit [Δx/Δt]
            else:                                              # "auto"
                act = np.asarray(blade.marker_active, dtype=bool)
                pos = u_n[act & (u_n > 0.0)]
                w_floor = float(pos.mean()) if pos.size else 0.0
            tanphi_floor = w_floor / np.maximum(np.abs(omega) * r_edge, 1e-30)
            tanphi_e = np.where(tanphi_e > 0.0, tanphi_e, tanphi_floor)

        # Optional shed subset (Q2): trail the helix only from a subset of edges,
        # e.g. wake_markers="tip" → ONLY the tip-vortex closure edge (index N). E is
        # restricted so A = B_subset @ E_subset; per-edge geometry follows suit.
        e_idx = self._shed_edge_idx(blade)
        if e_idx is not None:
            E = E[e_idx, :]; r_edge = r_edge[e_idx]; eps_edge = eps_edge[e_idx]
            edge3d = edge3d[e_idx]; tanphi_e = tanphi_e[e_idx]

        # Axial DOWNWASH direction = OPPOSITE the thrust. BOTH the wake descent AND
        # the induced-velocity projection use it. The 2026-07 first cut used
        # −sign(ω)·axis for both — but for the HVAB convention thrust = −axis, so
        # −sign(ω)·axis is the THRUST side, i.e. (a) the wake descended the WRONG way
        # (verified from the wake VTP: nodes climbed toward −thrust) and (b) the
        # induced velocity was projected onto the thrust direction → the tip w_corr
        # came out −(upwash). Using −thrust_axis fixes both: the wake descends on the
        # downwash side and w_corr>0 (added downwash) de-loads the tip like the
        # straight kernel. See patch_notes/alm_dag_edge_fix/.
        td = getattr(rotor, 'thrust_axis', None)
        if td is not None:
            downwash = -np.asarray(td, dtype=np.float64)
            downwash = downwash / np.sqrt(downwash @ downwash)
        else:
            downwash = rsign * axis                            # fallback (opposite the old −rsign·axis)
        descent = downwash                                     # wake convects on the downwash side
        proj = downwash                                        # project induced velocity onto downwash

        # Prescribed helix: rotate each edge back by −sign(ω)·ψ about (hub,axis) and
        # descend axially by r·tanφ·ψ. rings newest-first (ψ=0 at the blade).
        dpsi = np.radians(self._dag_helix_azimuth_deg)
        nw = self._kleine_wake_nw                              # ring-count override (n_w)
        if nw:
            n_ring = max(2, int(nw))
        else:
            n_ring = max(2, int(round(self._dag_helix_revs * 2.0 * np.pi / dpsi)) + 1)
        rel = edge3d - hub[None, :]                            # (n+1,3)
        axcomp = (rel @ axis)[:, None] * axis[None, :]          # axial part (rotation-invariant)
        perp = rel - axcomp                                     # radial part (rotates)
        cross = np.cross(np.broadcast_to(axis, perp.shape), perp)  # axis × perp
        rings = []
        for a in range(n_ring):
            psi = a * dpsi
            ang = -rsign * psi
            rot_perp = perp * np.cos(ang) + cross * np.sin(ang)   # Rodrigues (perp ⟂ axis)
            node = (hub[None, :] + axcomp + rot_perp
                    + descent[None, :] * (r_edge * tanphi_e * psi)[:, None])
            rings.append(node)

        self._dag_helix_rings[k] = rings                       # cache for VTK export
        B = freewake_influence(ctrl3d, rings, eps_edge, proj)   # (n, n+1 or subset)
        M = B @ E                                              # (n, n) operator
        self._dag_helix_M[k] = M                               # cache for fast mode
        return M @ np.asarray(Gamma, dtype=np.float64)          # (n,) [Δx/Δt]

    def _dag_straight_wake_viz(self, k, blade):
        """Build a VISUALIZATION of the straight-kernel trailed wake (Dağ Eq. 17:
        semi-infinite straight trailed vortices shed from each panel EDGE in the
        downwash direction). Stored in `_dag_helix_rings[k]` so the wake VTK export
        renders it — VIZ ONLY, it does NOT enter the (analytic, radial) correction.
        Drawn to a finite length (≈2R) for ParaView; contrast with the helix.
        """
        n = len(blade.marker_r)
        ctrl3d = np.asarray(self._last_positions[k * n:(k + 1) * n], float)
        edge3d = self._edge_positions_3d(ctrl3d)               # (n+1, 3)
        rotor = self.rotor
        td = getattr(rotor, 'thrust_axis', None)
        if td is not None:
            dw = -np.asarray(td, dtype=np.float64)
        else:
            dw = np.sign(rotor.omega) * np.asarray(rotor.rotation_axis, float)
        dw = dw / np.sqrt(dw @ dw)                             # downwash direction
        R = float(getattr(rotor, 'radius', np.max(blade.marker_r)))
        n_node = 25
        step = (2.0 * R) / (n_node - 1)                        # finite draw length ≈2R
        self._dag_helix_rings[k] = [edge3d + dw[None, :] * (a * step)
                                    for a in range(n_node)]     # straight filaments

    @staticmethod
    def _smooth_active(x, active, iters):
        """Light 3-point [0.25, 0.5, 0.25] spanwise smoothing over the ACTIVE markers
        (`iters` passes, endpoints held). The edge trailed-vorticity kernel gives
        w ∝ −Γ″ in the near field, so on dense ALM markers (ε/Δr≈2) it amplifies any
        marker-to-marker Γ ripple into a sawtooth (odd-even) instability. Smoothing Γ
        before the correction suppresses the Nyquist mode (×0.5 per pass) and breaks
        the feedback, without touching the smooth (physical) part of the loading.
        Inactive markers are left untouched. iters<=0 returns x unchanged.
        """
        if iters is None or int(iters) <= 0:
            return x
        x = np.asarray(x, dtype=np.float64).copy()
        idx = np.where(active)[0] if active is not None else np.arange(len(x))
        if len(idx) < 3:
            return x
        v = x[idx]
        for _ in range(int(iters)):
            vn = v.copy()
            vn[1:-1] = 0.25 * v[:-2] + 0.5 * v[1:-1] + 0.25 * v[2:]
            v = vn
        x[idx] = v
        return x

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
        shed_every, nw = self._kleine_wake_params()

        for k in range(rotor.n_blades):
            if self._kleine_wake.get(k) is None:
                self._kleine_wake[k] = FreeWake(nw)

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

        # Shed a new ring at the panel-EDGE positions ONLY every `shed_every`
        # steps = every `wake_azimuth_deg` (2°) of rotation, matching Dağ's 2°
        # azimuthal discretization (rings convect every step regardless). The
        # trailed vortices live on the N+1 edges (incl. tip/root closure), matching
        # the A = B_edge @ E build. Q2: shed only the selected edges (None = all).
        if self._kleine_wake_steps % shed_every == 0:
            e_idx = self._shed_edge_idx(rotor.blades[0])
            for k in range(rotor.n_blades):
                pos_k = positions[k * npb:(k + 1) * npb]
                edge_pos = self._edge_positions_3d(pos_k)          # (npb+1, 3)
                self._kleine_wake[k].shed(
                    edge_pos if e_idx is None else edge_pos[e_idx])

    # -----------------------------------------------------------------
    # §2.2 BEM Force Computation
    # -----------------------------------------------------------------

    @staticmethod
    def _le_normal_resolve(u_n, u_rel, phi_deg, twist_deg, cos_sweep):
        """Resolve the section velocity into the plane normal to the swept LE.

        Rotor tip sweep lies in the ROTOR plane, so the axial (through-disk)
        velocity u_n is always perpendicular to the LE and is NOT reduced; only
        the in-plane tangential component (u_rel·cos φ) is scaled by cosΛ. This
        is the independence/cross-flow principle for a rotor (not the translating-
        wing V·cosΛ). Returns (V_n, φ_n[deg], α_n[deg]); for Λ=0 (cos_sweep=1)
        it reduces to (u_rel, φ, α) up to floating-point round-off.
        """
        phi = np.radians(phi_deg)
        u_tan_n = u_rel * np.cos(phi) * cos_sweep      # tangential ⊥ LE [Δx/Δt]
        V_n = np.hypot(u_n, u_tan_n)                    # normal-plane speed
        phi_n = np.degrees(np.arctan2(u_n, u_tan_n))   # normal-plane flow angle
        alpha_n = twist_deg - phi_n                     # rotorcraft α (LE-normal)
        return V_n, phi_n, alpha_n

    def _compute_bem_forces(
        self,
        u_markers: 'npt.NDArray'
    ) -> BEMResult:
        """Compute BEM aerodynamic forces at all markers

        Physical Process — Leishman ROTORCRAFT (propeller) sign convention.
        Watanabe et al. Eq. 8-19 use the OPPOSITE wind-turbine convention; the
        two differ in the sign of α and of the drag term in F_n/F_θ. Our signs
        are the thrust-producing-rotor ones (correct for HVAB/HART2 hover), so
        do NOT "fix" them to Watanabe's without also flipping the twist sign.
            1. Decompose velocity: u_n (axial/normal), u_θ (tangential) [Eq. 8-9]
            2. Relative velocity:  u_rel = √(u_n² + (ωr - u_θ)²)         [Eq. 10]
            3. Flow angle:         φ = atan2(u_n, ωr - u_θ)              [Eq. 11]
            4. Angle of attack:    α = γ - φ   (twist - flow angle;
                                   rotor sign — Watanabe Eq. 12 is φ - γ)
            5. Reynolds number:    Re = u_rel · c_a / ν
            6. Airfoil lookup:     CL(Re, α), CD(Re, α)
            7. Lift/drag forces:   F_L = ½·ρ·u_rel²·c·Δr·CL              [Eq. 13]
                                   F_D = ½·ρ·u_rel²·c·Δr·CD              [Eq. 14]
            8. Normal/tangential:  F_n = FL·cosφ - FD·sinφ   (thrust)
                                   F_θ = FL·sinφ + FD·cosφ   (in-plane)
                                   (Watanabe Eq. 15-16 has the opposite
                                   drag-term signs, for a turbine.)

        Args:
            u_markers: Velocity at markers, shape (N_total, 3)  [Δx/Δt]

        Returns:
            BEMResult with all intermediate and final quantities
        """
        rotor = self.rotor
        n_total = rotor.total_markers
        n_per_blade = rotor.markers_per_blade

        # P1 bem breakdown timers (env ALM_PROFILE_BEM; inert otherwise).
        _pb = getattr(self, '_pf_bem', False)
        if _pb:
            import time as _tm

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
                swept = bool(np.any(blade.marker_sweep != 0.0))
            else:
                cos_sweep = 1.0
                swept = False
            if swept and self._sweep_alpha_normal:
                # LE-normal resolution (Step 2): V_n, φ_n, α_n. De-loads the
                # swept tip vs the legacy cos²Λ path (α reduced ~0.3–0.5°).
                u_aero, phi_aero, alpha_aero = self._le_normal_resolve(
                    u_n, u_rel, phi_deg, blade.marker_twist, cos_sweep)
            else:
                # Legacy aero-only path (byte-identical): cos_sweep=1 → u_rel.
                u_aero = u_rel * cos_sweep   # [Δx/Δt] velocity normal to LE
                phi_aero = phi_deg
                alpha_aero = alpha_deg
            # Diagnostics reflect the aero (LE-normal when enabled) kinematics.
            phi_all[idx_start:idx_end] = phi_aero
            alpha_all[idx_start:idx_end] = alpha_aero

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

            # --- Airfoil polar lookup (α = α_n when LE-normal on; V_n → Mach/Re) ---
            if _pb: _t0 = _tm.perf_counter()
            CL, CD = self._lookup_cl_cd(alpha_aero, Re, u_aero, Mach,
                                        blade, n_per_blade)
            if _pb: self._pf_acc['polar'] += _tm.perf_counter() - _t0

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
                    # Dağ correction (Dağ & Sørensen 2020) via the standard
                    # one-update-per-step RELAXATION scheme (Martínez-Tossas &
                    # Meneveau 2019, as reviewed in Kleine 2022 §1): the applied
                    # correction is a relaxation-weighted average of the previous
                    # and the current correction velocity,
                    #     w = relax·w_new + (1−relax)·w_prev,
                    # w_new from the current-step Γ. The fixed point is the FULL
                    # self-consistent Dağ correction (w=w_new); relax only sets
                    # the convergence rate / stability (Kleine: relax→1 can go
                    # unstable, small relax is slower). The pre-2026-07 code used
                    # `relax·w_new` alone (no (1−relax)·w_prev), which converged
                    # to relax·full = UNDER-applied. Steady hover → converges to
                    # the self-consistent correction. See patch_notes/alm_dag_edge_fix/.
                    Gamma = np.where(active, 0.5 * chord * u_aero * CL, 0.0)  # [lu²/lt]
                    # Break the w ∝ −Γ″ sawtooth (odd-even) instability on dense ALM
                    # markers by lightly smoothing Γ before the trailed-vorticity
                    # correction (opt-in; 0 = paper-literal). See _smooth_active.
                    Gamma = self._smooth_active(Gamma, active, self._eps_corr_smooth)
                    if self._dag_wake == "prescribed_helix":
                        # Dağ §3.2 rotor: prescribed helical wake (Eq. 22).
                        w_new = self._dag_prescribed_helix_wcorr(
                            k, blade, Gamma, u_n, u_theta, dr)
                    else:
                        # Dağ Eq. 17 straight semi-infinite (translating-wing) kernel.
                        w_new = self._viscous_core_correction(
                            blade.marker_r, blade.marker_epsilon, Gamma, chord, dr)
                        self._dag_straight_wake_viz(k, blade)   # wake VTK viz only
                    w_prev = self._dag_w_prev.get(k)
                    if w_prev is None or len(w_prev) != len(w_new):
                        w_prev = np.zeros_like(w_new)
                    rlx = self._eps_corr_relax
                    w_corr = rlx * w_new + (1.0 - rlx) * w_prev
                    self._dag_w_prev[k] = w_corr
                u_n = u_n + np.where(active, w_corr, 0.0)        # added downwash
                u_rel, phi_deg, alpha_deg = \
                    rotor.recompute_velocity_triangle(k, u_n, u_theta)
                if swept and self._sweep_alpha_normal:
                    u_aero, phi_aero, alpha_aero = self._le_normal_resolve(
                        u_n, u_rel, phi_deg, blade.marker_twist, cos_sweep)
                else:
                    u_aero = u_rel * cos_sweep                   # sweep-corrected
                    phi_aero = phi_deg
                    alpha_aero = alpha_deg
                Re = u_aero * chord / (self.nu + 1e-30)
                if self._polar_wants_mach and self.a_phys:
                    Mach = u_aero * (self.dx_phys / self.dt_phys) / self.a_phys
                if _pb: _t0 = _tm.perf_counter()
                CL, CD = self._lookup_cl_cd(alpha_aero, Re, u_aero, Mach,
                                            blade, n_per_blade)
                if _pb: self._pf_acc['polar'] += _tm.perf_counter() - _t0
                # overwrite pre-correction kinematics in the output arrays
                u_n_all[idx_start:idx_end] = u_n
                u_rel_all[idx_start:idx_end] = u_rel
                phi_all[idx_start:idx_end] = phi_aero
                alpha_all[idx_start:idx_end] = alpha_aero
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
            phi_rad = np.radians(phi_aero)               # [radians] (φ_n if LE-normal)
            cos_phi = np.cos(phi_rad)                    # [dimensionless]
            sin_phi = np.sin(phi_rad)                    # [dimensionless]

            F_n = F_L * cos_phi - F_D * sin_phi          # [lattice force]
            F_theta = F_L * sin_phi + F_D * cos_phi      # [lattice force]

            # --- Prandtl tip/root loss correction ---
            if self.prandtl_loss:
                F_pr = self._compute_prandtl_factor(blade.marker_r, phi_aero)
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
        model._eps_corr_smooth = int(ec.get('smooth', 0))  # spanwise Γ smoothing passes
        model._eps_corr_method = ec.get('method', 'dag')  # "dag" | "kleine"
        model._eps_endpoint_closure = ec.get('endpoint_closure', False)  # option 3
        model._kleine_wake_mode = ec.get('wake', 'straight')  # "straight" | "free"
        # Dağ wake: same 'wake' key ("straight" | "prescribed_helix"), used when method="dag".
        model._dag_wake = ec.get('wake', 'straight')
        model._dag_helix_revs = float(ec.get('helix_revs', 2.0))
        model._dag_helix_azimuth_deg = float(ec.get('helix_azimuth_deg', 2.0))
        # Hover pitch guard: "auto" (default) | "off" (paper-literal) | float [Δx/Δt]
        model._dag_helix_pitch_floor = ec.get('helix_pitch_floor', 'auto')
        model._kleine_wake_markers = ec.get('wake_markers', 'all')  # Q2: "all"|"tip"|N|f|idx
        _nw = ec.get('n_w', None)                 # None → auto from revs & 2° interval
        model._kleine_wake_nw = None if _nw is None else int(_nw)
        model._kleine_wake_revs = float(ec.get('wake_revs', 2.0))       # Dağ: 2 rev
        model._kleine_wake_azimuth_deg = float(ec.get('wake_azimuth_deg', 2.0))  # Dağ: 2°
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
        model._sampling_ring_n = int(samp.get('ring_n', 20))          # ② ring
        model._sampling_ring_r_factor = float(samp.get('ring_r_factor', 1.0))
    elif isinstance(samp, str):
        model._sampling_mode = samp

    # Radial truncation + renormalisation of the tip/root Gaussian source
    # (Merabet & Laurendeau 2021): deposit each marker's force only within the
    # blade's geometric span [r_root, r_tip], renormalised to conserve total
    # force (relocated inboard). Default off → existing runs bit-identical.
    spreading_cfg = config.get('spreading', {})
    model._radial_trunc = bool(spreading_cfg.get('radial_truncation', False))

    # Anisotropic Gaussian force projection (Natelson Eq. 7; ① Step A). Per-axis
    # width factors relative to the isotropic marker ε: c=chord, t=thickness,
    # r=radial. c=t=r=1 reduces exactly to the isotropic spread. Default off.
    ani = spreading_cfg.get('anisotropic', None)
    if isinstance(ani, dict) and ani.get('enabled', True):
        model._aniso = {'c': float(ani.get('c', 1.0)),
                        't': float(ani.get('t', 1.0)),
                        'r': float(ani.get('r', 1.0))}

    # Swept-tip refinement (patch_notes/alm_tip_sweep_refine, Step 3).
    #   sweep_alpha_normal → LE-normal α resolution in the BEM (Step 2).
    #   sweep_geometric    → physically offset the swept markers (Step 1); the
    #     back-set direction is -sign(ω)·ê_θ, so set _sweep_sign on the LU blades
    #     (to_lattice_units does not carry these flags across). marker_sweep_offset
    #     was already recomputed in lattice units by the LU generate_markers.
    model._sweep_alpha_normal = bool(config.get('sweep_alpha_normal', False))
    if config.get('sweep_geometric', False):
        _sgn = -float(np.sign(rotor_lu.omega)) or -1.0   # ω=0 guard
        for _b in model.rotor.blades:
            _b.sweep_geometric = True
            _b._sweep_sign = _sgn

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