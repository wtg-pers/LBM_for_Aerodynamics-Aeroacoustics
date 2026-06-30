"""
LBM Simulation — Single Timestep Physics

This module defines the Simulation class, which encapsulates the core LBM
physics for one timestep. The `advance()` method performs all physical
operations in a clear, readable sequence matching the continuous equations:

    1. Macroscopic (ρ, u from f)
    2. Body Force  (ALM pipeline → F(x))
    3. Velocity Correction (Guo: u += F/(2ρ))
    4. Collision (equilibrium + forcing + relaxation, all internal)
    5. Streaming (pull: f' = f*(x - c_i))
    6. Boundary Conditions (domain faces + internal obstacles)
    7. Buffer Swap (f ← f')

Design Principle:
    "main은 실행 진입점, Simulation이 물리를 서술한다."
    → 물리적 흐름이 advance() 안에서 직관적으로 확인된다.

References:
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017
    - Guo, Zheng, Shi, Phys. Rev. E 65, 046308, 2002

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt
    from src.macroscopic.compute import Macroscopic
    from src.collision.base import CollisionOperator
    from src.streaming.stream import StreamingPull
    from src.boundary.domain_bc_manager import DomainBCManager
    from src.boundary.wall import HalfwayBounceBack


class Simulation:
    """Core LBM simulation state and single-step physics.

    This class owns the distribution function buffers (f, f_post)
    and all physical operators. One call to `advance()` performs exactly
    one LBM timestep. Streaming writes directly to f (no f_new buffer).

    Attributes:
        f:          Current distribution function (Q, Nx, Ny[, Nz])  [dimensionless]
        rho:        Density field (Nx, Ny[, Nz])                     [dimensionless]
        u:          Velocity field (dim, Nx, Ny[, Nz])               [Δx/Δt]
                    (Guo force-corrected when body_force is active)
        body_force: External body force (dim, Nx, Ny[, Nz])          [lattice force / lu³]
                    or None if no ALM is active
        f_post:     Post-collision distribution (read-only access)    [dimensionless]
                    Needed by MEM force calculation (ForceManager)
        step_count: Number of advance() calls since initialization

    Example:
        >>> sim = Simulation(xp=xp, macroscopic=macro, equilibrium=eq,
        ...                  collision=collision, streaming=streaming,
        ...                  forcing_scheme=forcing, bc_manager=bc_mgr,
        ...                  tau=0.6, domain_shape=(100, 50, 50))
        >>> sim.set_distribution(f_init)
        >>> for step in range(1000):
        ...     sim.advance()
        ...     # Access sim.rho, sim.u for output
    """

    def __init__(
        self,
        xp: 'ModuleType',
        macroscopic: 'Macroscopic',
        collision: 'CollisionOperator',
        streaming: 'StreamingPull',
        bc_manager: 'DomainBCManager',
        tau: float,
        domain_shape: Tuple[int, ...],
        obstacle_bc: Optional['HalfwayBounceBack'] = None,
        al_model: Optional[object] = None,
        sgs_cfg: Optional[dict] = None,
    ) -> None:
        """Initialize Simulation with all physical operators.

        Args:
            xp: Array module (numpy or cupy)
            macroscopic: Macroscopic variable calculator (ρ, u from f)
            collision: Collision operator (BGK, Cumulant, etc.)
                       Owns equilibrium and forcing internally.
            streaming: Streaming operator (pull scheme)
            bc_manager: Domain boundary condition manager
            tau: Relaxation time  [Δt]
                 ν = cs²·(τ - 0.5) in lattice units  [Δx²/Δt]
            domain_shape: Grid dimensions (Nx, Ny) or (Nx, Ny, Nz)  [lu]
            obstacle_bc: Internal obstacle BC (HalfwayBounceBack), optional
            al_model: Actuator Line model (ActuatorLineModel or
                      MultiRotorManager), optional
            sgs_cfg: Validated SGS config dict with keys
                     {enabled, model, Cs, Cw}, or None for SGS-disabled.
                     Produced by src.turbulence.sgs.parse_sgs_config.
        """
        # ── Array module ──
        self.xp = xp

        # ── Physical operators ──
        self.macroscopic = macroscopic
        self.collision = collision
        self.streaming = streaming

        # ── Boundary conditions ──
        self.bc_manager = bc_manager
        self.obstacle_bc = obstacle_bc

        # ── Actuator Line ──
        self.al_model = al_model

        # ── SGS / turbulence model ──
        # Default to disabled if caller did not supply a parsed dict.
        self._sgs_cfg = sgs_cfg if sgs_cfg is not None else {
            "enabled": False, "model": "off", "Cs": 0.18, "Cw": 0.5,
        }

        # ── Parameters ──
        self.tau: float = tau                        # [Δt]
        self.domain_shape: Tuple[int, ...] = domain_shape  # [lu]

        # ── State (populated by advance()) ──
        self.rho: Optional['npt.NDArray'] = None     # [dimensionless]
        self.u: Optional['npt.NDArray'] = None       # [Δx/Δt] (force-corrected)
        self.body_force: Optional['npt.NDArray'] = None  # [lattice force / lu³]
        # SGS eddy viscosity buffer (lattice units). Allocated in
        # set_distribution() iff SGS is enabled. None -> kernel skips writing.
        self.nu_t: Optional['npt.NDArray'] = None    # [lu²/lt]

        # ── Distribution buffers (populated by set_distribution()) ──
        self.f: Optional['npt.NDArray'] = None       # Current distribution  [dimensionless]
        self._f_post: Optional['npt.NDArray'] = None # Post-collision buffer [dimensionless]

        # ── Counters ──
        self.step_count: int = 0
        self._is_ready: bool = False

        # ── Fused CUDA kernel (lazy init) ──
        self._fused_kernel = None
        self._use_fused: bool = False

        # ── WALE pre-pass kernels (lazy init for sgs_model="wale") ──
        # The WALE eddy-viscosity needs u globally before collision so the
        # gradient stencil can read neighbours. We compute (rho, u) via a
        # macro pre-pass, then nu_t via a separate kernel, then run the
        # cumulant collide reading nu_t from a buffer.
        self._macro_kernel = None
        self._wale_kernel = None
        self._u_buf = None      # (dim, N) flat for pre-pass writes
        self._rho_buf = None    # (N,)     flat
        self._nu_t_in = None    # (N,)     fed to cumulant WALE branch

        # ── Obstacle BC kernel (lazy init in set_distribution) ──
        # Only one of _hwbb_kernel / _ibb_kernel is set at a time, depending
        # on isinstance(obstacle_bc, ...). Both None → Python path.
        self._ibb_kernel = None

    # =====================================================================
    # Public Interface
    # =====================================================================

    def set_distribution(self, f: 'npt.NDArray') -> None:
        """Set distribution function and allocate work buffer.

        This is the bridge between Initialization and Execution:
        after calling this, advance() can be called.

        Only one work buffer (f_post) is allocated. Streaming writes
        directly back to f (pull scheme: source=f_post, dest=f, no conflict).

        If running on GPU with BGK D3Q27, a fused CUDA kernel is used
        for macroscopic + collision (single kernel launch instead of ~15).

        Args:
            f: Initial distribution function, shape (Q, Nx, Ny[, Nz])
               [dimensionless]
        """
        self.f = f
        self._f_post = self.xp.empty_like(f)
        self._is_ready = True

        # ── SGS eddy-viscosity output buffer ──
        # Allocate iff a non-trivial SGS model is in use, so the kernel can
        # write nu_t per cell for diagnostics / VTK export.
        if self._sgs_cfg["model"] in ("smagorinsky", "wale", "dyn_smag"):
            self.nu_t = self.xp.zeros(self.domain_shape, dtype=f.dtype)
        else:
            self.nu_t = None

        # ── WALE pre-pass scratch buffers ──
        # The cumulant collide WALE branch reads nu_t from a flat (N,) buffer.
        # The pre-pass kernels write into flat (N,) and (dim, N) views.
        if self._sgs_cfg["model"] in ("wale", "dyn_smag"):
            N = 1
            for d in self.domain_shape:
                N *= int(d)
            dim = len(self.domain_shape)
            self._rho_buf = self.xp.empty(N, dtype=f.dtype)
            self._u_buf   = self.xp.empty((dim, N), dtype=f.dtype)
            self._nu_t_in = self.xp.empty(N, dtype=f.dtype)

        # ── Try to initialize CUDA kernels ──
        self._use_fused = False
        self._fused_is_cumulant = False
        # (streaming CUDA kernel is now handled inside StreamingPull)
        self._hwbb_kernel = None

        # Track whether the fused kernel is the D2Q9 variant (different
        # launch signature: omega_3 + omega_4 instead of omega_high).
        self._fused_is_d2q9 = False

        if self.xp.__name__ == 'cupy' and len(self.domain_shape) == 3:
            from src.collision.bgk import BGKCollision
            from src.collision.cumulant import CumulantCollision

            sgs_model = self._sgs_cfg["model"]   # "off" | "smagorinsky" | "wale" | "dyn_smag"
            # Cumulant collide branch: dyn_smag reuses the WALE block (both
            # just read nu_t from a buffer). sgs_model_kernel -> what the
            # cumulant kernel template is built with.
            sgs_model_kernel = "wale" if sgs_model == "dyn_smag" else sgs_model

            # Collision kernel
            if isinstance(self.collision, BGKCollision):
                if self._sgs_cfg["enabled"]:
                    raise NotImplementedError(
                        "SGS turbulence models are wired to the cumulant "
                        "collision only; BGK + SGS is not supported."
                    )
                try:
                    from src.kernels.bgk_d3q27 import BGKCollideKernelD3Q27
                    self._fused_kernel = BGKCollideKernelD3Q27()
                    self._use_fused = True
                except Exception:
                    pass
            elif isinstance(self.collision, CumulantCollision):
                try:
                    from src.kernels.cumulant_d3q27 import CumulantCollideKernelD3Q27
                    self._fused_kernel = CumulantCollideKernelD3Q27(sgs_model=sgs_model_kernel)
                    self._use_fused = True
                    self._fused_is_cumulant = True
                    if sgs_model != "off":
                        print(f"  [fused kernel] D3Q27 cumulant + SGS={sgs_model}")
                except Exception:
                    pass

                # WALE / dyn_smag pre-pass kernels (3D)
                # Both feed nu_t to the cumulant collide WALE branch.
                if sgs_model == "wale":
                    from src.kernels.macro_d3q27 import MacroKernelD3Q27
                    from src.kernels.wale_d3q27 import WALEKernelD3Q27
                    self._macro_kernel = MacroKernelD3Q27()
                    self._wale_kernel  = WALEKernelD3Q27()
                elif sgs_model == "dyn_smag":
                    from src.kernels.macro_d3q27 import MacroKernelD3Q27
                    from src.kernels.dyn_smag_d3q27 import DynSmagKernelD3Q27
                    self._macro_kernel = MacroKernelD3Q27()
                    self._wale_kernel  = DynSmagKernelD3Q27()

            # Obstacle BC kernel: dispatch on type (HWBB vs IBB).
            if self.obstacle_bc is not None:
                from src.boundary.wall import HalfwayBounceBack
                if type(self.obstacle_bc) is HalfwayBounceBack:
                    try:
                        from src.kernels.bounce_back_d3q27 import HWBBKernelD3Q27
                        self._hwbb_kernel = HWBBKernelD3Q27()
                    except Exception:
                        pass
                else:
                    try:
                        from src.boundary.interpolated_wall import InterpolatedBounceBack
                        if isinstance(self.obstacle_bc, InterpolatedBounceBack):
                            from src.kernels.interpolated_wall_d3q27 import IBBKernelD3Q27
                            self._ibb_kernel = IBBKernelD3Q27()
                    except Exception:
                        pass

        elif self.xp.__name__ == 'cupy' and len(self.domain_shape) == 2:
            # D2Q9 fused path (currently only Cumulant). Kernel is float32-only.
            import numpy as _np
            is_f32 = (self.f.dtype == _np.float32)

            if is_f32:
                from src.collision.cumulant_d2q9 import CumulantCollisionD2Q9

                if isinstance(self.collision, CumulantCollisionD2Q9):
                    sgs_model = self._sgs_cfg["model"]
                    sgs_model_kernel = "wale" if sgs_model == "dyn_smag" else sgs_model
                    try:
                        from src.kernels.cumulant_d2q9 import CumulantCollideKernelD2Q9
                        self._fused_kernel = CumulantCollideKernelD2Q9(
                            sgs_model=sgs_model_kernel,
                        )
                        self._use_fused = True
                        self._fused_is_cumulant = True
                        self._fused_is_d2q9 = True
                    except Exception:
                        pass

                # WALE / dyn_smag pre-pass kernels (2D)
                if self._sgs_cfg["model"] == "wale":
                    from src.kernels.macro_d2q9 import MacroKernelD2Q9
                    from src.kernels.wale_d2q9 import WALEKernelD2Q9
                    self._macro_kernel = MacroKernelD2Q9()
                    self._wale_kernel  = WALEKernelD2Q9()
                elif self._sgs_cfg["model"] == "dyn_smag":
                    from src.kernels.macro_d2q9 import MacroKernelD2Q9
                    from src.kernels.dyn_smag_d2q9 import DynSmagKernelD2Q9
                    self._macro_kernel = MacroKernelD2Q9()
                    self._wale_kernel  = DynSmagKernelD2Q9()

                # Obstacle BC kernel: dispatch on exact type.
                # Plain HalfwayBounceBack → HWBBKernelD2Q9
                # InterpolatedBounceBack  → IBBKernelD2Q9
                # MovingWallBounceBack / others → Python path.
                if self.obstacle_bc is not None:
                    from src.boundary.wall import HalfwayBounceBack
                    if type(self.obstacle_bc) is HalfwayBounceBack:
                        try:
                            from src.kernels.bounce_back_d2q9 import HWBBKernelD2Q9
                            self._hwbb_kernel = HWBBKernelD2Q9()
                        except Exception:
                            pass
                    else:
                        try:
                            from src.boundary.interpolated_wall import InterpolatedBounceBack
                            if isinstance(self.obstacle_bc, InterpolatedBounceBack):
                                from src.kernels.interpolated_wall_d2q9 import IBBKernelD2Q9
                                self._ibb_kernel = IBBKernelD2Q9()
                        except Exception:
                            pass

    @property
    def is_ready(self) -> bool:
        """Whether set_distribution() has been called."""
        return self._is_ready

    @property
    def f_post(self) -> Optional['npt.NDArray']:
        """Post-collision distribution (read-only).

        Needed by MEM (Momentum Exchange Method) force calculation.
        Valid after the most recent advance() call.

        In streaming-fused mode (ping-pong), f itself is the post-collision
        state after swap, so f_post returns f.

        Returns:
            Post-collision f, shape (Q, Nx, Ny[, Nz])  [dimensionless]
        """
        return self._f_post

    # =====================================================================
    # Diagnostic: NaN/Inf trap
    # =====================================================================

    def _check_nan(self, where: str) -> None:
        """Detect non-finite values in rho/u and abort with location info.

        Runs every step when enabled; lightweight single reductions on GPU.
        Set Simulation.nan_trap_enabled = False on class to disable.
        """
        if not getattr(self, 'nan_trap_enabled', True):
            return
        xp = self.xp
        if self.rho is None or self.u is None:
            return

        rho_bad = bool(xp.any(~xp.isfinite(self.rho)))
        u_bad = bool(xp.any(~xp.isfinite(self.u)))
        if not (rho_bad or u_bad):
            return

        if rho_bad:
            bad_mask = ~xp.isfinite(self.rho)
            field = 'rho'
        else:
            bad_mask = xp.any(~xp.isfinite(self.u), axis=0)
            field = 'u'

        n_bad = int(bad_mask.sum())
        bad_idx = xp.argwhere(bad_mask)
        loc = tuple(int(v) for v in bad_idx[0])

        rho_finite = xp.where(xp.isfinite(self.rho), self.rho, xp.float64(1.0))
        u_sq = xp.where(xp.isfinite(self.u), self.u, xp.float64(0.0)) ** 2
        u_mag_sq = xp.sum(u_sq, axis=0)
        rho_min = float(xp.min(rho_finite))
        rho_max = float(xp.max(rho_finite))
        u_max = float(xp.sqrt(xp.max(u_mag_sq)))

        raise RuntimeError(
            f"[NaN trap @ {where}] step={self.step_count}, "
            f"domain={self.domain_shape}, bad_field='{field}', "
            f"first_bad_loc(x,y,z)={loc}, n_bad={n_bad}, "
            f"rho_finite_range=[{rho_min:.4g}, {rho_max:.4g}], "
            f"|u|_max_finite={u_max:.4g}"
        )

    def advance(self) -> None:
        """Perform one LBM timestep.

        Physical Process (matches continuous equations):
            1. Macroscopic:     ρ, u = moments(f)
            2. Body Force:      F(x) = ALM pipeline (if active)
            3. Velocity Corr:   u += F/(2ρ)         (Guo, 2002)
            4. Collision:       f* = collide(f, ρ, u, τ, F)
                                (equilibrium + forcing internally)
            5. Streaming:       f = f*(x - c_i)     (pull, writes to f directly)
            6. Boundary Cond:   domain faces + obstacles

        After calling advance():
            - self.rho, self.u, self.body_force are updated
            - self.f contains the new distribution
            - self.f_post contains post-collision distribution
            - self.step_count is incremented

        Raises:
            RuntimeError: If set_distribution() has not been called.
        """
        if not self._is_ready:
            raise RuntimeError(
                "Simulation not ready: call set_distribution() first"
            )

        if self._use_fused and self.al_model is None:
            self._advance_fused()
        elif self._use_fused and self.al_model is not None:
            self._advance_fused_with_alm()
        else:
            self._advance_default()

    def _advance_default(self) -> None:
        """Default advance path using CuPy array operations."""

        # ─── Step 1: Macroscopic Variables ───────────────────────────
        self.rho, self.u = self.macroscopic.compute(self.f)
        self._check_nan("default:post-macro")

        # ─── Step 2: Body Force (ALM pipeline) ──────────────────────
        self.body_force = self._compute_body_force(self.u, self.rho)

        # ─── Step 3: Guo Velocity Correction ────────────────────────
        u = self.u
        if self.body_force is not None:
            u = u + self.body_force / (2.0 * self.rho[None, ...])
            self.u = u

        # ─── Step 4: Collision ───────────────────────────────────────
        self.collision.collide(
            self.f, self._f_post, self.rho, u, self.tau, self.body_force
        )

        # ─── Step 5: Streaming ───────────────────────────────────────
        self.streaming.compute(self._f_post, self.f)

        # ─── Step 6: Boundary Conditions ─────────────────────────────
        self.bc_manager.apply_all(self.f, self._f_post)
        if self.obstacle_bc is not None:
            self.obstacle_bc.apply_with_reset(self.f, self._f_post)

        self.step_count += 1

    def _advance_fused(self) -> None:
        """Fused CUDA kernel path (no ALM): macro + collision in 1 launch."""
        xp = self.xp
        N = 1
        for d in self.domain_shape:
            N *= d

        # Allocate rho/u if needed (first call)
        if self.rho is None:
            self.rho = xp.empty(self.domain_shape, dtype=self.f.dtype)
            self.u = xp.empty((len(self.domain_shape),) + self.domain_shape,
                              dtype=self.f.dtype)

        # ─── WALE / dyn_smag pre-pass: f -> rho, u (flat) -> nu_t_in ──
        if self._sgs_cfg["model"] in ("wale", "dyn_smag"):
            self._wale_pre_pass(N)

        # ─── Fused: macro + collision (1 kernel launch) ─────────────
        omega = 1.0 / self.tau
        if self._fused_is_cumulant:
            if self._fused_is_d2q9:
                ob = self.collision.omega_bulk
                self._fused_kernel.launch(
                    f_in=self.f,
                    f_post=self._f_post,
                    rho_out=self.rho,
                    u_out=self.u,
                    force=None,
                    omega_1=omega,
                    omega_bulk=omega if ob is None else ob,
                    omega_3=self.collision.omega_3,
                    omega_4=self.collision.omega_4,
                    N=N,
                    Cs=self._sgs_cfg["Cs"],
                    nu_t_out=self.nu_t,
                    nu_t_in=self._nu_t_in,
                )
            else:
                self._fused_kernel.launch(
                    f_in=self.f,
                    f_post=self._f_post,
                    rho_out=self.rho,
                    u_out=self.u,
                    force=None,
                    omega_1=omega,
                    omega_bulk=self.collision.omega_bulk,
                    omega_high=self.collision.omega_3,
                    N=N,
                    Cs=self._sgs_cfg["Cs"],
                    nu_t_out=self.nu_t,
                    nu_t_in=self._nu_t_in,
                )
        else:
            self._fused_kernel.launch(
                f_in=self.f,
                f_post=self._f_post,
                rho_out=self.rho,
                u_out=self.u,
                force=None,
                omega=omega,
                N=N,
            )
        self.body_force = None
        self._check_nan("fused:post-collision")

        # ─── Streaming ───────────────────────────────────────────────
        self.streaming.compute(self._f_post, self.f)

        # ─── Boundary Conditions ─────────────────────────────────────
        self.bc_manager.apply_all(self.f, self._f_post)
        if self.obstacle_bc is not None:
            self._apply_obstacle_bc()

        self.step_count += 1

    def _wale_pre_pass(self, N: int) -> None:
        """Compute (rho, u) from f and nu_t from u via the SGS pre-pass —
        feeds the cumulant collide WALE branch. Same buffer pipeline
        works for both WALE and Dynamic Smagorinsky; only the second
        kernel launch differs."""
        model = self._sgs_cfg["model"]
        if len(self.domain_shape) == 2:
            Nx, Ny = self.domain_shape
            self._macro_kernel.launch(
                self.f, self._rho_buf,
                self._u_buf[0], self._u_buf[1], N,
            )
            if model == "wale":
                self._wale_kernel.launch(
                    self._u_buf[0], self._u_buf[1], self._nu_t_in,
                    Nx, Ny, Cw=float(self._sgs_cfg["Cw"]), dx=1.0,
                )
            else:  # dyn_smag
                self._wale_kernel.launch(
                    self._u_buf[0], self._u_buf[1], self._nu_t_in,
                    Nx, Ny,
                    dx=1.0,
                    Cs_max=float(self._sgs_cfg["Cs_max"]),
                    alpha_sq=float(self._sgs_cfg["alpha_sq"]),
                )
        else:
            Nx, Ny, Nz = self.domain_shape
            self._macro_kernel.launch(
                self.f, self._rho_buf,
                self._u_buf[0], self._u_buf[1], self._u_buf[2], N,
            )
            if model == "wale":
                self._wale_kernel.launch(
                    self._u_buf[0], self._u_buf[1], self._u_buf[2],
                    self._nu_t_in,
                    Nx, Ny, Nz, Cw=float(self._sgs_cfg["Cw"]), dx=1.0,
                )
            else:  # dyn_smag
                self._wale_kernel.launch(
                    self._u_buf[0], self._u_buf[1], self._u_buf[2],
                    self._nu_t_in,
                    Nx, Ny, Nz,
                    dx=1.0,
                    Cs_max=float(self._sgs_cfg["Cs_max"]),
                    alpha_sq=float(self._sgs_cfg["alpha_sq"]),
                )

    def _apply_obstacle_bc(self) -> None:
        """Dispatch obstacle BC application across HWBB / IBB / Python paths."""
        N = 1
        for d in self.domain_shape:
            N *= d

        if self._hwbb_kernel is not None:
            self._hwbb_kernel.apply(
                self.f, self._f_post,
                self.obstacle_bc.needs_bounce, N,
            )
            self._hwbb_kernel.reset_solid(
                self.f, self.obstacle_bc.solid_mask, N,
            )
        elif self._ibb_kernel is not None:
            if len(self.domain_shape) == 2:
                Nx, Ny = self.domain_shape
                self._ibb_kernel.apply(
                    self.f, self._f_post,
                    self.obstacle_bc.needs_bounce,
                    self.obstacle_bc.q_fraction,
                    Nx, Ny,
                )
            else:
                Nx, Ny, Nz = self.domain_shape
                self._ibb_kernel.apply(
                    self.f, self._f_post,
                    self.obstacle_bc.link_cell,
                    self.obstacle_bc.link_dir,
                    self.obstacle_bc.link_q,
                    self.obstacle_bc.n_links,
                    Nx, Ny, Nz,
                )
            self._ibb_kernel.reset_solid(
                self.f, self.obstacle_bc.solid_mask, N,
            )
        else:
            self.obstacle_bc.apply_with_reset(self.f, self._f_post)

    def _advance_fused_with_alm(self) -> None:
        """Fused path with ALM: macro → ALM → fused collision."""
        xp = self.xp
        N = 1
        for d in self.domain_shape:
            N *= d

        # ─── Step 1: Macroscopic (needed by ALM before collision) ───
        self.rho, self.u = self.macroscopic.compute(self.f)

        # Ensure rho/u dtype matches f for fused CUDA kernel
        # (macroscopic.compute uses tensordot which promotes to float64
        #  via int32 c × float32 f, but the kernel writes float32)
        dtype = self.f.dtype
        if self.rho.dtype != dtype:
            self.rho = self.rho.astype(dtype)
        if self.u.dtype != dtype:
            self.u = self.u.astype(dtype)

        self._check_nan("fused+alm:post-macro")

        # ─── Step 2: Body Force (ALM pipeline) ──────────────────────
        self.body_force = self._compute_body_force(self.u, self.rho)

        # Ensure dtype matches f (ALM returns float64, kernel expects float32)
        if self.body_force is not None and self.body_force.dtype != self.f.dtype:
            self.body_force = self.body_force.astype(self.f.dtype)

        # ─── Step 3+4: Guo correction + collision (fused kernel) ────
        omega = 1.0 / self.tau
        if self._fused_is_cumulant:
            if self._fused_is_d2q9:
                ob = self.collision.omega_bulk
                self._fused_kernel.launch(
                    f_in=self.f,
                    f_post=self._f_post,
                    rho_out=self.rho,
                    u_out=self.u,
                    force=self.body_force,
                    omega_1=omega,
                    omega_bulk=omega if ob is None else ob,
                    omega_3=self.collision.omega_3,
                    omega_4=self.collision.omega_4,
                    N=N,
                    Cs=self._sgs_cfg["Cs"],
                    nu_t_out=self.nu_t,
                )
            else:
                self._fused_kernel.launch(
                    f_in=self.f,
                    f_post=self._f_post,
                    rho_out=self.rho,
                    u_out=self.u,
                    force=self.body_force,
                    omega_1=omega,
                    omega_bulk=self.collision.omega_bulk,
                    omega_high=self.collision.omega_3,
                    N=N,
                    Cs=self._sgs_cfg["Cs"],
                    nu_t_out=self.nu_t,
                )
        else:
            self._fused_kernel.launch(
                f_in=self.f,
                f_post=self._f_post,
                rho_out=self.rho,
                u_out=self.u,
                force=self.body_force,
                omega=omega,
                N=N,
            )

        # ─── Step 5: Streaming ───────────────────────────────────────
        self.streaming.compute(self._f_post, self.f)

        # ─── Step 6: Boundary Conditions ─────────────────────────────
        self.bc_manager.apply_all(self.f, self._f_post)
        if self.obstacle_bc is not None:
            self._apply_obstacle_bc()

        self.step_count += 1

    # =====================================================================
    # Internal
    # =====================================================================

    def _compute_body_force(
        self,
        u: 'npt.NDArray',
        rho: 'npt.NDArray',
    ) -> Optional['npt.NDArray']:
        """Compute body force from Actuator Line Model.

        The ALM pipeline:
            1. Advance rotor (rotation by ωΔt)
            2. Interpolate LBM velocity at blade markers (Gaussian kernel)
            3. Compute BEM forces at each blade element (CL, CD lookup)
            4. Spread forces back to LBM grid (Gaussian kernel)

        Args:
            u: Velocity field (raw, before Guo correction)
               shape (dim, Nx, Ny[, Nz])  [Δx/Δt]
            rho: Density field
               shape (Nx, Ny[, Nz])  [dimensionless]

        Returns:
            Body force field, shape (dim, Nx, Ny[, Nz])  [lattice force / lu³]
            or None if no ALM is active.
        """
        if self.al_model is None:
            return None

        # al_model.step() handles:
        #   rotor.advance(dt) → interpolation → BEM → spreading
        return self.al_model.step(u, dt=1.0)  # [lattice force / lu³]

    # =====================================================================
    # Info / Debug
    # =====================================================================

    def print_info(self) -> None:
        """Print simulation configuration summary."""
        dim = len(self.domain_shape)
        total_cells = 1
        for n in self.domain_shape:
            total_cells *= n

        nu = (1.0 / 3.0) * (self.tau - 0.5)  # [Δx²/Δt]

        print(f"\n{'='*60}")
        print(f"  Simulation Info")
        print(f"{'='*60}")
        print(f"  Domain: {'×'.join(str(n) for n in self.domain_shape)} "
              f"({dim}D, {total_cells:,} cells)")
        print(f"  τ = {self.tau:.6f}, ν = {nu:.6f} [Δx²/Δt]")
        print(f"  Collision: {self.collision.__class__.__name__}")
        print(f"  Streaming: {self.streaming.__class__.__name__}")
        print(f"  ALM: {'active' if self.al_model is not None else 'inactive'}")
        print(f"  Obstacle BC: "
              f"{'active' if self.obstacle_bc is not None else 'none'}")
        print(f"  Ready: {self.is_ready}")
        print(f"{'='*60}")

    def __repr__(self) -> str:
        dim = len(self.domain_shape)
        shape_str = '×'.join(str(n) for n in self.domain_shape)
        return (f"Simulation({dim}D, {shape_str}, "
                f"τ={self.tau:.4f}, "
                f"step={self.step_count}, "
                f"ready={self.is_ready})")