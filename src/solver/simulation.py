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

        # ── Parameters ──
        self.tau: float = tau                        # [Δt]
        self.domain_shape: Tuple[int, ...] = domain_shape  # [lu]

        # ── State (populated by advance()) ──
        self.rho: Optional['npt.NDArray'] = None     # [dimensionless]
        self.u: Optional['npt.NDArray'] = None       # [Δx/Δt] (force-corrected)
        self.body_force: Optional['npt.NDArray'] = None  # [lattice force / lu³]

        # ── Distribution buffers (populated by set_distribution()) ──
        self.f: Optional['npt.NDArray'] = None       # Current distribution  [dimensionless]
        self._f_post: Optional['npt.NDArray'] = None # Post-collision buffer [dimensionless]

        # ── Counters ──
        self.step_count: int = 0
        self._is_ready: bool = False

        # ── CUDA kernel mode (lazy init) ──
        self._fused_kernel = None
        self._use_fused: bool = False
        self._use_esoteric: bool = False
        self._esoteric_kernel = None
        self._esoteric_step: int = 0

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

        # ── Try to initialize CUDA kernels ──
        self._use_fused = False
        self._use_esoteric = False
        self._fused_is_cumulant = False
        self._hwbb_kernel = None

        if self.xp.__name__ == 'cupy' and len(self.domain_shape) == 3:
            from src.collision.bgk import BGKCollision
            from src.collision.cumulant import CumulantCollision

            # Try Esoteric Pull first (highest priority for BGK)
            if isinstance(self.collision, BGKCollision):
                try:
                    self._init_esoteric(f)
                except Exception:
                    pass

            # Fallback: fused collision kernel
            if not self._use_esoteric:
                if isinstance(self.collision, BGKCollision):
                    try:
                        from src.kernels.bgk_d3q27 import BGKCollideKernelD3Q27
                        self._fused_kernel = BGKCollideKernelD3Q27()
                        self._use_fused = True
                    except Exception:
                        pass
                elif isinstance(self.collision, CumulantCollision):
                    try:
                        from src.kernels.cumulant_d3q27 import CumulantCollideKernelD3Q27
                        self._fused_kernel = CumulantCollideKernelD3Q27()
                        self._use_fused = True
                        self._fused_is_cumulant = True
                    except Exception:
                        pass

                # HWBB kernel (only needed in non-Esoteric mode)
                if self.obstacle_bc is not None:
                    try:
                        from src.kernels.bounce_back_d3q27 import HWBBKernelD3Q27
                        self._hwbb_kernel = HWBBKernelD3Q27()
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

        In Esoteric mode, f_post is not available (single buffer).
        MEM force needs adaptation for Esoteric ordering.

        Returns:
            Post-collision f, shape (Q, Nx, Ny[, Nz])  [dimensionless]
        """
        if self._use_esoteric:
            return None
        return self._f_post

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

        if self._use_esoteric:
            self._advance_esoteric()
        elif self._use_fused and self.al_model is None:
            self._advance_fused()
        elif self._use_fused and self.al_model is not None:
            self._advance_fused_with_alm()
        else:
            self._advance_default()

    # =====================================================================
    # Esoteric Pull path
    # =====================================================================

    def _init_esoteric(self, f: 'npt.NDArray') -> None:
        """Initialize Esoteric Pull mode.

        Converts f to Esoteric memory layout, creates node_type and BC
        parameter arrays, and initializes the Esoteric kernel.

        Args:
            f: Initial distribution in standard D3Q27 ordering (27, Nx, Ny, Nz)
        """
        from src.kernels.esoteric_d3q27 import (
            EsotericBGKKernelD3Q27,
            convert_f_std_to_esoteric,
            init_f_esoteric,
            NODE_FLUID, NODE_SOLID, NODE_EQ_BC, NODE_NEUMANN,
        )
        xp = self.xp
        Nx, Ny, Nz = self.domain_shape
        N = Nx * Ny * Nz

        # Convert f from standard to Esoteric direction ordering
        # (skip if f is already in Esoteric layout, e.g. checkpoint restart)
        if not getattr(self, '_esoteric_f_already_set', False):
            f_eso = convert_f_std_to_esoteric(xp, f)
            self.f = init_f_esoteric(xp, f_eso, t_start=0)
        else:
            self.f = f
        self._f_post = None  # not needed in Esoteric mode

        # Build node_type array from BC config
        node_type = xp.zeros((Nx, Ny, Nz), dtype=xp.int8)  # all FLUID

        # Mark solid nodes: internal obstacle
        if self.obstacle_bc is not None:
            solid = self.obstacle_bc.solid_mask  # (Nx, Ny, Nz) bool
            node_type[solid] = NODE_SOLID

        # Mark domain boundary faces from bc_manager
        bc_rho = xp.ones((Nx, Ny, Nz), dtype=xp.float32)
        bc_ux = xp.zeros((Nx, Ny, Nz), dtype=xp.float32)
        bc_uy = xp.zeros((Nx, Ny, Nz), dtype=xp.float32)
        bc_uz = xp.zeros((Nx, Ny, Nz), dtype=xp.float32)

        for face_bc in self.bc_manager.face_bcs:
            loc = face_bc.location.value
            cfg = face_bc.config

            # Determine face slice
            if loc in ('xmin', 'west'):
                sl = (0, slice(None), slice(None))
            elif loc in ('xmax', 'east'):
                sl = (Nx-1, slice(None), slice(None))
            elif loc in ('ymin', 'south'):
                sl = (slice(None), 0, slice(None))
            elif loc in ('ymax', 'north'):
                sl = (slice(None), Ny-1, slice(None))
            elif loc in ('zmin', 'bottom'):
                sl = (slice(None), slice(None), 0)
            elif loc in ('zmax', 'top'):
                sl = (slice(None), slice(None), Nz-1)
            else:
                continue

            # Map BC type to node_type
            method = cfg.method if hasattr(cfg, 'method') else ''
            bc_type = cfg.bc_type.value if hasattr(cfg, 'bc_type') else ''

            if 'wall' in method or 'bounce' in method:
                node_type[sl] = NODE_SOLID
            elif 'neumann' in method:
                node_type[sl] = NODE_NEUMANN
            else:
                # Equilibrium or regularized inlet/outlet → EQ_BC
                node_type[sl] = NODE_EQ_BC
                # Set BC parameters
                if hasattr(cfg, 'density') and cfg.density is not None:
                    bc_rho[sl] = float(cfg.density)
                if hasattr(cfg, 'velocity') and cfg.velocity is not None:
                    vel = cfg.velocity
                    if isinstance(vel, (int, float)):
                        bc_ux[sl] = float(vel)
                    elif isinstance(vel, (list, tuple)):
                        if len(vel) > 0:
                            bc_ux[sl] = float(vel[0])
                        if len(vel) > 1:
                            bc_uy[sl] = float(vel[1])
                        if len(vel) > 2:
                            bc_uz[sl] = float(vel[2])

        # Sponge layers: mark slab nodes as NODE_SPONGE
        from src.kernels.esoteric_d3q27 import NODE_SPONGE
        for sponge_bc in self.bc_manager.sponge_layers:
            sb = sponge_bc
            loc = sb.location.value if hasattr(sb, 'location') else ''
            L = sb.thickness if hasattr(sb, 'thickness') else 0
            if L <= 0:
                continue
            # Sponge sigma profile and target
            sigma_1d = sb._sigma if hasattr(sb, '_sigma') else None
            if sigma_1d is None:
                continue
            sigma_1d_np = sigma_1d.get() if hasattr(sigma_1d, 'get') else sigma_1d
            rho_inf = float(sb.rho_inf) if hasattr(sb, 'rho_inf') else 1.0
            u_inf = sb.u_inf.get() if hasattr(sb.u_inf, 'get') else sb.u_inf

            for layer_i in range(L):
                sigma_val = float(sigma_1d_np[layer_i])
                if loc in ('xmax', 'east'):
                    ix = Nx - 1 - layer_i
                    sl = (ix, slice(None), slice(None))
                elif loc in ('xmin', 'west'):
                    ix = layer_i
                    sl = (ix, slice(None), slice(None))
                else:
                    continue  # TODO: y/z sponge

                node_type[sl] = NODE_SPONGE
                bc_rho[sl] = rho_inf
                bc_ux[sl] = float(u_inf[0])
                bc_uy[sl] = float(u_inf[1]) if len(u_inf) > 1 else 0.0
                # Reuse bc_uz as sigma for sponge nodes
                bc_uz[sl] = sigma_val

        # Store as flat arrays
        self._eso_node_type = node_type.ravel()
        self._eso_bc_rho = bc_rho.ravel()
        self._eso_bc_ux = bc_ux.ravel()
        self._eso_bc_uy = bc_uy.ravel()
        self._eso_bc_uz = bc_uz.ravel()
        self._esoteric_step = 0

        # Allocate macroscopic arrays
        self.rho = xp.empty(self.domain_shape, dtype=xp.float32)
        self.u = xp.empty((3,) + self.domain_shape, dtype=xp.float32)

        # Convert needs_bounce to Esoteric ordering + force accumulator
        self._eso_needs_bounce = None
        self._eso_force_out = None
        if self.obstacle_bc is not None and hasattr(self.obstacle_bc, 'needs_bounce'):
            from src.kernels.esoteric_d3q27 import _STD_TO_ESO
            nb_std = self.obstacle_bc.needs_bounce  # (27, Nx, Ny, Nz) bool
            nb_eso = xp.empty_like(nb_std)
            for eso_q in range(27):
                std_q = _STD_TO_ESO[eso_q]
                nb_eso[eso_q] = nb_std[std_q]
            self._eso_needs_bounce = nb_eso.reshape(27, N)
            self._eso_force_out = xp.zeros(3, dtype=xp.float32)

        # Create kernel
        self._esoteric_kernel = EsotericBGKKernelD3Q27()
        self._use_esoteric = True

    def _advance_esoteric(self) -> None:
        """Esoteric Pull advance: single kernel launch per step.

        Physical process (inside kernel):
            1. Load (Esoteric Pull streaming part 2/2)
            2. Macroscopic: rho, u
            3. BC check + Collision (or BC override)
            4. MEM Force (optional, at boundary links)
            5. Store (Esoteric Pull streaming part 1/2)
        """
        Nx, Ny, Nz = self.domain_shape
        omega = 1.0 / self.tau

        # Force calculation: only compute at force_interval steps
        # (to avoid unnecessary atomicAdd overhead every step)
        force_out = None
        if self._eso_force_out is not None:
            self._eso_force_out.fill(0)
            force_out = self._eso_force_out

        self._esoteric_kernel.launch(
            self.f, self.rho, self.u,
            self._eso_node_type,
            self._eso_bc_rho, self._eso_bc_ux,
            self._eso_bc_uy, self._eso_bc_uz,
            omega, Nx, Ny, Nz,
            t_step=self._esoteric_step,
            needs_bounce=self._eso_needs_bounce,
            force_out=force_out,
        )

        self.body_force = None
        self._esoteric_step += 1
        self.step_count += 1

    # =====================================================================
    # Standard (non-Esoteric) paths
    # =====================================================================

    def _advance_default(self) -> None:
        """Default advance path using CuPy array operations."""

        # ─── Step 1: Macroscopic Variables ───────────────────────────
        self.rho, self.u = self.macroscopic.compute(self.f)

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

        # ─── Fused: macro + collision (1 kernel launch) ─────────────
        omega = 1.0 / self.tau
        if self._fused_is_cumulant:
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

        # ─── Streaming ───────────────────────────────────────────────
        self.streaming.compute(self._f_post, self.f)

        # ─── Boundary Conditions ─────────────────────────────────────
        self.bc_manager.apply_all(self.f, self._f_post)
        if self.obstacle_bc is not None:
            if self._hwbb_kernel is not None:
                N = 1
                for d in self.domain_shape:
                    N *= d
                self._hwbb_kernel.apply(
                    self.f, self._f_post,
                    self.obstacle_bc.needs_bounce, N,
                )
                self._hwbb_kernel.reset_solid(
                    self.f, self.obstacle_bc.solid_mask, N,
                )
            else:
                self.obstacle_bc.apply_with_reset(self.f, self._f_post)

        self.step_count += 1

    def _advance_fused_with_alm(self) -> None:
        """Fused path with ALM: macro → ALM → fused collision."""
        xp = self.xp
        N = 1
        for d in self.domain_shape:
            N *= d

        # ─── Step 1: Macroscopic (needed by ALM before collision) ───
        self.rho, self.u = self.macroscopic.compute(self.f)

        # ─── Step 2: Body Force (ALM pipeline) ──────────────────────
        self.body_force = self._compute_body_force(self.u, self.rho)

        # ─── Step 3+4: Guo correction + collision (fused kernel) ────
        omega = 1.0 / self.tau
        if self._fused_is_cumulant:
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
            if self._hwbb_kernel is not None:
                self._hwbb_kernel.apply(
                    self.f, self._f_post,
                    self.obstacle_bc.needs_bounce, N,
                )
                self._hwbb_kernel.reset_solid(
                    self.f, self.obstacle_bc.solid_mask, N,
                )
            else:
                self.obstacle_bc.apply_with_reset(self.f, self._f_post)

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