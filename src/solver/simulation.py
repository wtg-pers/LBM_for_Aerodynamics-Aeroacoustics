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

        # ── Fused CUDA kernel (lazy init) ──
        self._fused_kernel = None
        self._use_fused: bool = False

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

        # ── Try to initialize fused CUDA kernel ──
        self._use_fused = False
        self._fused_is_cumulant = False
        if self.xp.__name__ == 'cupy' and len(self.domain_shape) == 3:
            from src.collision.bgk import BGKCollision
            from src.collision.cumulant import CumulantCollision
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

        # ─── Streaming (1 CuPy operation) ───────────────────────────
        self.streaming.compute(self._f_post, self.f)

        # ─── Boundary Conditions ─────────────────────────────────────
        self.bc_manager.apply_all(self.f, self._f_post)
        if self.obstacle_bc is not None:
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