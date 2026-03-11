"""
LBM Simulation — Single Timestep Physics

This module defines the Simulation class, which encapsulates the core LBM
physics for one timestep. The `advance()` method performs all physical
operations in a clear, readable sequence matching the continuous equations:

    1. Macroscopic (ρ, u from f)
    2. Body Force  (ALM pipeline → F(x))
    3. Velocity Correction (Guo: u += F/(2ρ))
    4. Equilibrium (f_eq from ρ, u)
    5. Forcing Source Term (S_i from F, ρ, u, τ)
    6. Collision (BGK: f* = f - ω(f - f_eq) + S_i)
    7. Streaming (pull: f' = f*(x - c_i))
    8. Boundary Conditions (domain faces + internal obstacles)
    9. Buffer Swap (f ← f')

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
    from src.equilibrium.equilibrium import Maxwellian
    from src.collision.bgk import BGK
    from src.streaming.stream import StreamingPull
    from src.forcing.guo_forcing import GuoForcing
    from src.boundary.domain_bc_manager import DomainBCManager
    from src.boundary.wall import HalfwayBounceBack


class Simulation:
    """Core LBM simulation state and single-step physics.

    This class owns the distribution function buffers (f, f_post, f_new)
    and all physical operators. One call to `advance()` performs exactly
    one LBM timestep.

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
        equilibrium: 'Maxwellian',
        collision: 'BGK',
        streaming: 'StreamingPull',
        forcing_scheme: 'GuoForcing',
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
            equilibrium: Equilibrium distribution calculator (f_eq from ρ, u)
            collision: Collision operator (BGK)
            streaming: Streaming operator (pull scheme)
            forcing_scheme: Forcing source term calculator (Guo)
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
        self.equilibrium = equilibrium
        self.collision = collision
        self.streaming = streaming
        self.forcing_scheme = forcing_scheme

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
        self._f_new: Optional['npt.NDArray'] = None  # Post-streaming buffer [dimensionless]
        self._f_post: Optional['npt.NDArray'] = None # Post-collision buffer [dimensionless]

        # ── Counters ──
        self.step_count: int = 0
        self._is_ready: bool = False

    # =====================================================================
    # Public Interface
    # =====================================================================

    def set_distribution(self, f: 'npt.NDArray') -> None:
        """Set distribution function and allocate work buffers.

        This is the bridge between Initialization and Execution:
        after calling this, advance() can be called.

        Args:
            f: Initial distribution function, shape (Q, Nx, Ny[, Nz])
               [dimensionless]
        """
        self.f = f
        self._f_new = self.xp.empty_like(f)
        self._f_post = self.xp.empty_like(f)
        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        """Whether set_distribution() has been called."""
        return self._is_ready

    @property
    def f_post(self) -> Optional['npt.NDArray']:
        """Post-collision distribution (read-only).

        Needed by MEM (Momentum Exchange Method) force calculation.
        Valid after the most recent advance() call.

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
            4. Equilibrium:     f_eq = Maxwellian(ρ, u)
            5. Forcing:         S_i = Guo source(F, ρ, u, τ)
            6. Collision:       f* = f - ω(f - f_eq) + S_i
            7. Streaming:       f' = f*(x - c_i)    (pull scheme)
            8. Boundary Cond:   domain faces + obstacles
            9. Buffer Swap:     f ← f'

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

        # ─── Step 1: Macroscopic Variables ───────────────────────────
        #   ρ(x) = Σ_i f_i(x)                    [dimensionless]
        #   u(x) = Σ_i c_i·f_i(x) / ρ(x)        [Δx/Δt]
        self.rho, self.u = self.macroscopic.compute(self.f)

        # ─── Step 2: Body Force (ALM pipeline) ──────────────────────
        #   F(x) from Actuator Line Model (BEM → Gaussian spreading)
        #   Returns None if no ALM is active
        self.body_force = self._compute_body_force(self.u, self.rho)

        # ─── Step 3: Guo Velocity Correction ────────────────────────
        #   u_corrected = u_raw + F / (2ρ)        [Δx/Δt]
        #   (Guo et al. 2002, Eq.4: ρu = Σξ_i·f_i + F·Δt/2)
        #   In lattice units (Δt=1): u += F/(2ρ)
        u = self.u
        if self.body_force is not None:
            u = u + self.body_force / (2.0 * self.rho[None, ...])
            self.u = u  # Store force-corrected velocity for output

        # ─── Step 4: Equilibrium Distribution ────────────────────────
        #   f_eq = w_i · ρ · (1 + 3(c_i·u) + 4.5(c_i·u)² - 1.5|u|²)
        #   [dimensionless]
        f_eq = self.equilibrium.compute(self.rho, u)

        # ─── Step 5: Forcing Source Term ─────────────────────────────
        #   S_i = (1 - 1/(2τ)) · w_i · [(c_i - u)/cs² + (c_i·u)·c_i/cs⁴] · F
        #   [dimensionless]
        if self.body_force is not None:
            S_i = self.forcing_scheme.compute_source_term(
                self.body_force, self.rho, u, self.tau
            )
        else:
            S_i = None

        # ─── Step 6: Collision (BGK + source term) ───────────────────
        #   f* = f - ω(f - f_eq) + S_i
        #   where ω = 1/τ  [1/Δt]
        self._f_post[:] = self.collision.collide(
            self.f, f_eq, self.tau, source=S_i
        )

        # ─── Step 7: Streaming (Pull scheme) ─────────────────────────
        #   f'(x, t+1) = f*(x - c_i, t)
        self.streaming.compute(self._f_post, self._f_new)

        # ─── Step 8: Boundary Conditions ─────────────────────────────
        #   Phase 1: Domain faces (equilibrium, Neumann, etc.)
        self.bc_manager.apply_all(self._f_new, self._f_post)

        #   Phase 2: Internal obstacles (HalfwayBounceBack)
        if self.obstacle_bc is not None:
            self.obstacle_bc.apply_with_reset(self._f_new, self._f_post)

        # ─── Step 9: Buffer Swap ─────────────────────────────────────
        #   f ← f'  (new becomes current)
        self.f, self._f_new = self._f_new, self.f
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