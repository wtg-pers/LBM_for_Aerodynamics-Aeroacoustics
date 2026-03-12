"""
Collision Operator — Abstract Base Class

Defines the unified interface for ALL LBM collision models.
Each collision operator owns its equilibrium and forcing internally.

Physical Context:
    The Boltzmann collision term Ω_i encapsulates the entire relaxation
    process toward equilibrium, including any external forcing:

        f_i(x + c_i, t+1) = f_i(x, t) + Ω_i(f, F)

    For BGK:    Ω_i = -ω(f_i - f_i^eq) + S_i(F)
    For MRT:    Ω_i = -M⁻¹ · S · M · (f - f^eq) + S_i(F)
    For Cumulant: Ω_i computed entirely in cumulant space

    In all cases, equilibrium and forcing are INTERNAL to the collision.

Design:
    collide() performs the full collision step in-place:
        f_out[:] = f_in + Ω(f_in, ρ, u, τ, F)

    compute_equilibrium() is exposed for the Initializer to create
    initial f from (ρ₀, u₀), since every solver needs f_eq at t=0.

References:
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017
    - Geier et al., Comp. & Math. with Appl. 70(4), 507-547, 2015 (Cumulant)

Author: LBM Development Team
Date: 2026-03
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class CollisionOperator(ABC):
    """Abstract interface for LBM collision operators.

    Every collision model (BGK, MRT, Cumulant, ...) implements this
    interface. The collision operator OWNS its equilibrium computation
    and forcing scheme internally — the caller only provides
    macroscopic fields and the force field.

    Subclasses:
        BGKCollision:      f* = f - ω(f - f_eq) + S_i
        CumulantCollision: (future) cumulant-space relaxation

    Usage in Simulation.advance():
        self.collision.collide(f, f_post, rho, u, tau, body_force)
    """

    def __init__(self, xp: 'ModuleType', lattice: 'Lattice') -> None:
        """Initialize collision operator with lattice model.

        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model with c, w, cs2, dim, Q
        """
        self.xp = xp
        self.lattice = lattice
        self.dim: int = lattice.dim
        self.Q: int = lattice.Q

    @abstractmethod
    def collide(
        self,
        f_in: 'npt.NDArray',
        f_out: 'npt.NDArray',
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
        tau: float,
        force: Optional['npt.NDArray'] = None,
    ) -> None:
        """Perform collision step in-place.

        This is the COMPLETE collision operation:
            equilibrium + forcing + relaxation

        Physical Formula (generic):
            f_out[:] = f_in + Ω(f_in, ρ, u, τ, F)

        Args:
            f_in:  Pre-collision distribution (Q, Nx, Ny[, Nz])     [dimensionless]
            f_out: Post-collision result — written in-place          [dimensionless]
            rho:   Density field (Nx, Ny[, Nz])                     [dimensionless]
            u:     Velocity (Guo-corrected) (dim, Nx, Ny[, Nz])     [Δx/Δt]
            tau:   Relaxation time                                   [Δt]
            force: Body force (dim, Nx, Ny[, Nz]) or None           [lattice force / lu³]
        """
        ...

    @abstractmethod
    def compute_equilibrium(
        self,
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
    ) -> 'npt.NDArray':
        """Compute equilibrium distribution f_eq(ρ, u).

        Exposed for the SolverInitializer to create initial f
        at t=0. Each collision model may define its own equilibrium.

        For BGK: standard Maxwellian
            f_eq = w_i · ρ · (1 + 3(c·u) + 4.5(c·u)² - 1.5|u|²)

        For Cumulant: may differ in higher-order terms.

        Args:
            rho: Density field (Nx, Ny[, Nz])                  [dimensionless]
            u:   Velocity field (dim, Nx, Ny[, Nz])             [Δx/Δt]

        Returns:
            f_eq: Equilibrium distribution (Q, Nx, Ny[, Nz])   [dimensionless]
        """
        ...