"""
BGK Collision Operator with Integrated Equilibrium and Guo Forcing

This module implements the complete BGK collision step as a single
operation, following the standard LBM formulation where equilibrium
and forcing are integral parts of the collision process.

Physical Formula:
    f_i* = f_i − ω(f_i − f_i^eq) + S_i

where:
    ω = 1/τ                         [1/Δt]   relaxation rate
    f_i^eq = Maxwellian equilibrium  [dimensionless]
    S_i = Guo forcing source term    [dimensionless]

Recovered macroscopic equations (to second order in Δt):
    ∂ρ/∂t + ∇·(ρu) = 0                          (continuity)
    ∂(ρu)/∂t + ∇·(ρuu) = −∇p + ∇·(μ∇u) + F    (momentum)

where:
    p = ρ · cs²                     (equation of state)
    ν = cs² · (τ − 0.5)            (kinematic viscosity, unaffected by F)

References:
    - Bhatnagar, Gross, Krook, Phys. Rev. 94, 511, 1954
    - Guo, Zheng, Shi, Phys. Rev. E 65, 046308, 2002
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 4 & 6

Author: LBM Development Team
Date: 2026-03
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from src.collision.base import CollisionOperator


class BGKCollision(CollisionOperator):
    """BGK (Single Relaxation Time) Collision with Guo Forcing.

    Computes the full collision step including:
        1. Maxwellian equilibrium f_eq(ρ, u)
        2. BGK relaxation: f − ω(f − f_eq)
        3. Guo forcing source term S_i (if force is provided)

    The equilibrium and forcing are computed internally — the caller
    only provides macroscopic fields (ρ, u) and the force field F.

    Example:
        >>> collision = BGKCollision(xp, lattice)
        >>> collision.collide(f, f_post, rho, u, tau)           # No force
        >>> collision.collide(f, f_post, rho, u, tau, force=F)  # With force
        >>> f_eq = collision.compute_equilibrium(rho0, u0)      # For init
    """

    def __init__(self, xp: 'ModuleType', lattice: 'Lattice') -> None:
        """Initialize BGK collision operator.

        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model with c, w, cs2, dim, Q
        """
        super().__init__(xp, lattice)

        # ── Precompute lattice constants on device ──
        self._c = xp.asarray(lattice.c, dtype=xp.float64)   # (dim, Q)  [Δx/Δt]
        self._w = xp.asarray(lattice.w, dtype=xp.float64)   # (Q,)      [dimensionless]
        self._cs2: float = float(lattice.cs2)                # 1/3       [Δx²/Δt²]

        # Guo forcing constants
        self._inv_cs2: float = 1.0 / self._cs2               # 3.0  [Δt²/Δx²]
        self._inv_cs4: float = 1.0 / (self._cs2 ** 2)        # 9.0  [Δt⁴/Δx⁴]

    # =====================================================================
    # CollisionOperator interface
    # =====================================================================

    def collide(
        self,
        f_in: 'npt.NDArray',
        f_out: 'npt.NDArray',
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
        tau: float,
        force: Optional['npt.NDArray'] = None,
    ) -> None:
        """Perform BGK collision step in-place.

        Physical Process:
            1. Compute equilibrium:  f_eq = Maxwellian(ρ, u)
            2. BGK relaxation:       f* = f − ω(f − f_eq)
            3. Add forcing:          f* += S_i(F, ρ, u, τ)   (if F given)

        Args:
            f_in:  Pre-collision distribution (Q, Nx, Ny[, Nz])     [dimensionless]
            f_out: Post-collision result — written in-place          [dimensionless]
            rho:   Density field (Nx, Ny[, Nz])                     [dimensionless]
            u:     Velocity (Guo-corrected) (dim, Nx, Ny[, Nz])     [Δx/Δt]
            tau:   Relaxation time                                   [Δt]
            force: Body force (dim, Nx, Ny[, Nz]) or None           [lattice force / lu³]
        """
        # ── Step 1: Equilibrium ──────────────────────────────────
        #   f_eq = w_i · ρ · (1 + 3(c·u) + 4.5(c·u)² − 1.5|u|²)
        f_eq = self.compute_equilibrium(rho, u)

        # ── Step 2: BGK relaxation ───────────────────────────────
        #   f* = f − ω(f − f_eq)
        omega = 1.0 / tau   # [1/Δt]
        f_out[:] = f_in - omega * (f_in - f_eq)

        # ── Step 3: Guo forcing source term ──────────────────────
        #   S_i = (1 − 1/(2τ)) · w_i · [(c_i−u)/cs² + (c_i·u)·c_i/cs⁴] · F
        if force is not None:
            f_out += self._guo_source_term(force, rho, u, tau)

    def compute_equilibrium(
        self,
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
    ) -> 'npt.NDArray':
        """Compute Maxwellian equilibrium distribution.

        Physical Formula:
            f_i^eq = w_i · ρ · [1 + (c_i · u)/cs²
                                  + (c_i · u)²/(2·cs⁴)
                                  − |u|²/(2·cs²)]

        With cs² = 1/3:
            f_i^eq = w_i · ρ · (1 + 3(c_i·u) + 4.5(c_i·u)² − 1.5|u|²)

        Args:
            rho: Density field (Nx, Ny[, Nz])               [dimensionless]
            u:   Velocity field (dim, Nx, Ny[, Nz])          [Δx/Δt]

        Returns:
            f_eq: Equilibrium distribution (Q, Nx, Ny[, Nz]) [dimensionless]
        """
        xp = self.xp

        # |u|² = Σ_α u_α²  → (Nx, Ny[, Nz])  [Δx²/Δt²]
        usqr = xp.sum(u ** 2, axis=0)

        # c_i · u  → (Q, Nx, Ny[, Nz])  [Δx²/Δt²]
        cu = xp.einsum('dq,d...->q...', self._c, u)

        # Broadcast weights: (Q,) → (Q, 1, 1[, 1])
        w_bc = self._w.reshape((-1,) + (1,) * rho.ndim)

        # f_eq = w_i · ρ · (1 + 3·(c·u) + 4.5·(c·u)² − 1.5·|u|²)
        return w_bc * rho * (
            1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * usqr
        )  # [dimensionless]

    # =====================================================================
    # Internal: Guo forcing
    # =====================================================================

    def _guo_source_term(
        self,
        force: 'npt.NDArray',
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
        tau: float,
    ) -> 'npt.NDArray':
        """Compute Guo forcing source term S_i.

        Physical Formula (Guo et al., 2002):
            S_i = (1 − 1/(2τ)) · w_i ·
                  [(c_iα − u_α)/cs² + (c_iβ·u_β)·c_iα/cs⁴] · F_α

        The prefactor (1 − 1/(2τ)) ensures the forcing does not
        alter the recovered kinematic viscosity.

        Args:
            force: Body force (dim, Nx, Ny[, Nz])   [lattice force / lu³]
            rho:   Density field (Nx, Ny[, Nz])      [dimensionless]
            u:     Velocity (dim, Nx, Ny[, Nz])       [Δx/Δt]
            tau:   Relaxation time                     [Δt]

        Returns:
            S_i: Source term (Q, Nx, Ny[, Nz])        [dimensionless]
        """
        xp = self.xp
        c = self._c   # (dim, Q)  [Δx/Δt]
        w = self._w   # (Q,)      [dimensionless]

        # Prefactor: (1 − 1/(2τ))  [dimensionless]
        prefactor = 1.0 - 0.5 / tau

        # ── Term 1: (c_iα − u_α) · F_α / cs² ───────────────────
        # c_i · F : (Q, Nx, Ny[, Nz])
        ci_dot_F = xp.einsum(
            'dq, d... -> q...', c.astype(xp.float64), force,
        )
        # u · F : (Nx, Ny[, Nz])
        u_dot_F = xp.sum(u * force, axis=0)
        # Term1 = (c_i·F − u·F) / cs²
        term1 = (ci_dot_F - u_dot_F[None, ...]) * self._inv_cs2

        # ── Term 2: (c_iβ · u_β) · (c_iα · F_α) / cs⁴ ─────────
        # c_i · u : (Q, Nx, Ny[, Nz])
        ci_dot_u = xp.einsum(
            'dq, d... -> q...', c.astype(xp.float64), u,
        )
        term2 = ci_dot_u * ci_dot_F * self._inv_cs4

        # ── Combine: S_i = prefactor · w_i · (term1 + term2) ────
        w_bc = w.reshape((-1,) + (1,) * rho.ndim)

        return prefactor * w_bc * (term1 + term2)  # [dimensionless]

    # =====================================================================
    # Info
    # =====================================================================

    def __repr__(self) -> str:
        return (f"BGKCollision(D{self.dim}Q{self.Q}, "
                f"cs²={self._cs2:.4f}, Guo forcing integrated)")