"""
BGK (Bhatnagar-Gross-Krook) Collision Operator for LBM

This module implements the single-relaxation-time (SRT) collision model,
also known as the BGK collision operator.

Physical Formula:
================
Standard BGK (without forcing):
    f_i^post = f_i - ω · (f_i - f_i^eq)

    where ω = 1/τ is the relaxation rate  [1/Δt]

With Guo forcing source term S_i:
    f_i^post = f_i - ω · (f_i - f_i^eq) + S_i

    where S_i is computed by the forcing scheme (e.g., GuoForcing)

Recovered viscosity:
    ν = cs² · (τ - 1/2) · Δt = (1/3) · (τ - 1/2)  [Δx²/Δt]
    (in lattice units with Δt = 1, Δx = 1)

Stability constraint:
    τ > 0.5  (otherwise ν < 0, which is unphysical)
    Practical: τ > 0.505 for numerical stability

References:
    - Bhatnagar, Gross, Krook, Phys. Rev. 94, 511, 1954
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 4

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class BGK:
    """BGK (Single Relaxation Time) Collision Operator
    
    The simplest and most commonly used collision model.
    Single parameter τ controls the kinematic viscosity.
    
    Supports optional external forcing source term for body forces
    (gravity, actuator line, sponge layers, etc.).
    
    Example:
        >>> collision = BGK(xp)
        >>> f_post = collision.collide(f, f_eq, tau)              # Standard
        >>> f_post = collision.collide(f, f_eq, tau, source=S_i)  # With forcing
    """
    
    def __init__(self, xp: 'ModuleType') -> None:
        """Initialize BGK collision operator
        
        Args:
            xp: Array module (numpy or cupy)
        """
        self.xp = xp
    
    def collide(
        self,
        f: 'npt.NDArray',
        f_eq: 'npt.NDArray',
        tau: float,
        source: Optional['npt.NDArray'] = None
    ) -> 'npt.NDArray':
        """Perform BGK collision step
        
        Physical Formula:
            f_i^post = f_i - (1/τ) · (f_i - f_i^eq) [+ S_i]
        
        Args:
            f: Pre-collision distribution, shape (Q, Nx, Ny[, Nz])  [dimensionless]
            f_eq: Equilibrium distribution, shape (Q, Nx, Ny[, Nz])  [dimensionless]
            tau: Relaxation time  [Δt]
                 ν = cs²·(τ - 0.5) in lattice units
            source: Forcing source term (optional),
                    shape (Q, Nx, Ny[, Nz])  [dimensionless]
                    Computed by ForcingScheme.compute_source_term()
        
        Returns:
            Post-collision distribution, shape (Q, Nx, Ny[, Nz])  [dimensionless]
            
        Note:
            The source term S_i already includes the (1 - 1/(2τ)) prefactor
            from the Guo forcing scheme. It is simply ADDED to the
            post-collision distribution, not further scaled.
        """
        omega = 1.0 / tau  # Relaxation rate  [1/Δt]
        
        # Standard BGK collision
        f_neq = f - f_eq   # Non-equilibrium part  [dimensionless]
        f_post = f - omega * f_neq  # [dimensionless]
        
        # Add forcing source term (if provided)
        if source is not None:
            f_post = f_post + source  # [dimensionless]
        
        return f_post