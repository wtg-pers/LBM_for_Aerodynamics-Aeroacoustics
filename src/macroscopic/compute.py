"""
Macroscopic Variable Computation for LBM

This module computes macroscopic quantities (density, velocity) from
the velocity distribution function.

Physical Formulae:
=================
Without external force:
    ρ(x, t) = Σ_i f_i(x, t)                    [dimensionless]  (Eq.3)
    ρu(x, t) = Σ_i ξ_i f_i(x, t)              [Δx/Δt]          (Eq.4a)

With external force F (Guo forcing):
    ρ(x, t) = Σ_i f_i(x, t)                    [dimensionless]  (unchanged)
    ρu(x, t) = Σ_i ξ_i f_i(x, t) + F·Δt/2    [Δx/Δt]          (Eq.4b)

    The F·Δt/2 correction ensures second-order accuracy in the
    recovered Navier-Stokes equations (Guo et al., 2002).
    In lattice units (Δt = 1): ρu = Σ ξ_i f_i + F/2

Note:
    Density is ALWAYS computed without force correction.
    Only momentum/velocity receives the half-force correction.

References:
    - Guo et al., Phys. Rev. E 65, 046308, 2002 (Eq.4)
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Sec. 6.3

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class Macroscopic:
    """Compute macroscopic variables from distribution function
    
    Supports both force-free and body-force-corrected computations.
    Compatible with 2D (D2Q9) and 3D (D3Q27) lattice models.
    
    Usage:
        >>> macro = Macroscopic(xp, lattice)
        >>> rho, u = macro.compute(f)                  # Without force
        >>> rho, u = macro.compute(f, force=F)         # With force correction
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice') -> None:
        """Initialize macroscopic calculator
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model with velocity vectors c
        """
        self.xp = xp
        self.lattice = lattice
        self.c = lattice.c   # (dim, Q)  [Δx/Δt]
        self.dim = lattice.dim
    
    def compute(
        self,
        f: 'npt.NDArray',
        force: Optional['npt.NDArray'] = None
    ) -> Tuple['npt.NDArray', 'npt.NDArray']:
        """Compute density and velocity from distribution function
        
        Physical Process:
            1. Density:   ρ = Σ_i f_i
            2. Momentum:  ρu = Σ_i c_i f_i  (+F/2 if force is given)
            3. Velocity:  u = momentum / ρ
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny[, Nz])  [dimensionless]
            force: External body force (optional),
                   shape (dim, Nx, Ny[, Nz])  [lattice force units]
                   If provided, velocity is corrected: u = (Σξ_i f_i + F/2) / ρ
        
        Returns:
            (rho, u): Tuple of density and velocity
                rho: shape (Nx, Ny[, Nz])  [dimensionless]
                u: shape (dim, Nx, Ny[, Nz])  [Δx/Δt]
        """
        # Step 1: Density — always without force correction
        # ρ = Σ_i f_i
        rho = self.xp.sum(f, axis=0)  # (Nx, Ny[, Nz])  [dimensionless]
        
        # Step 2: Momentum
        # ρu = Σ_i c_i · f_i  → tensordot: c(dim, Q) · f(Q, ...) = momentum(dim, ...)
        momentum = self.xp.tensordot(
            self.c, f, axes=(1, 0)
        )  # (dim, Nx, Ny[, Nz])  [Δx/Δt × dimensionless]
        
        # Step 3: Force correction (if applicable)
        # ρu_corrected = ρu_raw + F · Δt/2  (Δt = 1 in lattice units)
        if force is not None:
            momentum = momentum + 0.5 * force  # [Δx/Δt]
        
        # Step 4: Velocity
        # u = momentum / ρ
        u = momentum / rho[None, ...]  # (dim, Nx, Ny[, Nz])  [Δx/Δt]
        
        return rho, u
    
    def compute_density(self, f: 'npt.NDArray') -> 'npt.NDArray':
        """Compute density only (no force correction needed)
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny[, Nz])  [dimensionless]
            
        Returns:
            rho: Density field, shape (Nx, Ny[, Nz])  [dimensionless]
        """
        return self.xp.sum(f, axis=0)
    
    def compute_momentum(
        self,
        f: 'npt.NDArray',
        force: Optional['npt.NDArray'] = None
    ) -> 'npt.NDArray':
        """Compute momentum (with optional force correction)
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny[, Nz])  [dimensionless]
            force: External body force (optional),
                   shape (dim, Nx, Ny[, Nz])  [lattice force units]
            
        Returns:
            momentum: shape (dim, Nx, Ny[, Nz])  [Δx/Δt]
        """
        momentum = self.xp.tensordot(self.c, f, axes=(1, 0))
        
        if force is not None:
            momentum = momentum + 0.5 * force
        
        return momentum