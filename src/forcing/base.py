"""
Base Class for LBM Forcing Schemes

This module defines the abstract interface that all forcing schemes must implement.

Physical Context:
================
In the continuous Boltzmann equation with external force F:

    ∂f/∂t + ξ·∇f = Ω(f) + S(F)

where S(F) is the source term due to the external force.

When discretized on a lattice, the forcing affects:
    1. The collision step: additional source term S_i
    2. The macroscopic velocity: ρu = Σ ξ_i f_i + F·Δt/2

Different forcing schemes (Guo, EDM, Shan-Chen, etc.) differ in how
they construct S_i, but all share the velocity correction formula.

References:
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 6
    - Guo et al., Phys. Rev. E 65, 046308, 2002

Author: LBM Development Team
Date: 2026-02
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class ForcingScheme(ABC):
    """Abstract base class for LBM forcing schemes
    
    All forcing schemes must implement:
        - compute_source_term(): S_i added to collision
        - correct_velocity(): velocity correction ρu += F·Δt/2
    
    The forcing modifies the standard LBM time loop:
    
    Standard:  Macro → Eq → Collision → Stream → BC
    With Force: Macro(F) → Eq(u_corrected) → Collision + S_i → Stream → BC
    
    Attributes:
        xp: Array module (numpy or cupy)
        lattice: Lattice model
        dim: Spatial dimension (2 or 3)
        Q: Number of discrete velocities
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice') -> None:
        """Initialize forcing scheme
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
        """
        self.xp = xp
        self.lattice = lattice
        self.dim = lattice.dim
        self.Q = lattice.Q
        self.c = xp.asarray(lattice.c, dtype=xp.float64)   # (dim, Q) [Δx/Δt]
        self.w = xp.asarray(lattice.w, dtype=xp.float64)    # (Q,)     [dimensionless]
        self.cs2 = lattice.cs2                                # 1/3     [Δx²/Δt²]
    
    @abstractmethod
    def compute_source_term(
        self,
        F: 'npt.NDArray',
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
        tau: float
    ) -> 'npt.NDArray':
        """Compute forcing source term S_i
        
        This term is added to the post-collision distribution:
            f_i^post = f_i - ω(f_i - f_i^eq) + S_i
        
        Args:
            F: External body force field, shape (dim, Nx, Ny[, Nz])  [lattice force units]
                Force per unit volume in lattice units.
                Physical force relationship: F_lattice = F_physical * Δx³·Δt / (ρ₀·Δx³·(Δx/Δt))
                Simplified: F_lattice = F_physical * Δt² / (ρ₀·Δx)
            rho: Density field, shape (Nx, Ny[, Nz])  [dimensionless]
            u: Velocity field (force-corrected), shape (dim, Nx, Ny[, Nz])  [Δx/Δt]
            tau: Relaxation time  [Δt]
            
        Returns:
            Source term S_i, shape (Q, Nx, Ny[, Nz])  [dimensionless]
        """
        pass
    
    def correct_velocity(
        self,
        momentum: 'npt.NDArray',
        rho: 'npt.NDArray',
        F: 'npt.NDArray'
    ) -> 'npt.NDArray':
        """Compute force-corrected macroscopic velocity
        
        Physical Formula (Δt = 1 in lattice units):
            ρu = Σ ξ_i f_i + F/2
            u = (Σ ξ_i f_i + F/2) / ρ
        
        This correction ensures second-order accuracy for the
        macroscopic Navier-Stokes equations recovered from LBM.
        
        Args:
            momentum: Raw momentum Σ ξ_i f_i, shape (dim, Nx, Ny[, Nz])  [Δx/Δt]
            rho: Density field, shape (Nx, Ny[, Nz])  [dimensionless]
            F: External body force, shape (dim, Nx, Ny[, Nz])  [lattice force units]
            
        Returns:
            Corrected velocity, shape (dim, Nx, Ny[, Nz])  [Δx/Δt]
            
        Note:
            This formula is universal across all forcing schemes
            (Guo, EDM, Shan-Chen, etc.).
        """
        # u = (momentum + F * Δt/2) / ρ
        # In lattice units, Δt = 1
        corrected_momentum = momentum + 0.5 * F  # [Δx/Δt * dimensionless]
        u = corrected_momentum / rho[None, ...]   # [Δx/Δt]
        return u
    
    def get_info(self) -> str:
        """Return human-readable description of the forcing scheme"""
        return f"{self.__class__.__name__}(dim={self.dim}, Q={self.Q})"
    
    def __repr__(self) -> str:
        return self.get_info()