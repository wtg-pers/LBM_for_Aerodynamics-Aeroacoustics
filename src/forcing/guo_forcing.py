"""
Guo Forcing Scheme for LBM

Implements the forcing method of Guo, Zheng & Shi (2002) to incorporate
external body forces into the Lattice Boltzmann equation.

Physical Derivation:
===================
Starting from the Boltzmann equation with force term:
    ∂f/∂t + ξ·∇f = Ω(f) + S

The discrete forcing source term S_i is constructed to recover the
correct macroscopic equations (Navier-Stokes) to second order in Δt:

    S_i = (1 - 1/(2τ)) · w_i · [
        (c_iα - u_α) / cs²  +  (c_iβ · u_β) · c_iα / cs⁴
    ] · F_α

where:
    c_i = lattice velocity vector i              [Δx/Δt]
    u   = macroscopic velocity (force-corrected)  [Δx/Δt]
    F   = external body force per unit volume     [lattice force units]
    τ   = relaxation time                         [Δt]
    cs² = 1/3                                     [Δx²/Δt²]
    w_i = lattice weight                          [dimensionless]

The factor (1 - 1/(2τ)) ensures correct viscosity in the recovered
Navier-Stokes equations. Without it, the kinematic viscosity would
be incorrectly modified by the forcing.

Modified LBM Time Evolution:
===========================
    f_i(x + c_i·Δt, t + Δt) = f_i(x, t) - ω·[f_i - f_i^eq] + S_i·Δt

With Δt = 1 in lattice units:
    f_i^post = f_i - ω·(f_i - f_i^eq) + S_i

Modified Macroscopic Velocity:
    ρu = Σ ξ_i f_i + F·Δt/2 = Σ ξ_i f_i + F/2

Recovered Navier-Stokes (to second order):
    ∂ρ/∂t + ∇·(ρu) = 0                                    (continuity)
    ∂(ρu)/∂t + ∇·(ρuu) = -∇p + ∇·(μ∇u) + F              (momentum)

where:
    p = ρ·cs²     (equation of state)
    ν = cs²·(τ - 1/2)·Δt  (kinematic viscosity, UNAFFECTED by forcing)

Verification:
============
This implementation can be verified with gravity-driven Poiseuille flow:
    u_x(y) = F_x/(2ν) · y · (H - y)
    
where H is the channel height and F_x is the body force magnitude.

References:
==========
    - Guo Z, Zheng C, Shi B, Phys. Rev. E 65, 046308, 2002
      "Discrete lattice effects on the forcing term in the lattice
       Boltzmann method"
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017,
      Section 6.3.2 (Guo forcing)
    - Mohamad A, "Lattice Boltzmann Method", Springer 2011, Ch. 9

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .base import ForcingScheme


class GuoForcing(ForcingScheme):
    """Guo Forcing Scheme (Guo et al., 2002)
    
    Second-order accurate forcing for incorporating external body forces
    into the LBM collision operator. The most widely used and recommended
    forcing scheme for general LBM applications.
    
    Key Properties:
        - Recovers correct Navier-Stokes to O(Δt²)
        - Viscosity is independent of the applied force
        - Compatible with BGK, MRT, and other collision operators
        - Suitable for both uniform and spatially varying forces
    
    Usage in Time Loop:
        1. Compute raw moments: ρ = Σf_i, momentum = Σξ_i·f_i
        2. Correct velocity: u = (momentum + F/2) / ρ
        3. Compute equilibrium: f_eq(ρ, u)
        4. Compute source term: S_i = GuoForcing.compute_source_term(F, ρ, u, τ)
        5. Collision with forcing: f_post = f - ω(f - f_eq) + S_i
        6. Stream and apply BCs as usual
    
    Example:
        >>> import numpy as np
        >>> from src.lattice import D2Q9
        >>> from src.forcing import GuoForcing
        >>>
        >>> lattice = D2Q9(np)
        >>> guo = GuoForcing(np, lattice)
        >>>
        >>> # Gravity in x-direction
        >>> F = np.zeros((2, Nx, Ny))
        >>> F[0, :, :] = 1e-6  # [lattice force units]
        >>>
        >>> # In time loop:
        >>> S = guo.compute_source_term(F, rho, u, tau)
        >>> f_post = collision.collide(f, f_eq, tau) + S
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice') -> None:
        """Initialize Guo forcing scheme
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
        """
        super().__init__(xp, lattice)
        
        # Precompute constants for efficiency
        self._inv_cs2 = 1.0 / self.cs2          # = 3.0  [Δt²/Δx²]
        self._inv_cs4 = 1.0 / (self.cs2 ** 2)   # = 9.0  [Δt⁴/Δx⁴]
    
    def compute_source_term(
        self,
        F: 'npt.NDArray',
        rho: 'npt.NDArray',
        u: 'npt.NDArray',
        tau: float
    ) -> 'npt.NDArray':
        """Compute Guo forcing source term S_i
        
        Physical Formula:
        -----------------
        S_i = (1 - 1/(2τ)) · w_i · [(c_iα - u_α)/cs² + (c_iβ·u_β)·c_iα/cs⁴] · F_α
        
        Using Einstein summation convention (α, β summed over spatial dimensions).
        
        Vectorized Implementation:
        -------------------------
        Let: A_α = (c_iα - u_α)/cs² + (c_iβ·u_β)·c_iα/cs⁴
        
        Term 1: (c_i - u)/cs²
            shape: (Q, dim, ...) → contracted with F → (Q, ...)
            
        Term 2: (c_i·u) · c_i / cs⁴
            c_i·u: (Q, ...) — dot product of velocity vectors with u
            Then multiply by c_i and contract with F
            
        Final: S_i = prefactor · w_i · (term1_contracted + term2_contracted)
        
        Args:
            F: Body force, shape (dim, Nx, Ny[, Nz])  [lattice force units]
            rho: Density, shape (Nx, Ny[, Nz])  [dimensionless]
            u: Corrected velocity, shape (dim, Nx, Ny[, Nz])  [Δx/Δt]
            tau: Relaxation time  [Δt]
            
        Returns:
            S_i: Source term, shape (Q, Nx, Ny[, Nz])  [dimensionless]
        """
        xp = self.xp
        c = self.c        # (dim, Q)   [Δx/Δt]
        w = self.w        # (Q,)       [dimensionless]
        
        # Prefactor: (1 - 1/(2τ))  [dimensionless]
        # This factor ensures the forcing does not alter the viscosity
        prefactor = 1.0 - 0.5 / tau
        
        # =====================================================================
        # Term 1: (c_iα - u_α) · F_α / cs²
        # =====================================================================
        # c_i broadcast to spatial dims: c[α, i] → (dim, Q, 1, 1[, 1])
        # u has shape (dim, Nx, Ny[, Nz])
        # F has shape (dim, Nx, Ny[, Nz])
        #
        # (c_iα - u_α) · F_α  summed over α → shape (Q, Nx, Ny[, Nz])
        # =====================================================================
        
        # c_i · F : dot product over spatial dimension → (Q, Nx, Ny[, Nz])
        # einsum: 'dq, d... -> q...'   (d=dim, q=Q, ...=spatial)
        ci_dot_F = xp.einsum('dq, d... -> q...', c.astype(xp.float64), F)
        
        # u · F : dot product over spatial dimension → (Nx, Ny[, Nz])
        u_dot_F = xp.sum(u * F, axis=0)
        
        # Term 1 contribution: (c_i · F - u · F) / cs²
        # = (c_iα - u_α) · F_α / cs²
        term1 = (ci_dot_F - u_dot_F[None, ...]) * self._inv_cs2
        
        # =====================================================================
        # Term 2: (c_iβ · u_β) · (c_iα · F_α) / cs⁴
        # =====================================================================
        # c_i · u : (Q, Nx, Ny[, Nz])
        # c_i · F : already computed as ci_dot_F
        # =====================================================================
        ci_dot_u = xp.einsum('dq, d... -> q...', c.astype(xp.float64), u)
        
        term2 = ci_dot_u * ci_dot_F * self._inv_cs4
        
        # =====================================================================
        # Combine: S_i = prefactor · w_i · (term1 + term2)
        # =====================================================================
        # Broadcast w to (Q, 1, 1[, 1])
        w_bc = w.reshape((-1,) + (1,) * rho.ndim)
        
        S = prefactor * w_bc * (term1 + term2)  # (Q, Nx, Ny[, Nz]) [dimensionless]
        
        return S
    
    def get_info(self) -> str:
        """Return human-readable description"""
        return (
            f"GuoForcing (Guo et al., 2002):\n"
            f"  Lattice: D{self.dim}Q{self.Q}\n"
            f"  cs² = {self.cs2:.4f}\n"
            f"  Properties: 2nd-order accurate, viscosity-independent"
        )