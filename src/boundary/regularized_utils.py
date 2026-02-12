"""
Regularized Distribution Function Utilities

Low-level functions for computing equilibrium and non-equilibrium
components of the LBM distribution function at boundary nodes.

Physical Basis (Chapman-Enskog Expansion):
    f_i ≈ f_i^eq(ρ, u) + f_i^(1)(Π^neq)

    Equilibrium:
        f_i^eq = w_i ρ (1 + c_i·u/cs² + (c_i·u)²/(2cs⁴) - u²/(2cs²))

    First-order non-equilibrium:
        f_i^(1) = (w_i / 2cs⁴) Σ_αβ Q_iαβ Π^neq_αβ
        Q_iαβ = c_iα c_iβ - cs² δ_αβ

    where Π^neq is the non-equilibrium stress tensor:
        Π^neq_αβ = Σ_i (f_i - f_i^eq) c_iα c_iβ

References:
    - Latt & Chopard, Math. Comp. Sim. 72, 2006
    - Malaspinas, Chopard, Latt, Comp. Fluids 49, 2011

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


def compute_f_eq(xp: 'ModuleType',
                 rho: 'npt.NDArray',
                 u: 'npt.NDArray',
                 c: 'npt.NDArray',
                 w: 'npt.NDArray',
                 cs2: float) -> 'npt.NDArray':
    """Compute equilibrium distribution function.
    
    f_i^eq = w_i ρ (1 + c_i·u/cs² + (c_i·u)²/(2cs⁴) - |u|²/(2cs²))
    
    This is a standalone utility that does NOT require a Maxwellian object
    and works with arbitrary spatial shapes.
    
    Args:
        xp: Array module (numpy or cupy)
        rho: Density field, shape (...)  [dimensionless]
        u: Velocity field, shape (dim, ...)  [Δx/Δt]
        c: Lattice velocities, shape (dim, Q)  [Δx/Δt]
        w: Lattice weights, shape (Q,)  [dimensionless]
        cs2: Speed of sound squared = 1/3  [Δx²/Δt²]
        
    Returns:
        f_eq: Equilibrium distribution, shape (Q, ...)  [dimensionless density]
    """
    c_float = c.astype(xp.float64)
    
    usqr = xp.sum(u * u, axis=0)                              # |u|² [Δx²/Δt²]
    cu = xp.einsum('di,d...->i...', c_float, u)               # c·u   [Δx²/Δt²]
    
    # Broadcast weights: (Q,) → (Q, 1, 1, ...)
    ndim_extra = rho.ndim
    w_bc = w.reshape((-1,) + (1,) * ndim_extra)
    
    f_eq = w_bc * rho * (
        1.0 
        + cu / cs2 
        + 0.5 * cu * cu / (cs2 * cs2) 
        - 0.5 * usqr / cs2
    )
    
    return f_eq


def compute_Pi_neq(xp: 'ModuleType',
                   f: 'npt.NDArray',
                   c: 'npt.NDArray',
                   w: 'npt.NDArray',
                   cs2: float) -> 'npt.NDArray':
    """Compute non-equilibrium stress tensor Π^neq from distribution function.
    
    Π^neq_αβ = Σ_i (f_i - f_i^eq) c_iα c_iβ
    
    Physical meaning:
        In the Navier-Stokes limit:
            Π^neq_αβ ≈ -2ρ cs² (τ - 0.5) S_αβ
        where S_αβ = (∂u_α/∂x_β + ∂u_β/∂x_α) / 2 is the strain rate tensor.
    
    Args:
        xp: Array module (numpy or cupy)
        f: Distribution function, shape (Q, ...)  [density]
        c: Lattice velocities, shape (dim, Q)  [Δx/Δt]
        w: Lattice weights, shape (Q,)  [dimensionless]
        cs2: Speed of sound squared = 1/3  [Δx²/Δt²]
        
    Returns:
        Π^neq tensor, shape (dim, dim, ...)  [density × Δx²/Δt²]
    """
    # --- Macroscopic quantities ---
    rho = xp.sum(f, axis=0)                                          # [density]
    c_float = c.astype(xp.float64)
    u = xp.einsum('di,i...->d...', c_float, f) / (rho + 1e-30)     # [Δx/Δt]
    
    # --- Equilibrium ---
    usqr = xp.sum(u * u, axis=0)                                     # [Δx²/Δt²]
    cu = xp.einsum('di,d...->i...', c_float, u)                      # [Δx²/Δt²]
    
    ndim_extra = f.ndim - 1
    w_bc = w.reshape((-1,) + (1,) * ndim_extra)
    
    f_eq = w_bc * rho * (1.0 + cu / cs2 + 0.5 * cu**2 / (cs2**2) - 0.5 * usqr / cs2)
    
    # --- Non-equilibrium ---
    f_neq = f - f_eq  # [density]
    
    # --- Stress tensor: Π^neq_αβ = Σ_i f_neq_i × c_iα × c_iβ ---
    dim = c.shape[0]
    spatial_shape = f.shape[1:]
    Pi_neq = xp.zeros((dim, dim) + spatial_shape, dtype=xp.float64)
    
    for alpha in range(dim):
        for beta in range(alpha, dim):
            cc = c_float[alpha] * c_float[beta]  # (Q,)
            cc_bc = cc.reshape((-1,) + (1,) * ndim_extra)
            val = xp.sum(f_neq * cc_bc, axis=0)
            Pi_neq[alpha, beta] = val
            if alpha != beta:
                Pi_neq[beta, alpha] = val  # symmetric tensor
    
    return Pi_neq


def reconstruct_f_regularized(xp: 'ModuleType',
                               rho: 'npt.NDArray',
                               u: 'npt.NDArray',
                               Pi_neq: 'npt.NDArray',
                               c: 'npt.NDArray',
                               w: 'npt.NDArray',
                               cs2: float) -> 'npt.NDArray':
    """Reconstruct ALL distributions using regularized approach.
    
    f_i = f_i^eq(ρ, u) + f_i^(1)(Π^neq)
    
    where:
        f_i^(1) = (w_i / 2cs⁴) Σ_αβ Q_iαβ Π^neq_αβ
        Q_iαβ = c_iα c_iβ - cs² δ_αβ
    
    Used for flat nodes on regularized face BCs (non-equilibrium inlet,
    pressure outlet, regularized wall). Edge/corner nodes should use
    compute_f_eq() instead (f_neq = 0).
    
    Args:
        xp: Array module
        rho: Target density, shape (...)  [density]
        u: Target velocity, shape (dim, ...)  [Δx/Δt]
        Pi_neq: Stress tensor from interior, shape (dim, dim, ...)  [density×Δx²/Δt²]
        c: Lattice velocities, shape (dim, Q)  [Δx/Δt]
        w: Lattice weights, shape (Q,)  [dimensionless]
        cs2: Speed of sound squared  [Δx²/Δt²]
        
    Returns:
        Reconstructed distribution, shape (Q, ...)  [density]
    """
    # Equilibrium part
    f_eq = compute_f_eq(xp, rho, u, c, w, cs2)
    
    # Non-equilibrium part: f^(1) = (w_i / 2cs⁴) Σ_αβ Q_iαβ Π^neq_αβ
    Q = w.shape[0]
    dim = c.shape[0]
    spatial_shape = rho.shape
    ndim_extra = len(spatial_shape)
    
    c_float = c.astype(xp.float64)
    cs4 = cs2 * cs2
    
    f_neq = xp.zeros((Q,) + spatial_shape, dtype=xp.float64)
    
    for alpha in range(dim):
        for beta in range(dim):
            # Q_iαβ = c_iα c_iβ - cs² δ_αβ
            cc = c_float[alpha] * c_float[beta]
            delta = cs2 if alpha == beta else 0.0
            Q_iab = cc - delta  # (Q,)
            
            Q_iab_bc = Q_iab.reshape((-1,) + (1,) * ndim_extra)
            f_neq += Q_iab_bc * Pi_neq[alpha, beta]
    
    w_bc = w.reshape((-1,) + (1,) * ndim_extra)
    f_neq = w_bc / (2.0 * cs4) * f_neq
    
    return f_eq + f_neq