# =============================================================================
# LBM Lattice Constants
# =============================================================================
#
# Common constants for all lattice models (D2Q9, D3Q19, D3Q27, etc.)
#
# These values arise from the isotropy conditions of the lattice and are
# fundamental to the LBM equation of state p = ρ cs².
#
# References:
#   - Krüger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 3-4
#   - Qian, d'Humières, Lallemand, Europhys. Lett. 17, 1992
#
# =============================================================================

import numpy as np

# =============================================================================
# Speed of Sound
# =============================================================================
#
# For all standard lattice models (D2Q9, D3Q15, D3Q19, D3Q27), the speed
# of sound squared is:
#
#   c_s² = 1/3   [Δx²/Δt²]
#
# This determines:
#   1. Equation of state: p = ρ c_s²
#   2. Relaxation time:   τ = ν / c_s² + 0.5 = 3ν + 0.5
#   3. Mach number:       Ma = u / c_s
#   4. Compressibility:   Ma < 0.1 for accurate incompressible flow
#
# =============================================================================

CS2: float = 1.0 / 3.0           # c_s² = 1/3  [Δx²/Δt²]
CS: float = np.sqrt(CS2)         # c_s ≈ 0.577  [Δx/Δt]
INV_CS2: float = 3.0             # 1/c_s² = 3  [Δt²/Δx²]
CS4: float = CS2 * CS2           # c_s⁴ = 1/9  [Δx⁴/Δt⁴]
INV_CS4: float = 9.0             # 1/c_s⁴ = 9  [Δt⁴/Δx⁴]


# =============================================================================
# Derived Relations
# =============================================================================

def tau_from_nu(nu_lu: float) -> float:
    """
    Compute relaxation time τ from lattice viscosity ν_lu.
    
    τ = ν_lu / c_s² + 0.5 = 3 ν_lu + 0.5
    
    Args:
        nu_lu: Kinematic viscosity in lattice units [lu²/lt]
    
    Returns:
        Relaxation time τ [dimensionless]
    """
    return nu_lu * INV_CS2 + 0.5


def nu_from_tau(tau: float) -> float:
    """
    Compute lattice viscosity ν_lu from relaxation time τ.
    
    ν_lu = (τ - 0.5) × c_s² = (τ - 0.5) / 3
    
    Args:
        tau: Relaxation time τ [dimensionless]
    
    Returns:
        Kinematic viscosity in lattice units [lu²/lt]
    """
    return (tau - 0.5) * CS2


def omega_from_tau(tau: float) -> float:
    """
    Compute relaxation frequency ω from relaxation time τ.
    
    ω = 1/τ
    
    Args:
        tau: Relaxation time τ [dimensionless]
    
    Returns:
        Relaxation frequency ω [dimensionless]
    """
    return 1.0 / tau


def mach_number(u_lu: float) -> float:
    """
    Compute Mach number from lattice velocity.
    
    Ma = u / c_s
    
    Args:
        u_lu: Velocity in lattice units [lu/lt]
    
    Returns:
        Mach number [dimensionless]
    """
    return u_lu / CS


def max_stable_velocity(target_ma: float = 0.1) -> float:
    """
    Compute maximum lattice velocity for target Mach number.
    
    u_max = Ma × c_s
    
    Args:
        target_ma: Target Mach number (default 0.1 for low compressibility)
    
    Returns:
        Maximum lattice velocity [lu/lt]
    """
    return target_ma * CS


# =============================================================================
# Stability Limits
# =============================================================================

TAU_MIN: float = 0.5             # Theoretical minimum (unstable at equality)
TAU_SAFE_MIN: float = 0.52       # Practical minimum for stability
TAU_MAX: float = 2.0             # Practical maximum (accuracy degrades)

MA_MAX_ACCURATE: float = 0.1     # Mach limit for accurate incompressible flow
MA_MAX_STABLE: float = 0.3       # Mach limit before severe compressibility


# =============================================================================
# Module Info
# =============================================================================

__all__ = [
    'CS2', 'CS', 'INV_CS2', 'CS4', 'INV_CS4',
    'tau_from_nu', 'nu_from_tau', 'omega_from_tau',
    'mach_number', 'max_stable_velocity',
    'TAU_MIN', 'TAU_SAFE_MIN', 'TAU_MAX',
    'MA_MAX_ACCURATE', 'MA_MAX_STABLE',
]