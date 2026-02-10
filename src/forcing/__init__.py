"""
External Force (Body Force) Framework for LBM

This module provides forcing schemes to incorporate external body forces
into the LBM time evolution equation.

Physical Background:
===================
The LBM with external force modifies two aspects of the standard algorithm:

1. Macroscopic velocity correction:
   ρu = Σ ξ_i f_i + F·Δt/2

2. Forcing source term in collision:
   f_i^post = f_i - ω(f_i - f_i^eq) + S_i

where S_i depends on the specific forcing scheme.

Available Schemes:
    - GuoForcing: Guo et al. (2002) - second-order accurate, most widely used
    - (Future) EDMForcing: Exact Difference Method (He et al., 1998)

Applications:
    - Gravity-driven flows (Poiseuille with body force)
    - Actuator Line model (wind turbine blade forces)
    - Sponge layers (damping forces)
    - Buoyancy-driven flows (thermal LBM)

References:
    - Guo et al., Phys. Rev. E 65, 046308, 2002
    - He et al., J. Stat. Phys. 87, 115-136, 1997
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 6

Author: LBM Development Team
Date: 2026-02
"""

from .base import ForcingScheme
from .guo_forcing import GuoForcing

__all__ = ['ForcingScheme', 'GuoForcing']


def get_forcing_scheme(name: str, xp, lattice, **kwargs):
    """Factory function to get forcing scheme by name
    
    Args:
        name: Scheme name ('guo', 'edm', 'none')
        xp: Array module (numpy or cupy)
        lattice: Lattice model (D2Q9, D3Q27)
        **kwargs: Additional scheme-specific parameters
        
    Returns:
        ForcingScheme instance, or None if name is 'none'
        
    Raises:
        ValueError: If scheme name is not recognized
        
    Example:
        >>> forcing = get_forcing_scheme('guo', np, lattice)
    """
    schemes = {
        'guo': GuoForcing,
        # 'edm': EDMForcing,  # Future
    }
    
    name_lower = name.lower().strip()
    
    if name_lower in ('none', '', 'disabled'):
        return None
    
    if name_lower not in schemes:
        available = ', '.join(sorted(schemes.keys()))
        raise ValueError(
            f"Unknown forcing scheme: '{name}'. Available: {available}"
        )
    
    return schemes[name_lower](xp, lattice, **kwargs)