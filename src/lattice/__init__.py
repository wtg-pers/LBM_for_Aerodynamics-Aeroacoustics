"""
Lattice Models for LBM

This module provides lattice model implementations for the Lattice Boltzmann Method.

Available Models:
    - D2Q9: 2D lattice with 9 velocities (standard 2D)
    - D3Q27: 3D lattice with 27 velocities (full 3D, most accurate)

Future Models (planned):
    - D3Q19: 3D lattice with 19 velocities (reduced memory, slightly less accurate)
    - D3Q15: 3D lattice with 15 velocities (minimal 3D)

Usage:
    >>> from src.lattice import D2Q9, D3Q27
    >>> import numpy as np
    >>> 
    >>> # For 2D simulations
    >>> lattice_2d = D2Q9(np)
    >>> 
    >>> # For 3D simulations
    >>> lattice_3d = D3Q27(np)

Each lattice class provides:
    - c: Velocity vectors, shape (dim, Q)  [Δx/Δt]
    - w: Weights, shape (Q,)  [dimensionless]
    - cs2: Speed of sound squared = 1/3  [Δx²/Δt²]
    - opp: Opposite direction indices  [dimensionless]
    - Q: Number of discrete velocities
    - dim: Spatial dimensions (2 or 3)

Author: LBM Development Team
Date: 2026-02
"""

from .d2q9 import D2Q9
from .d3q27 import D3Q27

__all__ = ['D2Q9', 'D3Q27']


def get_lattice(model_name: str, xp):
    """Factory function to get lattice by name
    
    Args:
        model_name: Lattice model name ('D2Q9', 'D3Q27')
        xp: Array module (numpy or cupy)
        
    Returns:
        Lattice instance
        
    Raises:
        ValueError: If model_name is not recognized
        
    Example:
        >>> lattice = get_lattice('D2Q9', np)
    """
    models = {
        'D2Q9': D2Q9,
        'd2q9': D2Q9,
        'D3Q27': D3Q27,
        'd3q27': D3Q27,
    }
    
    if model_name not in models:
        available = ', '.join(sorted(set(k.upper() for k in models.keys())))
        raise ValueError(f"Unknown lattice model: '{model_name}'. Available: {available}")
    
    return models[model_name](xp)