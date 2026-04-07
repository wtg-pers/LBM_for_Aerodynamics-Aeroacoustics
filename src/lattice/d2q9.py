"""
D2Q9 Lattice Model for 2D Lattice Boltzmann Method

This module defines the D2Q9 (2 dimensions, 9 velocities) lattice model,
the standard choice for 2D incompressible flow simulations.

Lattice Structure:
    
         6  2  5         y (North)
          \ | /          |
        3 - 0 - 1    ->  x (East)
          / | \          
         7  4  8        

Velocity Vectors (c_i):
    i = 0 : ( 0,  0)  - Rest
    i = 1 : ( 1,  0)  - East
    i = 2 : ( 0,  1)  - North
    i = 3 : (-1,  0)  - West
    i = 4 : ( 0, -1)  - South
    i = 5 : ( 1,  1)  - NE (diagonal)
    i = 6 : (-1,  1)  - NW (diagonal)
    i = 7 : (-1, -1)  - SW (diagonal)
    i = 8 : ( 1, -1)  - SE (diagonal)

Weights (w_i):
    w_0     = 4/9   ≈ 0.4444  (rest)
    w_1-4   = 1/9   ≈ 0.1111  (axis-aligned)
    w_5-8   = 1/36  ≈ 0.0278  (diagonal)

Isotropy Conditions Satisfied:
    1. Σ w_i = 1                          (normalization)
    2. Σ w_i c_iα = 0                     (zero momentum at rest)
    3. Σ w_i c_iα c_iβ = c_s² δ_αβ        (pressure tensor isotropy)
    4. Σ w_i c_iα c_iβ c_iγ = 0           (odd moments vanish)
    5. Σ w_i c_iα c_iβ c_iγ c_iδ = c_s⁴ Δ (viscous stress isotropy)

    where c_s² = 1/3 (speed of sound squared) [Δx²/Δt²]

Physical Applications:
    - 2D channel flow (Poiseuille)
    - Lid-driven cavity
    - Flow around 2D cylinder
    - Vortex shedding studies (Karman vortex street)

References:
    - Qian, d'Humières, Lallemand, Europhys. Lett. 17, 1992
    - Chen & Doolen, Annu. Rev. Fluid Mech. 30, 1998
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 3

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class D2Q9:
    """D2Q9 Lattice Model for 2D LBM Simulations
    
    Standard 2D lattice with 9 discrete velocities. Provides all necessary
    lattice parameters for LBM: velocity vectors, weights, speed of sound,
    and opposite direction indices.
    
    Attributes:
        xp: Array module (numpy or cupy)
        Q: Number of discrete velocities (9)  [dimensionless]
        dim: Spatial dimensions (2)  [dimensionless]
        c: Lattice velocity vectors, shape (2, 9)  [Δx/Δt]
        w: Lattice weights, shape (9,)  [dimensionless]
        cs2: Speed of sound squared = 1/3  [Δx²/Δt²]
        opp: Opposite direction indices, shape (9,)  [dimensionless]
        
    Example:
        >>> import numpy as np
        >>> lattice = D2Q9(np)
        >>> print(lattice.c[:, 1])  # East direction
        [1 0]
        >>> print(lattice.w[0])     # Rest weight
        0.4444444444444444
    """
    
    def __init__(self, xp: 'ModuleType', dtype=np.float64) -> None:
        """Initialize D2Q9 lattice model

        Args:
            xp: Array module (numpy or cupy) for GPU/CPU compatibility
            dtype: Computation precision (np.float32 or np.float64)
        """
        self.xp = xp
        self.Q = 9       # Number of discrete velocities  [dimensionless]
        self.dim = 2     # Spatial dimensions  [dimensionless]
        self.dtype = dtype  # Computation precision (float32 or float64)
        
        # =====================================================================
        # Lattice Velocity Vectors c_i
        # =====================================================================
        # Shape: (dim, Q) = (2, 9)
        # Units: [Δx/Δt] (lattice velocity units)
        #
        # Ordering convention:
        #   0: rest
        #   1-4: axis-aligned (E, N, W, S)
        #   5-8: diagonals (NE, NW, SW, SE)
        #
        #       6   2   5
        #         \ | /
        #       3 - 0 - 1
        #         / | \
        #       7   4   8
        # =====================================================================
        self.c = xp.array([
            # i:   0   1   2   3   4   5   6   7   8
            [0,  1,  0, -1,  0,  1, -1, -1,  1],   # c_x  [Δx/Δt]
            [0,  0,  1,  0, -1,  1,  1, -1, -1]    # c_y  [Δx/Δt]
        ], dtype=xp.int32)
        
        # =====================================================================
        # Lattice Weights w_i
        # =====================================================================
        # Units: [dimensionless]
        #
        # Derived from isotropy conditions:
        #   - Rest (|c|² = 0):     w_0 = 4/9
        #   - Axis (|c|² = 1):     w_1-4 = 1/9
        #   - Diagonal (|c|² = 2): w_5-8 = 1/36
        #
        # Normalization: Σ w_i = 4/9 + 4×(1/9) + 4×(1/36) = 1 ✓
        # =====================================================================
        self.w = xp.array([
            4.0/9.0,                              # i=0: rest
            1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,  # i=1-4: axis
            1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0  # i=5-8: diagonal
        ], dtype=dtype)
        
        # =====================================================================
        # Speed of Sound Squared
        # =====================================================================
        # c_s² = 1/3 for all standard lattice models (D2Q9, D3Q19, D3Q27)
        # Units: [Δx²/Δt²]
        #
        # This determines the equation of state: p = ρ c_s²
        # And limits the maximum flow velocity (Ma = u/c_s < 0.1 for accuracy)
        # =====================================================================
        self.cs2 = 1.0 / 3.0  # [Δx²/Δt²]
        
        # =====================================================================
        # Opposite Direction Indices
        # =====================================================================
        # For bounce-back boundary conditions: f_opp[i] ← f_i
        # Property: c[:, opp[i]] = -c[:, i]
        #
        # Mapping:
        #   0 ↔ 0 (rest → rest)
        #   1 ↔ 3 (E ↔ W)
        #   2 ↔ 4 (N ↔ S)
        #   5 ↔ 7 (NE ↔ SW)
        #   6 ↔ 8 (NW ↔ SE)
        # =====================================================================
        self.opp = self._compute_opposite_indices()
    
    def _compute_opposite_indices(self) -> 'npt.NDArray':
        """Compute opposite direction indices automatically
        
        For each direction i, find j such that c[:, j] = -c[:, i].
        This is used in bounce-back boundary conditions.
        
        Returns:
            Array of opposite indices, shape (Q,)  [dimensionless]
            
        Note:
            Uses vectorized computation for efficiency, though for Q=9
            the performance difference is negligible.
        """
        xp = self.xp
        opp = xp.zeros(self.Q, dtype=xp.int32)
        
        for i in range(self.Q):
            # Find j where c[:, j] + c[:, i] = 0
            diff = xp.abs(self.c + self.c[:, i:i+1]).sum(axis=0)
            opp[i] = xp.argmin(diff)
        
        return opp
    
    def get_direction_info(self) -> dict:
        """Get human-readable direction information
        
        Returns:
            Dictionary mapping direction index to name and properties
            
        Example:
            >>> info = lattice.get_direction_info()
            >>> print(info[1])  # East direction
            {'name': 'E', 'velocity': (1, 0), 'weight': 0.111..., 'opposite': 3}
        """
        xp = self.xp
        names = ['Rest', 'E', 'N', 'W', 'S', 'NE', 'NW', 'SW', 'SE']
        
        info = {}
        for i in range(self.Q):
            c_i = self.c[:, i]
            if hasattr(c_i, 'get'):  # CuPy array
                c_i = c_i.get()
            info[i] = {
                'name': names[i],
                'velocity': tuple(c_i),
                'weight': float(self.w[i]),
                'opposite': int(self.opp[i]),
                'speed_squared': int((c_i**2).sum())
            }
        return info
    
    def __repr__(self) -> str:
        """String representation of the lattice model"""
        return f"D2Q9(Q={self.Q}, dim={self.dim}, cs²={self.cs2:.4f})"