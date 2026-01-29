"""
Domain Wall Boundary Condition (Half-way Bounce-Back) for LBM

This module implements the half-way bounce-back scheme for domain boundaries
(south, north, bottom, top walls). Essential for internal flow simulations
like channel flow or DFG cylinder benchmark.

Physical Principle:
    The wall is located halfway between the boundary node and the ghost node.
    For a node at y=0 (SOUTH wall), the wall is at y=-0.5.
    
    Half-way bounce-back:
        f_ī(x_boundary, t+Δt) = f_i*(x_boundary, t)
    
    where ī is the opposite direction of i, and f_i* is post-collision.

Boundary Overlap Strategy (exclude_inlet_outlet=True):
    To avoid conflicts between wall BC and inlet/outlet BC at domain edges:
    
    ┌─────────────────────────────────────────────────────────┐
    │                     TOP (z=Nz-1)                        │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │                                                 │   │
    │ I │     Wall BC applied to x ∈ [1, Nx-2]           │ O │
    │ n │     (excludes inlet/outlet columns)             │ u │
    │ l │                                                 │ t │
    │ e │                                                 │ l │
    │ t │                                                 │ e │
    │   │                                                 │ t │
    │ x │                                                 │   │
    │ = │                                                 │ x │
    │ 0 │                                                 │ = │
    │   │                                                 │ N │
    │   │                                                 │ x │
    │   │                                                 │ - │
    │   │                                                 │ 1 │
    │   └─────────────────────────────────────────────────┘   │
    │                    BOTTOM (z=0)                         │
    └─────────────────────────────────────────────────────────┘
    
    This ensures:
    - Inlet (x=0): handles all y, z  
    - Outlet (x=Nx-1): handles all y, z
    - Wall: handles x ∈ [1, Nx-2] only → NO OVERLAP

Accuracy:
    - Second-order accurate when wall is exactly at q = 0.5
    - Standard choice for rectangular channel boundaries

References:
    - Ladd, J. Fluid Mech. 271, 1994
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 5

Author: LBM Development Team
Date: 2026-01
"""

from typing import TYPE_CHECKING, Optional, Tuple, List
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .base import BoundaryCondition, BoundaryLocation


class DomainWallBounceBack(BoundaryCondition):
    """Half-way Bounce-Back for Domain Wall Boundaries
    
    Implements no-slip wall boundary condition at domain faces
    (SOUTH, NORTH, BOTTOM, TOP). Uses the half-way bounce-back scheme.
    
    Key Feature - exclude_inlet_outlet option:
        When True (default), wall BC is applied only to x ∈ [1, Nx-2],
        avoiding overlap with inlet (x=0) and outlet (x=Nx-1) boundaries.
    
    Attributes:
        xp: Array module (numpy or cupy)
        lattice: Lattice model
        location: Which domain face (SOUTH, NORTH, BOTTOM, TOP)
        shape: Domain shape (Nx, Ny, Nz)
        exclude_inlet_outlet: If True, excludes x=0 and x=Nx-1 from wall BC
        
    Note:
        This BC should NOT be used for WEST/EAST faces in typical
        inlet-outlet configurations. Use inlet/outlet BCs instead.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 shape: Tuple[int, int, int],
                 exclude_inlet_outlet: bool = True) -> None:
        """Initialize domain wall bounce-back
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q19, D3Q27)
            location: Boundary location (SOUTH, NORTH, BOTTOM, TOP)
            shape: Domain shape (Nx, Ny, Nz)  [lattice units]
            exclude_inlet_outlet: If True (default), wall BC excludes x=0 and x=Nx-1
                                  to prevent overlap with inlet/outlet BCs.
                                  If False, wall BC covers entire boundary face.
        """
        super().__init__(xp, lattice, location)
        
        self.shape = shape
        self.opp = xp.asarray(lattice.opp)
        self.dim = lattice.dim
        self.exclude_inlet_outlet = exclude_inlet_outlet
        
        # Store opposite indices for incoming directions
        self.incoming_opp = self.opp[self.incoming_indices]
        
        # Determine x-slice based on exclude_inlet_outlet
        self._setup_x_slice()
    
    def _setup_x_slice(self) -> None:
        """Setup x-axis slice based on exclude_inlet_outlet option"""
        Nx = self.shape[0]
        
        # Only apply exclusion for y/z boundary walls (ymin, ymax, zmin, zmax)
        # These are walls that could overlap with inlet (xmin) and outlet (xmax)
        is_yz_wall = self.location.value in ['ymin', 'ymax', 'zmin', 'zmax']
        
        if self.exclude_inlet_outlet and is_yz_wall:
            # Exclude x=0 (inlet) and x=Nx-1 (outlet)
            self.x_slice = slice(1, Nx - 1)
            self.x_range_desc = f"[1, {Nx-2}]"
        else:
            # Full x range
            self.x_slice = slice(None)
            self.x_range_desc = f"[0, {Nx-1}]"
    
    def apply(self, f: 'npt.NDArray', f_post: Optional['npt.NDArray'] = None,
              **kwargs) -> None:
        """Apply half-way bounce-back at domain wall
        
        IMPORTANT: Must be called AFTER streaming.
        
        The bounce-back operation:
            f[opp[i], x_slice, boundary, :] = f_post[i, x_slice, boundary, :]
        
        Args:
            f: Distribution function after streaming, shape (Q, Nx, Ny, Nz)
               Modified in-place.
            f_post: Post-collision distribution from before streaming.
                   If None, uses f itself (approximation for steady flows).
            **kwargs: Additional parameters (unused)
        """
        # Use f_post if provided
        f_source = f_post if f_post is not None else f
        
        incoming = self.incoming_indices
        incoming_opp = self.incoming_opp
        x_sl = self.x_slice
        
        Nx, Ny, Nz = self.shape
        loc = self.location.value  # Get string value for comparison
        
        # Apply bounce-back based on wall location
        if loc == 'ymin':
            # Wall at y=0, incoming directions have c_y > 0
            for i, i_opp in zip(incoming, incoming_opp):
                f[i_opp, x_sl, 0, :] = f_source[i, x_sl, 0, :]
                
        elif loc == 'ymax':
            # Wall at y=Ny-1, incoming directions have c_y < 0
            for i, i_opp in zip(incoming, incoming_opp):
                f[i_opp, x_sl, Ny-1, :] = f_source[i, x_sl, Ny-1, :]
                
        elif loc == 'zmin':
            # Wall at z=0, incoming directions have c_z > 0
            for i, i_opp in zip(incoming, incoming_opp):
                f[i_opp, x_sl, :, 0] = f_source[i, x_sl, :, 0]
                
        elif loc == 'zmax':
            # Wall at z=Nz-1, incoming directions have c_z < 0
            for i, i_opp in zip(incoming, incoming_opp):
                f[i_opp, x_sl, :, Nz-1] = f_source[i, x_sl, :, Nz-1]
                
        elif loc == 'xmin':
            # Wall at x=0 (not typical, but supported)
            for i, i_opp in zip(incoming, incoming_opp):
                f[i_opp, 0, :, :] = f_source[i, 0, :, :]
                
        elif loc == 'xmax':
            # Wall at x=Nx-1 (not typical, but supported)
            for i, i_opp in zip(incoming, incoming_opp):
                f[i_opp, Nx-1, :, :] = f_source[i, Nx-1, :, :]
    
    def apply_vectorized(self, f: 'npt.NDArray', 
                         f_post: Optional['npt.NDArray'] = None) -> None:
        """Fully vectorized bounce-back using advanced indexing
        
        Alternative implementation with potentially better GPU performance.
        
        Args:
            f: Distribution after streaming (modified in-place)
            f_post: Post-collision distribution (optional)
        """
        f_source = f_post if f_post is not None else f
        
        incoming = self.incoming_indices
        incoming_opp = self.incoming_opp
        x_sl = self.x_slice
        
        Nx, Ny, Nz = self.shape
        loc = self.location.value
        
        if loc == 'ymin':
            f[incoming_opp, x_sl, 0, :] = f_source[incoming, x_sl, 0, :]
        elif loc == 'ymax':
            f[incoming_opp, x_sl, Ny-1, :] = f_source[incoming, x_sl, Ny-1, :]
        elif loc == 'zmin':
            f[incoming_opp, x_sl, :, 0] = f_source[incoming, x_sl, :, 0]
        elif loc == 'zmax':
            f[incoming_opp, x_sl, :, Nz-1] = f_source[incoming, x_sl, :, Nz-1]
        elif loc == 'xmin':
            f[incoming_opp, 0, :, :] = f_source[incoming, 0, :, :]
        elif loc == 'xmax':
            f[incoming_opp, Nx-1, :, :] = f_source[incoming, Nx-1, :, :]
    
    def get_info(self) -> str:
        """Return information string about this boundary condition"""
        n_incoming = len(self.incoming_indices)
        
        info = f"Domain Wall (HWBB) at {self.location.value}:"
        info += f"\n    Incoming directions: {n_incoming}"
        info += f"\n    X-range: {self.x_range_desc}"
        
        if self.exclude_inlet_outlet:
            info += " (inlet/outlet excluded)"
        
        return info


class DomainWallManager:
    """Manager for multiple domain wall boundaries
    
    Convenience class to handle all four walls (ymin, ymax, zmin, zmax)
    in a channel flow configuration.
    
    Example:
        >>> wall_mgr = DomainWallManager(xp, lattice, shape)
        >>> wall_mgr.apply_all(f, f_post)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 shape: Tuple[int, int, int],
                 walls: Optional[List[str]] = None,
                 exclude_inlet_outlet: bool = True) -> None:
        """Initialize wall manager
        
        Args:
            xp: Array module
            lattice: Lattice model
            shape: Domain shape (Nx, Ny, Nz)
            walls: List of wall locations. Default: ['ymin', 'ymax', 'zmin', 'zmax']
                   Accepts both coordinate ('ymin') and legacy ('south') names.
            exclude_inlet_outlet: If True, walls exclude x=0 and x=Nx-1
        """
        self.xp = xp
        self.shape = shape
        self.exclude_inlet_outlet = exclude_inlet_outlet
        self.walls: List[DomainWallBounceBack] = []
        
        if walls is None:
            walls = ['ymin', 'ymax', 'zmin', 'zmax']
        
        for wall_name in walls:
            location = BoundaryLocation.from_string(wall_name)
            wall_bc = DomainWallBounceBack(
                xp, lattice, location, shape,
                exclude_inlet_outlet=exclude_inlet_outlet
            )
            self.walls.append(wall_bc)
    
    def apply_all(self, f: 'npt.NDArray', 
                  f_post: Optional['npt.NDArray'] = None) -> None:
        """Apply all wall boundary conditions
        
        Args:
            f: Distribution after streaming (modified in-place)
            f_post: Post-collision distribution (optional but recommended)
        """
        for wall in self.walls:
            wall.apply(f, f_post)
    
    def get_info(self) -> str:
        """Return information about all walls"""
        Nx = self.shape[0]
        
        info_lines = [
            "Domain Wall Manager:",
            f"  Walls: {[w.location.value for w in self.walls]}",
        ]
        
        if self.exclude_inlet_outlet:
            info_lines.append(f"  X-range: [1, {Nx-2}] (inlet/outlet excluded)")
        else:
            info_lines.append(f"  X-range: [0, {Nx-1}] (full)")
        
        return "\n".join(info_lines)
    
    def __len__(self) -> int:
        return len(self.walls)
    
    def __iter__(self):
        return iter(self.walls)