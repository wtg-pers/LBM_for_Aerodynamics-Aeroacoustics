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
    (ymin, ymax, zmin, zmax). Uses the half-way bounce-back scheme.
    
    Key Feature - BC Application Order:
        Wall BCs should be applied BEFORE inlet/outlet BCs.
        This allows inlet/outlet to overwrite wall values at corners,
        ensuring proper handling of all directions without gaps.
    
    Attributes:
        xp: Array module (numpy or cupy)
        lattice: Lattice model
        location: Which domain face (ymin, ymax, zmin, zmax)
        shape: Domain shape (Nx, Ny, Nz)
        
    Note:
        This BC should NOT be used for xmin/xmax faces in typical
        inlet-outlet configurations. Use inlet/outlet BCs instead.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 shape: Tuple[int, int, int],
                 exclude_inlet_outlet: bool = False) -> None:
        """Initialize domain wall bounce-back
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q19, D3Q27)
            location: Boundary location (ymin, ymax, zmin, zmax)
            shape: Domain shape (Nx, Ny, Nz)  [lattice units]
            exclude_inlet_outlet: If True, wall BC excludes x=0 and x=Nx-1.
                                  Default is False - apply to full boundary,
                                  and let inlet/outlet overwrite as needed.
        """
        # Store lattice info before super().__init__() for _get_incoming_indices
        self._lattice = lattice
        self._location_str = location.value if hasattr(location, 'value') else str(location).lower()
        
        super().__init__(xp, lattice, location)
        
        self.shape = shape
        self.opp = xp.asarray(lattice.opp)
        self.dim = lattice.dim
        self.exclude_inlet_outlet = exclude_inlet_outlet
        
        # Store opposite indices for incoming directions
        self.incoming_opp = self.opp[self.incoming_indices]
        
        # Determine x-slice based on exclude_inlet_outlet
        self._setup_x_slice()
    
    def _get_incoming_indices(self) -> 'npt.NDArray':
        """Get indices of distributions entering the domain at this boundary
        
        IMPORTANT: For domain walls, we use NON-OVERLAPPING direction sets
        to avoid duplicate bounce-back at edges/corners.
        
        Direction assignment strategy:
            - ymin/ymax walls: handle ALL directions with c_y != 0
            - zmin/zmax walls: handle ONLY directions with c_y == 0 AND c_z != 0
            - xmin/xmax walls: handle ONLY directions with c_y == 0 AND c_z == 0 AND c_x != 0
        
        This ensures each direction is handled by EXACTLY ONE wall,
        preventing mass creation/destruction at edges and corners.
        
        Returns:
            Array of velocity direction indices
        """
        xp = self.xp
        c = self.c
        
        # Get location string
        if hasattr(self, '_location_str'):
            loc_value = self._location_str
        else:
            loc_value = self.location.value
        
        if loc_value in ['ymin', 'south']:
            # y=0 wall: handle ALL c_y > 0 directions (including edges/corners)
            mask = c[1, :] > 0
            
        elif loc_value in ['ymax', 'north']:
            # y=Ny-1 wall: handle ALL c_y < 0 directions
            mask = c[1, :] < 0
            
        elif loc_value in ['zmin', 'bottom']:
            # z=0 wall: handle c_z > 0 directions, but ONLY those with c_y == 0
            # (directions with c_y != 0 are already handled by ymin/ymax walls)
            mask = (c[2, :] > 0) & (c[1, :] == 0)
            
        elif loc_value in ['zmax', 'top']:
            # z=Nz-1 wall: handle c_z < 0 directions, but ONLY those with c_y == 0
            mask = (c[2, :] < 0) & (c[1, :] == 0)
            
        elif loc_value in ['xmin', 'west']:
            # x=0 wall (rare for walls): c_x > 0 with c_y == 0 and c_z == 0
            mask = (c[0, :] > 0) & (c[1, :] == 0) & (c[2, :] == 0)
            
        elif loc_value in ['xmax', 'east']:
            # x=Nx-1 wall (rare for walls): c_x < 0 with c_y == 0 and c_z == 0
            mask = (c[0, :] < 0) & (c[1, :] == 0) & (c[2, :] == 0)
            
        else:
            raise ValueError(f"Unknown location value: {loc_value}")
        
        return xp.where(mask)[0]
    
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
            f[i, boundary] = f_post[opp[i], boundary]
            
        Where i is an incoming direction (would have come from outside domain).
        The outgoing distribution (opp[i]) bounces back to become incoming.
        
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
        # Key: f[incoming] = f_source[outgoing] (outgoing bounces back to incoming)
        if loc == 'ymin':
            # Wall at y=0, incoming directions have c_y > 0
            # Particles going out (c_y < 0) bounce back to incoming (c_y > 0)
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, x_sl, 0, :] = f_source[i_opp, x_sl, 0, :]
                
        elif loc == 'ymax':
            # Wall at y=Ny-1, incoming directions have c_y < 0
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, x_sl, Ny-1, :] = f_source[i_opp, x_sl, Ny-1, :]
                
        elif loc == 'zmin':
            # Wall at z=0, incoming directions have c_z > 0
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, x_sl, :, 0] = f_source[i_opp, x_sl, :, 0]
                
        elif loc == 'zmax':
            # Wall at z=Nz-1, incoming directions have c_z < 0
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, x_sl, :, Nz-1] = f_source[i_opp, x_sl, :, Nz-1]
                
        elif loc == 'xmin':
            # Wall at x=0 (not typical, but supported)
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, 0, :, :] = f_source[i_opp, 0, :, :]
                
        elif loc == 'xmax':
            # Wall at x=Nx-1 (not typical, but supported)
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, Nx-1, :, :] = f_source[i_opp, Nx-1, :, :]
    
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
        
        # Key: f[incoming] = f_source[outgoing]
        if loc == 'ymin':
            f[incoming, x_sl, 0, :] = f_source[incoming_opp, x_sl, 0, :]
        elif loc == 'ymax':
            f[incoming, x_sl, Ny-1, :] = f_source[incoming_opp, x_sl, Ny-1, :]
        elif loc == 'zmin':
            f[incoming, x_sl, :, 0] = f_source[incoming_opp, x_sl, :, 0]
        elif loc == 'zmax':
            f[incoming, x_sl, :, Nz-1] = f_source[incoming_opp, x_sl, :, Nz-1]
        elif loc == 'xmin':
            f[incoming, 0, :, :] = f_source[incoming_opp, 0, :, :]
        elif loc == 'xmax':
            f[incoming, Nx-1, :, :] = f_source[incoming_opp, Nx-1, :, :]
    
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
    
    Important: BC Application Order
        Wall BCs should be applied BEFORE inlet/outlet BCs in main.py:
        
        domain_walls.apply_all(f_new, f_post)  # First
        bc_manager.apply_all(f_new)            # Second (overwrites corners)
    
    Example:
        >>> wall_mgr = DomainWallManager(xp, lattice, shape)
        >>> wall_mgr.apply_all(f, f_post)
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 shape: Tuple[int, int, int],
                 walls: Optional[List[str]] = None,
                 exclude_inlet_outlet: bool = False) -> None:
        """Initialize wall manager
        
        Args:
            xp: Array module
            lattice: Lattice model
            shape: Domain shape (Nx, Ny, Nz)
            walls: List of wall locations. Default: ['ymin', 'ymax', 'zmin', 'zmax']
                   Accepts both coordinate ('ymin') and legacy ('south') names.
            exclude_inlet_outlet: If True, walls exclude x=0 and x=Nx-1.
                                  Default is False - let inlet/outlet overwrite.
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