"""
Domain Wall Boundary Conditions for LBM

This module implements half-way bounce-back boundary conditions
at domain walls (ymin, ymax for 2D; ymin, ymax, zmin, zmax for 3D).

Supports both 2D and 3D simulations.

Physical Principle:
    No-slip wall boundary using half-way bounce-back scheme.
    The wall is assumed to be located halfway between fluid and solid nodes.

Key Feature - BC Application Order:
    Wall BCs should be applied BEFORE inlet/outlet BCs.
    This allows inlet/outlet to overwrite wall values at corners.

References:
    - Ladd, J. Fluid Mech. 271, 1994
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 5

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Tuple, List
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

from .base import BoundaryCondition, BoundaryLocation


class DomainWallBounceBack(BoundaryCondition):
    """Half-way Bounce-Back for Domain Wall Boundaries
    
    Implements no-slip wall boundary condition at domain faces.
    Uses the half-way bounce-back scheme.
    
    Supports both 2D and 3D domains.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation,
                 shape: tuple,
                 exclude_inlet_outlet: bool = False) -> None:
        """Initialize domain wall bounce-back
        
        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
            location: Boundary location (ymin, ymax, zmin, zmax)
            shape: Domain shape (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
            exclude_inlet_outlet: If True, wall BC excludes x=0 and x=Nx-1
        """
        # Store lattice info before super().__init__()
        self._lattice = lattice
        self._location_str = location.value if hasattr(location, 'value') else str(location).lower()
        
        super().__init__(xp, lattice, location)
        
        self.shape = shape
        self.dim = lattice.dim
        self.opp = xp.asarray(lattice.opp)
        self.exclude_inlet_outlet = exclude_inlet_outlet
        
        # Store opposite indices for incoming directions
        self.incoming_opp = self.opp[self.incoming_indices]
        
        # Determine x-slice based on exclude_inlet_outlet
        self._setup_x_slice()
    
    def _get_incoming_indices(self) -> 'npt.NDArray':
        """Get indices of distributions entering the domain at this boundary
        
        Uses NON-OVERLAPPING direction sets to avoid duplicate bounce-back
        at edges/corners.
        """
        xp = self.xp
        c = self.c
        loc_value = self._location_str
        
        if self.dim == 2:
            # 2D: Only handle y directions for lateral walls
            if loc_value in ['ymin', 'south']:
                mask = c[1, :] > 0
            elif loc_value in ['ymax', 'north']:
                mask = c[1, :] < 0
            elif loc_value in ['xmin', 'west']:
                mask = (c[0, :] > 0) & (c[1, :] == 0)
            elif loc_value in ['xmax', 'east']:
                mask = (c[0, :] < 0) & (c[1, :] == 0)
            else:
                raise ValueError(f"Invalid 2D wall location: {loc_value}")
        else:
            # 3D: Non-overlapping direction assignment
            if loc_value in ['ymin', 'south']:
                mask = c[1, :] > 0
            elif loc_value in ['ymax', 'north']:
                mask = c[1, :] < 0
            elif loc_value in ['zmin', 'bottom']:
                mask = (c[2, :] > 0) & (c[1, :] == 0)
            elif loc_value in ['zmax', 'top']:
                mask = (c[2, :] < 0) & (c[1, :] == 0)
            elif loc_value in ['xmin', 'west']:
                mask = (c[0, :] > 0) & (c[1, :] == 0) & (c[2, :] == 0)
            elif loc_value in ['xmax', 'east']:
                mask = (c[0, :] < 0) & (c[1, :] == 0) & (c[2, :] == 0)
            else:
                raise ValueError(f"Unknown location value: {loc_value}")
        
        return xp.where(mask)[0]
    
    def _setup_x_slice(self) -> None:
        """Setup x-axis slice based on exclude_inlet_outlet option"""
        Nx = self.shape[0]
        
        is_lateral_wall = self.location.value in ['ymin', 'ymax', 'zmin', 'zmax',
                                                    'south', 'north', 'bottom', 'top']
        
        if self.exclude_inlet_outlet and is_lateral_wall:
            self.x_slice = slice(1, Nx - 1)
            self.x_range_desc = f"[1, {Nx-2}]"
        else:
            self.x_slice = slice(None)
            self.x_range_desc = f"[0, {Nx-1}]"
    
    def apply(self, f: 'npt.NDArray', f_post: Optional['npt.NDArray'] = None,
              **kwargs) -> None:
        """Apply half-way bounce-back at domain wall
        
        Args:
            f: Distribution after streaming (modified in-place)
            f_post: Post-collision distribution (optional)
        """
        if self.dim == 2:
            self._apply_2d(f, f_post)
        else:
            self._apply_3d(f, f_post)
    
    def _apply_2d(self, f: 'npt.NDArray', f_post: Optional['npt.NDArray']) -> None:
        """Apply for 2D domain"""
        f_source = f_post if f_post is not None else f
        
        incoming = self.incoming_indices
        incoming_opp = self.incoming_opp
        x_sl = self.x_slice
        
        Nx, Ny = f.shape[1], f.shape[2]
        loc = self.location.value
        
        if loc in ['ymin', 'south']:
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, x_sl, 0] = f_source[i_opp, x_sl, 0]
        elif loc in ['ymax', 'north']:
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, x_sl, Ny-1] = f_source[i_opp, x_sl, Ny-1]
        elif loc in ['xmin', 'west']:
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, 0, :] = f_source[i_opp, 0, :]
        elif loc in ['xmax', 'east']:
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, Nx-1, :] = f_source[i_opp, Nx-1, :]
    
    def _apply_3d(self, f: 'npt.NDArray', f_post: Optional['npt.NDArray']) -> None:
        """Apply for 3D domain"""
        f_source = f_post if f_post is not None else f
        
        incoming = self.incoming_indices
        incoming_opp = self.incoming_opp
        x_sl = self.x_slice
        
        Nx, Ny, Nz = f.shape[1], f.shape[2], f.shape[3]
        loc = self.location.value
        
        if loc in ['ymin', 'south']:
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, x_sl, 0, :] = f_source[i_opp, x_sl, 0, :]
        elif loc in ['ymax', 'north']:
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, x_sl, Ny-1, :] = f_source[i_opp, x_sl, Ny-1, :]
        elif loc in ['zmin', 'bottom']:
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, x_sl, :, 0] = f_source[i_opp, x_sl, :, 0]
        elif loc in ['zmax', 'top']:
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, x_sl, :, Nz-1] = f_source[i_opp, x_sl, :, Nz-1]
        elif loc in ['xmin', 'west']:
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, 0, :, :] = f_source[i_opp, 0, :, :]
        elif loc in ['xmax', 'east']:
            for i, i_opp in zip(incoming, incoming_opp):
                f[i, Nx-1, :, :] = f_source[i_opp, Nx-1, :, :]
    
    def get_info(self) -> str:
        """Return information string"""
        n_incoming = len(self.incoming_indices)
        dim_str = "2D" if self.dim == 2 else "3D"
        
        info = f"Domain Wall (HWBB, {dim_str}) at {self.location.value}:"
        info += f"\n    Incoming directions: {n_incoming}"
        info += f"\n    X-range: {self.x_range_desc}"
        
        if self.exclude_inlet_outlet:
            info += " (inlet/outlet excluded)"
        
        return info


class DomainWallManager:
    """Manager for multiple domain wall boundaries
    
    Convenience class to handle all walls in a channel flow configuration.
    
    For 2D: ymin, ymax (default)
    For 3D: ymin, ymax, zmin, zmax (default)
    
    Supports both 2D and 3D simulations.
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 shape: tuple,
                 walls: Optional[List[str]] = None,
                 exclude_inlet_outlet: bool = False) -> None:
        """Initialize wall manager
        
        Args:
            xp: Array module
            lattice: Lattice model
            shape: Domain shape (Nx, Ny) or (Nx, Ny, Nz)
            walls: List of wall locations. Default depends on dimension.
            exclude_inlet_outlet: If True, walls exclude x=0 and x=Nx-1.
        """
        self.xp = xp
        self.shape = shape
        self.dim = lattice.dim
        self.exclude_inlet_outlet = exclude_inlet_outlet
        self.walls: List[DomainWallBounceBack] = []
        
        # Default walls based on dimension
        if walls is None:
            if self.dim == 2:
                walls = ['ymin', 'ymax']
            else:
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
        dim_str = "2D" if self.dim == 2 else "3D"
        
        info_lines = [
            f"Domain Wall Manager ({dim_str}):",
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