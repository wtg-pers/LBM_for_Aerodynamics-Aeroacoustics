"""
Boundary Conditions Base Module for LBM

This module defines the abstract base class for all boundary conditions
and provides utility functions for boundary handling.

Design Philosophy:
    - All boundary conditions modify the distribution function in-place
    - Boundary conditions are applied AFTER streaming (pull scheme)
    - Each boundary condition class handles one face/region

Coordinate System:
    3D domain with shape (Nx, Ny, Nz):
    
        z (zmax)
        ↑   
        │   y (ymax)
        │  ╱
        │ ╱
        └──────→ x (xmax)
    
    Boundary faces:
        - xmin: x = 0       (typical inlet)
        - xmax: x = Nx-1    (typical outlet)
        - ymin: y = 0       
        - ymax: y = Ny-1    
        - zmin: z = 0       
        - zmax: z = Nz-1    

Author: LBM Development Team
Date: 2026-01
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, List, Optional, Dict, Any, Union
from enum import Enum

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class BoundaryLocation(Enum):
    """Boundary face locations for 3D domain
    
    Coordinate-based naming (primary):
        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX
    
    Legacy naming (aliases, for backward compatibility):
        WEST=XMIN, EAST=XMAX, SOUTH=YMIN, NORTH=YMAX, BOTTOM=ZMIN, TOP=ZMAX
    """
    # Primary: Coordinate-based names
    XMIN = 'xmin'    # x = 0
    XMAX = 'xmax'    # x = Nx-1
    YMIN = 'ymin'    # y = 0
    YMAX = 'ymax'    # y = Ny-1
    ZMIN = 'zmin'    # z = 0
    ZMAX = 'zmax'    # z = Nz-1
    
    # Aliases (same values, for backward compatibility)
    WEST = 'xmin'
    EAST = 'xmax'
    SOUTH = 'ymin'
    NORTH = 'ymax'
    BOTTOM = 'zmin'
    TOP = 'zmax'
    
    @classmethod
    def from_string(cls, name: str) -> 'BoundaryLocation':
        """Convert string to BoundaryLocation enum
        
        Accepts both coordinate-based names (xmin, xmax, ...) and
        legacy names (west, east, south, north, bottom, top).
        
        Args:
            name: Location name (case-insensitive)
        
        Returns:
            BoundaryLocation enum value
            
        Raises:
            ValueError: If name is not a valid location
            
        Examples:
            >>> BoundaryLocation.from_string('xmin')
            BoundaryLocation.XMIN
            >>> BoundaryLocation.from_string('west')  # alias
            BoundaryLocation.XMIN
        """
        name_lower = name.lower().strip()
        
        mapping = {
            # Primary: Coordinate-based
            'xmin': cls.XMIN,
            'xmax': cls.XMAX,
            'ymin': cls.YMIN,
            'ymax': cls.YMAX,
            'zmin': cls.ZMIN,
            'zmax': cls.ZMAX,
            
            # Aliases: Cardinal directions (legacy)
            'west': cls.XMIN,
            'east': cls.XMAX,
            'south': cls.YMIN,
            'north': cls.YMAX,
            'bottom': cls.ZMIN,
            'top': cls.ZMAX,
            
            # Additional aliases
            'left': cls.XMIN,
            'right': cls.XMAX,
            'front': cls.YMIN,
            'back': cls.YMAX,
            'down': cls.ZMIN,
            'up': cls.ZMAX,
            
            # Numeric index style
            'x0': cls.XMIN,
            'x1': cls.XMAX,
            'y0': cls.YMIN,
            'y1': cls.YMAX,
            'z0': cls.ZMIN,
            'z1': cls.ZMAX,
        }
        
        if name_lower not in mapping:
            valid_names = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax',
                          'west', 'east', 'south', 'north', 'bottom', 'top']
            raise ValueError(f"Unknown boundary location: '{name}'. "
                           f"Valid options: {valid_names}")
        
        return mapping[name_lower]
    
    def get_axis(self) -> int:
        """Get the axis index for this boundary
        
        Returns:
            0 for x-boundaries, 1 for y-boundaries, 2 for z-boundaries
        """
        if self.value in ['xmin', 'xmax']:
            return 0
        elif self.value in ['ymin', 'ymax']:
            return 1
        else:  # zmin, zmax
            return 2
    
    def is_min(self) -> bool:
        """Check if this is a minimum boundary (x=0, y=0, or z=0)"""
        return self.value in ['xmin', 'ymin', 'zmin']
    
    def is_max(self) -> bool:
        """Check if this is a maximum boundary"""
        return self.value in ['xmax', 'ymax', 'zmax']


def parse_boundary_location(location: Union[str, 'BoundaryLocation']) -> 'BoundaryLocation':
    """Parse boundary location from string or enum
    
    Convenience function that accepts both string and BoundaryLocation.
    
    Args:
        location: Either a string ('xmin', 'west', ...) or BoundaryLocation enum
        
    Returns:
        BoundaryLocation enum value
    """
    if isinstance(location, BoundaryLocation):
        return location
    elif isinstance(location, str):
        return BoundaryLocation.from_string(location)
    else:
        raise TypeError(f"Expected str or BoundaryLocation, got {type(location)}")


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions
    
    All boundary conditions must implement the apply() method which
    modifies the distribution function in-place after streaming.
    
    Attributes:
        xp: Array module (numpy or cupy)
        lattice: Lattice model
        location: Boundary location
    """
    
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice', 
                 location: Union[str, BoundaryLocation]) -> None:
        """Initialize boundary condition
        
        Args:
            xp: Array module
            lattice: Lattice model
            location: Which face this BC applies to (string or BoundaryLocation)
        """
        self.xp = xp
        self.lattice = lattice
        
        # Accept both string and BoundaryLocation
        self.location = parse_boundary_location(location)
        
        self.c = xp.asarray(lattice.c)
        self.w = xp.asarray(lattice.w)
        self.Q = lattice.Q
        self.cs2 = lattice.cs2
        
        # Determine unknown (incoming) directions for this boundary
        self.incoming_indices = self._get_incoming_indices()
    
    def _get_incoming_indices(self) -> 'npt.NDArray':
        """Get indices of distributions entering the domain at this boundary
        
        For pull scheme after streaming, these are the directions that
        would have come from outside the domain.
        
        Returns:
            Array of velocity direction indices
        """
        xp = self.xp
        c = self.c
        
        # Normal vector pointing INTO the domain
        # Use .value to get the actual string for comparison
        loc_value = self.location.value
        
        if loc_value == 'xmin':    # x=0, normal = +x
            mask = c[0, :] > 0
        elif loc_value == 'xmax':  # x=Nx-1, normal = -x
            mask = c[0, :] < 0
        elif loc_value == 'ymin':  # y=0, normal = +y
            mask = c[1, :] > 0
        elif loc_value == 'ymax':  # y=Ny-1, normal = -y
            mask = c[1, :] < 0
        elif loc_value == 'zmin':  # z=0, normal = +z
            mask = c[2, :] > 0
        elif loc_value == 'zmax':  # z=Nz-1, normal = -z
            mask = c[2, :] < 0
        else:
            raise ValueError(f"Unknown location value: {loc_value}")
        
        return xp.where(mask)[0]
    
    @abstractmethod
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply boundary condition to distribution function
        
        This method modifies f in-place for the boundary nodes.
        
        Args:
            f: Distribution function, shape (Q, Nx, Ny, Nz)
            **kwargs: Additional parameters (rho, u, etc.)
        """
        pass
    
    def get_boundary_slice(self, shape: Tuple[int, ...]) -> Tuple[slice, ...]:
        """Get array slice for this boundary
        
        Args:
            shape: Domain shape (Nx, Ny, Nz)
            
        Returns:
            Tuple of slices to index boundary nodes
        """
        Nx, Ny, Nz = shape
        loc_value = self.location.value
        
        if loc_value == 'xmin':
            return (slice(None), 0, slice(None), slice(None))
        elif loc_value == 'xmax':
            return (slice(None), Nx-1, slice(None), slice(None))
        elif loc_value == 'ymin':
            return (slice(None), slice(None), 0, slice(None))
        elif loc_value == 'ymax':
            return (slice(None), slice(None), Ny-1, slice(None))
        elif loc_value == 'zmin':
            return (slice(None), slice(None), slice(None), 0)
        elif loc_value == 'zmax':
            return (slice(None), slice(None), slice(None), Nz-1)


class BoundaryManager:
    """Manages multiple boundary conditions
    
    Applies all boundary conditions in sequence after streaming.
    """
    
    def __init__(self) -> None:
        self.boundaries: List[BoundaryCondition] = []
    
    def add(self, bc: BoundaryCondition) -> None:
        """Add a boundary condition"""
        self.boundaries.append(bc)
    
    def apply_all(self, f: 'npt.NDArray', **kwargs) -> None:
        """Apply all boundary conditions in order"""
        for bc in self.boundaries:
            bc.apply(f, **kwargs)
    
    def __len__(self) -> int:
        return len(self.boundaries)
    
    def __iter__(self):
        return iter(self.boundaries)


# =============================================================================
# Factory function for creating BCs from config dictionary
# =============================================================================

def create_boundary_from_config(xp: 'ModuleType', 
                                 lattice: 'Lattice',
                                 bc_name: str,
                                 config: Dict[str, Any],
                                 shape: Tuple[int, ...]) -> Optional[BoundaryCondition]:
    """Create a boundary condition from config dictionary
    
    Factory function that creates the appropriate BC based on config.
    
    Args:
        xp: Array module
        lattice: Lattice model
        bc_name: User-defined name for this boundary (for logging)
        config: Boundary config dictionary with keys:
                - type: 'inlet', 'outlet', 'wall', 'periodic'
                - location: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'
                - (type-specific parameters)
        shape: Domain shape (Nx, Ny, Nz)
        
    Returns:
        BoundaryCondition instance, or None if type is periodic/unknown
        
    Example config:
        >>> config = {
        ...     "type": "inlet",
        ...     "location": "xmin",
        ...     "velocity": 0.1,
        ...     "profile": "uniform"
        ... }
        >>> bc = create_boundary_from_config(np, lattice, "my_inlet", config, shape)
    """
    # Import here to avoid circular imports
    from .inlet import EquilibriumInlet, EquilibriumInletProfile
    from .outlet import CharacteristicOutlet, ConvectiveOutlet, ExtrapolationOutlet
    from .domain_wall import DomainWallBounceBack
    
    bc_type = config.get('type', '').lower()
    location_str = config.get('location', bc_name)  # fallback to bc_name if no location
    
    try:
        location = BoundaryLocation.from_string(location_str)
    except ValueError as e:
        print(f"Warning: {e}")
        return None
    
    # =========================================================================
    # Inlet BC
    # =========================================================================
    if bc_type == 'inlet':
        velocity = config.get('velocity', 0.1)
        density = config.get('density', 1.0)
        profile = config.get('profile', 'uniform')
        
        if profile == 'parabolic':
            return EquilibriumInletProfile(xp, lattice, location,
                                           u_max=velocity, shape=shape,
                                           density=density)
        else:  # uniform
            return EquilibriumInlet(xp, lattice, location,
                                    velocity=velocity, density=density,
                                    shape=shape)
    
    # =========================================================================
    # Outlet BC
    # =========================================================================
    elif bc_type == 'outlet':
        method = config.get('method', 'characteristic').lower()
        rho_target = config.get('pressure', config.get('rho', config.get('rho_target', 1.0)))
        
        if method == 'characteristic' or method == 'open':
            relax_coeff = config.get('relax_coeff', config.get('k', 0.1))
            return CharacteristicOutlet(xp, lattice, location,
                                        rho_target=rho_target,
                                        relax_coeff=relax_coeff,
                                        shape=shape)
        elif method == 'convective':
            u_conv = config.get('convective_velocity', 0.1)
            return ConvectiveOutlet(xp, lattice, location,
                                    convective_velocity=u_conv,
                                    shape=shape)
        elif method == 'extrapolation':
            order = config.get('order', 1)
            return ExtrapolationOutlet(xp, lattice, location, order=order)
        else:
            print(f"Warning: Unknown outlet method '{method}', using characteristic")
            return CharacteristicOutlet(xp, lattice, location,
                                        rho_target=rho_target, shape=shape)
    
    # =========================================================================
    # Wall BC (domain boundary)
    # =========================================================================
    elif bc_type == 'wall':
        method = config.get('method', 'bounce_back').lower()
        exclude_io = config.get('exclude_inlet_outlet', True)
        
        if method in ['bounce_back', 'hwbb', 'halfway']:
            return DomainWallBounceBack(xp, lattice, location, shape,
                                        exclude_inlet_outlet=exclude_io)
        else:
            print(f"Warning: Wall method '{method}' not yet implemented, using bounce_back")
            return DomainWallBounceBack(xp, lattice, location, shape,
                                        exclude_inlet_outlet=exclude_io)
    
    # =========================================================================
    # Periodic BC (handled by streaming, no explicit BC needed)
    # =========================================================================
    elif bc_type == 'periodic':
        return None
    
    else:
        print(f"Warning: Unknown boundary type '{bc_type}' for {bc_name}")
        return None


def create_all_boundaries_from_config(xp: 'ModuleType',
                                       lattice: 'Lattice',
                                       boundaries_config: Dict[str, Dict],
                                       shape: Tuple[int, ...],
                                       verbose: bool = True) -> BoundaryManager:
    """Create BoundaryManager with all boundaries from config
    
    Args:
        xp: Array module
        lattice: Lattice model
        boundaries_config: Dictionary of boundary configs
                          e.g., {"inlet": {...}, "outlet": {...}, ...}
        shape: Domain shape
        verbose: Print info about created boundaries
        
    Returns:
        BoundaryManager with all boundaries added
        
    Example:
        >>> boundaries_config = {
        ...     "inlet": {"type": "inlet", "location": "xmin", "velocity": 0.1},
        ...     "outlet": {"type": "outlet", "location": "xmax", "rho": 1.0},
        ...     "wall_y0": {"type": "wall", "location": "ymin"},
        ... }
        >>> bc_manager = create_all_boundaries_from_config(np, lattice, 
        ...                                                 boundaries_config, shape)
    """
    manager = BoundaryManager()
    
    for bc_name, bc_config in boundaries_config.items():
        bc = create_boundary_from_config(xp, lattice, bc_name, bc_config, shape)
        if bc is not None:
            manager.add(bc)
            if verbose:
                loc = bc_config.get('location', bc_name)
                print(f"    {bc_name}: {bc_config.get('type', 'unknown')} at {loc}")
    
    return manager