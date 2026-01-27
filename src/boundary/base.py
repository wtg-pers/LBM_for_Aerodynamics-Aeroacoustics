"""
Boundary Conditions Base Module for LBM

This module defines the abstract base class for all boundary conditions
and provides utility functions for boundary handling.

Design Philosophy:
    - All boundary conditions modify the distribution function in-place
    - Boundary conditions are applied AFTER streaming (pull scheme)
    - Each boundary condition class handles one face/region

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
    """Boundary face locations for 3D domain"""
    WEST = 'west'      # x = 0
    EAST = 'east'      # x = Nx-1
    SOUTH = 'south'    # y = 0
    NORTH = 'north'    # y = Ny-1
    BOTTOM = 'bottom'  # z = 0
    TOP = 'top'        # z = Nz-1
    
    @classmethod
    def from_string(cls, name: str) -> 'BoundaryLocation':
        """Convert string to BoundaryLocation enum
        
        Args:
            name: Location name ('west', 'east', 'south', 'north', 'bottom', 'top')
                  Case-insensitive
        
        Returns:
            BoundaryLocation enum value
            
        Raises:
            ValueError: If name is not a valid location
            
        Examples:
            >>> BoundaryLocation.from_string('west')
            BoundaryLocation.WEST
            >>> BoundaryLocation.from_string('EAST')
            BoundaryLocation.EAST
        """
        name_lower = name.lower().strip()
        
        mapping = {
            'west': cls.WEST,
            'east': cls.EAST,
            'south': cls.SOUTH,
            'north': cls.NORTH,
            'bottom': cls.BOTTOM,
            'top': cls.TOP,
            # Alternative names
            'left': cls.WEST,
            'right': cls.EAST,
            'front': cls.SOUTH,
            'back': cls.NORTH,
            'down': cls.BOTTOM,
            'up': cls.TOP,
            # Index-based (x=0, x=Nx-1, etc.)
            'x0': cls.WEST,
            'x1': cls.EAST,
            'xmin': cls.WEST,
            'xmax': cls.EAST,
            'y0': cls.SOUTH,
            'y1': cls.NORTH,
            'ymin': cls.SOUTH,
            'ymax': cls.NORTH,
            'z0': cls.BOTTOM,
            'z1': cls.TOP,
            'zmin': cls.BOTTOM,
            'zmax': cls.TOP,
        }
        
        if name_lower not in mapping:
            valid_names = ['west', 'east', 'south', 'north', 'bottom', 'top']
            raise ValueError(f"Unknown boundary location: '{name}'. "
                           f"Valid options: {valid_names}")
        
        return mapping[name_lower]


def parse_boundary_location(location: Union[str, 'BoundaryLocation']) -> 'BoundaryLocation':
    """Parse boundary location from string or enum
    
    Convenience function that accepts both string and BoundaryLocation.
    
    Args:
        location: Either a string ('west', 'east', ...) or BoundaryLocation enum
        
    Returns:
        BoundaryLocation enum value
        
    Examples:
        >>> parse_boundary_location('west')
        BoundaryLocation.WEST
        >>> parse_boundary_location(BoundaryLocation.EAST)
        BoundaryLocation.EAST
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
        if self.location == BoundaryLocation.WEST:    # x=0, normal = +x
            mask = c[0, :] > 0
        elif self.location == BoundaryLocation.EAST:  # x=Nx-1, normal = -x
            mask = c[0, :] < 0
        elif self.location == BoundaryLocation.SOUTH: # y=0, normal = +y
            mask = c[1, :] > 0
        elif self.location == BoundaryLocation.NORTH: # y=Ny-1, normal = -y
            mask = c[1, :] < 0
        elif self.location == BoundaryLocation.BOTTOM: # z=0, normal = +z
            mask = c[2, :] > 0
        elif self.location == BoundaryLocation.TOP:   # z=Nz-1, normal = -z
            mask = c[2, :] < 0
        else:
            raise ValueError(f"Unknown location: {self.location}")
        
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
        
        if self.location == BoundaryLocation.WEST:
            return (slice(None), 0, slice(None), slice(None))
        elif self.location == BoundaryLocation.EAST:
            return (slice(None), Nx-1, slice(None), slice(None))
        elif self.location == BoundaryLocation.SOUTH:
            return (slice(None), slice(None), 0, slice(None))
        elif self.location == BoundaryLocation.NORTH:
            return (slice(None), slice(None), Ny-1, slice(None))
        elif self.location == BoundaryLocation.BOTTOM:
            return (slice(None), slice(None), slice(None), 0)
        elif self.location == BoundaryLocation.TOP:
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
                                 location_name: str,
                                 config: Dict[str, Any],
                                 shape: Tuple[int, ...]) -> Optional[BoundaryCondition]:
    """Create a boundary condition from config dictionary
    
    Factory function that creates the appropriate BC based on config['type'].
    
    Args:
        xp: Array module
        lattice: Lattice model
        location_name: Boundary name ('west', 'east', ...)
        config: Boundary config dictionary with keys:
                - type: 'inlet', 'outlet', 'wall', etc.
                - velocity: for inlet
                - pressure/rho_target: for outlet
                - method: for wall ('bounce_back', 'bouzidi', etc.)
        shape: Domain shape (Nx, Ny, Nz)
        
    Returns:
        BoundaryCondition instance, or None if type is not recognized
        
    Example config:
        >>> config = {"type": "inlet", "velocity": 0.1, "profile": "uniform"}
        >>> bc = create_boundary_from_config(np, lattice, 'west', config, (100, 40, 40))
    """
    # Import here to avoid circular imports
    from .inlet import EquilibriumInlet, EquilibriumInletProfile
    from .outlet import CharacteristicOutlet, ConvectiveOutlet, ExtrapolationOutlet
    
    bc_type = config.get('type', '').lower()
    location = BoundaryLocation.from_string(location_name)
    
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
    
    elif bc_type == 'outlet':
        method = config.get('method', 'characteristic').lower()
        rho_target = config.get('pressure', config.get('rho_target', 1.0))
        
        if method == 'characteristic' or method == 'open':
            relax_coeff = config.get('relax_coeff', 0.1)
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
    
    elif bc_type == 'wall':
        # Wall BCs will be implemented later
        # method = config.get('method', 'bounce_back')
        print(f"Warning: Wall BC not yet implemented for {location_name}")
        return None
    
    elif bc_type == 'periodic':
        # Periodic is handled by streaming, no explicit BC needed
        return None
    
    else:
        print(f"Warning: Unknown boundary type '{bc_type}' for {location_name}")
        return None


def create_all_boundaries_from_config(xp: 'ModuleType',
                                       lattice: 'Lattice',
                                       boundaries_config: Dict[str, Dict],
                                       shape: Tuple[int, ...]) -> BoundaryManager:
    """Create BoundaryManager with all boundaries from config
    
    Args:
        xp: Array module
        lattice: Lattice model
        boundaries_config: Dictionary of boundary configs
                          e.g., {"west": {...}, "east": {...}, ...}
        shape: Domain shape
        
    Returns:
        BoundaryManager with all boundaries added
        
    Example:
        >>> boundaries_config = {
        ...     "west": {"type": "inlet", "velocity": 0.1},
        ...     "east": {"type": "outlet", "pressure": 1.0},
        ... }
        >>> bc_manager = create_all_boundaries_from_config(np, lattice, 
        ...                                                 boundaries_config, shape)
    """
    manager = BoundaryManager()
    
    for location_name, bc_config in boundaries_config.items():
        bc = create_boundary_from_config(xp, lattice, location_name, bc_config, shape)
        if bc is not None:
            manager.add(bc)
            print(f"  Added {bc_config.get('type', 'unknown')} BC at {location_name}")
    
    return manager