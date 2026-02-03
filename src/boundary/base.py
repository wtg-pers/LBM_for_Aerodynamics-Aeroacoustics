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
    
    Factory function that creates the appropriate BC based on 'method'.
    
    The 'method' parameter determines the BC type:
        - 'equilibrium': Equilibrium inlet (f = f_eq)
        - 'non_equilibrium': Non-eq extrapolation inlet (preserves viscous info)
        - 'characteristic': Non-reflecting open BC (for outlet)
        - 'farfield': Far-field BC maintaining U∞ (for external flow)
        - 'ambient': Ambient BC for hover (pressure-based, velocity free)
        - 'sponge': Sponge layer (buffer zone damping)
        - 'convective': Advective outlet
        - 'extrapolation': Zero-gradient outlet
        - 'bounce_back': Half-way bounce-back wall
        - 'periodic': No explicit BC (handled by streaming)
    
    Args:
        xp: Array module
        lattice: Lattice model
        bc_name: User-defined name for this boundary (for logging)
        config: Boundary config dictionary with keys:
                - location: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax' (required)
                - method: BC method (required)
                - (method-specific parameters)
        shape: Domain shape (Nx, Ny, Nz)
        
    Returns:
        BoundaryCondition instance, or None if method is periodic/unknown
        
    Example configs:
        >>> # Inlet
        >>> config = {"location": "xmin", "method": "non_equilibrium", 
        ...           "velocity": 0.1, "rho": 1.0}
        
        >>> # Far-field (maintains freestream for external flow)
        >>> config = {"location": "ymin", "method": "farfield",
        ...           "rho": 1.0, "u_inf": 0.1, "k": 0.1}
        
        >>> # Ambient (hover, velocity free)
        >>> config = {"location": "zmax", "method": "ambient",
        ...           "rho": 1.0, "k": 0.5}
        
        >>> # Sponge layer
        >>> config = {"location": "xmax", "method": "sponge",
        ...           "thickness": 20, "strength": 0.8, "rho": 1.0, "u_inf": 0.1}
        
        >>> # Wall
        >>> config = {"location": "ymax", "method": "bounce_back"}
    """
    # Import here to avoid circular imports
    from .inlet import EquilibriumInlet, NonEquilibriumInlet
    from .outlet import CharacteristicOutlet, ConvectiveOutlet, ExtrapolationOutlet
    from .domain_wall import DomainWallBounceBack
    from .farfield import CharacteristicFarfield, AmbientBC, SpongeLayer
    
    # Get location (required)
    location_str = config.get('location')
    if location_str is None:
        print(f"Warning: 'location' not specified for boundary '{bc_name}'")
        return None
    
    try:
        location = BoundaryLocation.from_string(location_str)
    except ValueError as e:
        print(f"Warning: {e}")
        return None
    
    # Get method (required)
    method = config.get('method', '').lower()
    if not method:
        # Legacy support: check for 'type' field
        bc_type = config.get('type', '').lower()
        if bc_type == 'inlet':
            method = config.get('method', 'non_equilibrium').lower()
        elif bc_type == 'outlet':
            method = config.get('method', 'characteristic').lower()
        elif bc_type == 'wall':
            method = config.get('method', 'bounce_back').lower()
        elif bc_type == 'open':
            method = 'characteristic'
        elif bc_type == 'periodic':
            method = 'periodic'
        else:
            print(f"Warning: No 'method' specified for boundary '{bc_name}'")
            return None
    
    # =========================================================================
    # Equilibrium Inlet
    # =========================================================================
    if method in ['equilibrium', 'eq']:
        velocity = config.get('velocity', 0.1)
        density = config.get('rho', config.get('density', 1.0))
        
        return EquilibriumInlet(xp, lattice, location,
                                velocity=velocity, density=density,
                                shape=shape)
    
    # =========================================================================
    # Non-Equilibrium Inlet (recommended for mass conservation)
    # =========================================================================
    elif method in ['non_equilibrium', 'non_eq', 'neq']:
        velocity = config.get('velocity', 0.1)
        density = config.get('rho', config.get('density', 1.0))
        
        return NonEquilibriumInlet(xp, lattice, location,
                                   velocity=velocity, density=density,
                                   shape=shape)
    
    # =========================================================================
    # Characteristic Outlet (for outlet, not far-field)
    # =========================================================================
    elif method in ['characteristic', 'open', 'non_reflecting']:
        rho_target = config.get('rho', config.get('rho_target', config.get('pressure', 1.0)))
        relax_coeff = config.get('k', config.get('relax_coeff', 0.1))
        
        return CharacteristicOutlet(xp, lattice, location,
                                    rho_target=rho_target,
                                    relax_coeff=relax_coeff,
                                    shape=shape)
    
    # =========================================================================
    # Characteristic Far-field (maintains U∞ for external flow)
    # =========================================================================
    elif method in ['farfield', 'far_field', 'freestream']:
        rho_inf = config.get('rho', config.get('rho_inf', 1.0))
        u_inf = config.get('u_inf', config.get('velocity', 0.1))
        relax_coeff = config.get('k', config.get('relax_coeff', 0.1))
        
        return CharacteristicFarfield(xp, lattice, location,
                                      rho_inf=rho_inf,
                                      u_inf=u_inf,
                                      relax_coeff=relax_coeff,
                                      shape=shape)
    
    # =========================================================================
    # Ambient BC (hover - pressure-based, velocity free)
    # =========================================================================
    elif method in ['ambient', 'pressure_open', 'hover']:
        rho_ambient = config.get('rho', config.get('rho_ambient', 1.0))
        relax_coeff = config.get('k', config.get('relax_coeff', 0.5))
        
        return AmbientBC(xp, lattice, location,
                        rho_ambient=rho_ambient,
                        relax_coeff=relax_coeff,
                        shape=shape)
    
    # =========================================================================
    # Sponge Layer (buffer zone damping)
    # =========================================================================
    elif method in ['sponge', 'sponge_layer', 'buffer']:
        thickness = config.get('thickness', 10)
        strength = config.get('strength', config.get('sigma_max', 0.5))
        rho_inf = config.get('rho', config.get('rho_inf', 1.0))
        u_inf = config.get('u_inf', config.get('velocity', 0.0))
        profile = config.get('profile', 'polynomial')
        
        return SpongeLayer(xp, lattice, location,
                          thickness=thickness,
                          strength=strength,
                          rho_inf=rho_inf,
                          u_inf=u_inf,
                          profile=profile,
                          shape=shape)
    
    # =========================================================================
    # Convective Outlet
    # =========================================================================
    elif method in ['convective', 'advective']:
        u_conv = config.get('convective_velocity', config.get('velocity', 0.1))
        
        return ConvectiveOutlet(xp, lattice, location,
                                convective_velocity=u_conv,
                                shape=shape)
    
    # =========================================================================
    # Extrapolation Outlet (zero-gradient)
    # =========================================================================
    elif method in ['extrapolation', 'zero_gradient', 'neumann']:
        order = config.get('order', 1)
        
        return ExtrapolationOutlet(xp, lattice, location, order=order)
    
    # =========================================================================
    # Bounce-Back Wall
    # =========================================================================
    elif method in ['bounce_back', 'hwbb', 'halfway', 'wall']:
        exclude_io = config.get('exclude_inlet_outlet', False)
        
        return DomainWallBounceBack(xp, lattice, location, shape,
                                    exclude_inlet_outlet=exclude_io)
    
    # =========================================================================
    # Periodic (handled by streaming, no explicit BC)
    # =========================================================================
    elif method in ['periodic', 'none']:
        return None
    
    # =========================================================================
    # Unknown method
    # =========================================================================
    else:
        print(f"Warning: Unknown method '{method}' for boundary '{bc_name}'")
        print(f"  Available methods: equilibrium, non_equilibrium, characteristic,")
        print(f"                     farfield, ambient, sponge,")
        print(f"                     convective, extrapolation, bounce_back, periodic")
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
        ...     "inlet": {"location": "xmin", "method": "non_equilibrium", "velocity": 0.1},
        ...     "outlet": {"location": "xmax", "method": "characteristic", "rho": 1.0},
        ...     "farfield_y": {"location": "ymin", "method": "farfield", "u_inf": 0.1},
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
                method = bc_config.get('method', 'unknown')
                print(f"    {bc_name}: {method} at {loc}")
    
    return manager