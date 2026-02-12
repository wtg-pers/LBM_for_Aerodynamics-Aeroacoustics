"""
Boundary Conditions Base Module for LBM

This module defines the abstract base class for all boundary conditions
and provides utility functions for boundary handling.

Design Philosophy:
    - All boundary conditions modify the distribution function in-place
    - Boundary conditions are applied AFTER streaming (pull scheme)
    - Each boundary condition class handles one face/region

Corner Safety:
    When a BC face meets wall faces (e.g., inlet at xmin + wall at ymin),
    the corner/edge nodes can accumulate density because:
      1. f_neq extracted from interior includes wall nodes (unphysical values)
      2. Wall bounce-back traps mass at corners
    
    Solution: set_wall_info() builds a mask that zeros f_neq at wall-adjacent
    nodes on the BC face. Corner nodes fall back to pure equilibrium (safe).

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
Date: 2026-02
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
        """
        name_lower = name.lower().strip()
        
        # Direct mapping
        name_map = {
            'xmin': cls.XMIN, 'xmax': cls.XMAX,
            'ymin': cls.YMIN, 'ymax': cls.YMAX,
            'zmin': cls.ZMIN, 'zmax': cls.ZMAX,
            'west': cls.WEST, 'east': cls.EAST,
            'south': cls.SOUTH, 'north': cls.NORTH,
            'bottom': cls.BOTTOM, 'top': cls.TOP,
        }
        
        if name_lower in name_map:
            return name_map[name_lower]
        else:
            raise ValueError(
                f"Unknown boundary location: '{name}'. "
                f"Valid: {list(name_map.keys())}"
            )


def parse_boundary_location(location: Union[str, 'BoundaryLocation']) -> 'BoundaryLocation':
    """Parse boundary location from string or enum
    
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
    
    Corner Safety:
        Call set_wall_info() after construction to enable f_neq sanitization
        at wall-adjacent nodes. This prevents density accumulation at corners
        where this BC face meets wall boundaries.
    
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
        
        # Wall-adjacent node masking for f_neq sanitization (corner safety)
        # None = no masking (backward compatible, all nodes use f_neq)
        # ndarray = boolean mask, True = fluid (use f_neq), False = wall (f_neq = 0)
        self._neq_mask: Optional['npt.NDArray'] = None
        self._wall_faces: List[str] = []
    
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
    
    # =========================================================================
    # Corner Safety: Wall-adjacent f_neq sanitization
    # =========================================================================
    
    def set_wall_info(self, wall_faces: List[str], domain_shape: tuple) -> None:
        """Register domain wall faces for corner-safe BC application.
        
        When wall faces are registered, apply() methods will skip
        wall nodes on their face, leaving them to domain_wall bounce-back.
        
        Args:
            wall_faces: List of wall locations, e.g. ['ymin', 'ymax', 'zmin', 'zmax']
            domain_shape: Domain shape (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
        """
        # Normalize names
        name_map = {
            'south': 'ymin', 'north': 'ymax',
            'bottom': 'zmin', 'top': 'zmax',
            'west': 'xmin', 'east': 'xmax',
        }
        self._wall_faces = [name_map.get(w.lower(), w.lower()) for w in wall_faces]
        self._domain_shape = domain_shape

    def _get_face_fluid_slices(self) -> tuple:
        """Get index slices that exclude wall nodes on this BC's face.
        
        For an xmin/xmax face with walls at ymin/ymax/zmin/zmax,
        returns (slice(1,-1), slice(1,-1)) to skip y=0, y=Ny-1, z=0, z=Nz-1.
        
        Returns:
            2D: (secondary_slice,)
            3D: (secondary_slice_1, secondary_slice_2)
            If no wall info set: all slice(None) → full face (backward compatible)
        """
        if not self._wall_faces:
            # No wall info → full slices (backward compatible)
            if hasattr(self, 'dim') and self.dim == 2:
                return (slice(None),)
            return (slice(None), slice(None))
        
        wf = self._wall_faces
        loc = self.location.value
        dim = getattr(self, 'dim', 3)
        
        if dim == 2:
            # 2D faces: xmin/xmax → exclude y-walls; ymin/ymax → exclude x-walls
            if loc in ('xmin', 'xmax'):
                ys = slice(1 if 'ymin' in wf else None,
                          -1 if 'ymax' in wf else None)
                return (ys,)
            else:  # ymin, ymax
                xs = slice(1 if 'xmin' in wf else None,
                          -1 if 'xmax' in wf else None)
                return (xs,)
        else:
            # 3D faces: each face has two transverse dimensions
            if loc in ('xmin', 'xmax'):
                # Face spans (y, z) → exclude y-walls and z-walls
                ys = slice(1 if 'ymin' in wf else None,
                          -1 if 'ymax' in wf else None)
                zs = slice(1 if 'zmin' in wf else None,
                          -1 if 'zmax' in wf else None)
                return (ys, zs)
            elif loc in ('ymin', 'ymax'):
                # Face spans (x, z)
                xs = slice(1 if 'xmin' in wf else None,
                          -1 if 'xmax' in wf else None)
                zs = slice(1 if 'zmin' in wf else None,
                          -1 if 'zmax' in wf else None)
                return (xs, zs)
            else:  # zmin, zmax
                # Face spans (x, y)
                xs = slice(1 if 'xmin' in wf else None,
                          -1 if 'xmax' in wf else None)
                ys = slice(1 if 'ymin' in wf else None,
                          -1 if 'ymax' in wf else None)
                return (xs, ys)
    
    def _build_neq_mask(self, shape: tuple, dim: int) -> Optional['npt.NDArray']:
        """Build boolean mask for f_neq sanitization on this BC face
        
        The mask has the same shape as the BC face (the 2 or 1 axes
        perpendicular to the BC normal direction).
        
        Returns:
            Boolean array (True=fluid, False=wall-adjacent), or None if no walls
        """
        xp = self.xp
        loc = self.location.value
        walls = self._wall_faces
        
        if not walls:
            return None
        
        # Determine which axes form this BC face, and the face shape
        # ---------------------------------------------------------------
        # BC at xmin/xmax → face axes are (y, z) for 3D, (y,) for 2D
        # BC at ymin/ymax → face axes are (x, z) for 3D, (x,) for 2D
        # BC at zmin/zmax → face axes are (x, y) for 3D only
        # ---------------------------------------------------------------
        
        if dim == 2:
            Nx, Ny = shape
            
            if loc in ['xmin', 'xmax']:
                # Face axis: y, shape (Ny,)
                face_shape = (Ny,)
                # Which walls affect this face?
                wall_at_start = 'ymin' in walls  # y=0
                wall_at_end = 'ymax' in walls    # y=Ny-1
                
            elif loc in ['ymin', 'ymax']:
                # Face axis: x, shape (Nx,)
                face_shape = (Nx,)
                wall_at_start = 'xmin' in walls  # x=0
                wall_at_end = 'xmax' in walls    # x=Nx-1
            else:
                return None
            
            if not wall_at_start and not wall_at_end:
                return None
            
            mask = xp.ones(face_shape, dtype=bool)
            if wall_at_start:
                mask[0] = False
            if wall_at_end:
                mask[-1] = False
            return mask
        
        else:  # 3D
            Nx, Ny, Nz = shape
            
            if loc in ['xmin', 'xmax']:
                # Face axes: (y, z), shape (Ny, Nz)
                face_shape = (Ny, Nz)
                # Axis 0 of face = y
                axis0_walls = ('ymin' in walls, 'ymax' in walls)
                # Axis 1 of face = z
                axis1_walls = ('zmin' in walls, 'zmax' in walls)
                
            elif loc in ['ymin', 'ymax']:
                # Face axes: (x, z), shape (Nx, Nz)
                face_shape = (Nx, Nz)
                axis0_walls = ('xmin' in walls, 'xmax' in walls)
                axis1_walls = ('zmin' in walls, 'zmax' in walls)
                
            elif loc in ['zmin', 'zmax']:
                # Face axes: (x, y), shape (Nx, Ny)
                face_shape = (Nx, Ny)
                axis0_walls = ('xmin' in walls, 'xmax' in walls)
                axis1_walls = ('ymin' in walls, 'ymax' in walls)
            else:
                return None
            
            # Check if any walls are relevant
            any_wall = any(axis0_walls) or any(axis1_walls)
            if not any_wall:
                return None
            
            mask = xp.ones(face_shape, dtype=bool)
            
            # Axis 0: start/end
            if axis0_walls[0]:   # min wall
                mask[0, :] = False
            if axis0_walls[1]:   # max wall
                mask[-1, :] = False
            
            # Axis 1: start/end
            if axis1_walls[0]:   # min wall
                mask[:, 0] = False
            if axis1_walls[1]:   # max wall
                mask[:, -1] = False
            
            return mask
    
    def _sanitize_f_neq(self, f_neq: 'npt.NDArray') -> 'npt.NDArray':
        """Zero out f_neq at wall-adjacent nodes to prevent corner contamination
        
        Physical principle:
            At wall-adjacent nodes on the BC face, the interior reference node
            is a wall node whose macroscopic variables are unphysical (artifacts
            of bounce-back). Using this f_neq would inject spurious density.
            Setting f_neq=0 makes those nodes use pure equilibrium (safe).
        
        Args:
            f_neq: Non-equilibrium distribution, shape (Q_or_n, *face_shape)
                   First axis is velocity directions, remaining axes match BC face
        
        Returns:
            Sanitized f_neq (zero at wall-adjacent positions)
            If no wall info set, returns f_neq unchanged.
        """
        if self._neq_mask is None:
            return f_neq
        
        # Broadcasting: f_neq has shape (n_directions, *face_shape)
        #               mask has shape (*face_shape)
        # NumPy/CuPy broadcast: mask auto-expands along axis 0
        return f_neq * self._neq_mask
    
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
    
    def set_wall_info_all(self, wall_faces: List[str], domain_shape: tuple) -> None:
        """Propagate wall info to all registered BCs for corner-safe slicing.
        
        Args:
            wall_faces: List of wall face names, e.g. ['ymin', 'ymax', 'zmin', 'zmax']
            domain_shape: Domain shape
        """
        for bc in self.boundaries:
            bc.set_wall_info(wall_faces, domain_shape)
    
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
                                 shape: Tuple[int, ...]) -> Optional['BoundaryCondition']:
    """Create a boundary condition from config dictionary
    
    Factory function that creates the appropriate BC based on 'method'.
    
    Available Methods:
    ==================
    
    Inlet BCs:
        - 'equilibrium' / 'eq': EquilibriumInlet (simple, f = f^eq)
        - 'non_equilibrium' / 'neq': NonEquilibriumInlet (preserves f^neq)
        - 'regularized_inlet' / 'reg_inlet': RegularizedInlet (full reconstruction)
    
    Outlet BCs:
        - 'pressure_relaxation': PressureRelaxationOutlet (recommended)
        - 'regularized_outlet' / 'reg_outlet': RegularizedOutlet (full reconstruction)
        - 'convective': ConvectiveOutlet (advection-based)
        - 'neumann' / 'extrapolation': NeumannOutlet (zero-gradient)
    
    Wall BCs:
        - 'bounce_back' / 'hwbb' / 'wall': DomainWallBounceBack
        - 'regularized_wall' / 'reg_wall': RegularizedWall (full reconstruction)
    
    Far-field BCs:
        - 'freestream' / 'farfield': FreestreamBC (velocity FIXED to U∞)
        - 'sponge': SpongeLayer (buffer zone damping)
    
    Other:
        - 'periodic' / 'none': No explicit BC (handled by streaming)
    
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
        
        >>> # Outlet (pressure relaxation)
        >>> config = {"location": "xmax", "method": "pressure_relaxation",
        ...           "rho": 1.0, "k": 0.1}
        
        >>> # Far-field (freestream)
        >>> config = {"location": "ymin", "method": "freestream",
        ...           "rho": 1.0, "u_inf": 0.1, "k": 0.1}
        
        >>> # Sponge layer
        >>> config = {"location": "xmax", "method": "sponge",
        ...           "thickness": 20, "strength": 0.5, "rho": 1.0, "u_inf": 0.1}
        
        >>> # Wall
        >>> config = {"location": "ymax", "method": "bounce_back"}
    """
    # Import here to avoid circular imports
    from .inlet import EquilibriumInlet, NonEquilibriumInlet
    from .outlet import PressureRelaxationOutlet, ConvectiveOutlet, NeumannOutlet
    from .domain_wall import DomainWallBounceBack
    from .farfield import FreestreamBC, SpongeLayer
    from .regularized import RegularizedInlet, RegularizedOutlet, RegularizedWall
    
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
            method = config.get('method', 'pressure_relaxation').lower()
        elif bc_type == 'wall':
            method = config.get('method', 'bounce_back').lower()
        elif bc_type == 'open':
            method = 'pressure_relaxation'
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
    # Pressure Relaxation Outlet (formerly CharacteristicOutlet)
    # =========================================================================
    elif method in ['pressure_relaxation', 'characteristic', 'open', 'non_reflecting']:
        rho_target = config.get('rho', config.get('rho_target', config.get('pressure', 1.0)))
        relax_coeff = config.get('k', config.get('relax_coeff', 0.1))
        
        # Deprecation warning for old name
        if method == 'characteristic':
            import warnings
            warnings.warn(
                "method='characteristic' is deprecated. Use 'pressure_relaxation' instead.",
                DeprecationWarning
            )
        
        return PressureRelaxationOutlet(xp, lattice, location,
                                         rho_target=rho_target,
                                         relax_coeff=relax_coeff,
                                         shape=shape)
    
    # =========================================================================
    # Freestream BC (formerly CharacteristicFarfield)
    # =========================================================================
    elif method in ['freestream', 'farfield', 'far_field']:
        rho_inf = config.get('rho', config.get('rho_inf', 1.0))
        u_inf = config.get('u_inf', config.get('velocity', 0.1))
        relax_coeff = config.get('k', config.get('relax_coeff', 0.1))
        
        # Deprecation warning for old name
        if method in ['farfield', 'far_field']:
            import warnings
            warnings.warn(
                "method='farfield' is deprecated. Use 'freestream' instead.",
                DeprecationWarning
            )
        
        return FreestreamBC(xp, lattice, location,
                            u_inf=u_inf, rho_inf=rho_inf,
                            relax_coeff=relax_coeff,
                            shape=shape)
    
    # =========================================================================
    # Sponge Layer
    # =========================================================================
    elif method in ['sponge', 'sponge_layer']:
        rho_inf = config.get('rho', config.get('rho_inf', 1.0))
        u_inf = config.get('u_inf', config.get('velocity', 0.1))
        thickness = config.get('thickness', 20)
        strength = config.get('strength', config.get('sigma_max', 0.5))
        
        return SpongeLayer(xp, lattice, location,
                           u_inf=u_inf, rho_inf=rho_inf,
                           thickness=thickness, sigma_max=strength,
                           shape=shape)
    
    # =========================================================================
    # Convective Outlet
    # =========================================================================
    elif method in ['convective', 'advective']:
        u_conv = config.get('u_conv', config.get('velocity', 0.1))
        
        return ConvectiveOutlet(xp, lattice, location,
                                u_conv=u_conv, shape=shape)
    
    # =========================================================================
    # Neumann (Zero-gradient) Outlet
    # =========================================================================
    elif method in ['neumann', 'extrapolation', 'zero_gradient']:
        # Deprecation warning for old name
        if method == 'extrapolation':
            import warnings
            warnings.warn(
                "method='extrapolation' is deprecated. "
                "Use 'neumann' instead.",
                DeprecationWarning
            )
        
        return NeumannOutlet(xp, lattice, location, shape=shape)
    
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
    # Regularized Inlet (full reconstruction, no corner conflicts)
    # =========================================================================
    elif method in ['regularized_inlet', 'regularized_velocity', 'reg_inlet']:
        velocity = config.get('velocity', 0.1)
        density = config.get('rho', config.get('density', 1.0))
        
        return RegularizedInlet(xp, lattice, location,
                                 velocity=velocity, density=density,
                                 shape=shape)
    
    # =========================================================================
    # Regularized Outlet (full reconstruction, pressure relaxation)
    # =========================================================================
    elif method in ['regularized_outlet', 'regularized_pressure', 'reg_outlet']:
        rho_target = config.get('rho', config.get('rho_target', 1.0))
        relax_coeff = config.get('k', config.get('relax_coeff', 0.1))
        
        return RegularizedOutlet(xp, lattice, location,
                                  rho_target=rho_target,
                                  relax_coeff=relax_coeff,
                                  shape=shape)
    
    # =========================================================================
    # Regularized Wall (full reconstruction, no-slip or moving)
    # =========================================================================
    elif method in ['regularized_wall', 'reg_wall']:
        return RegularizedWall(xp, lattice, location, shape=shape)
    
    # =========================================================================
    # Unknown method
    # =========================================================================
    else:
        print(f"Warning: Unknown method '{method}' for boundary '{bc_name}'")
        print(f"  Available methods:")
        print(f"    Inlet: equilibrium, non_equilibrium, regularized_inlet")
        print(f"    Outlet: pressure_relaxation, regularized_outlet, convective, neumann")
        print(f"    Wall: bounce_back, regularized_wall")
        print(f"    Far-field: freestream, sponge")
        print(f"    Other: periodic")
        return None


def create_all_boundaries_from_config(xp: 'ModuleType',
                                       lattice: 'Lattice',
                                       boundaries_config: Dict[str, Dict],
                                       shape: Tuple[int, ...],
                                       verbose: bool = True) -> 'BoundaryManager':
    """Create BoundaryManager with all boundaries from config
    
    Automatically detects wall faces and applies corner safety
    (set_wall_info) to all f_neq-based BCs.
    
    Args:
        xp: Array module
        lattice: Lattice model
        boundaries_config: Dictionary of boundary configs
        shape: Domain shape
        verbose: Print info about created BCs
        
    Returns:
        BoundaryManager with all boundaries configured
    """
    manager = BoundaryManager()
    
    # ── Phase 1: Identify wall faces ──
    wall_methods = {'bounce_back', 'hwbb', 'halfway', 'wall'}
    wall_face_list: List[str] = []
    
    for bc_name, bc_config in boundaries_config.items():
        method = bc_config.get('method', '').lower()
        if method in wall_methods:
            loc = bc_config.get('location', '')
            if loc:
                wall_face_list.append(loc.lower())
    
    if verbose and wall_face_list:
        print(f"  Detected wall faces: {wall_face_list}")
    
    # ── Phase 2: Create BCs ──
    for bc_name, bc_config in boundaries_config.items():
        bc = create_boundary_from_config(xp, lattice, bc_name, bc_config, shape)
        if bc is not None:
            manager.add(bc)
    
    # ── Phase 3: Apply corner safety to all non-wall BCs ──
    if wall_face_list:
        manager.set_wall_info_all(wall_face_list, shape)
    
    return manager