"""
Geometry Manager for LBM Solver

Physical Process Flow:
1. Parse geometry configuration → Identify geometry type
2. Extract parameters (dimensions, position, orientation)
3. Validate parameters against domain constraints
4. Generate appropriate mask using geometry functions
5. Return boolean mask for boundary condition application

This module centralizes all geometry creation logic, removing it from main.py.

Location: src/boundary/geometry_manager.py

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Tuple, Dict, Any, Optional
import warnings

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

# Import geometry functions from src.boundary.geometry
from src.boundary.geometry import (
    create_sphere_mask,
    create_cylinder_mask,
    create_circle_mask,
    create_box_mask,
    create_airfoil_2d_mask,
)


def create_geometry_mask(
    xp: 'ModuleType',
    lattice: Any,  # Lattice object (D2Q9 or D3Q27)
    domain_shape: Tuple[int, ...],
    geometry_config: Dict[str, Any],
    characteristic_length: Optional[float] = None,
    verbose: bool = True
) -> Tuple['npt.NDArray', Dict[str, Any]]:
    """Create geometry mask from configuration
    
    Physical Process:
        1. Detect geometry type from config (priority order)
        2. Extract and validate parameters
        3. Call appropriate geometry generation function
        4. Return boolean mask (True = solid, False = fluid)
        
    Args:
        xp: Array module (numpy or cupy)
        lattice: Lattice object with dim attribute
        domain_shape: Domain shape (Nx, Ny) or (Nx, Ny, Nz)  [lattice units]
        geometry_config: Geometry configuration dictionary
        characteristic_length: Reference length for defaults  [lattice units, optional]
        verbose: If True, print geometry information
        
    Returns:
        (mask, info): Boolean mask and geometry information dictionary
        
    Geometry Priority (highest to lowest):
        2D: airfoil > circle > cylinder (as circle) > box
        3D: cylinder > sphere > box > (wing - future)
        
    Example Config (2D Airfoil):
        {
            "airfoil": {
                "enabled": True,
                "naca": "0012",
                "chord": 40,
                "angle_of_attack": 5.0,
                "center": (100, 200),
                "num_points": 150
            }
        }
        
    Example Config (3D Cylinder):
        {
            "cylinder": {
                "enabled": True,
                "center": (100, 150),
                "radius": 20,
                "axis": "z",
                "axis_range": (0, 50)
            }
        }
        
    Raises:
        ValueError: If no valid geometry is configured
        ValueError: If parameters are invalid
    """
    dim = lattice.dim
    
    # Default characteristic length
    if characteristic_length is None:
        characteristic_length = min(domain_shape) // 10
    
    # Geometry type detection order
    if dim == 2:
        # 2D: Check airfoil > circle > cylinder > box
        geometry_types = ['airfoil', 'circle', 'cylinder', 'box']
    else:
        # 3D: Check cylinder > sphere > box
        geometry_types = ['cylinder', 'sphere', 'box', 'airfoil']
    
    # Find first enabled geometry
    selected_type = None
    selected_config = None
    
    for geom_type in geometry_types:
        if geom_type in geometry_config:
            geom_config = geometry_config[geom_type]
            if geom_config.get('enabled', False):
                selected_type = geom_type
                selected_config = geom_config
                break
    
    # No geometry enabled → return empty mask
    if selected_type is None:
        if verbose:
            print("  Internal Obstacle: (none)")
        empty_mask = xp.zeros(domain_shape, dtype=bool)
        info = {'type': 'none', 'solid_nodes': 0}
        return empty_mask, info
    
    # Generate mask based on geometry type
    if dim == 2:
        mask, info = _create_2d_geometry(
            xp, domain_shape, selected_type, selected_config,
            characteristic_length, verbose
        )
    else:
        mask, info = _create_3d_geometry(
            xp, domain_shape, selected_type, selected_config,
            characteristic_length, verbose
        )
    
    return mask, info


def _create_2d_geometry(
    xp: 'ModuleType',
    domain_shape: Tuple[int, int],
    geom_type: str,
    config: Dict[str, Any],
    char_length: float,
    verbose: bool
) -> Tuple['npt.NDArray', Dict[str, Any]]:
    """Create 2D geometry mask
    
    Args:
        xp: Array module
        domain_shape: (Nx, Ny)  [lattice units]
        geom_type: Geometry type ('airfoil', 'circle', 'cylinder', 'box')
        config: Geometry configuration
        char_length: Characteristic length for defaults  [lattice units]
        verbose: Print information
        
    Returns:
        (mask, info): Mask and geometry info dictionary
    """
    Nx, Ny = domain_shape
    
    if geom_type == 'airfoil':
        # NACA Airfoil
        naca = config.get('naca', '0012')
        chord = config.get('chord', char_length)
        aoa = config.get('angle_of_attack', 0.0)
        center = config.get('center', (Nx//5, Ny//2))
        num_points = config.get('num_points', 150)
        
        mask = create_airfoil_2d_mask(
            xp, domain_shape,
            naca=naca,
            chord_length=chord,
            center=center,
            angle_of_attack=aoa,
            num_points=num_points
        )
        
        info = {
            'type': 'airfoil',
            'naca': naca,
            'chord': chord,
            'angle_of_attack': aoa,
            'center': center,
            'num_points': num_points,
            'solid_nodes': int(xp.sum(mask))
        }
        
        if verbose:
            print(f"  Internal Obstacle (NACA Airfoil, 2D):")
            print(f"    NACA {naca}, chord={chord} [lattice units]")
            print(f"    center={center}, AoA={aoa}° [degrees]")
            print(f"    num_points={num_points}")
            print(f"    {info['solid_nodes']} solid nodes")
    
    elif geom_type in ['circle', 'cylinder']:
        # Circle (or cylinder interpreted as circle in 2D)
        center = config.get('center', (Nx//5, Ny//2))
        radius = config.get('radius', char_length//2)
        
        mask = create_circle_mask(xp, domain_shape, center=center, radius=radius)
        
        info = {
            'type': 'circle',
            'center': center,
            'radius': radius,
            'solid_nodes': int(xp.sum(mask))
        }
        
        if verbose:
            print(f"  Internal Obstacle (Circle, 2D):")
            print(f"    center={center}, R={radius} [lattice units]")
            print(f"    {info['solid_nodes']} solid nodes")
    
    elif geom_type == 'box':
        # Rectangle (2D box)
        corner_min = config.get('corner_min', (Nx//4, Ny//3))
        corner_max = config.get('corner_max', (Nx//3, 2*Ny//3))
        
        # 2D box: create rectangular mask
        import numpy as np_local
        x_min, y_min = corner_min
        x_max, y_max = corner_max
        
        x = np_local.arange(Nx)
        y = np_local.arange(Ny)
        X, Y = np_local.meshgrid(x, y, indexing='ij')
        
        mask_np = ((X >= x_min) & (X <= x_max) & 
                   (Y >= y_min) & (Y <= y_max))
        
        if xp.__name__ == 'cupy':
            import cupy as cp
            mask = cp.asarray(mask_np)
        else:
            mask = mask_np
        
        info = {
            'type': 'box',
            'corner_min': corner_min,
            'corner_max': corner_max,
            'solid_nodes': int(xp.sum(mask))
        }
        
        if verbose:
            print(f"  Internal Obstacle (Box, 2D):")
            print(f"    corners=({corner_min}, {corner_max}) [lattice units]")
            print(f"    {info['solid_nodes']} solid nodes")
    
    else:
        raise ValueError(f"Unknown 2D geometry type: {geom_type}")
    
    return mask, info


def _create_3d_geometry(
    xp: 'ModuleType',
    domain_shape: Tuple[int, int, int],
    geom_type: str,
    config: Dict[str, Any],
    char_length: float,
    verbose: bool
) -> Tuple['npt.NDArray', Dict[str, Any]]:
    """Create 3D geometry mask
    
    Args:
        xp: Array module
        domain_shape: (Nx, Ny, Nz)  [lattice units]
        geom_type: Geometry type ('cylinder', 'sphere', 'box', 'airfoil')
        config: Geometry configuration
        char_length: Characteristic length for defaults  [lattice units]
        verbose: Print information
        
    Returns:
        (mask, info): Mask and geometry info dictionary
    """
    Nx, Ny, Nz = domain_shape
    
    if geom_type == 'cylinder':
        # Cylinder
        center = config.get('center', (Nx//5, Ny//2))
        radius = config.get('radius', char_length//2)
        axis = config.get('axis', 'z')
        axis_range = config.get('axis_range', None)
        height = config.get('height', None)
        axis_center = config.get('axis_center', None)
        
        mask = create_cylinder_mask(
            xp, domain_shape,
            center=center,
            radius=radius,
            axis=axis,
            height=height,
            axis_center=axis_center,
            axis_range=axis_range
        )
        
        info = {
            'type': 'cylinder',
            'center': center,
            'radius': radius,
            'axis': axis,
            'axis_range': axis_range,
            'solid_nodes': int(xp.sum(mask))
        }
        
        if verbose:
            print(f"  Internal Obstacle (Cylinder, 3D):")
            print(f"    center={center}, R={radius} [lattice units]")
            print(f"    axis={axis}, range={axis_range}")
            print(f"    {info['solid_nodes']} solid nodes")
    
    elif geom_type == 'sphere':
        # Sphere
        center = config.get('center', (Nx//5, Ny//2, Nz//2))
        radius = config.get('radius', char_length//2)
        
        mask = create_sphere_mask(
            xp, domain_shape,
            center=center,
            radius=radius
        )
        
        info = {
            'type': 'sphere',
            'center': center,
            'radius': radius,
            'solid_nodes': int(xp.sum(mask))
        }
        
        if verbose:
            print(f"  Internal Obstacle (Sphere, 3D):")
            print(f"    center={center}, R={radius} [lattice units]")
            print(f"    {info['solid_nodes']} solid nodes")
    
    elif geom_type == 'box':
        # Box
        corner_min = config.get('corner_min', (Nx//4, Ny//3, Nz//3))
        corner_max = config.get('corner_max', (Nx//3, 2*Ny//3, 2*Nz//3))
        
        mask = create_box_mask(
            xp, domain_shape,
            corner_min=corner_min,
            corner_max=corner_max
        )
        
        info = {
            'type': 'box',
            'corner_min': corner_min,
            'corner_max': corner_max,
            'solid_nodes': int(xp.sum(mask))
        }
        
        if verbose:
            print(f"  Internal Obstacle (Box, 3D):")
            print(f"    corners=({corner_min}, {corner_max}) [lattice units]")
            print(f"    {info['solid_nodes']} solid nodes")
    
    elif geom_type == 'airfoil':
        # 3D wing not yet supported
        warnings.warn(
            "NACA airfoil is currently 2D-only. "
            "3D wing support will be added in future updates. "
            "Returning empty mask.",
            UserWarning
        )
        mask = xp.zeros(domain_shape, dtype=bool)
        info = {
            'type': 'airfoil_unsupported',
            'solid_nodes': 0,
            'message': '3D wing not yet implemented'
        }
        
        if verbose:
            print(f"  WARNING: NACA airfoil is 2D-only.")
            print(f"           3D wing support coming in future update.")
            print(f"  Internal Obstacle: (none)")
    
    else:
        raise ValueError(f"Unknown 3D geometry type: {geom_type}")
    
    return mask, info


def validate_geometry_config(
    geometry_config: Dict[str, Any],
    domain_shape: Tuple[int, ...],
    lattice_dim: int
) -> Tuple[bool, str]:
    """Validate geometry configuration
    
    Checks:
        - Geometry fits within domain
        - Parameters are physically reasonable
        - Required fields are present
        
    Args:
        geometry_config: Geometry configuration dictionary
        domain_shape: Domain shape  [lattice units]
        lattice_dim: Lattice dimension (2 or 3)
        
    Returns:
        (is_valid, message): Validation result and message
        
    Example:
        >>> is_valid, msg = validate_geometry_config(config, (500, 200), 2)
        >>> if not is_valid:
        ...     print(f"Invalid config: {msg}")
    """
    # Find enabled geometry
    enabled_geoms = []
    for geom_type, geom_config in geometry_config.items():
        if isinstance(geom_config, dict) and geom_config.get('enabled', False):
            enabled_geoms.append(geom_type)
    
    if len(enabled_geoms) == 0:
        return True, "No geometry enabled (valid)"
    
    if len(enabled_geoms) > 1:
        return False, f"Multiple geometries enabled: {enabled_geoms}. Only one allowed."
    
    geom_type = enabled_geoms[0]
    config = geometry_config[geom_type]
    
    # Type-specific validation
    if geom_type in ['circle', 'cylinder', 'sphere']:
        radius = config.get('radius', 0)
        if radius <= 0:
            return False, f"{geom_type}: radius must be positive"
        
        if radius > min(domain_shape) / 2:
            return False, f"{geom_type}: radius too large for domain"
    
    elif geom_type == 'airfoil':
        if lattice_dim != 2:
            return False, "Airfoil only supported in 2D (use dimension=2)"
        
        chord = config.get('chord', 0)
        if chord <= 0:
            return False, "Airfoil: chord must be positive"
        
        if chord > min(domain_shape) * 0.8:
            return False, "Airfoil: chord too large (should be < 80% of domain)"
        
        naca = config.get('naca', '')
        if len(naca) != 4:
            return False, f"Airfoil: NACA must be 4 digits, got '{naca}'"
    
    elif geom_type == 'box':
        corner_min = config.get('corner_min', None)
        corner_max = config.get('corner_max', None)
        
        if corner_min is None or corner_max is None:
            return False, "Box: corner_min and corner_max required"
        
        if len(corner_min) != lattice_dim or len(corner_max) != lattice_dim:
            return False, f"Box: corners must match dimension {lattice_dim}D"
    
    return True, "Valid"


# =============================================================================
# Convenience functions for common geometries
# =============================================================================

def create_cylinder_obstacle(
    xp: 'ModuleType',
    domain_shape: Tuple[int, int, int],
    center: Tuple[float, float],
    radius: float,
    axis: str = 'z',
    axis_range: Optional[Tuple[float, float]] = None
) -> 'npt.NDArray':
    """Quick helper to create cylinder mask
    
    Args:
        xp: Array module
        domain_shape: (Nx, Ny, Nz)  [lattice units]
        center: (cx, cy) perpendicular to axis  [lattice units]
        radius: Cylinder radius  [lattice units]
        axis: Cylinder axis ('x', 'y', 'z')
        axis_range: Optional (min, max) along axis  [lattice units]
        
    Returns:
        Boolean mask  [lattice units]
    """
    return create_cylinder_mask(
        xp, domain_shape, center, radius, axis, axis_range=axis_range
    )


def create_airfoil_obstacle(
    xp: 'ModuleType',
    domain_shape: Tuple[int, int],
    naca: str,
    chord: float,
    center: Tuple[float, float],
    aoa: float = 0.0
) -> 'npt.NDArray':
    """Quick helper to create airfoil mask
    
    Args:
        xp: Array module
        domain_shape: (Nx, Ny)  [lattice units]
        naca: NACA 4-digit designation
        chord: Chord length  [lattice units]
        center: (cx, cy)  [lattice units]
        aoa: Angle of attack  [degrees]
        
    Returns:
        Boolean mask  [lattice units]
    """
    return create_airfoil_2d_mask(
        xp, domain_shape, naca, chord, center, aoa, num_points=150
    )