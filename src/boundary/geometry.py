"""
Geometry Creation Utilities for LBM

This module provides utility functions for creating solid masks
for common geometries (spheres, cylinders, boxes, etc.).

All functions return boolean masks where:
    True  = solid (wall)
    False = fluid

Author: LBM Development Team
Date: 2026-01
"""

from typing import TYPE_CHECKING, Tuple, Optional, Union
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


def create_sphere_mask(xp: 'ModuleType', 
                       shape: Tuple[int, int, int], 
                       center: Tuple[float, float, float], 
                       radius: float) -> 'npt.NDArray':
    """Create a spherical solid mask
    
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        center: Sphere center (x, y, z)  [lattice units]
        radius: Sphere radius  [lattice units]
        
    Returns:
        Boolean mask, shape (Nx, Ny, Nz), True inside sphere
        
    Example:
        >>> mask = create_sphere_mask(np, (100, 40, 40), (50, 20, 20), 5.0)
    """
    Nx, Ny, Nz = shape
    cx, cy, cz = center
    
    x = xp.arange(Nx, dtype=xp.float64)
    y = xp.arange(Ny, dtype=xp.float64)
    z = xp.arange(Nz, dtype=xp.float64)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    distance = xp.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
    
    return distance <= radius


def create_cylinder_mask(xp: 'ModuleType', 
                         shape: Tuple[int, int, int],
                         center: Tuple[float, float],
                         radius: float,
                         axis: str = 'z',
                         height: Optional[float] = None,
                         axis_center: Optional[float] = None,
                         axis_range: Optional[Tuple[float, float]] = None) -> 'npt.NDArray':
    """Create a cylindrical solid mask
    
    Creates a cylinder aligned with the specified axis. Can be:
    - Infinite: spans the entire domain along the axis (default)
    - Finite: limited height specified by height/axis_center or axis_range
    
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        center: Cylinder center in the plane perpendicular to axis  [lattice units]
                For axis='z': (x_center, y_center)
                For axis='x': (y_center, z_center)
                For axis='y': (x_center, z_center)
        radius: Cylinder radius  [lattice units]
        axis: Cylinder axis direction ('x', 'y', or 'z')
        height: Cylinder height/length along axis  [lattice units, optional]
                If None, cylinder spans entire domain (infinite)
        axis_center: Center position along axis  [lattice units, optional]
                     Only used if height is specified
                     If None, defaults to domain center
        axis_range: Explicit range (min, max) along axis  [lattice units, optional]
                    Alternative to height/axis_center specification
        
    Returns:
        Boolean mask, shape (Nx, Ny, Nz), True inside cylinder
        
    Examples:
        # Infinite cylinder (spans entire z)
        >>> mask = create_cylinder_mask(np, (100, 40, 40), (25, 20), 4.0, axis='z')
        
        # Finite cylinder with height=20, centered at z=20
        >>> mask = create_cylinder_mask(np, (100, 40, 40), (25, 20), 4.0, 
        ...                             axis='z', height=20, axis_center=20)
        
        # Finite cylinder with explicit z range
        >>> mask = create_cylinder_mask(np, (100, 40, 40), (25, 20), 4.0,
        ...                             axis='z', axis_range=(10, 30))
    """
    Nx, Ny, Nz = shape
    
    x = xp.arange(Nx, dtype=xp.float64)
    y = xp.arange(Ny, dtype=xp.float64)
    z = xp.arange(Nz, dtype=xp.float64)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    # Compute radial distance from cylinder axis
    if axis == 'z':
        cx, cy = center
        radial_distance = xp.sqrt((X - cx)**2 + (Y - cy)**2)
        axis_coord = Z
        axis_length = Nz
    elif axis == 'x':
        cy, cz = center
        radial_distance = xp.sqrt((Y - cy)**2 + (Z - cz)**2)
        axis_coord = X
        axis_length = Nx
    elif axis == 'y':
        cx, cz = center
        radial_distance = xp.sqrt((X - cx)**2 + (Z - cz)**2)
        axis_coord = Y
        axis_length = Ny
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
    
    # Radial condition: inside cylinder radius
    radial_mask = radial_distance <= radius
    
    # Axial condition: within height limits
    if axis_range is not None:
        # Explicit range specified
        z_min, z_max = axis_range
        axial_mask = (axis_coord >= z_min) & (axis_coord <= z_max)
    elif height is not None:
        # Height and center specified
        if axis_center is None:
            axis_center = (axis_length - 1) / 2.0  # Default: domain center
        half_height = height / 2.0
        z_min = axis_center - half_height
        z_max = axis_center + half_height
        axial_mask = (axis_coord >= z_min) & (axis_coord <= z_max)
    else:
        # Infinite cylinder (spans entire axis)
        axial_mask = True
    
    return radial_mask & axial_mask


def create_box_mask(xp: 'ModuleType', 
                    shape: Tuple[int, int, int],
                    corner_min: Tuple[int, int, int],
                    corner_max: Tuple[int, int, int]) -> 'npt.NDArray':
    """Create a rectangular box solid mask
    
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        corner_min: Minimum corner (x_min, y_min, z_min)  [lattice units]
        corner_max: Maximum corner (x_max, y_max, z_max)  [lattice units]
        
    Returns:
        Boolean mask, shape (Nx, Ny, Nz), True inside box
        
    Example:
        >>> mask = create_box_mask(np, (100, 40, 40), (20, 10, 10), (30, 30, 30))
    """
    Nx, Ny, Nz = shape
    x_min, y_min, z_min = corner_min
    x_max, y_max, z_max = corner_max
    
    x = xp.arange(Nx)
    y = xp.arange(Ny)
    z = xp.arange(Nz)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    mask = ((X >= x_min) & (X <= x_max) &
            (Y >= y_min) & (Y <= y_max) &
            (Z >= z_min) & (Z <= z_max))
    
    return mask


def create_channel_walls_mask(xp: 'ModuleType', 
                               shape: Tuple[int, int, int],
                               wall_directions: str = 'yz') -> 'npt.NDArray':
    """Create wall masks for channel boundaries
    
    Creates solid walls at the domain boundaries in specified directions.
    
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        wall_directions: Which directions have walls
                        'y' = walls at y=0 and y=Ny-1
                        'z' = walls at z=0 and z=Nz-1
                        'yz' = both (rectangular channel)
                        'xyz' = all directions (closed box)
                        
    Returns:
        Boolean mask, shape (Nx, Ny, Nz), True at wall nodes
        
    Example:
        >>> mask = create_channel_walls_mask(np, (100, 40, 40), 'yz')
    """
    Nx, Ny, Nz = shape
    mask = xp.zeros(shape, dtype=bool)
    
    if 'y' in wall_directions:
        mask[:, 0, :] = True
        mask[:, -1, :] = True
    
    if 'z' in wall_directions:
        mask[:, :, 0] = True
        mask[:, :, -1] = True
    
    if 'x' in wall_directions:
        mask[0, :, :] = True
        mask[-1, :, :] = True
    
    return mask


def create_ellipsoid_mask(xp: 'ModuleType',
                          shape: Tuple[int, int, int],
                          center: Tuple[float, float, float],
                          semi_axes: Tuple[float, float, float]) -> 'npt.NDArray':
    """Create an ellipsoidal solid mask
    
    Ellipsoid equation: (x-cx)²/a² + (y-cy)²/b² + (z-cz)²/c² <= 1
    
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        center: Ellipsoid center (cx, cy, cz)  [lattice units]
        semi_axes: Semi-axes lengths (a, b, c)  [lattice units]
                   a = x-direction, b = y-direction, c = z-direction
        
    Returns:
        Boolean mask, shape (Nx, Ny, Nz), True inside ellipsoid
        
    Example:
        >>> mask = create_ellipsoid_mask(np, (100, 40, 40), (50, 20, 20), (10, 5, 5))
    """
    Nx, Ny, Nz = shape
    cx, cy, cz = center
    a, b, c = semi_axes
    
    x = xp.arange(Nx, dtype=xp.float64)
    y = xp.arange(Ny, dtype=xp.float64)
    z = xp.arange(Nz, dtype=xp.float64)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    # Ellipsoid equation
    normalized_distance = ((X - cx)/a)**2 + ((Y - cy)/b)**2 + ((Z - cz)/c)**2
    
    return normalized_distance <= 1.0


def combine_masks(xp: 'ModuleType', 
                  *masks, 
                  operation: str = 'union') -> 'npt.NDArray':
    """Combine multiple masks with boolean operations
    
    Args:
        xp: Array module (numpy or cupy)
        *masks: Variable number of boolean masks (same shape)
        operation: 'union' (OR), 'intersection' (AND), or 'difference' (first - others)
        
    Returns:
        Combined boolean mask
        
    Example:
        >>> sphere = create_sphere_mask(np, shape, (50, 20, 20), 10)
        >>> cylinder = create_cylinder_mask(np, shape, (50, 20), 5, 'x')
        >>> combined = combine_masks(np, sphere, cylinder, operation='union')
    """
    if len(masks) == 0:
        raise ValueError("At least one mask required")
    
    result = masks[0].copy()
    
    if operation == 'union':
        for mask in masks[1:]:
            result = result | mask
    elif operation == 'intersection':
        for mask in masks[1:]:
            result = result & mask
    elif operation == 'difference':
        for mask in masks[1:]:
            result = result & (~mask)
    else:
        raise ValueError(f"Unknown operation: {operation}. Use 'union', 'intersection', or 'difference'")
    
    return result


def get_geometry_info(mask: 'npt.NDArray', xp: 'ModuleType' = None) -> dict:
    """Get information about a geometry mask
    
    Args:
        mask: Boolean solid mask
        xp: Array module (auto-detected if None)
        
    Returns:
        Dictionary with geometry statistics
    """
    if xp is None:
        xp = np
    
    solid_count = int(xp.sum(mask))
    total_nodes = mask.size
    fluid_count = total_nodes - solid_count
    
    return {
        'solid_nodes': solid_count,
        'fluid_nodes': fluid_count,
        'total_nodes': total_nodes,
        'solid_fraction': solid_count / total_nodes,
        'shape': mask.shape
    }