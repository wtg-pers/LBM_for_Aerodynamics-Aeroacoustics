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
    
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        center: Cylinder center in the plane perpendicular to axis  [lattice units]
        radius: Cylinder radius  [lattice units]
        axis: Cylinder axis direction ('x', 'y', or 'z')
        height: Cylinder height/length along axis  [lattice units, optional]
        axis_center: Center position along axis  [lattice units, optional]
        axis_range: Explicit range (min, max) along axis  [lattice units, optional]
        
    Returns:
        Boolean mask, shape (Nx, Ny, Nz), True inside cylinder
    """
    Nx, Ny, Nz = shape
    
    x = xp.arange(Nx, dtype=xp.float64)
    y = xp.arange(Ny, dtype=xp.float64)
    z = xp.arange(Nz, dtype=xp.float64)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
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
    
    radial_mask = radial_distance <= radius
    
    if axis_range is not None:
        z_min, z_max = axis_range
        axial_mask = (axis_coord >= z_min) & (axis_coord <= z_max)
    elif height is not None:
        if axis_center is None:
            axis_center = (axis_length - 1) / 2.0
        half_height = height / 2.0
        z_min = axis_center - half_height
        z_max = axis_center + half_height
        axial_mask = (axis_coord >= z_min) & (axis_coord <= z_max)
    else:
        axial_mask = True
    
    return radial_mask & axial_mask


def create_circle_mask(xp: 'ModuleType',
                       shape: Tuple[int, int],
                       center: Tuple[float, float],
                       radius: float) -> 'npt.NDArray':
    """Create a 2D circular solid mask
    
    For 2D simulations, creates a circle (cross-section of infinite cylinder).
    
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny)  [lattice units]
        center: Circle center (cx, cy)  [lattice units]
        radius: Circle radius  [lattice units]
        
    Returns:
        Boolean mask, shape (Nx, Ny), True inside circle
    """
    Nx, Ny = shape
    cx, cy = center
    
    x = xp.arange(Nx, dtype=xp.float64)
    y = xp.arange(Ny, dtype=xp.float64)
    X, Y = xp.meshgrid(x, y, indexing='ij')
    
    distance = xp.sqrt((X - cx)**2 + (Y - cy)**2)
    
    return distance <= radius


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
    
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        wall_directions: Which directions have walls ('y', 'z', 'yz', 'xyz')
                        
    Returns:
        Boolean mask, shape (Nx, Ny, Nz), True at wall nodes
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
    
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        center: Ellipsoid center (cx, cy, cz)  [lattice units]
        semi_axes: Semi-axes lengths (a, b, c)  [lattice units]
        
    Returns:
        Boolean mask, shape (Nx, Ny, Nz), True inside ellipsoid
    """
    Nx, Ny, Nz = shape
    cx, cy, cz = center
    a, b, c = semi_axes
    
    x = xp.arange(Nx, dtype=xp.float64)
    y = xp.arange(Ny, dtype=xp.float64)
    z = xp.arange(Nz, dtype=xp.float64)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    normalized_distance = ((X - cx)/a)**2 + ((Y - cy)/b)**2 + ((Z - cz)/c)**2
    
    return normalized_distance <= 1.0


def combine_masks(xp: 'ModuleType', 
                  *masks, 
                  operation: str = 'union') -> 'npt.NDArray':
    """Combine multiple masks with boolean operations
    
    Args:
        xp: Array module (numpy or cupy)
        *masks: Variable number of boolean masks (same shape)
        operation: 'union' (OR), 'intersection' (AND), or 'difference'
        
    Returns:
        Combined boolean mask
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
        raise ValueError(f"Unknown operation: {operation}")
    
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


def create_domain_wall_mask(xp: 'ModuleType',
                            shape: Tuple[int, int, int],
                            walls: Optional[list] = None,
                            exclude_x: bool = True) -> 'npt.NDArray':
    """Create solid mask for domain boundary walls
    
    Creates a mask where boundary layers are marked as solid.
    This allows using HalfwayBounceBack (wall.py) for domain walls,
    which properly handles the periodic streaming issue.
    
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny, Nz)  [lattice units]
        walls: List of walls to include. Options:
               - 'ymin' or 'south': y = 0
               - 'ymax' or 'north': y = Ny-1
               - 'zmin' or 'bottom': z = 0
               - 'zmax' or 'top': z = Nz-1
               - 'xmin' or 'west': x = 0 (usually inlet, excluded by default)
               - 'xmax' or 'east': x = Nx-1 (usually outlet, excluded by default)
               Default: ['ymin', 'ymax', 'zmin', 'zmax']
        exclude_x: If True (default), exclude xmin/xmax from walls
                   even if specified. Set False for closed box.
        
    Returns:
        Boolean mask, shape (Nx, Ny, Nz), True = solid (wall)
        
    Note:
        Using this mask with HalfwayBounceBack means the wall is at the
        boundary layer (e.g., y=0), not between y=0 and y=1.
        The effective fluid domain becomes (Nx, Ny-2, Nz-2) for channel flow.
        
    Example:
        >>> wall_mask = create_domain_wall_mask(xp, (100, 40, 40))
        >>> wall_bc = HalfwayBounceBack(xp, lattice, wall_mask)
    """
    Nx, Ny, Nz = shape
    mask = xp.zeros(shape, dtype=bool)
    
    # Default walls
    if walls is None:
        walls = ['ymin', 'ymax', 'zmin', 'zmax']
    
    # Normalize wall names
    wall_map = {
        'south': 'ymin', 'north': 'ymax',
        'bottom': 'zmin', 'top': 'zmax',
        'west': 'xmin', 'east': 'xmax'
    }
    walls = [wall_map.get(w.lower(), w.lower()) for w in walls]
    
    # Apply walls
    for wall in walls:
        if wall == 'ymin':
            mask[:, 0, :] = True
        elif wall == 'ymax':
            mask[:, -1, :] = True
        elif wall == 'zmin':
            mask[:, :, 0] = True
        elif wall == 'zmax':
            mask[:, :, -1] = True
        elif wall == 'xmin' and not exclude_x:
            mask[0, :, :] = True
        elif wall == 'xmax' and not exclude_x:
            mask[-1, :, :] = True
    
    return mask