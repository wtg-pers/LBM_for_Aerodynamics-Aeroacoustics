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


# =============================================================================
# NACA Airfoil Generation (2D)
# =============================================================================

def _parse_naca_4digit(naca_str: str) -> Tuple[float, float, float]:
    """Parse NACA 4-digit designation into geometric parameters
    
    NACA MPXX format:
        M: Maximum camber as % of chord (first digit)
        P: Position of max camber in tenths of chord (second digit)
        XX: Maximum thickness as % of chord (last two digits)
    
    Args:
        naca_str: 4-digit string (e.g., "0012", "2412")
        
    Returns:
        (max_camber, camber_position, max_thickness)
        All values are fractions of chord (0.0 to 1.0)  [dimensionless]
        
    Examples:
        NACA 0012: (0.00, 0.0, 0.12) - symmetric, 12% thick
        NACA 2412: (0.02, 0.4, 0.12) - 2% camber at 40% chord, 12% thick
    """
    if len(naca_str) != 4:
        raise ValueError(f"NACA 4-digit must be exactly 4 digits, got '{naca_str}'")
    
    try:
        m = int(naca_str[0]) / 100.0  # Max camber [dimensionless]
        p = int(naca_str[1]) / 10.0   # Position of max camber [dimensionless]
        t = int(naca_str[2:4]) / 100.0  # Max thickness [dimensionless]
    except ValueError:
        raise ValueError(f"Invalid NACA digits: '{naca_str}'")
    
    return m, p, t


def _naca_thickness_distribution(x: 'npt.NDArray', t_max: float) -> 'npt.NDArray':
    """Compute NACA airfoil thickness distribution
    
    Uses the standard NACA 4-digit thickness formula with closed trailing edge.
    
    Physics:
        y_t = 5 * t_max * [a0*√x + a1*x + a2*x² + a3*x³ + a4*x⁴]
        Coefficients give smooth thickness from leading to trailing edge.
        
    Args:
        x: Chordwise position, normalized [0, 1]  [dimensionless]
        t_max: Maximum thickness as fraction of chord  [dimensionless]
        
    Returns:
        Half-thickness perpendicular to camber line  [dimensionless]
        
    Note:
        This formula gives a closed trailing edge (y_t(x=1) = 0).
        For open trailing edge, use a4 = -0.1036 instead.
    """
    # Import numpy (not xp, as this is always CPU-side coordinate generation)
    import numpy as np_local
    
    # Coefficients for closed trailing edge
    a0 =  0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 =  0.2843
    a4 = -0.1015  # Closed TE (use -0.1036 for open TE)
    
    # Half-thickness distribution [dimensionless]
    y_t = 5.0 * t_max * (
        a0 * np_local.sqrt(x) + 
        a1 * x + 
        a2 * x**2 + 
        a3 * x**3 + 
        a4 * x**4
    )
    
    return y_t


def _naca_camber_line(x: 'npt.NDArray', m: float, p: float) -> Tuple['npt.NDArray', 'npt.NDArray']:
    """Compute NACA 4-digit mean camber line and slope
    
    Physics:
        For symmetric airfoils (m=0): y_c = 0, dy_c/dx = 0
        For cambered airfoils (m>0): parabolic arcs before and after max camber
        
    Args:
        x: Chordwise position, normalized [0, 1]  [dimensionless]
        m: Maximum camber as fraction of chord  [dimensionless]
        p: Position of max camber as fraction of chord  [dimensionless]
        
    Returns:
        (y_c, dy_c_dx): Camber line height and slope  [dimensionless]
    """
    import numpy as np_local
    
    y_c = np_local.zeros_like(x)
    dy_c_dx = np_local.zeros_like(x)
    
    if m == 0.0 or p == 0.0:
        # Symmetric airfoil: no camber
        return y_c, dy_c_dx
    
    # Forward arc (0 ≤ x < p)
    mask_forward = x < p
    if np_local.any(mask_forward):
        y_c[mask_forward] = (m / p**2) * (2*p*x[mask_forward] - x[mask_forward]**2)
        dy_c_dx[mask_forward] = (2*m / p**2) * (p - x[mask_forward])
    
    # Aft arc (p ≤ x ≤ 1)
    mask_aft = ~mask_forward
    if np_local.any(mask_aft):
        y_c[mask_aft] = (m / (1-p)**2) * (
            (1 - 2*p) + 2*p*x[mask_aft] - x[mask_aft]**2
        )
        dy_c_dx[mask_aft] = (2*m / (1-p)**2) * (p - x[mask_aft])
    
    return y_c, dy_c_dx


def generate_naca_4digit_coordinates(
    naca: str,
    num_points: int = 100,
    cosine_spacing: bool = True
) -> Tuple['npt.NDArray', 'npt.NDArray']:
    """Generate NACA 4-digit airfoil coordinates (normalized, 0-1 chord)
    
    Physical Process:
        1. Parse NACA digits → (camber, position, thickness)
        2. Generate x-coordinates along chord (with optional cosine clustering at LE/TE)
        3. Calculate thickness distribution perpendicular to chord
        4. Calculate mean camber line and its slope
        5. Compute upper/lower surface coordinates normal to camber line
        
    Args:
        naca: 4-digit NACA designation (e.g., "0012", "2412")
        num_points: Number of points on upper/lower surface each
        cosine_spacing: If True, cluster points at LE/TE for better resolution
        
    Returns:
        (x_coords, y_coords): Airfoil coordinates  [dimensionless, 0-1]
        Shape: (2*num_points,) - ordered from TE lower → LE → TE upper
        
    Example:
        >>> x, y = generate_naca_4digit_coordinates("0012", num_points=50)
        >>> # x, y are normalized [0, 1], ready to scale by chord_length
    """
    import numpy as np_local
    
    m, p, t = _parse_naca_4digit(naca)
    
    # Generate chordwise distribution [dimensionless, 0 to 1]
    if cosine_spacing:
        # Cosine spacing: cluster points at leading/trailing edges
        beta = np_local.linspace(0, np_local.pi, num_points)
        x = 0.5 * (1.0 - np_local.cos(beta))  # Maps [0,π] → [0,1] with clustering
    else:
        # Uniform spacing
        x = np_local.linspace(0, 1, num_points)
    
    # Thickness distribution [dimensionless]
    y_t = _naca_thickness_distribution(x, t)
    
    # Mean camber line [dimensionless]
    y_c, dy_c_dx = _naca_camber_line(x, m, p)
    
    # Angle between camber line and chord [radians]
    theta = np_local.arctan(dy_c_dx)
    
    # Upper and lower surface coordinates (perpendicular to camber line)
    # [dimensionless]
    x_upper = x - y_t * np_local.sin(theta)
    y_upper = y_c + y_t * np_local.cos(theta)
    
    x_lower = x + y_t * np_local.sin(theta)
    y_lower = y_c - y_t * np_local.cos(theta)
    
    # Concatenate: Lower surface (TE→LE) + Upper surface (LE→TE)
    # This creates a closed contour for polygon operations
    x_airfoil = np_local.concatenate([x_lower[::-1], x_upper])
    y_airfoil = np_local.concatenate([y_lower[::-1], y_upper])
    
    return x_airfoil, y_airfoil


def _point_in_polygon_mask(
    X: 'npt.NDArray',
    Y: 'npt.NDArray',
    poly_x: 'npt.NDArray',
    poly_y: 'npt.NDArray'
) -> 'npt.NDArray':
    """Test if grid points are inside polygon using ray casting
    
    Physics Interpretation:
        Cast a horizontal ray from each grid point to infinity.
        Count intersections with polygon edges.
        Odd count = inside, even count = outside.
        
    Args:
        X, Y: Grid coordinates, shape (Nx, Ny)  [lattice units]
        poly_x, poly_y: Polygon vertices (closed contour)  [lattice units]
        
    Returns:
        Boolean mask, shape (Nx, Ny), True = inside polygon
        
    Note:
        This is a vectorized implementation for performance.
        Handles edge cases (point on edge, horizontal edges).
    """
    import numpy as np_local
    
    Nx, Ny = X.shape
    mask = np_local.zeros((Nx, Ny), dtype=bool)
    
    n_vertices = len(poly_x)
    
    # Flatten grid for vectorized processing
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # Ray casting: count crossings for each point
    inside = np_local.zeros(len(X_flat), dtype=bool)
    
    for i in range(n_vertices):
        j = (i + 1) % n_vertices  # Next vertex (wrap around)
        
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        
        # Check if ray crosses edge (yi,yj) at X > point.x
        # Condition: point.y is between yi and yj, and crossing is to the right
        cond1 = (yi > Y_flat) != (yj > Y_flat)
        cond2 = X_flat < (xj - xi) * (Y_flat - yi) / (yj - yi + 1e-10) + xi
        
        # Toggle inside/outside status for each crossing
        inside ^= (cond1 & cond2)
    
    mask = inside.reshape((Nx, Ny))
    
    return mask


def create_airfoil_2d_mask(
    xp: 'ModuleType',
    shape: Tuple[int, int],
    naca: str,
    chord_length: float,
    center: Tuple[float, float],
    angle_of_attack: float = 0.0,
    num_points: int = 100
) -> 'npt.NDArray':
    """Create 2D NACA airfoil solid mask
    
    Physical Process:
        1. Generate normalized NACA coordinates (0-1 chord)
        2. Scale to physical chord length [lattice units]
        3. Rotate by angle of attack [degrees → radians]
        4. Translate to specified center position [lattice units]
        5. Create boolean mask via point-in-polygon test for each grid point
        
    Args:
        xp: Array module (numpy or cupy)
        shape: Domain shape (Nx, Ny)  [lattice units]
        naca: NACA 4-digit designation (e.g., "0012", "2412")
        chord_length: Airfoil chord length  [lattice units]
        center: Airfoil center (x, y), typically at mid-chord  [lattice units]
        angle_of_attack: AoA in degrees (+ = nose up)  [degrees]
        num_points: Number of points for airfoil discretization
        
    Returns:
        Boolean mask, shape (Nx, Ny), True = solid (inside airfoil)
        
    Example:
        >>> mask = create_airfoil_2d_mask(
        ...     np, (500, 200), "2412", chord_length=80, 
        ...     center=(100, 100), angle_of_attack=5.0
        ... )
        >>> # Use with HalfwayBounceBack for no-slip condition
        
    Note:
        - Center is at mid-chord (0.5c) by default
        - Positive AoA rotates nose upward (CCW)
        - Uses numpy for coordinate generation (CPU), then transfers to xp
    """
    import numpy as np_local
    
    # Step 1: Generate normalized airfoil coordinates [dimensionless, 0-1]
    x_norm, y_norm = generate_naca_4digit_coordinates(naca, num_points)
    
    # Step 2: Scale by chord length [lattice units]
    x_airfoil = x_norm * chord_length
    y_airfoil = y_norm * chord_length
    
    # Step 3: Rotate by angle of attack
    if angle_of_attack != 0.0:
        alpha_rad = np_local.deg2rad(angle_of_attack)  # [radians]
        cos_a = np_local.cos(alpha_rad)
        sin_a = np_local.sin(alpha_rad)
        
        # Rotation around mid-chord (0.5, 0)
        x_mid = 0.5 * chord_length
        x_rot = (x_airfoil - x_mid) * cos_a - y_airfoil * sin_a + x_mid
        y_rot = (x_airfoil - x_mid) * sin_a + y_airfoil * cos_a
        
        x_airfoil = x_rot
        y_airfoil = y_rot
    
    # Step 4: Translate to center position [lattice units]
    cx, cy = center
    x_mid = 0.5 * chord_length  # Mid-chord position
    x_airfoil = x_airfoil - x_mid + cx
    y_airfoil = y_airfoil + cy
    
    # Step 5: Create grid for mask [lattice units]
    Nx, Ny = shape
    x = np_local.arange(Nx, dtype=np_local.float64)
    y = np_local.arange(Ny, dtype=np_local.float64)
    X, Y = np_local.meshgrid(x, y, indexing='ij')
    
    # Step 6: Point-in-polygon test for each grid point
    mask_np = _point_in_polygon_mask(X, Y, x_airfoil, y_airfoil)
    
    # Step 7: Transfer to target array module (numpy → cupy if needed)
    if xp.__name__ == 'cupy':
        import cupy as cp
        mask = cp.asarray(mask_np)
    else:
        mask = mask_np
    
    return mask


