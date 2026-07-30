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
    create_airfoil_2d_mask_from_coords,
    load_selig_dat,
    transform_airfoil_coords_to_lu,
    generate_naca_4digit_coordinates,
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
        # Airfoil: NACA 4-digit (`naca` key) or arbitrary coords
        # from a Selig .dat file (`selig_file` key) / inline coords
        # (`x_coords`, `y_coords` keys).
        chord = config.get('chord', char_length)
        aoa = config.get('angle_of_attack', 0.0)
        center = config.get('center', (Nx//5, Ny//2))

        selig_file = config.get('selig_file')
        x_coords = config.get('x_coords')
        y_coords = config.get('y_coords')
        naca = config.get('naca')

        if selig_file is not None or (x_coords is not None and y_coords is not None):
            # Arbitrary airfoil geometry from coordinates
            if selig_file is not None:
                x_norm, y_norm = load_selig_dat(selig_file)
                source = f"selig:{selig_file}"
            else:
                import numpy as np_local
                x_norm = np_local.asarray(x_coords, dtype=np_local.float64)
                y_norm = np_local.asarray(y_coords, dtype=np_local.float64)
                source = f"inline({len(x_norm)} pts)"

            mask = create_airfoil_2d_mask_from_coords(
                xp, domain_shape,
                x_norm=x_norm,
                y_norm=y_norm,
                chord_length=chord,
                center=center,
                angle_of_attack=aoa,
            )

            x_poly_lu, y_poly_lu = transform_airfoil_coords_to_lu(
                x_norm=x_norm, y_norm=y_norm,
                chord_length=chord, center=center,
                angle_of_attack=aoa,
            )

            info = {
                'type': 'airfoil',
                'source': source,
                'num_points': int(len(x_norm)),
                'chord': chord,
                'angle_of_attack': aoa,
                'center': center,
                'solid_nodes': int(xp.sum(mask)),
                'polygon_lu': (x_poly_lu, y_poly_lu),
            }

            if verbose:
                print(f"  Internal Obstacle (Airfoil, 2D, arbitrary coords):")
                print(f"    source={source}, pts={info['num_points']}")
                print(f"    chord={chord} [lu], center={center}, AoA={aoa}°")
                print(f"    {info['solid_nodes']} solid nodes")
        else:
            # NACA 4-digit fallback
            naca = naca or '0012'
            num_points = config.get('num_points', 150)

            mask = create_airfoil_2d_mask(
                xp, domain_shape,
                naca=naca,
                chord_length=chord,
                center=center,
                angle_of_attack=aoa,
                num_points=num_points,
            )

            # Regenerate + transform polygon for IBB q-fraction use.
            _xn, _yn = generate_naca_4digit_coordinates(naca, num_points)
            x_poly_lu, y_poly_lu = transform_airfoil_coords_to_lu(
                x_norm=_xn, y_norm=_yn,
                chord_length=chord, center=center,
                angle_of_attack=aoa,
            )

            info = {
                'type': 'airfoil',
                'naca': naca,
                'chord': chord,
                'angle_of_attack': aoa,
                'center': center,
                'num_points': num_points,
                'solid_nodes': int(xp.sum(mask)),
                'polygon_lu': (x_poly_lu, y_poly_lu),
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
 
        # For cylinders, check against cross-section dimensions only
        # (exclude the extrusion axis from the size check)
        if geom_type == 'cylinder' and lattice_dim == 3:
            axis = config.get('axis', 'z').lower()
            # domain_shape = (Nx, Ny, Nz)
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            axis_idx = axis_map.get(axis, 2)
            cross_dims = [s for i, s in enumerate(domain_shape)
                          if i != axis_idx]  # [lu]
            check_size = min(cross_dims)
        else:
            check_size = min(domain_shape)
 
        if radius > check_size / 2:
            return False, (
                f"{geom_type}: radius {radius} too large "
                f"(max {check_size // 2} for cross-section)"
            )
    
    elif geom_type == 'airfoil':
        if lattice_dim != 2:
            return False, "Airfoil only supported in 2D (use dimension=2)"
        
        chord = config.get('chord', 0)
        if chord <= 0:
            return False, "Airfoil: chord must be positive"
        
        if chord > min(domain_shape) * 0.8:
            return False, "Airfoil: chord too large (should be < 80% of domain)"
        
        # Accept either a Selig .dat file, inline (x,y) coords, or NACA 4-digit.
        selig_file = config.get('selig_file')
        x_coords = config.get('x_coords')
        y_coords = config.get('y_coords')
        naca = config.get('naca')

        if selig_file is not None:
            import os as _os
            if not _os.path.exists(selig_file):
                return False, f"Airfoil: selig_file not found: {selig_file}"
        elif x_coords is not None and y_coords is not None:
            if len(x_coords) != len(y_coords) or len(x_coords) < 3:
                return False, "Airfoil: x_coords and y_coords must match length (≥ 3 points)"
        elif naca is not None:
            if len(naca) != 4:
                return False, f"Airfoil: NACA must be 4 digits, got '{naca}'"
        else:
            return False, ("Airfoil: must specify 'selig_file', ('x_coords', 'y_coords'), "
                           "or 'naca' (4-digit)")
    
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


# =============================================================================
# Fine-Level Geometry Coordinate Transformation (for MLG)
# =============================================================================

def create_fine_level_geometry_config(
    geometry_config: Dict[str, Any],
    fine_origin_phys: Tuple[float, float, float],
    fine_shape: Tuple[int, int, int],
    dx_fine: float,
    dx_coarse: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Transform L0 obstacle config to fine-level local coordinates.

    Physical Meaning:
        An obstacle defined in Level 0 physical coordinates must be
        re-expressed in the fine grid's local coordinate system before
        a mask can be generated at fine resolution. This involves two
        geometric operations:

        1. Origin shift:  subtract fine domain's physical start point
        2. Scale change:  divide by fine grid spacing (dx_k = dx_0 / 2^k)

    Coordinate Mapping (for sphere center as example):
        cx_local = (cx_L0 - ox) / dx_fine
        cy_local = (cy_L0 - oy) / dx_fine
        cz_local = (cz_L0 - oz) / dx_fine
        radius_local = radius_L0 / dx_fine

    At fine resolution the obstacle has MORE grid points:
        D_L0 = 2R/dx_0,  D_fine = 2R/dx_fine = D_L0 × (dx_0/dx_fine)

    Args:
        geometry_config: Original internal_geometry config dict (L0 coords).
            Example: {'sphere': {'enabled': True, 'center': (50,25,25),
                      'radius': 10}}
        fine_origin_phys: Physical origin (ox, oy, oz) of the fine domain
            in L0 lattice units. This is the overlap-EXPANDED start point
            (fine_domain_coarse.x_start, etc.), NOT the user-specified
            region start.  [L0 lattice units]
        fine_shape: Fine grid dimensions (Nx_f, Ny_f, Nz_f).
            [fine lattice units]
        dx_fine: Fine grid spacing in L0 lattice units (= dx_0 / 2^k).
            For Level 1: dx_fine = 0.5, Level 2: dx_fine = 0.25, etc.
        dx_coarse: Coarse (L0) grid spacing (= 1.0 for level 0).
        verbose: Print transformation details.

    Returns:
        New geometry config dict with coordinates in fine local units,
        or empty dict if obstacle doesn't intersect fine domain.

    Example:
        >>> # L0 sphere at (50, 25, 25), R=10
        >>> # Fine L1: origin=(28, 8, 8), dx=0.5
        >>> cfg = create_fine_level_geometry_config(
        ...     {'sphere': {'enabled': True, 'center': (50,25,25), 'radius': 10}},
        ...     fine_origin_phys=(28.0, 8.0, 8.0),
        ...     fine_shape=(108, 88, 88),
        ...     dx_fine=0.5,
        ... )
        >>> cfg['sphere']['center']   # (44.0, 34.0, 34.0)
        >>> cfg['sphere']['radius']   # 20.0
    """
    # ── Find active geometry type ──────────────────────────────
    geometry_types_3d = ['cylinder', 'sphere', 'box']
    active_type = None
    active_config = None

    for gtype in geometry_types_3d:
        if gtype in geometry_config:
            gcfg = geometry_config[gtype]
            if gcfg.get('enabled', False):
                active_type = gtype
                active_config = dict(gcfg)  # shallow copy
                break

    if active_type is None:
        return {}  # No obstacle configured

    ox, oy, oz = fine_origin_phys
    Nx_f, Ny_f, Nz_f = fine_shape
    new_config: Dict[str, Any] = {}

    # ── Sphere ─────────────────────────────────────────────────
    if active_type == 'sphere':
        cx, cy, cz = active_config['center']
        radius = active_config['radius']

        # Transform: L0 physical → fine local
        cx_local = (cx - ox) / dx_fine
        cy_local = (cy - oy) / dx_fine
        cz_local = (cz - oz) / dx_fine
        r_local = radius / dx_fine

        # Intersection test: sphere bounding box vs fine domain
        if (cx_local + r_local < 0 or cx_local - r_local > Nx_f - 1 or
            cy_local + r_local < 0 or cy_local - r_local > Ny_f - 1 or
            cz_local + r_local < 0 or cz_local - r_local > Nz_f - 1):
            if verbose:
                print(f"    Sphere does not intersect fine domain — skipped")
            return {}

        new_config = {
            'sphere': {
                'enabled': True,
                'center': (cx_local, cy_local, cz_local),
                'radius': r_local,
            }
        }
        if verbose:
            print(f"    Fine obstacle (sphere): center=({cx_local:.1f}, "
                  f"{cy_local:.1f}, {cz_local:.1f}), R={r_local:.1f} "
                  f"[fine lu]")
            print(f"      (L0 original: center=({cx}, {cy}, {cz}), "
                  f"R={radius} [L0 lu])")

    # ── Cylinder ───────────────────────────────────────────────
    elif active_type == 'cylinder':
        center_2d = active_config['center']  # cross-section center
        radius = active_config['radius']
        axis = active_config.get('axis', 'z')

        # The 2D center is in the plane perpendicular to the axis.
        # axis='z' → center=(cx, cy), axis='y' → center=(cx, cz),
        # axis='x' → center=(cy, cz)
        if axis == 'z':
            c0_local = (center_2d[0] - ox) / dx_fine
            c1_local = (center_2d[1] - oy) / dx_fine
        elif axis == 'y':
            c0_local = (center_2d[0] - ox) / dx_fine
            c1_local = (center_2d[1] - oz) / dx_fine
        elif axis == 'x':
            c0_local = (center_2d[0] - oy) / dx_fine
            c1_local = (center_2d[1] - oz) / dx_fine
        else:
            raise ValueError(f"Unknown cylinder axis: '{axis}'")

        r_local = radius / dx_fine

        new_config = {
            'cylinder': {
                'enabled': True,
                'center': (c0_local, c1_local),
                'radius': r_local,
                'axis': axis,
            }
        }

        # Transform axis_range if specified
        axis_range = active_config.get('axis_range', None)
        if axis_range is not None:
            axis_origin = {'x': ox, 'y': oy, 'z': oz}[axis]
            new_config['cylinder']['axis_range'] = (
                (axis_range[0] - axis_origin) / dx_fine,
                (axis_range[1] - axis_origin) / dx_fine,
            )

        # Transform height/axis_center if specified. This block was lost in
        # a duplicated definition of this function (the shadowed copy had it,
        # the surviving one did not) — a partial cylinder silently became an
        # INFINITE cylinder on every fine level. Manual 06 §6.9-②; gate:
        # patch_notes/entry_unification/gates/mlg_partial_cylinder_gate.py
        if 'height' in active_config:
            new_config['cylinder']['height'] = (
                active_config['height'] / dx_fine)
            if 'axis_center' in active_config:
                axis_origin = {'x': ox, 'y': oy, 'z': oz}[axis]
                new_config['cylinder']['axis_center'] = (
                    (active_config['axis_center'] - axis_origin) / dx_fine)
            else:
                # Without an explicit axis_center the mask builder defaults
                # to the DOMAIN axial midpoint — a different physical
                # location on L0 vs a fine level. Refuse instead of
                # silently moving the body.
                raise ValueError(
                    "cylinder 'height' without 'axis_center' is ambiguous "
                    "across MLG levels — specify axis_center explicitly "
                    "in internal_geometry.cylinder")

        if verbose:
            print(f"    Fine obstacle (cylinder): center=({c0_local:.1f}, "
                  f"{c1_local:.1f}), R={r_local:.1f}, axis={axis} [fine lu]")

    # ── Box ────────────────────────────────────────────────────
    elif active_type == 'box':
        cmin = active_config['corner_min']
        cmax = active_config['corner_max']

        new_min = (
            (cmin[0] - ox) / dx_fine,
            (cmin[1] - oy) / dx_fine,
            (cmin[2] - oz) / dx_fine,
        )
        new_max = (
            (cmax[0] - ox) / dx_fine,
            (cmax[1] - oy) / dx_fine,
            (cmax[2] - oz) / dx_fine,
        )

        # Intersection test
        if (new_max[0] < 0 or new_min[0] > Nx_f - 1 or
            new_max[1] < 0 or new_min[1] > Ny_f - 1 or
            new_max[2] < 0 or new_min[2] > Nz_f - 1):
            if verbose:
                print(f"    Box does not intersect fine domain — skipped")
            return {}

        new_config = {
            'box': {
                'enabled': True,
                'corner_min': new_min,
                'corner_max': new_max,
            }
        }
        if verbose:
            print(f"    Fine obstacle (box): min=({new_min[0]:.1f}, "
                  f"{new_min[1]:.1f}, {new_min[2]:.1f}), "
                  f"max=({new_max[0]:.1f}, {new_max[1]:.1f}, "
                  f"{new_max[2]:.1f}) [fine lu]")

    return new_config

# =============================================================================
# 2D Fine-Level Geometry Coordinate Transformation (for 2D MLG)
# =============================================================================

def create_fine_level_geometry_config_2d(
    geometry_config: Dict[str, Any],
    fine_origin_phys: Tuple[float, float],
    fine_shape: Tuple[int, int],
    dx_fine: float,
    dx_coarse: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Transform L0 obstacle config to fine-level local coordinates (2D).

    2D variant of create_fine_level_geometry_config. Supports:
        - 'circle' / 'cylinder' (2D interpretation)
        - 'box'
        - 'airfoil' (NACA 4-digit, inline coords, or selig .dat)

    Coordinate Mapping:
        cx_local    = (cx_L0 - ox) / dx_fine
        cy_local    = (cy_L0 - oy) / dx_fine
        radius_local = radius_L0 / dx_fine
        chord_local  = chord_L0 / dx_fine   (airfoil)

    Args:
        geometry_config: internal_geometry dict (2D coords).
        fine_origin_phys: (ox, oy) fine domain origin in L0 lattice units.
        fine_shape: (Nx_f, Ny_f) fine grid dimensions.
        dx_fine: Fine grid spacing in L0 units (= 1 / 2^k).
        dx_coarse: L0 spacing (= 1.0).
        verbose: Print transformation details.

    Returns:
        New geometry config dict with fine local coords, or empty dict
        if obstacle doesn't intersect fine domain / no obstacle configured.
    """
    # Find active 2D geometry type
    geometry_types_2d = ['circle', 'cylinder', 'box', 'airfoil']
    active_type = None
    active_config = None

    for gtype in geometry_types_2d:
        if gtype in geometry_config:
            gcfg = geometry_config[gtype]
            if gcfg.get('enabled', False):
                active_type = gtype
                active_config = dict(gcfg)
                break

    if active_type is None:
        return {}

    ox, oy = fine_origin_phys
    Nx_f, Ny_f = fine_shape

    # ── Circle / Cylinder (2D) ──────────────────────────────────
    if active_type in ('circle', 'cylinder'):
        cx, cy = active_config['center']
        radius = active_config['radius']

        cx_local = (cx - ox) / dx_fine
        cy_local = (cy - oy) / dx_fine
        r_local = radius / dx_fine

        # Intersection test
        if (cx_local + r_local < 0 or cx_local - r_local > Nx_f - 1 or
            cy_local + r_local < 0 or cy_local - r_local > Ny_f - 1):
            if verbose:
                print(f"    {active_type} does not intersect fine domain — skipped")
            return {}

        new_config = {
            active_type: {
                'enabled': True,
                'center': (cx_local, cy_local),
                'radius': r_local,
            }
        }
        # Carry the wall-BC selector so each MLG fine level honors HWBB / IBB
        # the user requested at L0.
        if 'wall_bc' in active_config:
            new_config[active_type]['wall_bc'] = active_config['wall_bc']
        if verbose:
            print(f"    Fine obstacle ({active_type}): "
                  f"center=({cx_local:.1f}, {cy_local:.1f}), "
                  f"R={r_local:.1f} [fine lu]")
        return new_config

    # ── Box (2D) ────────────────────────────────────────────────
    elif active_type == 'box':
        cmin = active_config['corner_min']
        cmax = active_config['corner_max']
        new_min = ((cmin[0] - ox) / dx_fine, (cmin[1] - oy) / dx_fine)
        new_max = ((cmax[0] - ox) / dx_fine, (cmax[1] - oy) / dx_fine)

        if (new_max[0] < 0 or new_min[0] > Nx_f - 1 or
            new_max[1] < 0 or new_min[1] > Ny_f - 1):
            if verbose:
                print(f"    Box does not intersect fine domain — skipped")
            return {}

        new_config = {
            'box': {
                'enabled': True,
                'corner_min': new_min,
                'corner_max': new_max,
            }
        }
        if verbose:
            print(f"    Fine obstacle (box 2D): min={new_min}, max={new_max}")
        return new_config

    # ── Airfoil (2D) ────────────────────────────────────────────
    elif active_type == 'airfoil':
        # Airfoil center + chord are in L0 lattice units. Rescale.
        cx, cy = active_config['center']
        chord = active_config['chord']
        aoa = active_config.get('angle_of_attack', 0.0)

        cx_local = (cx - ox) / dx_fine
        cy_local = (cy - oy) / dx_fine
        chord_local = chord / dx_fine

        # Bounding box check: airfoil is within ±chord/2 of center,
        # plus small thickness margin (~10% chord).
        half = 0.55 * chord_local
        if (cx_local + half < 0 or cx_local - half > Nx_f - 1 or
            cy_local + half < 0 or cy_local - half > Ny_f - 1):
            if verbose:
                print(f"    Airfoil does not intersect fine domain — skipped")
            return {}

        new_config = {
            'airfoil': {
                'enabled': True,
                'center': (cx_local, cy_local),
                'chord': chord_local,
                'angle_of_attack': aoa,
            }
        }
        # Preserve geometry-source keys (selig_file / naca / inline coords)
        # plus the wall-BC selector so each MLG tier picks up HWBB / IBB
        # the user requested at L0.
        for key in (
            'selig_file', 'naca', 'x_coords', 'y_coords', 'num_points',
            'wall_bc',
        ):
            if key in active_config:
                new_config['airfoil'][key] = active_config[key]

        if verbose:
            print(f"    Fine obstacle (airfoil 2D): "
                  f"center=({cx_local:.1f}, {cy_local:.1f}), "
                  f"chord={chord_local:.1f} [fine lu], AoA={aoa}°")
        return new_config

    return {}
