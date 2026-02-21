"""
Actuator Line Model for Wind Turbine Simulation

This package implements the Actuator Line (AL) model for simulating
wind turbine blades within the LBM framework, following the approach
of Watanabe et al. (Computers & Fluids, 305, 2026).

Data Flow per Timestep:
    Rotor.advance(dt)
        → get_all_marker_positions()         [rotor.py]
        → interpolate_velocity_batch()       [interpolation.py]
        → compute_relative_velocity()        [rotor.py]
        → CL/CD lookup                       [airfoil_data.py]
        → F_L, F_D → F_n, F_θ (Eq. 9-12)    [actuator_line.py]
        → project_all_forces()               [rotor.py]
        → spread_forces_to_grid() (Eq. 13)   [spreading.py]
        → F(x) enters Guo forcing            [guo_forcing.py]

Coordinate System Support (NEW):
    The package now supports arbitrary rotation axes via RotorCoordinateSystem.
    Presets include HAWT (X-axis or Z-axis rotation) and VAWT configurations.

Modules:
    - coordinates: Rotor coordinate system for arbitrary rotation axes (NEW)
    - airfoil_data: Airfoil polar data (CL, CD vs α, Re) management
    - blade: Single blade geometry and marker particle distribution
    - rotor: Multi-blade rotor kinematics and rotation
    - interpolation: Grid → Marker velocity interpolation (Gaussian)
    - spreading: Marker → Grid force distribution (Eq. 13)
    - actuator_line: Main AL controller orchestrating the full pipeline

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026
    - Sørensen & Shen, J. Fluids Eng. 124, 393-399, 2002
"""

# =============================================================================
# Coordinate System (NEW - required by blade.py and rotor.py)
# =============================================================================
from .coordinates import (
    RotorCoordinateSystem,
    create_coordinate_system,
)

# =============================================================================
# Airfoil Data
# =============================================================================
from .airfoil_data import (
    AirfoilPolarData, AirfoilDatabase,
    make_polar_query, gen_airfoil_polar,
    load_airfoil_from_csv, load_airfoil_sections_from_csv,
    create_flat_plate_polar, create_naca0012_polar,
    create_nrel_s826_database,
    # NEW: Config-based polar factory
    gen_airfoil_polar_extended,
    MultiAirfoilPolarManager,
    create_polar_from_config,
)

# =============================================================================
# Blade and Rotor Geometry
# =============================================================================
from .blade import Blade, BladeSection
from .rotor import Rotor

# =============================================================================
# Velocity Interpolation
# =============================================================================
from .interpolation import (
    gaussian_kernel_3d,
    interpolate_velocity_at_marker,
    interpolate_velocity_batch,
    interpolate_velocity_batch_fast,
    compute_interpolation_stencil_info,
)

# =============================================================================
# Force Spreading
# =============================================================================
from .spreading import (
    spread_force_single_marker,
    spread_forces_to_grid,
    spread_forces_uniform_epsilon,
    check_force_conservation,
)

# =============================================================================
# Main Controller
# =============================================================================
from .actuator_line import (
    ActuatorLineModel,
    MultiRotorManager,
    BEMResult,
    create_actuator_line_from_config,
    create_multi_rotor_from_config,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Coordinate system
    'RotorCoordinateSystem',
    'create_coordinate_system',
    # Airfoil data
    'AirfoilPolarData', 'AirfoilDatabase',
    'make_polar_query', 'gen_airfoil_polar',
    'load_airfoil_from_csv', 'load_airfoil_sections_from_csv',
    'create_flat_plate_polar', 'create_naca0012_polar',
    'create_nrel_s826_database',
    # NEW: Config-based polar factory
    'gen_airfoil_polar_extended',
    'MultiAirfoilPolarManager',
    'create_polar_from_config',
    # Blade and rotor
    'Blade', 'BladeSection',
    'Rotor',
    # Interpolation
    'gaussian_kernel_3d',
    'interpolate_velocity_at_marker',
    'interpolate_velocity_batch',
    'interpolate_velocity_batch_fast',
    'compute_interpolation_stencil_info',
    # Spreading
    'spread_force_single_marker',
    'spread_forces_to_grid',
    'spread_forces_uniform_epsilon',
    'check_force_conservation',
    # Main controller
    'ActuatorLineModel',
    'BEMResult',
    'create_actuator_line_from_config',
]