"""
Actuator Line Model for Wind Turbine Simulation

This package implements the Actuator Line (AL) model for simulating
wind turbine blades within the LBM framework, following the approach
of Watanabe et al. (Computers & Fluids, 305, 2026).

Coordinate System Support:
    The package now supports arbitrary rotation axes via RotorCoordinateSystem.
    Presets include HAWT (X-axis or Z-axis rotation) and VAWT configurations.

Modules:
    - coordinates: Rotor coordinate system for arbitrary rotation axes
    - blade: Single blade geometry and marker particle distribution
    - rotor: Multi-blade rotor kinematics and rotation
    - airfoil_data: Airfoil polar data (CL, CD vs α, Re) management
    - interpolation: Grid → Marker velocity interpolation (Gaussian)
    - spreading: Marker → Grid force distribution (Eq. 13)
    - actuator_line: Main AL controller orchestrating the full pipeline

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026
    - Sørensen & Shen, J. Fluids Eng. 124, 393-399, 2002
"""

# =============================================================================
# Core modules (always available)
# =============================================================================

# Coordinate system (NEW)
from .coordinates import (
    RotorCoordinateSystem,
    RotorAxisPreset,
    create_coordinate_system,
)

# Blade and rotor geometry
from .blade import Blade, BladeSection
from .rotor import Rotor

__all__ = [
    # Coordinate system
    'RotorCoordinateSystem',
    'RotorAxisPreset',
    'create_coordinate_system',
    # Blade and rotor
    'Blade', 'BladeSection',
    'Rotor',
]

# =============================================================================
# Optional modules (may not be present in all environments)
# =============================================================================

try:
    from .airfoil_data import (
        AirfoilPolarData, AirfoilDatabase,
        make_polar_query, gen_airfoil_polar,
        load_airfoil_from_csv, load_airfoil_sections_from_csv,
        create_flat_plate_polar, create_naca0012_polar,
        create_nrel_s826_database,
    )
    __all__.extend([
        'AirfoilPolarData', 'AirfoilDatabase',
        'make_polar_query', 'gen_airfoil_polar',
        'load_airfoil_from_csv', 'load_airfoil_sections_from_csv',
        'create_flat_plate_polar', 'create_naca0012_polar',
        'create_nrel_s826_database',
    ])
except ImportError:
    pass

try:
    from .interpolation import (
        gaussian_kernel_3d,
        interpolate_velocity_at_marker,
        interpolate_velocity_batch,
        interpolate_velocity_batch_fast,
        compute_interpolation_stencil_info,
    )
    __all__.extend([
        'gaussian_kernel_3d',
        'interpolate_velocity_at_marker',
        'interpolate_velocity_batch',
        'interpolate_velocity_batch_fast',
        'compute_interpolation_stencil_info',
    ])
except ImportError:
    pass

try:
    from .spreading import (
        spread_force_single_marker,
        spread_forces_to_grid,
        spread_forces_uniform_epsilon,
        check_force_conservation,
    )
    __all__.extend([
        'spread_force_single_marker',
        'spread_forces_to_grid',
        'spread_forces_uniform_epsilon',
        'check_force_conservation',
    ])
except ImportError:
    pass

try:
    from .actuator_line import (
        ActuatorLineModel,
        BEMResult,
        create_actuator_line_from_config,
    )
    __all__.extend([
        'ActuatorLineModel',
        'BEMResult',
        'create_actuator_line_from_config',
    ])
except ImportError:
    pass