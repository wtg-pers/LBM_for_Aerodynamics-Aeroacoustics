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
        → CL/CD lookup                      [airfoil_data.py]
        → F_L, F_D → F_n, F_θ (Eq. 9-12)  [actuator_line.py]
        → project_all_forces()              [rotor.py]
        → spread_forces_to_grid() (Eq. 13)  [spreading.py]
        → F(x) enters Guo forcing           [guo_forcing.py]

Modules:
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

from .airfoil_data import (
    AirfoilPolarData, AirfoilDatabase,
    make_polar_query, gen_airfoil_polar,
    load_airfoil_from_csv, load_airfoil_sections_from_csv,
    create_flat_plate_polar, create_naca0012_polar,
    create_nrel_s826_database,
)
from .blade import Blade, BladeSection
from .rotor import Rotor
from .interpolation import (
    gaussian_kernel_3d,
    interpolate_velocity_at_marker,
    interpolate_velocity_batch,
    interpolate_velocity_batch_fast,
    compute_interpolation_stencil_info,
)
from .spreading import (
    spread_force_single_marker,
    spread_forces_to_grid,
    spread_forces_uniform_epsilon,
    check_force_conservation,
)
from .actuator_line import (
    ActuatorLineModel,
    BEMResult,
    create_actuator_line_from_config,
)

__all__ = [
    'AirfoilPolarData', 'AirfoilDatabase',
    'make_polar_query', 'gen_airfoil_polar',
    'load_airfoil_from_csv', 'load_airfoil_sections_from_csv',
    'create_flat_plate_polar', 'create_naca0012_polar',
    'create_nrel_s826_database',
    'Blade', 'BladeSection',
    'Rotor',
    'gaussian_kernel_3d',
    'interpolate_velocity_at_marker',
    'interpolate_velocity_batch',
    'interpolate_velocity_batch_fast',
    'compute_interpolation_stencil_info',
    'spread_force_single_marker',
    'spread_forces_to_grid',
    'spread_forces_uniform_epsilon',
    'check_force_conservation',
    'ActuatorLineModel',
    'BEMResult',
    'create_actuator_line_from_config',
]