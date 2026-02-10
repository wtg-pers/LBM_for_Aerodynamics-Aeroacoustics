"""
Actuator Line Model for Wind Turbine Simulation

This package implements the Actuator Line (AL) model for simulating
wind turbine blades within the LBM framework, following the approach
of Watanabe et al. (Computers & Fluids, 305, 2026).

Physical Concept:
=================
The AL model replaces physical blade geometry with a line of discrete
marker particles. At each marker, aerodynamic forces (lift, drag) are
computed using Blade Element Momentum (BEM) theory and local flow
conditions, then spread onto the LBM grid via Gaussian filtering.

Modules:
    - airfoil_data: Airfoil polar data (CL, CD vs α, Re) management
    - blade: Single blade geometry and marker particle distribution
    - rotor: Multi-blade rotor kinematics and rotation
    - actuator_line: Main AL controller (velocity sampling, BEM, spreading)
    - interpolation: Grid → Marker velocity interpolation
    - spreading: Marker → Grid force distribution (Gaussian filter)

References:
    - Watanabe et al., Comp. & Fluids 305, 106901, 2026
    - Sørensen & Shen, J. Fluids Eng. 124, 393-399, 2002

Author: LBM Development Team
Date: 2026-02
"""

from .airfoil_data import (
    AirfoilPolarData, AirfoilDatabase,
    make_polar_query, gen_airfoil_polar,
    load_airfoil_from_csv, load_airfoil_sections_from_csv,
    create_flat_plate_polar, create_naca0012_polar,
    create_nrel_s826_database,
)

__all__ = [
    'AirfoilPolarData', 'AirfoilDatabase',
    'make_polar_query', 'gen_airfoil_polar',
    'load_airfoil_from_csv', 'load_airfoil_sections_from_csv',
    'create_flat_plate_polar', 'create_naca0012_polar',
    'create_nrel_s826_database',
    # Blade geometry
    'Blade', 'BladeSection',
]