# =============================================================================
# Domain Configuration
# =============================================================================
Nx = 200
Ny = 50
Nz = 50
# Cylinder parameters
diameter = 20      # [lattice units]
center_x = Nx // 4  # Located at 25% from inlet
center_y = Ny // 2  # Centered in y
center_z = Nz // 2


# =============================================================================
# Simulation Parameters
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "dimension": 3,
    "lattice_model": "D3Q27",
    "domain": {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz
    },
    "physics": {
        "Re": 100.0,
        "u_init": 0.1,
        "characteristic_length": diameter
    },
    "time": {
        "max_steps": 200000,
        "output_interval": 500,
        "checkpoint_interval": 10000,
        "probe_interval": 10                  # Force/probe sampling
    }
}


# =============================================================================
# Boundary Conditions (Method-Based Configuration)
# =============================================================================
# Available methods:
#   - 'non_equilibrium': Inlet with velocity specification (preserves f_neq)
#   - 'equilibrium': Simple inlet (may cause mass drift)
#   - 'characteristic': Non-reflecting open BC (outlet/far-field)
#   - 'convective': Advective outlet
#   - 'extrapolation': Zero-gradient outlet
#   - 'bounce_back': Half-way bounce-back wall
#   - 'periodic': No BC (handled by streaming)
#

boundaries = {
    "inlet": {"location": "xmin", "method": "non_equilibrium", "velocity": 0.1},
    "outlet": {"location": "xmax", "method": "characteristic", "rho": 1.0},
    "ymin": {"location": "ymin", "method": "ambient", "rho": 1.0, "k": 0.3},
    "ymax": {"location": "ymax", "method": "ambient", "rho": 1.0, "k": 0.3},
    "zmin": {"location": "zmin", "method": "ambient", "rho": 1.0, "k": 0.3},
    "zmax": {"location": "zmax", "method": "ambient", "rho": 1.0, "k": 0.3},
}
# boundaries = {
#     "top": {"location": "zmax", "method": "ambient", "rho": 1.0, "k": 0.5},
#     "bottom": {"location": "zmin", "method": "ambient", "rho": 1.0, "k": 0.5},
#     "sides": {"location": "xmin", "method": "ambient", "rho": 1.0, "k": 0.5},
#     # ... 나머지 경계도 동일
# }
# =============================================================================
# Alternative: Periodic z-direction (quasi-2D simulation)
# =============================================================================
# For faster quasi-2D simulations, comment out farfield_bottom/top above
# and z-direction will default to periodic.

# =============================================================================
# Internal Geometry (Obstacle)
# =============================================================================

# internal_geometry = {
#     "cylinder": {
#         "enabled": True,
#         "center": (cylinder_center_x, cylinder_center_y),
#         "radius": cylinder_diameter // 2,
#         "axis": "z",                  # Cylinder axis direction
#         "axis_range": (30, 70)     # Full span in z
#     }
# }

internal_geometry = {
    "sphere": {
        "enabled": True,
        "center": (center_x, center_y, center_z),
        "radius": diameter // 2
    }
}

# =============================================================================
# Output Configuration
# =============================================================================
output = {
    "output_dir": "./results/vtk",
    "checkpoint_dir": "./checkpoints",
    "csv_dir": "./results/csv",
    
    "clear_previous": True,          # Set True for fresh start
    
    "vtk": {
        "enabled": True,
        "precision": "float32",
        "compression_level": 6,
        "variables": ["density", "pressure", "velocity", "velocity_magnitude", "solid_mask"]
    },
    "checkpoint": {
        "enabled": True,
        "keep_last_n": 3
    },
    "probes": {
        "enabled": True,
        "force_on_obstacle": True,    # Compute drag/lift
        "wake_probes": [              # Velocity probes for Strouhal
            (center_x + 2 * diameter, center_y, center_z),
            (center_x + 5 * diameter, center_y, center_z)
        ]
    }
}


# =============================================================================
# Force Calculation (for future use)
# =============================================================================
force_calculation = {
    "enabled": False,
    "interval": 1,
    "start_step": 0,
    "reference": {
        "rho": 1.0,
        "velocity": 0.1,
        "area": 30  # characteristic_length
    }
}

# =============================================================================
# Final Config Dictionary
# =============================================================================
config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "output": output
}