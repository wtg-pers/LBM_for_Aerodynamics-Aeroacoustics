# =============================================================================
# Domain Configuration
# =============================================================================
Nx = 200
Ny = 200
Nz = 200
# Cylinder parameters
cylinder_diameter = 20      # [lattice units]
cylinder_center_x = Nx // 5  # Located at 20% from inlet
cylinder_center_y = Ny // 2  # Centered in y


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
        "characteristic_length": cylinder_diameter
    },
    "time": {
        "max_steps": 50000,
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
# For TRUE EXTERNAL FLOW: Use 'characteristic' on all non-inlet boundaries!

boundaries = {
    # Inlet at x = 0
    "inlet": {
        "location": "xmin",
        "velocity": 0.1,
        "method": "non_equilibrium",
        "profile": "uniform"
    },
    
    # Outlet at x = Nx-1
    "outlet": {
        "location": "xmax",
        "method": "characteristic",
        "rho": 1.0,
        "k": 0.1
    },
    
    "ymin": {
        "location": "ymin",
        "method": "characteristic",
        "rho": 1.0,
        "k": 0.1,
    },
    "ymax": {
        "location": "ymax",
        "method": "characteristic",
        "rho": 1.0,
        "k": 0.1,
    },
    
    "zmin": {
        "location": "zmin",
        "method": "characteristic",
        "rho": 1.0,
        "k": 0.1,
    },
    "zmax": {
        "location": "zmax",
        "method": "characteristic",
        "rho": 1.0,
        "k": 0.1,
    },
}

# =============================================================================
# Alternative: Periodic z-direction (quasi-2D simulation)
# =============================================================================
# For faster quasi-2D simulations, comment out farfield_bottom/top above
# and z-direction will default to periodic.

# =============================================================================
# Internal Geometry (Obstacle)
# =============================================================================

internal_geometry = {
    "cylinder": {
        "enabled": True,
        "center": (cylinder_center_x, cylinder_center_y),
        "radius": cylinder_diameter // 2,
        "axis": "z",                  # Cylinder axis direction
        "axis_range": (0, Nz - 1)     # Full span in z
    }
}

# =============================================================================
# Output Configuration
# =============================================================================
output = {
    "output_dir": "./results_external/vtk",
    "checkpoint_dir": "./checkpoints_external",
    "csv_dir": "./results_external/csv",
    
    "clear_previous": False,          # Set True for fresh start
    
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
            (cylinder_center_x + 2 * cylinder_diameter, cylinder_center_y, Nz // 2),
            (cylinder_center_x + 5 * cylinder_diameter, cylinder_center_y, Nz // 2)
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