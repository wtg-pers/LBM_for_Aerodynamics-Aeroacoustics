"""
User parameter settings
"""
Nx = 500
Ny = 80
Nz = 80

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
        "Re": 20.0,
        "u_init": 0.1,
        "characteristic_length": 30
    },
    "time": {
        "max_steps": 50000,
        "output_interval": 100,
        "checkpoint_interval": 10000
    }
}

boundaries = {
    # Inlet at x = 0
    "inlet": {
        "type": "inlet",
        "location": "xmin",
        "velocity": 0.1,
        "profile": "uniform"
    },
    
    # Outlet at x = Nx-1
    "outlet": {
        "type": "outlet",
        "location": "xmax",
        "rho": 1.0,
        "k": 0.1,
        "method": "characteristic"
    },
    
    # Side walls (y-direction)
    "wall_ymin": {
        "type": "wall",
        "location": "ymin",
        "method": "bounce_back",
        "exclude_inlet_outlet": True
    },
    "wall_ymax": {
        "type": "wall",
        "location": "ymax",
        "method": "bounce_back",
        "exclude_inlet_outlet": True
    },
    
    # Top/bottom walls z-direction)
    "wall_zmin": {
        "type": "wall",
        "location": "zmin",
        "method": "bounce_back",
        "exclude_inlet_outlet": True
    },
    "wall_zmax": {
        "type": "wall",
        "location": "zmax",
        "method": "bounce_back",
        "exclude_inlet_outlet": True
    },
}

# =============================================================================
# Internal Geometry (Obstacles)
# =============================================================================
internal_geometry = {
    "cylinder": {
        "enabled": True,
        "type": "cylinder",
        "center": (Nx // 5, Ny // 2),   # (x, y) center
        "radius": 10,                    # characteristic_length // 2
        "axis": "z",
        "axis_range": (0, Nz - 1)       # Full span in z
    }
}

# =============================================================================
# Output Configuration
# =============================================================================
output = {
    # Directory settings
    "output_dir": "./results/vtk",
    "checkpoint_dir": "./checkpoints",
    "csv_dir": "./results_test/csv",
    
    "clear_previous": True,
    
    "vtk": {
        "enabled": True,
        "precision": "float32",
        "compression_level": 6,
        "variables": ["density", "pressure", "velocity", "velocity_magnitude", "solid_mask"]
    },
    "checkpoint": {
        "enabled": True,
        "keep_last_n": 3
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
    "output": output,
    "force_calculation": force_calculation
}