# =============================================================================
# Domain Configuration
# =============================================================================
# Cylinder parameters
diameter = 20      # [lattice units]
Nx = diameter * 15
Ny = diameter * 15
Nz = 3
size = Nx*Ny*Nz / 10000
center_x = Nx // 5  # Located at 25% from inlet
center_y = Ny // 2  # Centered in y
center_z = Nz // 2

MACH_INLET = 0.1
RHO = 1.0
RE_TGT = 100.0


# =============================================================================
# Simulation Parameters
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "device_id": 0,
    "dimension": 3,
    "lattice_model": "D3Q27",
    "domain": {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz
    },
    "physics": {
        "Re": RE_TGT,
        "u_init": MACH_INLET,
        "characteristic_length": diameter
    },
    "time": {
        "max_steps": 50000,
        "output_interval": 100,
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
    "inlet": {"location": "xmin", 
              "method": "non_equilibrium", 
              "velocity": MACH_INLET},
    "outlet": {"location": "xmax", 
               "method": "pressure_relaxation", 
               "rho": RHO, "k":0.1},
    "ymin": {"location": "ymin", 
               "method": "pressure_relaxation", 
               "rho": RHO, "k":0.1},
    "ymax": {"location": "ymax", 
               "method": "pressure_relaxation", 
               "rho": RHO, "k":0.1},
    # "zmin": {"location": "zmin", 
    #          "method": "pressure_relaxation", 
    #          "rho": RHO, "k": 0.3},
    # "zmax": {"location": "zmax", 
    #          "method": "pressure_relaxation", 
    #          "rho": RHO, "k": 0.3},
}

# =============================================================================
# Internal Geometry (Obstacle)
# =============================================================================
internal_geometry = {
    "cylinder": {
        "enabled": True,
        "center": (center_x, center_y),
        "radius": diameter // 2,
        "axis": "z",
        "axis_range": (0, Nz - 1)
    },
    "sphere": {
        "enabled": False,
        "center": (center_x, center_y, center_z),
        "radius": diameter // 2
    }
}


# =============================================================================
# Conservation Check Configuration
# =============================================================================
# Monitors mass conservation during simulation.
# Supports domain-wide check and multiple control volumes (CVs).
#
# Usage:
#   - enabled: Turn on/off all conservation checks
#   - check_interval: How often to check (0 = use output_interval)
#   - verbose: 0=silent, 1=summary, 2=detailed
#   - log_to_csv: Save results to CSV for post-processing
#
# Control Volumes:
#   Each CV monitors mass within specified bounds.
#   Bounds use Python variables (Nx, Ny, Nz, diameter, center_x, etc.)

conservation = {
    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    "enabled": True,
    "check_interval": 1000,           # 0 = use output_interval
    "verbose": 1,                  # 0: silent, 1: summary, 2: detailed
    "log_to_csv": True,
    
    "tolerance": {
        "mass_drift_percent": 1.0,  # Warning threshold (%)
        "warn_on_exceed": True,
    },
    
    # -------------------------------------------------------------------------
    # Domain-wide Conservation Check
    # -------------------------------------------------------------------------
    "domain": {
        "enabled": True,
    },
    
    # -------------------------------------------------------------------------
    # Control Volume based Conservation Check
    # -------------------------------------------------------------------------
    # Multiple CVs can be defined. Each CV monitors mass within specified bounds.
    # Bounds are specified using Python variables directly.
    
    "control_volumes": [
        # CV 1: Region around obstacle
        {
            "name": "obstacle_region",
            "enabled": True,
            "bounds": {
                "xmin": center_x - 3 * diameter,
                "xmax": center_x + 5 * diameter,
                "ymin": center_y - 2 * diameter,
                "ymax": center_y + 2 * diameter,
                "zmin": 0,
                "zmax": Nz - 1,
            }
        },
        
        # CV 2: Wake region (downstream of obstacle)
        {
            "name": "wake_region",
            "enabled": True,
            "bounds": {
                "xmin": center_x + diameter,
                "xmax": center_x + 10 * diameter,
                "ymin": center_y - 2 * diameter,
                "ymax": center_y + 2 * diameter,
                "zmin": 0,
                "zmax": Nz - 1,
            }
        },
        
        # # CV 3: Example of disabled CV
        # {
        #     "name": "inlet_region",
        #     "enabled": False,
        #     "bounds": {
        #         "xmin": 0,
        #         "xmax": 50,
        #         "ymin": 0,
        #         "ymax": Ny - 1,
        #         "zmin": 0,
        #         "zmax": Nz - 1,
        #     }
        # },
    ],
}


# =============================================================================
# Force Calculation Configuration (UPDATED - Enabled)
# =============================================================================
# Computes hydrodynamic forces on obstacles using Momentum Exchange Method (MEM).
#
# Physical Principle:
#   F = Σ (2 * c_i * f_i^post) at all boundary links
#
# Force Coefficients:
#   C_D = F_x / (0.5 * ρ * U² * A)
#   C_L = F_y / (0.5 * ρ * U² * A)
#   where A = D * Lz (reference area)
#
# Expected values for Re=150 cylinder:
#   C_D ≈ 1.3 - 1.4
#   C_L oscillates around 0 with amplitude ~ 0.3-0.5
#   St ≈ 0.18 - 0.19

force_calculation = {
    "enabled": True,                   # Enable force calculation
    "interval": 10,                    # Compute every N steps
    "start_step": 1000,                # Skip initial transient
    
    "reference": {
        "rho": RHO,                    # Reference density [dimensionless]
        "velocity": MACH_INLET,        # Reference velocity [Δx/Δt]
        "char_length": diameter,       # Characteristic length D [Δx]
        "span_length": Nz,             # Span length Lz [Δx] (for 2D: Nz)
    },
    
    "log": {
        "enabled": True,
        "filename": "force_history",
    },
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
# Final Config Dictionary
# =============================================================================
config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "conservation": conservation,
    "force_calculation": force_calculation,
    "output": output
}