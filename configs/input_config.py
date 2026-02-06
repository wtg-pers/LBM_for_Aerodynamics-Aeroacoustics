# =============================================================================
# Domain Configuration
# =============================================================================
# Cylinder parameters
diameter = 20      # [lattice units]
Nx = diameter * 25
Ny = diameter * 20
# Nz = 3

center_x = Nx // 5  # Located at 25% from inlet
center_y = Ny // 2  # Centered in y
# center_z = Nz // 2

MACH_INLET = 0.1
RHO = 1.0
RE_TGT = 100.0


# =============================================================================
# Simulation Parameters
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "device_id": 0,
    "dimension": 2,
    "lattice_model": "D2Q9",
    "domain": {
        "Nx": Nx,
        "Ny": Ny,
        # "Nz": Nz
    },
    "physics": {
        "Re": RE_TGT,
        "u_init": MACH_INLET,
        "characteristic_length": diameter
    },
    "time": {
        "max_steps": 30000,
        "output_interval": 100,
        "checkpoint_interval": 5000,
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
        # "axis_range": (0, Nz - 1)
    },
    # "sphere": {
    #     "enabled": False,
    #     "center": (center_x, center_y, center_z),
    #     "radius": diameter // 2
    # }
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
                # "zmin": 0,
                # "zmax": Nz - 1,
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
                # "zmin": 0,
                # "zmax": Nz - 1,
            }
        },
    ],
}


# =============================================================================
# Convergence Detection Configuration
# =============================================================================
# CV-based convergence: CV = σ/μ < ε within rolling window
#
# Path A (no obstacle): CV(avg_energy) < ε
# Path B (with obstacle): CV(Cd) < ε_Cd  (energy & Cl are monitor-only)
#
# Window sizing (auto):
#   T_conv = D / U  [timesteps]
#   window_timesteps = T_conv × time_coverage
#   window_samples = window_timesteps / feed_interval
#
#   Example (D=20, U=0.1, feed_interval=10):
#     T_conv = 200, window = 200×50/10 = 1000 samples → fills at 10,000 steps

convergence = {
    "enabled": True,

    "statistical": {
        # --- Window Size ---
        "window_size": "auto",           # "auto" (recommended) | int (manual samples)

        # --- CV Thresholds ---
        "epsilon": 1e-4,                 # [dimensionless] avg kinetic energy
        "Cd_epsilon": 0.01,              # [dimensionless] drag coefficient
        #   Steady flow:     Cd_epsilon = 1e-3 ~ 1e-2
        #   Periodic flow:   Cd_epsilon = 0.02 ~ 0.05  (natural CV ≈ 0.01)

        # --- Auto Window Tuning ---
        "auto_window": {
            "time_coverage": 50.0,       # window spans this many T_conv [dimensionless]
            "min_samples": 50,           # minimum buffer size [samples]
        },
    },

    # --- Termination Actions ---
    "on_converged": "checkpoint_and_stop",   # "checkpoint_and_stop" | "stop" | "continue"
    "on_max_steps": "warn",                  # "warn" | "error"
    "on_diverged": "stop_with_checkpoint",   # "stop_with_checkpoint" | "stop"
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

force_calculation = {
    "enabled": True,                   # Enable force calculation
    "interval": 10,                    # Compute every N steps
    "start_step": 100,                # Skip initial transient
    
    "reference": {
        "rho": RHO,                    # Reference density [dimensionless]
        "velocity": MACH_INLET,        # Reference velocity [Δx/Δt]
        "char_length": diameter,       # Characteristic length D [Δx]
        "span_length": 1,             # Span length Lz [Δx] (for 2D: Nz)
    },
    
    "log": {
        "enabled": True,
        "filename": "force_history",
    },
}


# =============================================================================
# Output Configuration
# =============================================================================
d_str = str(int(diameter))
re_str = str(int(RE_TGT))
folder_name = f"result_D{d_str}_Re{re_str}"

output = {
    "output_dir": f"./{folder_name}/vtk",
    "checkpoint_dir": f"./{folder_name}/checkpoints",
    "csv_dir": f"./{folder_name}/csv",
    
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
            (center_x + 2 * diameter, center_y),# center_z),
            (center_x + 5 * diameter, center_y),# center_z)
        ]
    }
}


# =============================================================================
# Final Config Dictionary (UPDATED - add convergence)
# =============================================================================
config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "conservation": conservation,
    "force_calculation": force_calculation,
    "convergence": convergence,            
    "output": output
}