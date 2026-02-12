# =============================================================================
# LBM Solver — General Purpose Configuration
# =============================================================================
#
# This config supports all simulation types:
#   - External flow (cylinder, airfoil)  → internal_geometry + force_calculation
#   - Channel flow (Poiseuille)          → wall BCs, no obstacle
#   - Wind turbine (AL)                  → actuator_line.enabled = True
#
# Usage:
#   python main.py --config configs/input_config.py
#   python main.py --config configs/input_config.py --max-steps 5000
#
# Available BC Methods (DomainBCManager):
# ──────────────────────────────────────────────────────────────────────
#   Velocity Inlet:   'regularized_inlet' (reg_inlet), 'equilibrium' (eq)
#   Pressure Outlet:  'regularized_outlet' (reg_outlet)
#   Domain Wall:      'regularized_wall' (reg_wall), 'bounce_back' (hwbb)
#   Zero-Gradient:    'neumann' (zero_gradient)
#   Sponge Layer:     'sponge' (sponge_layer)
#   Periodic:         'periodic' (or omit from config)
# ──────────────────────────────────────────────────────────────────────

# =============================================================================
# Geometry & Domain Configuration
# =============================================================================
# L: characteristic length — the single reference scale for the simulation.
#    cylinder → diameter,  airfoil → chord,  sphere → diameter, etc.
#    All spatial dimensions are expressed in multiples of L.
L = 100              # [lu] characteristic length

Nx = L * 25         # [lu] streamwise
Ny = L * 16         # [lu] crosswise
# Nz = 3            # [lu] spanwise (uncomment for 3D)

center_x = Nx // 4  # [lu] obstacle at 25% from inlet
center_y = Ny // 2  # [lu] centered in y
# center_z = Nz // 2

U_INF = 0.1          # [Δx/Δt] freestream velocity (≡ lattice Mach number)
RHO = 1.0            # [dimensionless] reference density
RE = 1000.0          # [dimensionless] Reynolds number = U_INF * L / ν


# =============================================================================
# Simulation Parameters
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "dimension": 2,
    "lattice_model": "D2Q9",
    "domain": {
        "Nx": Nx,
        "Ny": Ny,
        # "Nz": Nz,                       # uncomment for 3D
    },
    "physics": {
        "Re": RE,                           # [dimensionless]
        "u_init": U_INF,                    # [Δx/Δt] freestream / initial velocity
        # "u_ref": U_INF,                   # [Δx/Δt] override for Re calc (default = u_init)
        "characteristic_length": L,         # [Δx] for Re = u_ref * L / ν
    },
    "time": {
        "max_steps": 100000,
        "output_interval": 100,
        "checkpoint_interval": 5000,
        "probe_interval": 10,               # [steps] force/probe sampling
    }
}


# =============================================================================
# Boundary Conditions
# =============================================================================
# External flow (eq + neumann): Latt et al. 2020 style
#   3-face Dirichlet + outlet Neumann → automatic mass conservation
boundaries = {
    "inlet":  {"location": "xmin", "method": "equilibrium",  "velocity": U_INF},
    "outlet": {"location": "xmax", "method": "neumann"},
    "south":  {"location": "ymin", "method": "equilibrium",  "velocity": U_INF},
    "north":  {"location": "ymax", "method": "equilibrium",  "velocity": U_INF},
}

# ── Alternative BC sets (uncomment one block) ──
#
# Regularized set (best accuracy for channel / external flow):
# boundaries = {
#     "inlet":  {"location": "xmin", "method": "regularized_inlet",
#                "velocity": U_INF, "rho": RHO},
#     "outlet": {"location": "xmax", "method": "regularized_outlet",
#                "rho": RHO, "k": 0.1},
#     "south":  {"location": "ymin", "method": "regularized_outlet",
#                "rho": RHO, "k": 0.1},
#     "north":  {"location": "ymax", "method": "regularized_outlet",
#                "rho": RHO, "k": 0.1},
# }
#
# Channel flow (walls + inlet/outlet):
# boundaries = {
#     "inlet":  {"location": "xmin", "method": "regularized_inlet",
#                "velocity": U_INF, "rho": RHO},
#     "outlet": {"location": "xmax", "method": "regularized_outlet",
#                "rho": RHO, "k": 0.1},
#     "south":  {"location": "ymin", "method": "regularized_wall"},
#     "north":  {"location": "ymax", "method": "regularized_wall"},
# }


# =============================================================================
# Internal Geometry (Obstacle)
# =============================================================================
# Each geometry type interprets L as its own reference scale:
#   cylinder → L = diameter
#   airfoil  → L = chord
#   sphere   → L = diameter
#
# Only one geometry should be enabled at a time.

internal_geometry = {
    "airfoil": {
        "enabled": True,
        "naca": "0012",
        "chord": L,                         # [Δx] = L
        "angle_of_attack": -28.0,           # [degrees]
        "center": (center_x, center_y),     # [Δx, Δx]
        "num_points": 150,
    }
}

# internal_geometry = {
#     "cylinder": {
#         "enabled": True,
#         "center": (center_x, center_y),   # [Δx, Δx]
#         "diameter": L,                     # [Δx] = L
#     }
# }

# internal_geometry = {
#     "sphere": {
#         "enabled": True,
#         "center": (center_x, center_y, center_z),   # [Δx, Δx, Δx]
#         "diameter": L,                               # [Δx] = L
#     }
# }

# No obstacle (channel flow or pure AL simulation):
# internal_geometry = {}


# =============================================================================
# Actuator Line Configuration (disabled — template for wind turbine sims)
# =============================================================================
# To enable: set "enabled": True and configure rotor/units.
# Requires 3D domain (D3Q27) with physical unit conversion.
#
# See configs/NTNU_BT1_config.py for a complete working example.
actuator_line = {
    "enabled": False,

    # --- Rotor ---
    # "rotor": {
    #     "preset": "ntnu_bt1",               # predefined blade geometry
    #     "tsr": 6.0,                          # [dimensionless] tip speed ratio
    #     "u_inf": 10.0,                       # [m/s] freestream for ω calc
    #     "hub_center": [3.66, 1.341, 0.817],  # [m] physical coordinates
    #     "resolution": 32,                     # [dimensionless] D/Δx
    #     "theta_0": 0.0,                       # [rad] initial azimuth
    # },

    # --- Physical → Lattice unit conversion ---
    # "units": {
    #     "dx_phys": 0.02794,                  # [m/lu]
    #     "dt_phys": 4.19e-5,                  # [s/lt]
    # },

    # --- Gaussian regularization (Eq. 13) ---
    # "gaussian_cutoff": 3.0,                  # [dimensionless] cutoff at n_cut × ε
    # "rho_ref": 1.0,                          # [dimensionless]
}


# =============================================================================
# Conservation Check Configuration
# =============================================================================
conservation = {
    "enabled": True,
    "check_interval": 1000,                 # [steps] 0 = use output_interval
    "verbose": 0,                           # 0: silent, 1: summary, 2: detailed
    "log_to_csv": True,

    "tolerance": {
        "mass_drift_percent": 1.0,          # [%] warning threshold
        "warn_on_exceed": True,
    },

    "domain": {
        "enabled": True,
    },

    "control_volumes": [
        {
            "name": "obstacle_region",
            "enabled": True,
            "bounds": {
                "xmin": center_x - 3 * L,
                "xmax": center_x + 5 * L,
                "ymin": center_y - 2 * L,
                "ymax": center_y + 2 * L,
            }
        },
        {
            "name": "wake_region",
            "enabled": True,
            "bounds": {
                "xmin": center_x + L,
                "xmax": center_x + 10 * L,
                "ymin": center_y - 2 * L,
                "ymax": center_y + 2 * L,
            }
        },
    ],
}


# =============================================================================
# Convergence Detection (Cauchy Criterion)
# =============================================================================
# Cauchy convergence: |μ_new - μ_old| / |μ_old| < ε
#   Path A (no obstacle): Cauchy(avg_energy) < ε
#   Path B (with obstacle): Cauchy(Cd) < ε_Cd
convergence = {
    "enabled": True,

    "cauchy": {
        "window_size": "auto",
        "epsilon": 1e-5,                    # [dimensionless] avg kinetic energy
        "Cd_epsilon": 1e-3,                 # [dimensionless] drag coefficient
        "n_required": 2,

        "auto_window": {
            "time_coverage": 10.0,          # [T_conv] window span (default: 50)
            "min_samples": 100,
        },
    },

    "on_converged": "checkpoint_and_stop",
    "on_max_steps": "warn",
    "on_diverged": "stop_with_checkpoint",
}


# =============================================================================
# Force Calculation (Momentum Exchange Method)
# =============================================================================
force_calculation = {
    "enabled": True,
    "interval": 10,                         # [steps]
    "start_step": 100,                      # [steps] skip initial transient

    "reference": {
        "rho": RHO,                         # [dimensionless]
        "velocity": U_INF,                  # [Δx/Δt]
        "char_length": L,                   # [Δx]
        "span_length": 1,                   # [Δx] (2D: 1, 3D: Nz)
    },

    "log": {
        "enabled": True,
        "filename": "force_history",
    },
}


# =============================================================================
# Output — Auto-generated folder name
# =============================================================================
# Naming convention:
#   result_{geometry}[_ALM]_L{L}_Re{Re}
#
# Examples:
#   result_naca0012_L20_Re1000
#   result_naca0012_ALM_L20_Re50
#   result_cylinder_L40_Re100
#   result_none_ALM_L32_Re200

def _make_folder_name(geom_cfg, al_cfg, L_val, Re_val):
    """Generate output folder name from config (config-internal helper)."""
    # --- Obstacle tag ---
    tag = "none"
    for geom_type, geom_params in geom_cfg.items():
        if not geom_params.get("enabled", False):
            continue
        if geom_type == "airfoil":
            naca = geom_params.get("naca", "airfoil")
            tag = f"naca{naca}"             # → "naca0012"
        elif geom_type == "cylinder":
            tag = "cylinder"
        elif geom_type == "sphere":
            tag = "sphere"
        else:
            tag = geom_type                 # user-defined name
        break                               # first enabled geometry wins

    # --- AL tag ---
    al_tag = "_ALM" if al_cfg.get("enabled", False) else ""

    # --- Re: int if whole, else float ---
    Re_str = str(int(Re_val)) if Re_val == int(Re_val) else f"{Re_val:.1f}"

    return f"result_{tag}{al_tag}_L{int(L_val)}_Re{Re_str}"

folder_name = _make_folder_name(internal_geometry, actuator_line, L, RE)

output = {
    "output_dir": f"./{folder_name}/vtk",
    "checkpoint_dir": f"./{folder_name}/checkpoints",
    "csv_dir": f"./{folder_name}/csv",

    "clear_previous": True,

    "vtk": {
        "enabled": True,
        "precision": "float32",
        "compression_level": 6,
        "variables": ["density", "velocity", "solid_mask"],
    },
    "checkpoint": {
        "enabled": True,
        "keep_last_n": 3,
    },
}


# =============================================================================
# Final Config Dictionary
# =============================================================================
config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "actuator_line": actuator_line,
    "conservation": conservation,
    "convergence": convergence,
    "force_calculation": force_calculation,
    "output": output,
}