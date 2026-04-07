"""
MLG Poiseuille 3D — 4-Level, Uniform Y/Z Extent

All fine levels share the same y,z range as Level 1.
Only x-direction narrows progressively.
Purpose: test wall-resolution effects by keeping fine grid
coverage of the wall boundary layer identical across levels.

    Level 0: full domain
    Level 1: x[40,80] y[6,23] z[3,6]   ← base fine region
    Level 2: x[45,75] y[6,23] z[3,6]   ← same y,z, narrower x
    Level 3: x[50,70] y[6,23] z[3,6]   ← same y,z, even narrower x

Usage:
    python main.py --config configs/mlg_poiseuille_4level_uniform_yz.py

Author: LBM Development Team
Date: 2026-04
"""

import numpy as np

# =============================================================================
# §1. PHYSICAL PARAMETERS
# =============================================================================
RE = 20
U_INLET = 0.02
RHO = 1.0

Nx = 120
Ny = 30
Nz = 10
RESOLUTION = Ny

# =============================================================================
# §2. DERIVED LATTICE PARAMETERS
# =============================================================================
NU_LU = U_INLET * RESOLUTION / RE
TAU = 0.5 + 3.0 * NU_LU

CS = 1.0 / np.sqrt(3.0)
MA = U_INLET / CS

assert TAU > 0.5, f"UNSTABLE: τ = {TAU:.4f}"
assert MA < 0.3, f"Ma = {MA:.3f} > 0.3"

print(f"  [MLG 4-Level Uniform Y/Z] Re={RE}, H={Ny}, τ={TAU:.4f}, Ma={MA:.4f}")

# =============================================================================
# §3. MLG CONFIGURATION
# =============================================================================
OVERLAP_WIDTH = 2
INTERP_SCHEME = "cubic"
FILTER_LEVEL = 1

# Common y,z range for all fine levels
Y_MIN = 6
Y_MAX = 23
Z_MIN = 3
Z_MAX = 6

# =============================================================================
# §4. SIMULATION SETTINGS
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "dimension": 3,
    "lattice_model": "D3Q27",
    "collision_model": "bgk",

    "domain": {"Nx": Nx, "Ny": Ny, "Nz": Nz},

    "physics": {
        "Re": RE,
        "tau": TAU,
        "omega": 1.0 / TAU,
        "nu_lu": NU_LU,
        "u_ref_lu": U_INLET,
        "L_ref_lu": float(RESOLUTION),
        "initial_flow_velocity": [U_INLET, 0.0, 0.0],
    },

    "time": {
        "max_steps": 3000,
        "output_interval": 1000,
        "checkpoint_interval": 1000,
        "probe_interval": 100,
    },
}

# =============================================================================
# §5. BOUNDARY CONDITIONS
# =============================================================================
boundaries = {
    "inlet":      {"location": "xmin", "method": "regularized_inlet",
                   "velocity": U_INLET, "rho": RHO},
    # "outlet":     {"location": "xmax", "method": "regularized_outlet",
    #                "rho": RHO, "k": 0.1},
    "outlet": {"location": "xmax", "method": "sponge", 
               "velocity": [U_INLET, 0, 0], "density": 1.0,
               "thickness": 5, "sigma_max": 0.2},
    "wall_south": {"location": "ymin", "method": "regularized_wall"},
    "wall_north": {"location": "ymax", "method": "regularized_wall"},
    "wall_zmin":  {"location": "zmin", "method": "regularized_wall"},
    "wall_zmax":  {"location": "zmax", "method": "regularized_wall"},
}

# =============================================================================
# §6. NO INTERNAL OBSTACLE
# =============================================================================
internal_geometry = {"type": "none"}

# =============================================================================
# §7. MLG — 4 Levels, Uniform Y/Z
# =============================================================================
mlg = {
    "enabled": True,
    "num_levels": 4,
    "overlap_width": OVERLAP_WIDTH,
    "interpolation": INTERP_SCHEME,
    "filter_level": FILTER_LEVEL,
    "levels": [
        {},  # Level 0 = full domain

        # Level 1: base fine region
        {"region": {"x_min": 40, "x_max": 80,
                    "y_min": Y_MIN, "y_max": Y_MAX,
                    "z_min": Z_MIN, "z_max": Z_MAX}},

        # Level 2: narrower x, same y/z
        {"region": {"x_min": 45, "x_max": 75,
                    "y_min": Y_MIN, "y_max": Y_MAX,
                    "z_min": Z_MIN, "z_max": Z_MAX}},

        # Level 3: even narrower x, same y/z
        {"region": {"x_min": 50, "x_max": 70,
                    "y_min": Y_MIN, "y_max": Y_MAX,
                    "z_min": Z_MIN, "z_max": Z_MAX}},
    ],
}

# =============================================================================
# §8. MONITORS & OUTPUT
# =============================================================================
conservation = {
    "enabled": True,
    "check_interval": 100,
    "verbose": 0,
    "log_to_csv": True,
}

convergence = {"enabled": False}
force_calculation = {"enabled": False}

_folder = f"result_4level_uniform_yz_Re{int(RE)}"

output = {
    "output_dir": f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir": f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": True},
    "checkpoint": {"enabled": True},
}

# =============================================================================
# §9. FINAL CONFIG
# =============================================================================
config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "mlg": mlg,
    "conservation": conservation,
    "convergence": convergence,
    "force_calculation": force_calculation,
    "output": output,
}