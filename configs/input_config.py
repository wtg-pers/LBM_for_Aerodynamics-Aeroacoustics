# =============================================================================
# LBM Solver — Hover Rotor Configuration (NTNU BT1 based)
# =============================================================================
#
# Actuator Line Model test: hovering rotor with ground effect.
#
# Usage:
#   python main.py --config configs/input_config.py
#   python main.py --config configs/input_config.py --max-steps 10000
#
# =============================================================================

import numpy as np

# =============================================================================
# §1. Physical Constants
# =============================================================================
R_PHYS     = 0.125          # [m]      Rotor radius
D_PHYS     = 2 * R_PHYS     # [m]      Diameter = 0.25 m
CHORD      = 0.025          # [m]      Constant chord
PITCH      = 10.0           # [deg]    Collective pitch (blade twist = -10° in code)
N_BLADES   = 2              # [-]      Number of blades

ROOT_CUT   = 0.20           # [-]  20%
R_ROOT     = R_PHYS * ROOT_CUT  # [m] = 0.025 m (inactive region ends here)

# Operating conditions
TSR      = 6.0             # [-]      Tip speed ratio
U_INF    = 10.0            # [m/s]    Reference velocity (for ω calculation)
RPM        = 3000           # [RPM]    Rotation speed
OMEGA      = RPM * 2 * np.pi / 60  # [rad/s] = 314.16 rad/s
V_TIP      = OMEGA * R_PHYS # [m/s]    Tip speed = 39.27 m/s

# Air properties
NU_AIR     = 1.512e-5       # [m²/s]   Kinematic viscosity (20°C)
RHO_AIR    = 1.205          # [kg/m³]  Air density

# =============================================================================
# §2. Grid & Lattice Scaling
# =============================================================================
RESOLUTION = 32            # [-]      D/Δx (grid cells per diameter)
L = RESOLUTION
DX_PHYS    = D_PHYS / RESOLUTION  # [m/lu] = 0.02235 m

MA_TIP     = 0.05          # [-]      Lattice Mach number at tip
U_TIP_LU   = MA_TIP        # [lu/lt]  Tip velocity in lattice units
DT_PHYS    = U_TIP_LU * DX_PHYS / V_TIP  # [s/lt] = 1.86e-5 s

TAU        = 0.55          # [-]      Relaxation time
NU_LU      = (TAU - 0.5) / 3  # [lu²/lt] Lattice viscosity

# Derived quantities
D_LU       = RESOLUTION    # [lu]     Rotor diameter in lattice units
R_LU       = D_LU / 2      # [lu]     Rotor radius in lattice units
OMEGA_LU   = OMEGA * DT_PHYS  # [rad/lt] Angular velocity in lattice
STEPS_PER_REV = int(2 * np.pi / OMEGA_LU)  # ≈ 2513 steps/revolution

# =============================================================================
# §3. Domain Configuration
# =============================================================================
# Domain size: 3D × 3D × 3D
Nx = 200
Ny = 200
Nz = 200

# Hub position in LATTICE UNITS (domain coordinates)
HUB_X_LU = Nx // 2       # 40 [lu]  - 1D from inlet
HUB_Y_LU = Ny // 2          # 60 [lu]  - centered
HUB_Z_LU = int(Nz * 0.4)          # 60 [lu]  - centered

# Hub position in PHYSICAL UNITS (for Rotor factory)
# This is what gets passed to Rotor, then converted back to lattice units
HUB_X_PHYS = HUB_X_LU * DX_PHYS  # [m]
HUB_Y_PHYS = HUB_Y_LU * DX_PHYS  # [m]
HUB_Z_PHYS = HUB_Z_LU * DX_PHYS  # [m]

# =============================================================================
# §4. Physics (for LBM)
# =============================================================================
# Reference length and Re for viscosity calculation
CHAR_LENGTH = CHORD / DX_PHYS  # [lu] chord in lattice units
RE = U_TIP_LU * CHAR_LENGTH / NU_LU  # Grid Reynolds number

# =============================================================================
# §5. Simulation Config
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "dimension": 3,
    "lattice_model": "D3Q27",
    "domain": {"Nx": Nx, "Ny": Ny, "Nz": Nz},
    "physics": {
        "Re": RE,
        "u_init": U_TIP_LU,
        "u_ref": U_TIP_LU,
        "characteristic_length": CHAR_LENGTH,
    },
    "time": {
        "max_steps": STEPS_PER_REV * 40,
        "output_interval": STEPS_PER_REV // 360,
        "checkpoint_interval": STEPS_PER_REV * 5,
        "probe_interval": 10,
    },
}

# =============================================================================
# §6. Boundary Conditions
# =============================================================================
# X-axis = rotation axis (wind direction for HAWT)
# xmin = inlet (upstream), xmax = outlet (downstream/ground for hover)
# boundaries = {
#     "inlet":  {"location": "xmin", "method": "equilibrium", "velocity": 0.1},
#     "outlet": {"location": "xmax", "method": "neumann"},
#     "ymin":   {"location": "ymin", "method": "regularized_wall", "density": 1.0},
#     "ymax":   {"location": "ymax", "method": "equilibrium", "velocity": 0.1},
#     "zmin":   {"location": "zmin", "method": "equilibrium", "velocity": 0.1},
#     "zmax":   {"location": "zmax", "method": "equilibrium", "velocity": 0.1},
# }

boundaries = {
    "ground": {"location": "zmin", "method": "regularized_wall",},

    "top": {"location": "zmax", "method": "equilibrium", 
            "velocity": 0.0, "density": 1.0,},
    "xmin": {"location": "xmin", "method": "equilibrium",
             "velocity": 0.0,"density": 1.0,},
    "xmax": {"location": "xmax","method": "equilibrium",
             "velocity": 0.0,"density": 1.0,},
    "ymin": {"location": "ymin","method": "equilibrium",
             "velocity": 0.0,"density": 1.0,},
    "ymax": {"location": "ymax","method": "equilibrium",
             "velocity": 0.0,"density": 1.0,},
}

# =============================================================================
# §7. Internal Geometry (none for AL-only simulation)
# =============================================================================
internal_geometry = {}

# =============================================================================
# §8. Actuator Line Model
# =============================================================================
actuator_line = {
    "enabled": True,
    
    "rotor": {
        "n_blades": N_BLADES,
        "hub_center": [HUB_X_PHYS, HUB_Y_PHYS, HUB_Z_PHYS],  # [m]
        "omega": OMEGA,               # [rad/s] NEGATIVE for CW from +Z
        "theta_0": 0.0,                # [rad]
        "rotation_axis": "hawt_z",     # Z-axis rotation
        
        # Blade geometry
        "blade": {
            "sections": [
                # Root region (inactive) - 0% to 20%
                {"r": 0.0,              "chord": CHORD, "twist": -PITCH, 
                 "airfoil": "NACA0012", "active": False},
                {"r": R_ROOT,           "chord": CHORD, "twist": -PITCH, 
                 "airfoil": "NACA0012", "active": False},
                # Active region - 20% to 100%
                {"r": R_ROOT + 1e-6,    "chord": CHORD, "twist": -PITCH, 
                 "airfoil": "NACA0012", "active": True},
                {"r": R_PHYS,           "chord": CHORD, "twist": -PITCH, 
                 "airfoil": "NACA0012", "active": True},
            ],
        },
        
        "grid": {
            "resolution": RESOLUTION,
            "dx": DX_PHYS,
        },
    },
    
    "units": {
        "dx_phys": DX_PHYS,
        "dt_phys": DT_PHYS,
        "nu_phys": NU_AIR,
    },
    
    "gaussian_cutoff": 3.0,
    "rho_ref": 1.0,
}
# actuator_line = {
#     "enabled": True,
    
#     "rotor": {
#         "preset": "ntnu_bt1",
#         "tsr": TSR,
#         "u_inf": U_INF,
#         # Hub center in PHYSICAL UNITS [m]
#         # Will be converted to lattice units by Rotor.to_lattice_units()
#         "hub_center": [HUB_X_PHYS, HUB_Y_PHYS, HUB_Z_PHYS],
#         "resolution": RESOLUTION,
#         "theta_0": 0.0,
#         "rotation_axis": "hawt_x",
#     },
    
#     "units": {
#         "dx_phys": DX_PHYS,
#         "dt_phys": DT_PHYS,
#         "nu_phys": NU_AIR,
#     },
    
#     "gaussian_cutoff": 3.0,
#     "rho_ref": 1.0,
# }

# =============================================================================
# §9. Conservation & Convergence
# =============================================================================
conservation = {
    "enabled": True,
    "check_interval": STEPS_PER_REV // 360,
    "verbose": 0,  # 0 : silent, 1 : simple, 2: detail
    "log_to_csv": True,
}

convergence = {
    "enabled": True,
    "monitor": {
        "energy": {"enabled": True, "threshold": 1e-5, "window": 500},
    },
    "on_converged": "checkpoint_and_stop",
    "on_diverged": "stop_with_checkpoint",
    "on_max_steps": "warn",
}

# =============================================================================
# §10. Force Calculation (disabled for AL simulation)
# =============================================================================
force_calculation = {"enabled": False}

# =============================================================================
# §11. Output
# =============================================================================
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

_folder = _make_folder_name(internal_geometry, actuator_line, L, RE)

output = {
    "output_dir": f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir": f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {
        "enabled": True,
        "precision": "float32",
        "compression_level": 6,
        "variables": ["density", "velocity"],
    },
    "checkpoint": {"enabled": True, "keep_last_n": 3},
}

# =============================================================================
# §12. Final Config Dictionary
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

# =============================================================================
# §13. Summary (printed when config is loaded)
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Hover Rotor Configuration Summary")
    print("=" * 60)
    print(f"  Domain:        {Nx} × {Ny} × {Nz} = {Nx*Ny*Nz:,} cells")
    print(f"  Resolution:    D/Δx = {RESOLUTION}")
    print(f"  Δx = {DX_PHYS*1000:.3f} mm,  Δt = {DT_PHYS*1e6:.3f} μs")
    print(f"  τ = {TAU},  ν_lu = {NU_LU:.5f}")
    print(f"  Ma_tip = {MA_TIP},  ω_lu = {OMEGA_LU:.6f} rad/lt")
    print(f"  Steps/rev ≈ {STEPS_PER_REV}")
    print("-" * 60)
    print(f"  Hub (lu):      ({HUB_X_LU:.1f}, {HUB_Y_LU:.1f}, {HUB_Z_LU:.1f})")
    print(f"  Hub (phys):    ({HUB_X_PHYS:.4f}, {HUB_Y_PHYS:.4f}, {HUB_Z_PHYS:.4f}) m")
    print(f"  Rotor R (lu):  {R_LU:.1f}")
    print("=" * 60)