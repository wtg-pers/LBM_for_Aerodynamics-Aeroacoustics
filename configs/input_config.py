import numpy as np
import warnings


L_REF      = 0.025           # [m]      Characteristic length (= rotor diameter D)
RHO_REF    = 1.205           # [kg/m³]  Reference density (air at 20°C)

# -----------------------------------------------------------------------------
# Rotor operating condition (for ALM)
# -----------------------------------------------------------------------------
RPM        = 1000            # [rpm]    Rotor rotational speed (USER INPUT)
R_ROTOR    = L_REF / 2       # [m]      Rotor radius = D/2
OMEGA_ROTOR = RPM * 2.0 * np.pi / 60.0   # [rad/s]  Angular velocity

# -----------------------------------------------------------------------------
# Reference velocity (for Re calculation) vs Inlet velocity
# -----------------------------------------------------------------------------
U_REF      = OMEGA_ROTOR * (R_ROTOR * 0.75)  # [m/s]    Reference velocity
U_INF      = 0.0             # [m/s]    Freestream/inlet velocity (0 for hover)

# -----------------------------------------------------------------------------
# Reynolds number OR viscosity (choose ONE, Re takes priority)
# -----------------------------------------------------------------------------
RE         = 100             # [-]      Reynolds number (PRIORITY)
NU_PHYS    = None            # [m²/s]   Kinematic viscosity (set if RE=None)

# --- Auto-calculate: Re ↔ ν_phys ---
if RE is not None and NU_PHYS is not None:
    _nu_computed = U_REF * L_REF / RE
    warnings.warn(
        f"Both RE and NU_PHYS defined. Using RE={RE} (priority).\n"
        f"  Computed ν = {_nu_computed:.6e} m²/s\n"
        f"  Provided ν = {NU_PHYS:.6e} m²/s (ignored)"
    )
    NU_PHYS = _nu_computed
elif RE is not None:
    NU_PHYS = U_REF * L_REF / RE
elif NU_PHYS is not None:
    RE = U_REF * L_REF / NU_PHYS
else:
    raise ValueError("Either 'RE' or 'NU_PHYS' must be defined!")

# =============================================================================
# §2. NUMERICAL PARAMETERS (User Input - Discretization)
# =============================================================================

RESOLUTION = 40              # [-]  Grid cells per L_REF (N = L_REF / Δx)
LATTICE_VELOCITY = 0.1      # [-]  u_lu (controls accuracy, recommend 0.02~0.1)

# =============================================================================
# §3. AUTO-CALCULATED LATTICE PARAMETERS
# =============================================================================
# DO NOT MODIFY - These are derived from user inputs above.

# --- Conversion factors ---
DX_PHYS    = L_REF / RESOLUTION                      # [m/lu]
DT_PHYS    = LATTICE_VELOCITY * DX_PHYS / U_REF      # [s/lt]

OMEGA_LU = OMEGA_ROTOR * DT_PHYS
STEPS_PER_REV = int(2 * np.pi / OMEGA_LU)

# --- Lattice parameters ---
L_REF_LU   = RESOLUTION                              # [lu]
U_REF_LU   = LATTICE_VELOCITY                        # [lu/lt]
U_INF_LU   = U_INF / U_REF * LATTICE_VELOCITY        # [lu/lt] inlet velocity in LU
NU_LU      = NU_PHYS * DT_PHYS / (DX_PHYS ** 2)      # [lu²/lt]
TAU        = 0.5 + 3.0 * NU_LU                       # [-]
OMEGA_LBM  = 1.0 / TAU                               # [-] relaxation frequency (NOT rotor omega!)

# --- Lattice Mach number (for reference) ---
CS = 1.0 / np.sqrt(3.0)      # Lattice sound speed ≈ 0.577
MA_LATTICE = LATTICE_VELOCITY / CS

# --- Stability checks ---
_stability_warnings = []
if TAU <= 0.5:
    raise ValueError(f"CRITICAL: τ = {TAU:.4f} ≤ 0.5 → UNSTABLE!")
if TAU < 0.52:
    _stability_warnings.append(f"⚠️  τ = {TAU:.4f} < 0.52 → marginally stable")
if TAU > 2.0:
    _stability_warnings.append(f"⚠️  τ = {TAU:.4f} > 2.0 → accuracy degradation")
if MA_LATTICE > 0.3:
    raise ValueError(f"CRITICAL: Ma_lattice = {MA_LATTICE:.3f} > 0.3 → compressibility error!")
if MA_LATTICE > 0.1:
    _stability_warnings.append(f"⚠️  Ma_lattice = {MA_LATTICE:.3f} > 0.1 → ~{MA_LATTICE**2*100:.1f}% density error")

for _w in _stability_warnings:
    print(_w)

# =============================================================================
# §4. DOMAIN CONFIGURATION
# =============================================================================
# Domain size in lattice units, or as multiples of L_REF

# DOMAIN_MULTIPLE_X = 10       # [-]  Domain = 10 × L_REF in x
# DOMAIN_MULTIPLE_Y = 5        # [-]  Domain = 5 × L_REF in y
# DOMAIN_MULTIPLE_Z = 5        # [-]  Domain = 5 × L_REF in z

# Nx = DOMAIN_MULTIPLE_X * RESOLUTION    # [lu]
# Ny = DOMAIN_MULTIPLE_Y * RESOLUTION    # [lu]
# Nz = DOMAIN_MULTIPLE_Z * RESOLUTION    # [lu]
Nx = 80
Ny = 80
Nz = 100

# =============================================================================
# §4. SIMULATION SETTINGS
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "dimension": 3,
    "lattice_model": "D3Q27",
    
    "domain": {"Nx": Nx, "Ny": Ny, "Nz": Nz,},
    
    "physics": {
        # Physical (SI)
        "L_ref": L_REF,            # [m]
        "U_ref": U_REF,            # [m/s]
        "U_inf": U_INF,            # [m/s]
        "rho_ref": RHO_REF,        # [kg/m³]
        "nu_phys": NU_PHYS,        # [m²/s]
        "Re": RE,                  # [-]
        
        # Lattice (auto-calculated)
        "tau": TAU,
        "omega": OMEGA_LBM,
        "nu_lu": NU_LU,
        "u_ref_lu": U_REF_LU,
        "L_ref_lu": L_REF_LU,
        
        # Conversion factors
        "dx": DX_PHYS,             # [m/lu]
        "dt": DT_PHYS,             # [s/lt]
    },
    
    "time": {
        "max_steps": STEPS_PER_REV * 20,
        "output_interval": STEPS_PER_REV // 18,
        "checkpoint_interval": STEPS_PER_REV * 2,
        "probe_interval": 10,
    },
}

# =============================================================================
# §5. BOUNDARY CONDITIONS
# =============================================================================
# Adjust based on your case type

# boundaries = {
#     # Example: External flow (cylinder, airfoil)
#     "inlet":  {"location": "xmin", "method": "equilibrium", 
#                "velocity": U_REF_LU, "density": 1.0},
#     "outlet": {"location": "xmax", "method": "neumann"},
#     "top":    {"location": "ymax", "method": "equilibrium", 
#                "velocity": U_REF_LU, "density": 1.0},
#     "bottom": {"location": "ymin", "method": "equilibrium", 
#                "velocity": U_REF_LU, "density": 1.0},
#     "front":  {"location": "zmin", "method": "periodic"},
#     "back":   {"location": "zmax", "method": "periodic"},
# }

boundaries = {
    # "ground": {"location": "zmin", "method": "regularized_wall",},
    "ground": {"location": "zmin", "method": "regularized_outlet", 
            "velocity": 0.0, "density": 1.0, "k": 0.1},
    "top": {"location": "zmax", "method": "regularized_inlet", 
            "velocity": 0.0, "density": 1.0, "k": 0.1},
    "xmin": {"location": "xmin", "method": "regularized_outlet",
             "velocity": 0.0,"density": 1.0, "k": 0.1},
    "xmax": {"location": "xmax","method": "regularized_outlet",
             "velocity": 0.0,"density": 1.0, "k": 0.1},
    "ymin": {"location": "ymin","method": "regularized_outlet",
             "velocity": 0.0,"density": 1.0, "k": 0.1},
    "ymax": {"location": "ymax","method": "regularized_outlet",
             "velocity": 0.0,"density": 1.0, "k": 0.1},
}

# =============================================================================
# §6. INTERNAL GEOMETRY (Optional Module)
# =============================================================================
# Enable/disable objects in the flow domain

internal_geometry = {
    # Example: Cylinder
    # "cylinder": {
    #     "enabled": True,
    #     "center": [Nx // 4, Ny // 2, Nz // 2],  # [lu]
    #     "diameter": RESOLUTION,                   # [lu] = L_REF
    #     "axis": "z",
    # },
    
    # Example: Airfoil
    # "airfoil": {
    #     "enabled": True,
    #     "naca": "0012",
    #     "chord": RESOLUTION,                      # [lu] = L_REF
    #     "center": [Nx // 4, Ny // 2, Nz // 2],
    #     "angle_of_attack": 5.0,                   # [deg]
    # },
}

# =============================================================================
# §7. ACTUATOR LINE MODEL (Optional Module)
# =============================================================================
ALM_ENABLED = True

if ALM_ENABLED:
    # Rotor geometry (R_ROTOR, OMEGA_ROTOR defined in §1)
    CHORD_BLADE = 0.025             # [m]  Blade chord
    PITCH       = 10.0              # [deg] Collective pitch
    N_BLADES    = 2                 # [-]
    ROOT_CUT    = 0.20              # [-]
    
    # Hub position
    HUB_X_LU    = Nx // 2
    HUB_Y_LU    = Ny // 2
    HUB_Z_LU    = Nz * 0.75
    
    HUB_X_PHYS  = HUB_X_LU * DX_PHYS   # [m]
    HUB_Y_PHYS  = HUB_Y_LU * DX_PHYS   # [m]
    HUB_Z_PHYS  = HUB_Z_LU * DX_PHYS   # [m]
    
    # Polar Re range
    RE_POLAR_MIN    = max(1.0, RE * 0.15)
    RE_POLAR_MAX    = RE * 2.5
    RE_POLAR_TARGET = RE

actuator_line = {
    "enabled": ALM_ENABLED,
}

if ALM_ENABLED:
    actuator_line.update({
        "rotor": {
            "n_blades": N_BLADES,
            "hub_center": [HUB_X_PHYS, HUB_Y_PHYS, HUB_Z_PHYS],  # [m]
            "omega": OMEGA_ROTOR,      # [rad/s] rotor angular velocity
            "theta_0": 0.0,            # [rad]
            "rotation_axis": "hawt_z",
            
            "blade": {
                "sections": [
                    {"r": 0.0,                       "chord": CHORD_BLADE, 
                     "twist": -PITCH, "airfoil": "naca0012", "active": False},
                    {"r": R_ROTOR * ROOT_CUT,        "chord": CHORD_BLADE, 
                     "twist": -PITCH, "airfoil": "naca0012", "active": False},
                    {"r": R_ROTOR * ROOT_CUT + 1e-6, "chord": CHORD_BLADE, 
                     "twist": -PITCH, "airfoil": "naca0012", "active": True},
                    {"r": R_ROTOR,                   "chord": CHORD_BLADE, 
                     "twist": -PITCH, "airfoil": "naca0012", "active": True},
                ],
            },
        },
        
        "units": {
            "dx_phys": DX_PHYS,        # [m/lu]
            "dt_phys": DT_PHYS,        # [s/lt]
            "nu_phys": NU_PHYS,        # [m²/s]
        },

        "grid": {
                "resolution": RESOLUTION,   # [-] D/Δx = 40
                "dx": DX_PHYS,              # [m/lu]
            },
        
        "rho_ref": RHO_REF,            # [kg/m³]
        "gaussian_cutoff": 3.0,
    })

# =============================================================================
# §9. AIRFOIL POLAR (Only if ALM enabled)
# =============================================================================
if ALM_ENABLED:
    airfoil_polar = {
        "method": "neuralfoil",
        "airfoil_name": "naca0012",
        "Re_target": RE_POLAR_TARGET,
        "Re_min": RE_POLAR_MIN,
        "Re_max": RE_POLAR_MAX,
        "mode": "asb",
        "ncrit": 9.0,
    }
else:
    airfoil_polar = {}

# =============================================================================
# §9. FORCE CALCULATION (Optional Module)
# =============================================================================
# Enable for drag/lift measurement on internal geometry

force_calculation = {
    "enabled": False,
    # "object": "cylinder",
    # "method": "momentum_exchange",
}

# =============================================================================
# §10. CONSERVATION & CONVERGENCE
# =============================================================================
conservation = {
    "enabled": True,
    "check_interval": 100,
    "verbose": 0,
    "log_to_csv": True,
}

convergence = {
    "enabled": True,
    "monitor": {
        "energy": {"enabled": True, "threshold": 1e-6, "window": 1000},
    },
    "on_converged": "checkpoint_and_stop",
    "on_diverged": "stop_with_checkpoint",
    "on_max_steps": "continue",
}

# =============================================================================
# §11. OUTPUT
# =============================================================================
_case_tag = "ALM" if ALM_ENABLED else "base"
_folder = f"result_{_case_tag}_Re{int(RE)}_L{RESOLUTION}"

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
# §12. FINAL CONFIG DICTIONARY
# =============================================================================
config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "actuator_line": actuator_line,
    "airfoil_polar": airfoil_polar,
    "conservation": conservation,
    "convergence": convergence,
    "force_calculation": force_calculation,
    "output": output,
}

# =============================================================================
# §13. SUMMARY
# =============================================================================
def print_summary():
    """Print configuration summary."""
    print()
    print("=" * 70)
    print(" LBM Solver Configuration Summary")
    print("=" * 70)
    print()
    print(" §1. Physical Parameters (User Input):")
    print(f"      L_ref   = {L_REF} m")
    print(f"      U_ref   = {U_REF} m/s")
    print(f"      ρ_ref   = {RHO_REF} kg/m³")
    print(f"      Re      = {RE}")
    print(f"      ν_phys  = {NU_PHYS:.6e} m²/s")
    print()
    print(" §2. Discretization (User Input):")
    print(f"      Resolution       = {RESOLUTION} (N = L_ref/Δx)")
    print(f"      Lattice velocity = {LATTICE_VELOCITY} (u_lu)")
    print()
    print(" §3. Lattice Parameters (Auto-calculated):")
    print(f"      Δx      = {DX_PHYS:.6e} m")
    print(f"      Δt      = {DT_PHYS:.6e} s")
    print(f"      ν_lu    = {NU_LU:.6f}")
    print(f"      τ       = {TAU:.6f}")
    print(f"      ω       = {OMEGA_LBM:.6f}")
    print(f"      Ma_lu   = {MA_LATTICE:.4f}")
    print()
    print(" §4. Domain:")
    print(f"      Grid    = {Nx} × {Ny} × {Nz} = {Nx*Ny*Nz:,} cells")
    print(f"      Size    = {Nx*DX_PHYS:.4f} × {Ny*DX_PHYS:.4f} × {Nz*DX_PHYS:.4f} m")
    print()
    print(" §5. Time:")
    max_steps = simulation['time']['max_steps']
    print(f"      Max steps      = {max_steps:,}")
    print(f"      Physical time  = {max_steps * DT_PHYS:.4f} s")
    print(f"      Convective t*  = {max_steps * DT_PHYS * U_REF / L_REF:.1f}")
    print()
    print(" §6. Modules:")
    print(f"      ALM             = {ALM_ENABLED}")
    print(f"      Force calc      = {force_calculation['enabled']}")
    print()
    print("=" * 70)
    
    # Stability indicator
    if TAU <= 0.5:
        print("❌ UNSTABLE: τ ≤ 0.5")
    elif TAU < 0.52:
        print(f"⚠️  MARGINAL: τ = {TAU:.4f}")
    else:
        print(f"✅ STABLE: τ = {TAU:.4f}, Ma_lu = {MA_LATTICE:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    print_summary()