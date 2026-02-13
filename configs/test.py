# =============================================================================
# LBM Solver — Small Hover Rotor Test Configuration
# =============================================================================
#
# Simple 2-blade hover rotor for AL model validation.
#
# Physical Setup:
#   - Rotor: R = 0.125 m, D = 0.25 m, 2 blades
#   - Blade: NACA0012, c = 0.025 m (constant), pitch = 10°
#   - Rotation: Z-axis, CW (viewed from +Z) → ω < 0
#   - Wake direction: -Z (downwash toward ground)
#   - Ground: zmin = wall
#   - Far-field: quiescent air (u = 0)
#
# Usage:
#   python main.py --config configs/hover_test_config.py
#   python main.py --config configs/hover_test_config.py --max-steps 5000
#
# =============================================================================

import numpy as np

# =============================================================================
# §1. Physical Parameters
# =============================================================================
# Rotor geometry
R_PHYS     = 0.125          # [m]      Rotor radius
D_PHYS     = 2 * R_PHYS     # [m]      Diameter = 0.25 m
CHORD      = 0.025          # [m]      Constant chord
PITCH      = 10.0           # [deg]    Collective pitch (blade twist = -10° in code)
N_BLADES   = 2              # [-]      Number of blades

ROOT_CUT   = 0.20           # [-]  20%
R_ROOT     = R_PHYS * ROOT_CUT  # [m] = 0.025 m (inactive region ends here)

# Operating conditions
RPM        = 3000           # [RPM]    Rotation speed
OMEGA      = RPM * 2 * np.pi / 60  # [rad/s] = 314.16 rad/s
V_TIP      = OMEGA * R_PHYS # [m/s]    Tip speed = 39.27 m/s

# Air properties
NU_AIR     = 1.512e-5       # [m²/s]   Kinematic viscosity (20°C)
RHO_AIR    = 1.205          # [kg/m³]  Air density

# =============================================================================
# §2. Grid & Lattice Scaling
# =============================================================================
RESOLUTION = 40             # [-]      D/Δx
DX_PHYS    = D_PHYS / RESOLUTION  # [m/lu] = 0.00625 m = 6.25 mm

# Lattice Mach at tip (keep < 0.1)
MA_TIP     = 0.05           # [-]
U_TIP_LU   = MA_TIP         # [lu/lt]
DT_PHYS    = U_TIP_LU * DX_PHYS / V_TIP  # [s/lt] ≈ 7.96e-6 s

# Relaxation time
TAU        = 0.55           # [-]
NU_LU      = (TAU - 0.5) / 3  # [lu²/lt]

# Derived lattice quantities
D_LU       = RESOLUTION     # [lu]
R_LU       = D_LU / 2       # [lu] = 20
OMEGA_LU   = OMEGA * DT_PHYS  # [rad/lt] ≈ 0.0025
STEPS_PER_REV = int(2 * np.pi / OMEGA_LU)  # ≈ 2513

# =============================================================================
# §3. Domain Configuration
# =============================================================================
# ⚠️ 20D × 20D × 20D = 800³ = 512M cells (VERY LARGE!)
# For testing, use smaller domain factor (e.g., 5)
DOMAIN_FACTOR = 4           # [-]  Change to 20 for full simulation

Nx = DOMAIN_FACTOR * RESOLUTION  # [lu] 5D = 200
Ny = DOMAIN_FACTOR * RESOLUTION  # [lu]
Nz = 3 * RESOLUTION  # [lu]

# Hub position in LATTICE UNITS
HUB_X_LU = Nx // 2          # [lu] centered
HUB_Y_LU = Ny // 2          # [lu] centered
HUB_Z_LU = int(Nz * 0.3)    # [lu] 30% from ground

# Hub position in PHYSICAL UNITS (for Rotor factory)
HUB_X_PHYS = HUB_X_LU * DX_PHYS  # [m]
HUB_Y_PHYS = HUB_Y_LU * DX_PHYS  # [m]
HUB_Z_PHYS = HUB_Z_LU * DX_PHYS  # [m]

# =============================================================================
# §4. Physics (for LBM)
# =============================================================================
CHAR_LENGTH = CHORD / DX_PHYS  # [lu] chord in lattice units
RE_GRID = U_TIP_LU * CHAR_LENGTH / NU_LU

# =============================================================================
# §5. Simulation Config
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "dimension": 3,
    "lattice_model": "D3Q27",
    "domain": {"Nx": Nx, "Ny": Ny, "Nz": Nz},
    "physics": {
        "Re": RE_GRID,
        "u_init": 0.0,                    # Quiescent air
        "u_ref": U_TIP_LU,
        "characteristic_length": CHAR_LENGTH,
    },
    "time": {
        "max_steps": STEPS_PER_REV * 40,  # 10 revolutions
        "output_interval": STEPS_PER_REV // 360,
        "checkpoint_interval": STEPS_PER_REV * 2,
        "probe_interval": 10,
    },
}

# =============================================================================
# §6. Boundary Conditions
# =============================================================================
# Z-axis rotation, wake going -Z (down)
#   zmin: Ground (wall)
#   zmax: Air enters from above (far-field)
#   x/y:  Far-field quiescent

boundaries = {
    # Ground wall
    "ground": {
        "location": "zmin",
        "method": "regularized_wall",
    },
    # Far-field boundaries (quiescent air, u=0)
    "top": {
        "location": "zmax",
        "method": "equilibrium",
        "velocity": 0.0,
        "density": 1.0,
    },
    "xmin": {
        "location": "xmin",
        "method": "equilibrium",
        "velocity": 0.0,
        "density": 1.0,
    },
    "xmax": {
        "location": "xmax",
        "method": "equilibrium",
        "velocity": 0.0,
        "density": 1.0,
    },
    "ymin": {
        "location": "ymin",
        "method": "equilibrium",
        "velocity": 0.0,
        "density": 1.0,
    },
    "ymax": {
        "location": "ymax",
        "method": "equilibrium",
        "velocity": 0.0,
        "density": 1.0,
    },
}

# =============================================================================
# §7. Internal Geometry
# =============================================================================
internal_geometry = {}

# =============================================================================
# §8. Actuator Line Model
# =============================================================================
# Rotation direction:
#   - hawt_z: rotation about Z-axis, blades in XY-plane
#   - CW when viewed from +Z → negative omega (right-hand rule)
#   - Wake goes -Z (downward)
#
# Blade twist convention:
#   - Physical pitch = +10° (leading edge up)
#   - In blade.py: twist is angle from rotor plane
#   - For pitch = +10°, use twist = -10° in sections

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

# =============================================================================
# §9. Conservation & Convergence
# =============================================================================
conservation = {
    "enabled": True,
    "check_interval": STEPS_PER_REV // 360,
    "verbose": 0,
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
# §10. Force Calculation
# =============================================================================
force_calculation = {"enabled": False}

# =============================================================================
# §11. Output
# =============================================================================
_folder = f"result_hover_test_D{RESOLUTION}_factor{DOMAIN_FACTOR}"

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
# §13. Summary
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Small Hover Rotor Test Configuration")
    print("=" * 65)
    print(f"  Rotor:         R = {R_PHYS*1000:.1f} mm, D = {D_PHYS*1000:.1f} mm")
    print(f"  Blades:        {N_BLADES} × NACA0012, c = {CHORD*1000:.1f} mm")
    print(f"  Pitch:         {PITCH}° (twist = -{PITCH}° in blade.py)")
    print(f"  RPM:           {RPM} → ω = {OMEGA:.2f} rad/s")
    print(f"  Rotation:      Z-axis, CW (ω < 0)")
    print(f"  V_tip:         {V_TIP:.2f} m/s")
    print("-" * 65)
    print(f"  Domain:        {Nx} × {Ny} × {Nz} = {Nx*Ny*Nz:,} cells")
    print(f"                 ({DOMAIN_FACTOR}D × {DOMAIN_FACTOR}D × {DOMAIN_FACTOR}D)")
    print(f"  Resolution:    D/Δx = {RESOLUTION}")
    print(f"  Δx = {DX_PHYS*1000:.3f} mm,  Δt = {DT_PHYS*1e6:.3f} μs")
    print(f"  τ = {TAU},  ν_lu = {NU_LU:.5f}")
    print(f"  Ma_tip = {MA_TIP},  ω_lu = {OMEGA_LU:.6f} rad/lt")
    print(f"  Steps/rev ≈ {STEPS_PER_REV}")
    print("-" * 65)
    print(f"  Hub (lu):      ({HUB_X_LU}, {HUB_Y_LU}, {HUB_Z_LU})")
    print(f"  Hub (phys):    ({HUB_X_PHYS*1000:.1f}, {HUB_Y_PHYS*1000:.1f}, {HUB_Z_PHYS*1000:.1f}) mm")
    print(f"  Rotor R (lu):  {R_LU:.1f}")
    print("-" * 65)
    print(f"  Ground:        zmin (wall)")
    print(f"  Wake:          -Z (downward)")
    print("=" * 65)
    
    if DOMAIN_FACTOR < 20:
        print(f"\n⚠️  DOMAIN_FACTOR = {DOMAIN_FACTOR} (test mode)")
        print(f"   Change to 20 for full 20D×20D×20D domain")
        print(f"   Full domain: {20*RESOLUTION}³ = {(20*RESOLUTION)**3:,} cells")