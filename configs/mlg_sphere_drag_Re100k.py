"""
Sphere Drag — Re=100,000, 3-Level MLG, Cumulant Collision

High-Re turbulent flow past a sphere.
Cumulant collision is REQUIRED (BGK unstable above Re~8,000).

Resolution:
  L0: D/dx = 20  (coarse, far-field)
  L1: D/dx = 40  (near-wake)
  L2: D/dx = 80  (sphere surface, force measurement)

Reference (Geier et al. 2015, Fig. 11):
  Re=100,000: Cd ≈ 0.4–0.5 (turbulent, unsteady)
  Geier used 5-level with D=256 at finest — our D=80 is coarser,
  so this is primarily a stability/code validation test.

Stability:
  τ_0 = 0.50003 (ω → 2), τ_2 = 0.50012 at finest level.
  Cumulant handles this; BGK would immediately diverge.

Usage:
    python main.py --config configs/mlg_sphere_drag_Re100k.py
    python main.py --config configs/mlg_sphere_drag_Re100k.py --verbose

Author: LBM Development Team
Date: 2026-04
"""

import numpy as np

# =============================================================================
# §1. PHYSICAL PARAMETERS
# =============================================================================
RE = 100000
D = 20                          # Sphere diameter  [lattice units]
U_INLET = 0.05                  # Inlet velocity   [Δx/Δt]
RHO = 1.0                       # Reference density [dimensionless]

# Domain: 20D × 8D × 8D
Nx = 300
Ny = 160
Nz = 160

# Sphere position: 3D from inlet, centered in y-z
CENTER_X = 3 * D                # = 60
CENTER_Y = Ny // 2              # = 80
CENTER_Z = Nz // 2              # = 80

RESOLUTION = D

# =============================================================================
# §2. DERIVED LATTICE PARAMETERS
#
#   ν = U·D / Re = 1e-5   (very small → τ close to 0.5)
#   τ = 0.5 + 3ν = 0.50003
#   Ma = U / cs ≈ 0.087
#
#   Level τ values (convective scaling):
#     L0: τ = 0.50003  (ω = 1.99988)
#     L1: τ = 0.50006  (ω = 1.99976)
#     L2: τ = 0.50012  (ω = 1.99952)
# =============================================================================
NU_LU = U_INLET * D / RE       # = 1e-5  [Δx²/Δt]
TAU = 0.5 + 3.0 * NU_LU       # = 0.50003

CS = 1.0 / np.sqrt(3.0)
MA = U_INLET / CS

assert TAU > 0.5, f"UNSTABLE: τ = {TAU:.8f}"
assert MA < 0.3, f"COMPRESSIBILITY: Ma = {MA:.3f} > 0.3"

# Sphere frontal area
A_FRONTAL = np.pi * (D / 2) ** 2
SPAN_FOR_SPHERE = A_FRONTAL / D     # πD/4
BLOCKAGE = A_FRONTAL / (Ny * Nz) * 100

print(f"  [Sphere Re=100k] D={D}, τ={TAU:.8f}, Ma={MA:.4f}")
print(f"  ν = {NU_LU:.2e}, τ-0.5 = {TAU-0.5:.2e}")
print(f"  Sphere center: ({CENTER_X}, {CENTER_Y}, {CENTER_Z})")
print(f"  Blockage: {BLOCKAGE:.2f}%")

# =============================================================================
# §3. MLG CONFIGURATION (3-Level)
#
#   Level 0: full domain, D/dx=20
#   Level 1: near-wake region, D/dx=40
#   Level 2: sphere surface region, D/dx=80
#
#   L1: 2D upstream, 8D downstream, ±2D lateral (captures turbulent wake)
#   L2: 1D upstream, 4D downstream, ±1D lateral (sphere + separation)
# =============================================================================
OVERLAP_WIDTH = 2
INTERP_SCHEME = "cubic"
FILTER_LEVEL = 1

# Level 1 region (compact: 1D up, 3D down, ±1D lateral)
L1_X_MIN = CENTER_X - 1 * D        # = 40
L1_X_MAX = CENTER_X + 5 * D        # = 120
L1_Y_MIN = CENTER_Y - 1 * D        # = 60
L1_Y_MAX = CENTER_Y + 1 * D        # = 100
L1_Z_MIN = CENTER_Z - 1 * D        # = 60
L1_Z_MAX = CENTER_Z + 1 * D        # = 100

# Level 2 region (tight: 0.6D up, 1.2D down, ±0.6D lateral)
L2_M = int(0.6 * D)                # = 12
L2_X_MIN = CENTER_X - L2_M         # = 48
L2_X_MAX = CENTER_X + int(1.2 * D) # = 84
L2_Y_MIN = CENTER_Y - L2_M         # = 68
L2_Y_MAX = CENTER_Y + L2_M         # = 92
L2_Z_MIN = CENTER_Z - L2_M         # = 68
L2_Z_MAX = CENTER_Z + L2_M         # = 92

# =============================================================================
# §4. SIMULATION SETTINGS
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "precision": "float32",
    "dimension": 3,
    "lattice_model": "D3Q27",
    "collision_model": "cumulant",      # REQUIRED for Re=100k (BGK unstable)

    "domain": {"Nx": Nx, "Ny": Ny, "Nz": Nz},

    "physics": {
        "Re": RE,
        "tau": TAU,
        "omega": 1.0 / TAU,
        "nu_lu": NU_LU,
        "u_ref_lu": U_INLET,
        "L_ref_lu": float(D),
        "initial_flow_velocity": [U_INLET, 0.0, 0.0],
    },

    "time": {
        "max_steps": 50000,
        "output_interval": 500,
        "checkpoint_interval": 10000,
    },
}

# =============================================================================
# §5. BOUNDARY CONDITIONS
#
#   Far-field: regularized_inlet on all sides (uniform freestream)
#   Outlet: sponge layer (absorbs turbulent wake structures)
# =============================================================================
boundaries = {
    "inlet":      {"location": "xmin", "method": "regularized_inlet",
                   "velocity": U_INLET, "rho": RHO},
    "outlet":     {"location": "xmax", "method": "sponge",
                   "velocity": [U_INLET, 0, 0], "density": RHO,
                   "thickness": 15, "sigma_max": 0.1},
    "wall_south": {"location": "ymin", "method": "regularized_inlet",
                   "velocity": U_INLET, "rho": RHO},
    "wall_north": {"location": "ymax", "method": "regularized_inlet",
                   "velocity": U_INLET, "rho": RHO},
    "wall_zmin":  {"location": "zmin", "method": "regularized_inlet",
                   "velocity": U_INLET, "rho": RHO},
    "wall_zmax":  {"location": "zmax", "method": "regularized_inlet",
                   "velocity": U_INLET, "rho": RHO},
}

# =============================================================================
# §6. INTERNAL GEOMETRY — Sphere
# =============================================================================
internal_geometry = {
    "sphere": {
        "enabled": True,
        "center": (CENTER_X, CENTER_Y, CENTER_Z),
        "radius": D // 2,
    },
}

# =============================================================================
# §7. MLG — 3 Levels
# =============================================================================
mlg = {
    "enabled": True,
    "num_levels": 3,
    "overlap_width": OVERLAP_WIDTH,
    "interpolation": INTERP_SCHEME,
    "filter_level": FILTER_LEVEL,
    "levels": [
        {},  # Level 0: D=20

        # Level 1: D=40, near-wake
        {"region": {"x_min": L1_X_MIN, "x_max": L1_X_MAX,
                    "y_min": L1_Y_MIN, "y_max": L1_Y_MAX,
                    "z_min": L1_Z_MIN, "z_max": L1_Z_MAX}},

        # Level 2: D=80, sphere surface
        {"region": {"x_min": L2_X_MIN, "x_max": L2_X_MAX,
                    "y_min": L2_Y_MIN, "y_max": L2_Y_MAX,
                    "z_min": L2_Z_MIN, "z_max": L2_Z_MAX}},
    ],
}

# =============================================================================
# §8. CONSERVATION MONITOR
# =============================================================================
conservation = {
    "enabled": True,
    "check_interval": 1000,
    "verbose": 0,
    "log_to_csv": True,
}

# =============================================================================
# §9. CONVERGENCE DETECTION
#
#   Re=100,000 is turbulent → Cd oscillates, no true steady state.
#   Cauchy criterion on Cd mean: detect when mean Cd stabilizes.
#   Use larger epsilon since periodic fluctuations exist.
#
#   T_conv = D / U = 400 steps
# =============================================================================
convergence = {
    "enabled": True,

    "cauchy": {
        "window_size": "auto",
        "epsilon": 1e-4,            # Kinetic energy (relaxed for turbulent)
        "Cd_epsilon": 5e-3,         # Cd mean drift threshold (relaxed)

        "n_required": 3,

        "auto_window": {
            "time_coverage": 50.0,
            "min_samples": 200,
        },
    },

    "on_converged": "checkpoint_and_stop",
    "on_max_steps": "warn",
    "on_diverged": "stop_with_checkpoint",
}

# =============================================================================
# §10. FORCE CALCULATION
# =============================================================================
force_calculation = {
    "enabled": True,
    "interval": 10,
    "start_step": 1000,             # Skip longer transient for high Re

    "reference": {
        "rho": RHO,
        "velocity": U_INLET,
        "char_length": D,
        "span_length": SPAN_FOR_SPHERE,
    },

    "log": {
        "enabled": True,
        "filename": "force_history",
    },
}

# =============================================================================
# §11. OUTPUT
# =============================================================================
_mlg_tag = f"mlg{mlg['num_levels']}" if mlg["enabled"] else "single"
_folder = f"result_sphere_D{D}_Re{RE}_{_mlg_tag}"

output = {
    "output_dir": f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir": f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {
        "enabled": True,
        "precision": "float32",
        "variables": ["density", "pressure", "velocity",
                      "velocity_magnitude", "solid_mask"],
    },
    "checkpoint": {
        "enabled": True,
        "keep_last_n": 3,
    },
}

# =============================================================================
# §12. FINAL CONFIG
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
