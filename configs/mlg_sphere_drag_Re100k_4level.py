"""
Sphere Drag — Re=100,000, 4-Level MLG, Cumulant Collision

High-Re turbulent flow past a sphere with 4-level grid refinement.
Finest level resolves D=128Δx for accurate boundary layer capture.

Resolution:
  L0: D/dx = 16  (far-field)
  L1: D/dx = 32  (wake)
  L2: D/dx = 64  (separation zone)
  L3: D/dx = 128 (sphere surface, force measurement)

Reference (Geier et al. 2015):
  Re=100,000: Cd ≈ 0.4–0.5
  Geier used 5-level D=256 at finest with ~27.6M nodes.
  This config: 4-level D=128 at finest with ~16M nodes.

Stability:
  τ_0 = 0.500024 — Cumulant required (BGK unstable above Re~8,000).
  Level τ values:
    L0: 0.500024  L1: 0.500048  L2: 0.500096  L3: 0.500192

Usage:
    python main.py --config configs/mlg_sphere_drag_Re100k_4level.py
    python main.py --config configs/mlg_sphere_drag_Re100k_4level.py --gpu 1

Author: LBM Development Team
Date: 2026-04
"""

import numpy as np

# =============================================================================
# §1. PHYSICAL PARAMETERS
# =============================================================================
RE = 100000
D = 16                          # Sphere diameter  [lattice units]
U_INLET = 0.05                  # Inlet velocity   [Δx/Δt]
RHO = 1.0

# Domain: 15D × 10D × 10D
Nx = 15 * D                     # = 240
Ny = 160
Nz = 160

# Sphere position: 3D from inlet, centered in y-z
CENTER_X = 3 * D                # = 48
CENTER_Y = Ny // 2              # = 80
CENTER_Z = Nz // 2              # = 80

RESOLUTION = D

# =============================================================================
# §2. DERIVED LATTICE PARAMETERS
#
#   ν = U·D / Re = 8e-6
#   τ = 0.5 + 3ν = 0.500024
# =============================================================================
NU_LU = U_INLET * D / RE
TAU = 0.5 + 3.0 * NU_LU

CS = 1.0 / np.sqrt(3.0)
MA = U_INLET / CS

assert TAU > 0.5, f"UNSTABLE: τ = {TAU:.8f}"
assert MA < 0.3, f"COMPRESSIBILITY: Ma = {MA:.3f} > 0.3"

# Sphere frontal area
A_FRONTAL = np.pi * (D / 2) ** 2
SPAN_FOR_SPHERE = A_FRONTAL / D
BLOCKAGE = A_FRONTAL / (Ny * Nz) * 100

print(f"  [Sphere Re=100k, 4-Level] D={D}, τ={TAU:.8f}, Ma={MA:.4f}")
print(f"  ν = {NU_LU:.2e}, τ-0.5 = {TAU-0.5:.2e}")
print(f"  Sphere center: ({CENTER_X}, {CENTER_Y}, {CENTER_Z})")
print(f"  Blockage: {BLOCKAGE:.2f}%")

# =============================================================================
# §3. MLG CONFIGURATION (4-Level)
#
#   L0: full domain, D=16
#   L1: near-wake,   D=32   (±1D lateral, 1D up / 3D down)
#   L2: separation,  D=64   (±0.75D lateral, 0.75D up / 1.5D down)
#   L3: surface,     D=128  (±0.6D lateral, 0.6D up / 1D down)
# =============================================================================
OVERLAP_WIDTH = 2
INTERP_SCHEME = "cubic"
FILTER_LEVEL = 1

# Level 1: wake capture (1.5D up, 7D down, ±1.5D lateral)
L1_X_MIN = CENTER_X - int(1.5 * D)     # = 24
L1_X_MAX = CENTER_X + 7 * D            # = 160  (7D downstream)
L1_Y_MIN = CENTER_Y - int(1.5 * D)     # = 56
L1_Y_MAX = CENTER_Y + int(1.5 * D)     # = 104
L1_Z_MIN = CENTER_Z - int(1.5 * D)     # = 56
L1_Z_MAX = CENTER_Z + int(1.5 * D)     # = 104

# Level 2: separation zone (1D up, 4D down, ±1D lateral)
L2_X_MIN = CENTER_X - 1 * D            # = 32
L2_X_MAX = CENTER_X + 4 * D            # = 112  (4D downstream)
L2_Y_MIN = CENTER_Y - 1 * D            # = 64
L2_Y_MAX = CENTER_Y + 1 * D            # = 96
L2_Z_MIN = CENTER_Z - 1 * D            # = 64
L2_Z_MAX = CENTER_Z + 1 * D            # = 96

# Level 3: sphere surface + near separation (1D up, 2D down, ±1D lateral)
L3_X_MIN = CENTER_X - 1 * D            # = 32
L3_X_MAX = CENTER_X + 2 * D            # = 80  (2D downstream)
L3_Y_MIN = CENTER_Y - 1 * D            # = 64
L3_Y_MAX = CENTER_Y + 1 * D            # = 96
L3_Z_MIN = CENTER_Z - 1 * D            # = 64
L3_Z_MAX = CENTER_Z + 1 * D            # = 96

# =============================================================================
# §4. SIMULATION SETTINGS
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "device_id": 0,                     # GPU node (override with --gpu N)
    "precision": "float32",
    "dimension": 3,
    "lattice_model": "D3Q27",
    "collision_model": "cumulant",

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
        "output_interval": 1000,
        "checkpoint_interval": 10000,
    },
}

# =============================================================================
# §5. BOUNDARY CONDITIONS
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
# §7. MLG — 4 Levels
# =============================================================================
mlg = {
    "enabled": True,
    "num_levels": 4,
    "overlap_width": OVERLAP_WIDTH,
    "interpolation": INTERP_SCHEME,
    "filter_level": FILTER_LEVEL,
    "levels": [
        {},  # Level 0: D=16

        # Level 1: D=32, wake
        {"region": {"x_min": L1_X_MIN, "x_max": L1_X_MAX,
                    "y_min": L1_Y_MIN, "y_max": L1_Y_MAX,
                    "z_min": L1_Z_MIN, "z_max": L1_Z_MAX}},

        # Level 2: D=64, separation
        {"region": {"x_min": L2_X_MIN, "x_max": L2_X_MAX,
                    "y_min": L2_Y_MIN, "y_max": L2_Y_MAX,
                    "z_min": L2_Z_MIN, "z_max": L2_Z_MAX}},

        # Level 3: D=128, sphere surface
        {"region": {"x_min": L3_X_MIN, "x_max": L3_X_MAX,
                    "y_min": L3_Y_MIN, "y_max": L3_Y_MAX,
                    "z_min": L3_Z_MIN, "z_max": L3_Z_MAX}},
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
# =============================================================================
convergence = {
    "enabled": True,

    "cauchy": {
        "window_size": "auto",
        "epsilon": 1e-4,
        "Cd_epsilon": 5e-3,
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
    "start_step": 1000,

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
        "variables": ["density", "velocity", "solid_mask"],
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
