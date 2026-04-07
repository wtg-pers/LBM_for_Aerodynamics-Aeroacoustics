"""
Sphere Drag Validation — Re=100, 2-Level MLG

Validates fine-level obstacle support by comparing drag coefficient:
  - MLG enabled:  L0 at D/dx=20, L1 at D/dx=40  (sphere in fine region)
  - MLG disabled: Single grid at D/dx=20          (set mlg.enabled = False)

Physical Setup:
  - Steady, axisymmetric flow past a sphere at Re=100
  - Uniform inlet velocity, no-slip walls (sides), sponge outlet
  - Blockage ratio: π(D/2)² / (Ny×Nz) ≈ 1.2%

Reference Values (Re=100):
  - Johnson & Patel 1999:  Cd ≈ 1.09
  - Geier et al. 2015:    Cd ≈ 1.09 (Cumulant LBM)
  - Clift, Grace, Weber:  Cd ≈ 1.087

Domain Layout:
  ──────────────────────────────────────────────────────
  │          ┌──────────────────────┐                  │
  │  inlet   │   Fine L1 region     │          outlet  │
  │ ───→     │      ● sphere        │           ───→   │
  │  x=0     │                      │          x=300   │
  │          └──────────────────────┘                  │
  ──────────────────────────────────────────────────────
  Sphere: D=20, center=(100, 80, 80)  [L0 lattice units]
  Fine:   x∈[60,200], y∈[40,120], z∈[40,120]
          → 2D upstream, 5D downstream, ±2D lateral

Usage:
    python main.py --config configs/mlg_sphere_drag_Re100.py
    python main.py --config configs/mlg_sphere_drag_Re100.py --verbose

Author: LBM Development Team
Date: 2026-04
"""

import numpy as np

# =============================================================================
# §1. PHYSICAL PARAMETERS
# =============================================================================
RE = 100
D = 10                          # Sphere diameter  [lattice units]  (coarse test)
U_INLET = 0.05                  # Inlet velocity   [Δx/Δt]
RHO = 1.0                       # Reference density [dimensionless]

# Domain dimensions
Nx = 150                        # 15D streamwise
Ny = 80                         # 8D  cross-stream (y)
Nz = 80                         # 8D  cross-stream (z)

# Sphere position: 5D from inlet, centered in y-z
CENTER_X = 5 * D                # = 50
CENTER_Y = Ny // 2              # = 40
CENTER_Z = Nz // 2              # = 40

RESOLUTION = D                  # Characteristic length = sphere diameter

# =============================================================================
# §2. DERIVED LATTICE PARAMETERS
#
#   Re = U·D / ν  →  ν = U·D / Re
#   τ = 0.5 + 3ν  (BGK relaxation time)
#   Ma = U / cs   (cs = 1/√3 for D3Q27)
# =============================================================================
NU_LU = U_INLET * D / RE       # = 0.01  [Δx²/Δt]
TAU = 0.5 + 3.0 * NU_LU       # = 0.53  [Δt]

CS = 1.0 / np.sqrt(3.0)
MA = U_INLET / CS              # ≈ 0.087

assert TAU > 0.5, f"UNSTABLE: τ = {TAU:.4f}"
assert MA < 0.3, f"COMPRESSIBILITY: Ma = {MA:.3f} > 0.3"

# Sphere frontal area: A = π(D/2)² = πD²/4
# ForceManager computes A_ref = char_length × span_length
# → span_length = πD/4 to yield correct A_ref for sphere
A_FRONTAL = np.pi * (D / 2) ** 2   # ≈ 314.16  [Δx²]
SPAN_FOR_SPHERE = A_FRONTAL / D     # = πD/4 ≈ 15.71  [Δx]

BLOCKAGE = A_FRONTAL / (Ny * Nz) * 100  # ≈ 1.23%

print(f"  [Sphere Drag] Re={RE}, D={D}, τ={TAU:.4f}, Ma={MA:.4f}")
print(f"  ν = {NU_LU:.6f}, U = {U_INLET}")
print(f"  Sphere center: ({CENTER_X}, {CENTER_Y}, {CENTER_Z})")
print(f"  Frontal area: {A_FRONTAL:.2f}, Blockage: {BLOCKAGE:.2f}%")

# =============================================================================
# §3. MLG CONFIGURATION (2-Level)
#
#   Level 0: full domain, D/dx = 20
#   Level 1: fine region around sphere, D/dx = 40 (2× resolution)
#
#   Fine region margins:
#     upstream:   2D before sphere center  (captures stagnation region)
#     downstream: 5D after sphere center   (captures near-wake)
#     lateral:    ±2D from sphere center   (captures separation zone)
# =============================================================================
OVERLAP_WIDTH = 2
INTERP_SCHEME = "cubic"
FILTER_LEVEL = 1

FINE_X_MIN = CENTER_X - 2 * D      # = 30
FINE_X_MAX = CENTER_X + 5 * D      # = 100
FINE_Y_MIN = CENTER_Y - 2 * D      # = 20
FINE_Y_MAX = CENTER_Y + 2 * D      # = 60
FINE_Z_MIN = CENTER_Z - 2 * D      # = 20
FINE_Z_MAX = CENTER_Z + 2 * D      # = 60

# =============================================================================
# §4. SIMULATION SETTINGS
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "precision": "float32",         # "float32" (fast test) or "float64" (production)
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
        "L_ref_lu": float(D),
        "initial_flow_velocity": [U_INLET, 0.0, 0.0],
    },

    "time": {
        "max_steps": 20000,
        "output_interval": 1000,
        "checkpoint_interval": 10000,
        "probe_interval": 10,           # Force sampling interval
    },
}

# =============================================================================
# §5. BOUNDARY CONDITIONS
#
#   Physical setup for external flow past a sphere:
#     - Inlet (xmin):  Uniform velocity  U = (U_inlet, 0, 0)
#     - Outlet (xmax): Sponge layer (non-reflecting, absorbs wake structures)
#     - Walls (y,z):   No-slip walls (blockage ~1.2%, acceptable for Re=100)
# =============================================================================
boundaries = {
    "inlet":      {"location": "xmin", "method": "regularized_inlet",
                   "velocity": U_INLET, "rho": RHO},
    "outlet":     {"location": "xmax", "method": "sponge",
                   "velocity": [U_INLET, 0, 0], "density": RHO,
                   "thickness": 10, "sigma_max": 0.1},
    "wall_south": {"location": "ymin", "method": "regularized_wall"},
    "wall_north": {"location": "ymax", "method": "regularized_wall"},
    "wall_zmin":  {"location": "zmin", "method": "regularized_wall"},
    "wall_zmax":  {"location": "zmax", "method": "regularized_wall"},
}

# =============================================================================
# §6. INTERNAL GEOMETRY — Sphere
#
#   Sphere at domain center (y,z), 5D from inlet (x).
#   L0: D/dx = 20  (20 grid points per diameter)
#   L1: D/dx = 40  (40 grid points per diameter, auto-generated by MLG)
# =============================================================================
internal_geometry = {
    "sphere": {
        "enabled": True,
        "center": (CENTER_X, CENTER_Y, CENTER_Z),
        "radius": D // 2,
    },
}

# =============================================================================
# §7. MLG — 2 Levels
#
#   To run single-grid comparison: set "enabled": False
# =============================================================================
mlg = {
    "enabled": True,
    "num_levels": 2,
    "overlap_width": OVERLAP_WIDTH,
    "interpolation": INTERP_SCHEME,
    "filter_level": FILTER_LEVEL,
    "levels": [
        {},  # Level 0 = full domain

        # Level 1: fine region around sphere
        {"region": {"x_min": FINE_X_MIN, "x_max": FINE_X_MAX,
                    "y_min": FINE_Y_MIN, "y_max": FINE_Y_MAX,
                    "z_min": FINE_Z_MIN, "z_max": FINE_Z_MAX}},
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
# §9. CONVERGENCE DETECTION (Cauchy criterion on Cd)
#
#   Re=100 is steady axisymmetric flow → Cd converges to constant value.
#   Cauchy criterion: |μ_new − μ_old| / |μ_old| < ε_Cd
#
#   Convective time scale: T_conv = D / U = 20 / 0.05 = 400 steps
#   Window = 50 × T_conv / probe_interval = 50 × 400 / 10 = 2000 samples
# =============================================================================
convergence = {
    "enabled": True,

    "cauchy": {
        "window_size": "auto",
        "epsilon": 1e-5,            # Kinetic energy threshold
        "Cd_epsilon": 1e-3,         # Drag coefficient threshold

        "n_required": 3,            # 3 consecutive passes for robust detection

        "auto_window": {
            "time_coverage": 50.0,  # Window spans 50 × T_conv
            "min_samples": 200,
        },
    },

    "on_converged": "checkpoint_and_stop",
    "on_max_steps": "warn",
    "on_diverged": "stop_with_checkpoint",
}

# =============================================================================
# §10. FORCE CALCULATION
#
#   Momentum Exchange Method (MEM) for sphere drag/lift:
#     Cd = Fx / (0.5 · ρ · U² · A_frontal)
#     A_frontal = π(D/2)²  (sphere frontal area)
#
#   In ForceManager: A_ref = char_length × span_length
#   → char_length = D, span_length = πD/4
#   → A_ref = D × πD/4 = πD²/4 = A_frontal  ✓
# =============================================================================
force_calculation = {
    "enabled": True,
    "interval": 10,
    "start_step": 500,              # Skip initial transient (~1.25 T_conv)

    "reference": {
        "rho": RHO,
        "velocity": U_INLET,
        "char_length": D,
        "span_length": SPAN_FOR_SPHERE,     # πD/4 ≈ 15.71  → A = πD²/4
    },

    "log": {
        "enabled": True,
        "filename": "force_history",
    },
}

# =============================================================================
# §11. OUTPUT
# =============================================================================
_folder = f"result_sphere_D{D}_Re{RE}_mlg2"

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
