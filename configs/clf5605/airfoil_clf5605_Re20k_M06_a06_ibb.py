"""
CLF5605 Airfoil — 2D D2Q9 Cumulant + 4-level MLG, α=+6°, IBB wall

IBB variant of the α=6 MLG case. Pair with the HWBB sibling
`airfoil_clf5605_Re20k_M06_a06_hwbb.py` to compare Bouzidi
interpolated bounce-back vs. the half-way bounce-back baseline.
Output directories carry a `_ibb` / `_hwbb` suffix so both runs can
coexist on disk for direct Cp / Cd / Cl comparison.

Usage:
    python main.py --config configs/airfoil_clf5605_Re20k_M06_a06_ibb.py
"""

import os
import numpy as np

# =============================================================================
# §1. USER KNOBS
# =============================================================================

WALL_BC = "ibb"        # this file. Sibling _hwbb.py sets "hwbb".

AOA_DEG = 6.0
MACH    = 0.6
RE      = 20_000.0

RHO_PHYS = 0.015
MU_PHYS  = 1.13e-5
A_SOUND  = 240.0
NU_PHYS  = MU_PHYS / RHO_PHYS
U_INF_PHYS = MACH * A_SOUND
CHORD_PHYS = RE * NU_PHYS / U_INF_PHYS

NUM_LEVELS = 4                     # L0 + L1 + L2 + L3
CPC_L0     = 64                   # cells/chord on L0 (coarsest)
REFINE     = 2                     # fixed refinement ratio
CPC_FINEST = CPC_L0 * (REFINE ** (NUM_LEVELS - 1))  # = 1024 for 4 levels

U_MAX_LU = 0.05                    # lattice u_max on L0 (coarsest)

# Domain extents (chord multiples), used on L0
# Matches Tohoku Univ. Mars wind tunnel test section (600 mm x 150 mm for c=50 mm,
# NASA/TM-20240004230 Fig. 4). Airfoil streamwise placement is user-chosen:
# upstream 5c + chord 1c + downstream 6c = 12c total.  Wall inclination (1.3 deg)
# ignored.
UPSTREAM_C   = 5.0
DOWNSTREAM_C = 6.0
LATERAL_C    = 1.5               # half-height; tunnel height 150 mm = 3c

# L1 region (chord multiples around airfoil, still in L0 units)
L1_UPSTREAM = 2.0
L1_DOWN     = 5.0                # keep ~1c margin to outlet sponge
L1_LATERAL  = 1.3                # inside LATERAL_C with buffer to wall

# L2 region (tight around airfoil, in L0 units)
L2_UPSTREAM = 0.8
L2_DOWN     = 1.5
L2_LATERAL  = 0.8

# L3 region (tightest — airfoil surface + near wake; must be inside L2)
# Note: region extents are measured from mid-chord (AIRFOIL_CENTER_L0).
# To cover LE (at mid-chord - 0.5c), L3_UPSTREAM must be > 0.5.
L3_UPSTREAM = 0.7    # = 0.5c (to LE) + 0.2c margin (BL capture in stagnation region)
L3_DOWN     = 1.1    # = 0.5c (to TE) + 0.6c near-wake
L3_LATERAL  = 0.4    # covers ±0.17c envelope at α = 6° + margin

HERE = os.path.dirname(os.path.abspath(__file__))
SELIG_FILE = os.path.abspath(os.path.join(HERE, '..', 'to_claude', 'clf5605.dat'))

OMEGA_3 = 0.6
OMEGA_4 = 1.4

# =============================================================================
# §2. DERIVED LATTICE / DOMAIN PARAMETERS (L0 coarsest)
# =============================================================================

DX_L0     = CHORD_PHYS / CPC_L0                           # [m]
DT_L0     = U_MAX_LU * DX_L0 / U_INF_PHYS                 # [s]
U_INF_LU  = U_MAX_LU                                       # L0 inflow velocity
CHORD_LU  = float(CPC_L0)                                  # chord in L0 lu

Nx = int(round((UPSTREAM_C + 1.0 + DOWNSTREAM_C) * CPC_L0))
Ny = int(round(2.0 * LATERAL_C * CPC_L0))

LE_X_L0 = int(round(UPSTREAM_C * CPC_L0))
AIRFOIL_CENTER_L0 = (LE_X_L0 + CPC_L0 // 2, Ny // 2)

# Shear viscosity / tau on L0
NU_LU = U_INF_LU * CHORD_LU / RE                           # L0 lattice viscosity
TAU   = 0.5 + 3.0 * NU_LU                                  # L0 tau
CS    = 1.0 / np.sqrt(3.0)
MA_LU = U_MAX_LU / CS

# =============================================================================
# §3. MLG REGION DEFINITIONS (all in L0 physical lattice coords)
# =============================================================================

cx, cy = AIRFOIL_CENTER_L0

L1_XMIN = cx - int(round(L1_UPSTREAM * CPC_L0))
L1_XMAX = cx + int(round(L1_DOWN     * CPC_L0))
L1_YMIN = cy - int(round(L1_LATERAL  * CPC_L0))
L1_YMAX = cy + int(round(L1_LATERAL  * CPC_L0))

L2_XMIN = cx - int(round(L2_UPSTREAM * CPC_L0))
L2_XMAX = cx + int(round(L2_DOWN     * CPC_L0))
L2_YMIN = cy - int(round(L2_LATERAL  * CPC_L0))
L2_YMAX = cy + int(round(L2_LATERAL  * CPC_L0))

L3_XMIN = cx - int(round(L3_UPSTREAM * CPC_L0))
L3_XMAX = cx + int(round(L3_DOWN     * CPC_L0))
L3_YMIN = cy - int(round(L3_LATERAL  * CPC_L0))
L3_YMAX = cy + int(round(L3_LATERAL  * CPC_L0))

# =============================================================================
# §4. SANITY CHECKS
# =============================================================================

assert TAU > 0.5, f"UNSTABLE: τ = {TAU:.6f}"
assert MA_LU < 0.3, f"Ma_lu = {MA_LU:.3f} ≥ 0.3"
assert os.path.exists(SELIG_FILE), f"Missing airfoil file: {SELIG_FILE}"

# Rough memory estimate (each deeper level is REFINE× its parent region;
# fine_shape_k = (region_k + 2·overlap) × REFINE_k + 1 ≈ region·2^k)
Nx_L1 = (L1_XMAX - L1_XMIN + 4) * REFINE + 1
Ny_L1 = (L1_YMAX - L1_YMIN + 4) * REFINE + 1
Nx_L2 = (L2_XMAX - L2_XMIN + 4) * (REFINE ** 2) + 1
Ny_L2 = (L2_YMAX - L2_YMIN + 4) * (REFINE ** 2) + 1
Nx_L3 = (L3_XMAX - L3_XMIN + 4) * (REFINE ** 3) + 1
Ny_L3 = (L3_YMAX - L3_YMIN + 4) * (REFINE ** 3) + 1
nodes_L0 = Nx * Ny
nodes_L1 = Nx_L1 * Ny_L1
nodes_L2 = Nx_L2 * Ny_L2
nodes_L3 = Nx_L3 * Ny_L3
tot_nodes = nodes_L0 + nodes_L1 + nodes_L2 + nodes_L3
# D2Q9 cumulant estimate ~120 B/cell × 1.5 overhead
mem_gb = tot_nodes * 120 * 1.5 / 1e9

print(f"  [CLF5605 α={AOA_DEG:+.1f}°  MLG-{NUM_LEVELS}]")
print(f"  CPC L0={CPC_L0}, L1={CPC_L0*REFINE}, "
      f"L2={CPC_L0 * REFINE**2}, L3={CPC_FINEST}")
print(f"  τ L0={TAU:.6f}  Ma_lat={MA_LU:.4f}")
print(f"  L0 {Nx}×{Ny}={nodes_L0/1e6:.2f}M  "
      f"L1~{Nx_L1}×{Ny_L1}={nodes_L1/1e6:.2f}M")
print(f"  L2~{Nx_L2}×{Ny_L2}={nodes_L2/1e6:.2f}M  "
      f"L3~{Nx_L3}×{Ny_L3}={nodes_L3/1e6:.2f}M")
print(f"  Total ~{tot_nodes/1e6:.1f}M cells, est. ~{mem_gb:.1f} GB")

# =============================================================================
# §5. CONFIG BLOCKS
# =============================================================================

simulation = {
    "device_mode": "gpu",
    "precision": "float32",
    "dimension": 2,
    "lattice_model": "D2Q9",
    "collision_model": "cumulant",
    "omega_3": OMEGA_3,
    "omega_4": OMEGA_4,
}

physics = {
    "rho": RHO_PHYS,
    "U_inf": U_INF_PHYS,
    "Re": RE,
    "L_char": CHORD_PHYS,
    "flow_direction": [1.0, 0.0],
}

grid = {
    "Nx": Nx,
    "Ny": Ny,
    "resolution": CPC_L0,           # L0 resolution (the only "shape" knob)
}

numerics = {
    "u_max": U_MAX_LU,
    "collision": "cumulant",
}

boundaries = {
    # NEW config format: velocities are in PHYSICAL units [m/s].
    # setup.py converts to lattice units via UnitConverter.phys_to_lu_velocity().
    "inlet":  {"location": "xmin", "method": "eq",
               "velocity": [U_INF_PHYS, 0.0]},
    "outlet": {"location": "xmax", "method": "sponge",
               "velocity": [U_INF_PHYS, 0.0], "density": 1.0,
               "thickness": 20, "strength": 0.5},
    # Tunnel walls: free-slip (specular). Matches the paper's intent: the
    # physical 1.3° inclination in the Tohoku test section was designed to
    # cancel flat-plate BL displacement growth, so the core flow sees
    # effectively no wall BL. We discard the inclination and use slip walls
    # for the same net effect.
    "ymin":   {"location": "ymin", "method": "slip"},
    "ymax":   {"location": "ymax", "method": "slip"},
}

internal_geometry = {
    "airfoil": {
        "enabled": True,
        "selig_file": SELIG_FILE,
        "chord": CHORD_LU,               # L0 lattice units
        "center": AIRFOIL_CENTER_L0,     # L0 lattice units
        "angle_of_attack": AOA_DEG,
        # ibb → Bouzidi interpolated bounce-back (sub-pixel q-fraction).
        # hwbb → half-way bounce-back baseline.
        "wall_bc": WALL_BC,
    },
}

mlg = {
    "enabled": True,
    "num_levels": NUM_LEVELS,
    "overlap_width": 2,
    "interpolation": "cubic",
    "filter_level": 1,
    "levels": [
        {},  # L0 = full domain
        {"region": {"x_min": L1_XMIN, "x_max": L1_XMAX,
                    "y_min": L1_YMIN, "y_max": L1_YMAX}},
        {"region": {"x_min": L2_XMIN, "x_max": L2_XMAX,
                    "y_min": L2_YMIN, "y_max": L2_YMAX}},
        {"region": {"x_min": L3_XMIN, "x_max": L3_XMAX,
                    "y_min": L3_YMIN, "y_max": L3_YMAX}},
    ],
}

force_calculation = {
    "enabled": True,
    "interval": 10,
    "start_step": 1000,
    # Per-link (F_link, ρ_link) snapshot at each sampled step, dumped at
    # run end to <csv_dir>/surface_link_forces.npz for density-based Cp
    # post-processing via `python -m src.utilities.surface_distribution`.
    "save_link_forces": True,
    "reference": {
        "rho": 1.0,                 # lattice ρ∞
        "velocity": U_INF_LU,
        "char_length": CHORD_LU,    # L0 chord (will be rescaled to finest by setup)
        "span_length": 1,
    },
    "log": {"enabled": True, "filename": "force_history"},
}

time = {
    "max_steps": 100_000,
    "output_interval": 10_000,
    "logging_interval": 1_000,
    "checkpoint_interval": 50_000,
    "conservation_interval": 5_000,
}

conservation = {"enabled": True, "verbose": 0, "log_to_csv": True}

convergence = {
    "enabled": True,
    "cauchy": {
        "window_size": "auto",
        "epsilon": 1e-5,
        "Cd_epsilon": 3e-4,
        "n_required": 3,
    },
    "on_converged": "checkpoint_and_stop",
    "on_diverged":  "stop_with_checkpoint",
    "on_max_steps": "continue",
}

_folder = (
    f"result_clf5605_MLG_Re20k_M06_a"
    f"{int(round(AOA_DEG)):+03d}_{WALL_BC}"
).replace('+', 'p').replace('-', 'n')

output = {
    "output_dir":      f"./{_folder}/vtk",
    "checkpoint_dir":  f"./{_folder}/checkpoints",
    "csv_dir":         f"./{_folder}/csv",
    "clear_previous":  True,
    "vtk": {
        "enabled": True,
        "precision": "float32",
        "variables": ["density", "pressure", "velocity",
                      "velocity_magnitude", "solid_mask"],
    },
    "checkpoint": {"enabled": True, "keep_last_n": 2},
}

config = {
    "simulation": simulation,
    "physics":    physics,
    "grid":       grid,
    "numerics":   numerics,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "mlg":        mlg,
    "conservation": conservation,
    "convergence": convergence,
    "force_calculation": force_calculation,
    "output":     output,
    "time":       time,
}

if __name__ == "__main__":
    print()
    print("=" * 60)
    print(f"  CLF5605 MLG-{NUM_LEVELS}  α={AOA_DEG}°  summary")
    print("=" * 60)
    print(f"  L0 coarse domain:  {Nx} × {Ny} = {nodes_L0/1e6:.2f} M cells  "
          f"({CPC_L0} cells/chord)")
    print(f"  L1 mid:            ≈{Nx_L1} × {Ny_L1} = {nodes_L1/1e6:.2f} M cells  "
          f"({CPC_L0*REFINE} cells/chord)")
    print(f"  L2:                ≈{Nx_L2} × {Ny_L2} = {nodes_L2/1e6:.2f} M cells  "
          f"({CPC_L0*REFINE**2} cells/chord)")
    print(f"  L3 finest:         ≈{Nx_L3} × {Ny_L3} = {nodes_L3/1e6:.2f} M cells  "
          f"({CPC_FINEST} cells/chord)")
    print(f"  Total ~ {tot_nodes/1e6:.1f} M  est. memory ~ {mem_gb:.1f} GB")
    print(f"  Output: ./{_folder}/")
    print("=" * 60)
