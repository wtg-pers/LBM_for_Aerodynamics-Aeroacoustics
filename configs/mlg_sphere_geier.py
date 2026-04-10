"""
Sphere Drag Benchmark — Re=100,000, 5-Level MLG, Cumulant

Reproduces the benchmark setup from:
  [1] Geier et al., "The cumulant lattice Boltzmann equation in three
      dimensions: Theory and validation", Comp. Math. Appl. 70(4), 2015
  [2] Stiebler et al., "Lattice Boltzmann large eddy simulation of
      subcritical flows around a sphere on non-uniform grids",
      Comp. Math. Appl. 61(11), 2011

Common benchmark parameters (both papers):
  - Blockage ratio: λ = h/d = 11  (channel width / sphere diameter)
  - Sphere: D = 16Δx₀ = 256Δx₄ at finest
  - Cd = 2Fd / (π ρ u² r²)  [Geier Eq.112, Stiebler Eq.19]
  - Cd averaging: 25 turnover times (d/u_in)  [Stiebler]

Differences:
  - Geier: D3Q27 Cumulant, 5 levels, u₀=0.071, no LES
  - Stiebler: D3Q19 MRT, 6-7 levels, Smagorinsky LES (Cs=0.18)
  - Stiebler: slip BC on sides; Geier: constant velocity on sides

Our setup follows Geier's parameters with λ=11 blockage.

Standalone execution (grid info + VTK preview):
    python configs/mlg_sphere_geier.py

Simulation:
    python main.py --config configs/mlg_sphere_geier.py
    python main.py --config configs/mlg_sphere_geier.py --gpu 1

Author: LBM Development Team
Date: 2026-04
"""

import numpy as np

# =============================================================================
# §1. PHYSICAL PARAMETERS
#
#   Geier: D=16Δx₀, u₀=0.071, Re via ν adjustment
#   Stiebler: λ = h/d = 11 (blockage)
#   Both: Cd averaged over 25 turnover times (d/u_in)
# =============================================================================
RE = 100000
D = 16                          # Sphere diameter  [Δx₀]  (= 256Δx₄)
U_INLET = 0.071                 # Inlet velocity   [Δx/Δt] (Geier: 0.071)
RHO = 1.0
R = D // 2                      # Sphere radius = 8

# Domain: λ = h/d = 11 → equivalent rectangular h = 11*D ≈ 156
# Upstream 3D + downstream 10D = 13D streamwise
LAMBDA = 11                     # Blockage ratio h/d (Geier & Stiebler)
Ny_eq = int(np.round(np.sqrt(np.pi * R**2 / (1.0 / LAMBDA**2))))  # = 156
Nx = 3 * D + 10 * D            # = 208  (3D upstream + 10D downstream)
Ny = Ny_eq                      # = 156
Nz = Ny_eq                      # = 156

# Sphere position
CENTER_X = 3 * D                # = 48
CENTER_Y = Ny // 2              # = 78
CENTER_Z = Nz // 2              # = 78

RESOLUTION = D

# =============================================================================
# §2. DERIVED LATTICE PARAMETERS
#
#   ν = u₀·D / Re = 0.071×16 / 100000 = 1.136e-5
#   τ₀ = 0.5 + 3ν = 0.50003408
#   Ma = u₀ / cs = 0.123
#
#   Turnover time: T_turn = D / U = 16 / 0.071 ≈ 225 Δt₀
#   25 turnovers (Stiebler averaging): 25 × 225 ≈ 5625 Δt₀
# =============================================================================
NU_LU = U_INLET * D / RE
TAU = 0.5 + 3.0 * NU_LU

CS = 1.0 / np.sqrt(3.0)
MA = U_INLET / CS

assert TAU > 0.5, f"UNSTABLE: tau = {TAU:.8f}"
assert MA < 0.3, f"COMPRESSIBILITY: Ma = {MA:.3f} > 0.3"

A_FRONTAL = np.pi * R**2
SPAN_FOR_SPHERE = A_FRONTAL / D
BLOCKAGE = A_FRONTAL / (Ny * Nz) * 100
T_TURNOVER = D / U_INLET

print(f"  [Geier Benchmark] Re={RE:,}, D={D}, u0={U_INLET}")
print(f"  tau0 = {TAU:.8f}, Ma = {MA:.4f}")
print(f"  nu = {NU_LU:.6e}")
print(f"  lambda = h/d = {LAMBDA}, blockage = {BLOCKAGE:.3f}%")
print(f"  L0: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz:,} nodes")
print(f"  Turnover: {T_TURNOVER:.1f} steps")

# =============================================================================
# §3. MLG CONFIGURATION (5-Level)
#
#   L0: D=16   far-field
#   L1: D=32   transition
#   L2: D=64   outer wake
#   L3: D=128  separation zone
#   L4: D=256  sphere surface (force measurement)
#
#   Region design: inside-out with downstream wake extension
#   L4: sphere surface (tight)
#   L3→L2→L1: progressively wider downstream wake capture
# =============================================================================
OVERLAP_WIDTH = 2
INTERP_SCHEME = "compact_second_order"  # Geier: compact 2nd order
FILTER_LEVEL = 1

# L4: sphere bounding box + 1 cell margin (0.06D)
L4_X_MIN = CENTER_X - R - 1            # = 39
L4_X_MAX = CENTER_X + R + 1            # = 57
L4_Y_MIN = CENTER_Y - R - 1            # = 69
L4_Y_MAX = CENTER_Y + R + 1            # = 87
L4_Z_MIN = CENTER_Z - R - 1            # = 69
L4_Z_MAX = CENTER_Z + R + 1            # = 87

# L3: contain L4 + 3 cells upstream/lateral, 3D downstream
L3_X_MIN = L4_X_MIN - 3                # = 36
L3_X_MAX = CENTER_X + 3 * D            # = 96  (3D downstream)
L3_Y_MIN = L4_Y_MIN - 3                # = 66
L3_Y_MAX = L4_Y_MAX + 3                # = 90
L3_Z_MIN = L4_Z_MIN - 3                # = 66
L3_Z_MAX = L4_Z_MAX + 3                # = 90

# L2: contain L3 + 3 cells upstream/lateral, 5D downstream
L2_X_MIN = L3_X_MIN - 3                # = 33
L2_X_MAX = CENTER_X + 5 * D            # = 128 (5D downstream)
L2_Y_MIN = L3_Y_MIN - 3                # = 63
L2_Y_MAX = L3_Y_MAX + 3                # = 93
L2_Z_MIN = L3_Z_MIN - 3                # = 63
L2_Z_MAX = L3_Z_MAX + 3                # = 93

# L1: contain L2 + 3 cells upstream/lateral, 7D downstream
L1_X_MIN = L2_X_MIN - 3                # = 30
L1_X_MAX = CENTER_X + 7 * D            # = 160 (7D downstream)
L1_Y_MIN = L2_Y_MIN - 3                # = 60
L1_Y_MAX = L2_Y_MAX + 3                # = 96
L1_Z_MIN = L2_Z_MIN - 3                # = 60
L1_Z_MAX = L2_Z_MAX + 3                # = 96

# =============================================================================
# §4. SIMULATION SETTINGS
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "device_id": 0,
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
        "max_steps": 30000,
        "output_interval": 1000,
        "checkpoint_interval": 5000,
    },
}

# =============================================================================
# §5. BOUNDARY CONDITIONS
#
#   Geier: constant velocity at inlet, sides, top, bottom
#          non-reflective extrapolation at outlet
#   Stiebler: slip wall on sides, constant velocity at inlet,
#             pressure outlet
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
        "radius": R,
    },
}

# =============================================================================
# §7. MLG — 5 Levels
# =============================================================================
mlg = {
    "enabled": True,
    "num_levels": 5,
    "overlap_width": OVERLAP_WIDTH,
    "interpolation": INTERP_SCHEME,
    "filter_level": FILTER_LEVEL,
    "levels": [
        {},  # Level 0: D=16

        # Level 1: D=32, transition
        {"region": {"x_min": L1_X_MIN, "x_max": L1_X_MAX,
                    "y_min": L1_Y_MIN, "y_max": L1_Y_MAX,
                    "z_min": L1_Z_MIN, "z_max": L1_Z_MAX}},

        # Level 2: D=64, outer wake
        {"region": {"x_min": L2_X_MIN, "x_max": L2_X_MAX,
                    "y_min": L2_Y_MIN, "y_max": L2_Y_MAX,
                    "z_min": L2_Z_MIN, "z_max": L2_Z_MAX}},

        # Level 3: D=128, separation
        {"region": {"x_min": L3_X_MIN, "x_max": L3_X_MAX,
                    "y_min": L3_Y_MIN, "y_max": L3_Y_MAX,
                    "z_min": L3_Z_MIN, "z_max": L3_Z_MAX}},

        # Level 4: D=256, sphere surface
        {"region": {"x_min": L4_X_MIN, "x_max": L4_X_MAX,
                    "y_min": L4_Y_MIN, "y_max": L4_Y_MAX,
                    "z_min": L4_Z_MIN, "z_max": L4_Z_MAX}},
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
#
#   Geier Eq.112: cD = 8F_D / (ρ u₀² π d²)
#                    = F_D / (0.5 ρ u₀² π(d/2)²)  — same as ours
#   Stiebler Eq.19: cd = 2Fd / (π ρ u²_in r²)     — identical
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
_folder = f"result_sphere_D{D}_Re{RE}_{_mlg_tag}_geier"

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


# =============================================================================
# §13. STANDALONE EXECUTION — Grid preview
#
#   python configs/mlg_sphere_geier.py
#   → Prints grid info and writes VTK files for ParaView inspection
# =============================================================================
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from src.grid.overlap_manager import OverlapManager, IndexBox
    from src.grid.level_scaling import LevelScaler
    from src.io.mlg_vtk_writer import MLGVTKWriter

    num_levels = mlg["num_levels"]
    levels_config = mlg["levels"]

    # ── Level scaler ─────────────────────────────────────────
    scaler = LevelScaler(tau_0=TAU, num_levels=num_levels)

    # ── Overlap manager ──────────────────────────────────────
    overlap_mgr = OverlapManager()
    coarse_shape = (Nx, Ny, Nz)
    level_origins = [(0.0, 0.0, 0.0)]
    level_spacings = [(1.0, 1.0, 1.0)]

    print(f"\n{'='*60}")
    print(f"  MLG Grid Preview — {num_levels} levels")
    print(f"{'='*60}")
    print(f"  L0: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz:,} nodes")

    total_nodes = Nx * Ny * Nz

    for k in range(1, num_levels):
        level_cfg = levels_config[k]
        region_cfg = level_cfg["region"]

        po = level_origins[k - 1]
        pd = level_spacings[k - 1]

        local_x_min = round((region_cfg["x_min"] - po[0]) / pd[0])
        local_x_max = round((region_cfg["x_max"] - po[0]) / pd[0])
        local_y_min = round((region_cfg["y_min"] - po[1]) / pd[1])
        local_y_max = round((region_cfg["y_max"] - po[1]) / pd[1])
        local_z_min = round((region_cfg["z_min"] - po[2]) / pd[2])
        local_z_max = round((region_cfg["z_max"] - po[2]) / pd[2])

        fine_region = IndexBox(
            x_start=local_x_min, x_end=local_x_max,
            y_start=local_y_min, y_end=local_y_max,
            z_start=local_z_min, z_end=local_z_max,
        )

        overlap_region = overlap_mgr.add_level_pair(
            coarse_shape=coarse_shape,
            fine_region=fine_region,
            overlap_width=OVERLAP_WIDTH,
        )

        fdc = overlap_region.fine_domain_coarse
        lu_k = scaler.get_level_units(k)
        origin = (
            po[0] + fdc.x_start * pd[0],
            po[1] + fdc.y_start * pd[1],
            po[2] + fdc.z_start * pd[2],
        )
        spacing = (lu_k.dx, lu_k.dx, lu_k.dx)
        level_origins.append(origin)
        level_spacings.append(spacing)

        fs = overlap_region.fine_shape
        nodes = fs[0] * fs[1] * fs[2]
        total_nodes += nodes
        D_k = D * (2 ** k)
        print(f"  L{k}: {fs[0]}x{fs[1]}x{fs[2]} = {nodes:,} nodes  "
              f"(D={D_k}, origin={origin})")

        coarse_shape = overlap_region.fine_shape

    mem_gb = total_nodes * 27 * 4 * 3 / 1e9
    print(f"\n  Total: {total_nodes:,} nodes")
    print(f"  Memory (float32, 3 buffers): {mem_gb:.1f} GB")

    # ── Write VTK for preview ────────────────────────────────
    vtk_dir = os.path.join(os.path.dirname(__file__), '..', _folder, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)

    writer = MLGVTKWriter(
        output_dir=vtk_dir,
        coarse_shape=(Nx, Ny, Nz),
        overlap_mgr=overlap_mgr,
        scaler=scaler,
        num_levels=num_levels,
        precision="float32",
    )

    # Write empty .vti + .vth (density=1, velocity=0) for grid structure
    for k in range(num_levels):
        info = writer._level_info[k]
        shape = info["shape"]
        rho_dummy = np.ones(shape, dtype=np.float32)
        u_dummy = np.zeros((3,) + shape, dtype=np.float32)

        vti_filename = f"grid_preview_level{k}.vti"
        vti_path = os.path.join(writer._level_dirs[k], vti_filename)
        writer._write_vti(
            filepath=vti_path,
            shape=shape,
            origin=info["origin"],
            spacing=info["spacing"],
            rho=rho_dummy,
            u=u_dummy,
        )

    # Write .vth referencing preview files
    vti_rel_paths = [
        f"../level{k}/grid_preview_level{k}.vti"
        for k in range(num_levels)
    ]
    vth_path = os.path.join(writer._vth_dir, "grid_preview.vth")
    writer._write_vth(vth_path, vti_rel_paths, step=0)

    print(f"\n  VTK preview written to: {vtk_dir}/")
    print(f"  Open in ParaView: {vtk_dir}/vth/grid_preview.vth")
