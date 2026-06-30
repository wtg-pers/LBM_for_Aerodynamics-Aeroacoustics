"""
APC 9x4.5MR Propeller -- Axial Flight J=0.2, Cumulant + ALM + MLG (3-Level)

UIUC blade geometry + dual airfoil (E63 + NACA4412).
Target (propeller convention): CT ~ 0.08, CP ~ 0.04

Grid (D=40, 3-level MLG):
  - L0: 800x200x200
  - L1: refined near rotor + near wake
  - L2: finest level around rotor disk

Usage:
    python main.py --config configs/alm_9x45e_J02_mlg3.py
"""

import numpy as np

# =============================================================================
# S1. PROPELLER & FLOW PARAMETERS
# =============================================================================
R_IN = 4.50                                      # [in] prop radius
IN_TO_M = 0.0254                                 # [m/in]
R_PHYS = R_IN * IN_TO_M                          # [m] = 0.1143
D_PHYS = 2 * R_PHYS                              # [m] = 0.2286
RPM = 5000
N_REV_S = RPM / 60.0                             # [rev/s] = 83.33
OMEGA_PHYS = RPM * 2.0 * np.pi / 60.0            # [rad/s] = 523.6
TIP_SPEED = OMEGA_PHYS * R_PHYS                  # [m/s] = 59.85
N_BLADES = 2
N_RADIAL = 40

# Advance ratio
J = 0.2
U_INF_PHYS = J * N_REV_S * D_PHYS                # [m/s] = 3.81

# Reference chord at r/R=0.75 (UIUC: c/R=0.150)
CHORD_REF = 0.150 * R_PHYS                       # [m] = 0.01715

# Reynolds number at 75% span
V_75 = 0.75 * TIP_SPEED                          # [m/s] = 44.89
RE = 52238
NU_PHYS = V_75 * CHORD_REF / RE                  # [m^2/s]

# =============================================================================
# S2. LATTICE PARAMETERS
# =============================================================================
D = 40                                            # Prop diameter [dx_0]
R = D // 2                                        # = 20
U_MAX_LU = 0.1                                    # tip_speed in lattice
RHO = 1.0

DX_PHYS = D_PHYS / D                             # [m/lu]
DT_PHYS = U_MAX_LU * DX_PHYS / TIP_SPEED        # [s/lt]

NU_LU = NU_PHYS * DT_PHYS / DX_PHYS**2
TAU = 0.5 + 3.0 * NU_LU
OMEGA_LU = OMEGA_PHYS * DT_PHYS
U_INF_LU = U_INF_PHYS * DT_PHYS / DX_PHYS
TIP_LU = OMEGA_LU * R

CS = 1.0 / np.sqrt(3.0)
MA_TIP = TIP_LU / CS
MA_INF = U_INF_LU / CS
STEPS_PER_REV = int(2 * np.pi / OMEGA_LU)

assert TAU > 0.5, f"UNSTABLE: tau = {TAU:.8f}"
assert MA_TIP < 0.3, f"Ma_tip = {MA_TIP:.3f} > 0.3"

# =============================================================================
# S3. DOMAIN (L0): same as hover config (Up=3D, Lat=2.5D, Down=7D)
# =============================================================================
Nx = 10 * D                                      # 400
Ny = 5 * D                                       # 200
Nz = 5 * D                                       # 200

HUB_X = 3 * D                                    # 120 (3D from inlet)
HUB_Y = Ny // 2                                  # 100
HUB_Z = Nz // 2                                  # 100

print(f"  [APC 9x4.5MR J={J}, MLG 3-Level]")
print(f"  Re_75={RE:,}, D={D}, {N_BLADES} blades, RPM={RPM}")
print(f"  J={J} -> U_inf={U_INF_PHYS:.2f} m/s ({U_INF_LU:.5f} lu)")
print(f"  tip_speed={TIP_SPEED:.2f} m/s, Ma_tip={MA_TIP:.4f}, Ma_inf={MA_INF:.4f}")
print(f"  tau={TAU:.8f}, nu_lu={NU_LU:.6e}")
print(f"  steps/rev={STEPS_PER_REV}")

# =============================================================================
# S4. MLG -- 3 Levels (same as hover MLG config)
# =============================================================================
OVERLAP_WIDTH = 4
INTERP_SCHEME = "cubic"
FILTER_LEVEL = 1

# L1: upstream 0.8D, downstream 4D, lateral +/-1.25D
L1_X_MIN = HUB_X - 0.8 * D
L1_X_MAX = HUB_X + 4 * D
L1_Y_MIN = HUB_Y - int(1.25 * D)
L1_Y_MAX = HUB_Y + int(1.25 * D)
L1_Z_MIN = HUB_Z - int(1.25 * D)
L1_Z_MAX = HUB_Z + int(1.25 * D)

# L2: upstream 0.4D, downstream 2D, lateral +/-0.625D
L2_X_MIN = HUB_X - int(D * 0.4)
L2_X_MAX = HUB_X + 2 * D
L2_Y_MIN = HUB_Y - int(0.625 * D)
L2_Y_MAX = HUB_Y + int(0.625 * D)
L2_Z_MIN = HUB_Z - int(0.625 * D)
L2_Z_MAX = HUB_Z + int(0.625 * D)

# =============================================================================
# S5. BLADE GEOMETRY (UIUC data for APC 9x4.5MR)
# =============================================================================
_SECTIONS_UIUC = [
    # r/R    c/R     beta   airfoil       active
    # Hub region (r < 0.95" = r/R < 0.211)
    (0.15,  0.157,  34.80, "e63",         False),  # r=0.675"
    (0.20,  0.163,  36.50, "e63",         False),  # r=0.900"
    # E63 region (0.95" <= r < 3.36", r/R: 0.211~0.747)
    (0.25,  0.187,  34.26, "e63",         True),   # r=1.125"
    (0.30,  0.206,  29.64, "e63",         True),
    (0.35,  0.217,  25.62, "e63",         True),
    (0.40,  0.222,  22.53, "e63",         True),
    (0.45,  0.222,  20.25, "e63",         True),
    (0.50,  0.217,  18.37, "e63",         True),
    (0.55,  0.209,  16.83, "e63",         True),
    (0.60,  0.197,  15.51, "e63",         True),
    (0.65,  0.183,  14.38, "e63",         True),
    (0.70,  0.167,  13.45, "e63",         True),   # r=3.15"
    # NACA4412 region (r >= 3.36", r/R >= 0.747)
    (0.75,  0.150,  12.56, "naca4412",    True),   # r=3.375"
    (0.80,  0.133,  12.09, "naca4412",    True),
    (0.85,  0.116,  11.25, "naca4412",    True),
    (0.90,  0.099,  10.46, "naca4412",    True),
    (0.95,  0.074,   9.68, "naca4412",    True),
    (1.00,  0.049,   8.90, "naca4412",    True),
]

# =============================================================================
# S6. CONFIG
# =============================================================================
simulation = {
    "device_mode": "gpu",
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
        "u_ref_lu": TIP_LU,
        "L_ref_lu": float(D),
        "initial_flow_velocity": [U_INF_LU, 0.0, 0.0],
        "U_inf": U_INF_PHYS,
        "dx_phys": DX_PHYS,
        "dt_phys": DT_PHYS,
    },

    "time": {
        "max_steps": 20 * STEPS_PER_REV,
    },
}

boundaries = {
    "inlet":      {"location": "xmin", "method": "regularized_inlet",
                   "velocity": U_INF_LU, "rho": RHO},
    "outlet":     {"location": "xmax", "method": "sponge",
                   "velocity": [U_INF_LU, 0, 0], "density": RHO,
                   "thickness": 20, "sigma_max": 0.1},
    "ymin":       {"location": "ymin", "method": "regularized_inlet",
                   "velocity": [U_INF_LU, 0, 0], "rho": RHO},
    "ymax":       {"location": "ymax", "method": "regularized_inlet",
                   "velocity": [U_INF_LU, 0, 0], "rho": RHO},
    "zmin":       {"location": "zmin", "method": "regularized_inlet",
                   "velocity": [U_INF_LU, 0, 0], "rho": RHO},
    "zmax":       {"location": "zmax", "method": "regularized_inlet",
                   "velocity": [U_INF_LU, 0, 0], "rho": RHO},
}

internal_geometry = {"type": "none"}

# Dual airfoil polar: E63 (inner) + NACA4412 (outer)
airfoil_polar = {
    "method": "multi",
    "default": "e63",
    "airfoils": {
        "e63": {
            "method": "neuralfoil",
            "airfoil_name": "e63",
            "Re_target": RE,
            "Re_min": 10000,
            "Re_max": 100000,
            "Re_steps": 10,
            "mode": "asb",
            "ncrit": 9.0,
        },
        "naca4412": {
            "method": "neuralfoil",
            "airfoil_name": "naca4412",
            "Re_target": RE,
            "Re_min": 10000,
            "Re_max": 100000,
            "Re_steps": 10,
            "mode": "asb",
            "ncrit": 9.0,
        },
    },
}

actuator_line = {
    "enabled": True,
    "units": {
        "dx_phys": DX_PHYS,
        "dt_phys": DT_PHYS,
    },
    "rotor": {
        "n_blades": N_BLADES,
        "hub_center": [float(HUB_X * DX_PHYS),
                       float(HUB_Y * DX_PHYS),
                       float(HUB_Z * DX_PHYS)],
        "omega": OMEGA_PHYS,
        "rotation_axis": [1, 0, 0],
        "thrust_direction": [-1, 0, 0],
        "theta_0": 0.0,
        "blade": {
            "sections": [
                {"r": float(rR * R_PHYS),
                 "chord": float(cR * R_PHYS),
                 "twist": float(tw),
                 "airfoil": af,
                 "active": act}
                for rR, cR, tw, af, act in _SECTIONS_UIUC
            ],
        },
        "grid": {
            "n_radial": N_RADIAL,
        },
    },
    "gaussian_cutoff": 3.0,
    "rho_ref": 1.0,
    "coeff_mode": "auto",
    "ramp_steps": 500,
    "prandtl_loss": True,
}

mlg = {
    "enabled": True,
    "num_levels": 3,
    "overlap_width": OVERLAP_WIDTH,
    "interpolation": INTERP_SCHEME,
    "filter_level": FILTER_LEVEL,
    "levels": [
        {},
        {"region": {"x_min": L1_X_MIN, "x_max": L1_X_MAX,
                    "y_min": L1_Y_MIN, "y_max": L1_Y_MAX,
                    "z_min": L1_Z_MIN, "z_max": L1_Z_MAX}},
        {"region": {"x_min": L2_X_MIN, "x_max": L2_X_MAX,
                    "y_min": L2_Y_MIN, "y_max": L2_Y_MAX,
                    "z_min": L2_Z_MIN, "z_max": L2_Z_MAX}},
    ],
}

interval = {
    "output": STEPS_PER_REV,
    "log": STEPS_PER_REV // 10,
    "checkpoint": 5 * STEPS_PER_REV,
    "conservation": STEPS_PER_REV // 2,
}

conservation = {
    "enabled": True,
    "verbose": 0,
    "log_to_csv": True,
}

convergence = {"enabled": False}
force_calculation = {"enabled": False}

_folder = f"result_apc9x45e_J{J:.1f}_Re{RE // 1000}k_mlg3"

output = {
    "output_dir": f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir": f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": True, "precision": "float32"},
    "checkpoint": {"enabled": True, "keep_last_n": 2},
}

config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "mlg": mlg,
    "interval": interval,
    "airfoil_polar": airfoil_polar,
    "actuator_line": actuator_line,
    "conservation": conservation,
    "convergence": convergence,
    "force_calculation": force_calculation,
    "output": output,
}

# =============================================================================
# S7. STANDALONE -- Grid preview
# =============================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from src.grid.overlap_manager import OverlapManager, IndexBox
    from src.grid.level_scaling import LevelScaler

    num_levels = mlg["num_levels"]
    levels_config = mlg["levels"]
    scaler = LevelScaler(tau_0=TAU, num_levels=num_levels)
    overlap_mgr = OverlapManager()
    coarse_shape = (Nx, Ny, Nz)
    level_origins = [(0.0, 0.0, 0.0)]
    level_spacings = [(1.0, 1.0, 1.0)]

    total_nodes = Nx * Ny * Nz
    print(f"\n  L0: {Nx}x{Ny}x{Nz} = {total_nodes:,}")
    for k in range(1, num_levels):
        region_cfg = levels_config[k]["region"]
        po, pd = level_origins[k - 1], level_spacings[k - 1]
        fine_region = IndexBox(
            round((region_cfg["x_min"] - po[0]) / pd[0]),
            round((region_cfg["x_max"] - po[0]) / pd[0]),
            round((region_cfg["y_min"] - po[1]) / pd[1]),
            round((region_cfg["y_max"] - po[1]) / pd[1]),
            round((region_cfg["z_min"] - po[2]) / pd[2]),
            round((region_cfg["z_max"] - po[2]) / pd[2]))
        overlap_region = overlap_mgr.add_level_pair(
            coarse_shape, fine_region, OVERLAP_WIDTH)
        fdc = overlap_region.fine_domain_coarse
        lu_k = scaler.get_level_units(k)
        origin = (po[0] + fdc.x_start * pd[0],
                  po[1] + fdc.y_start * pd[1],
                  po[2] + fdc.z_start * pd[2])
        level_origins.append(origin)
        level_spacings.append((lu_k.dx,) * 3)
        fs = overlap_region.fine_shape
        nodes = fs[0] * fs[1] * fs[2]
        total_nodes += nodes
        print(f"  L{k}: {fs[0]}x{fs[1]}x{fs[2]} = {nodes:,}  "
              f"(dx={DX_PHYS * 1000 / (2**k):.2f}mm)")
        coarse_shape = fs

    print(f"  Total: {total_nodes:,} nodes, "
          f"{total_nodes * 27 * 4 * 2 / 1e9:.1f} GB")
    print(f"  Output: ./{_folder}/")
