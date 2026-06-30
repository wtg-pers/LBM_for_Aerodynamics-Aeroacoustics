"""
APC 9x4.5MR Propeller -- Hovering (J=0), Cumulant + ALM + MLG (3-Level)

Hover in quiescent air. No freestream velocity (U_inf = 0).
All boundaries: quiescent far-field (u=0).

Propeller (9x45MR-PERF.PE0):
  - APC 9x4.5MR, R = 4.50 in = 0.1143 m, D = 0.2286 m
  - RPM = 5000, 2 blades
  - Dual airfoil: E63 (inner, r<2.57") + APC12/NACA4412 (outer, r>4.18")
  - Transition midpoint: r = 3.375" (r/R = 0.75)
  - Hub radius = 0.32", hub transition = 0.77"
  - Re_75 = 52,238 (V_75 * c_75 / nu, standard ~20C)

Grid (D=40, 3-level MLG):
  - L0: 400x200x200  Up=3D  Lat=2.5D  Down=7D
  - L1: refined near rotor
  - L2: finest level around rotor disk

Usage:
    python main.py --config configs/alm_9x45mr_hover_mlg3.py
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
U_INF_PHYS = 0.0                                  # [m/s] HOVERING
N_BLADES = 2
N_RADIAL = 40

# Reference chord at r/R=0.75 (interpolated from PE0)
#   r=3.3293" -> c=0.6877", r=3.4476" -> c=0.6513"
#   r=3.375":   c ~ 0.6736"
CHORD_REF = 0.6736 * IN_TO_M                     # [m] = 0.01711

# Reynolds number at 75% span (V_75 * c_75 / nu)
V_75 = 0.75 * TIP_SPEED                          # [m/s] = 44.89
RE = 52238
NU_PHYS = V_75 * CHORD_REF / RE                  # [m/s] = 1.470e-5

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
U_INF_LU = 0.0                                   # HOVERING
TIP_LU = OMEGA_LU * R

CS = 1.0 / np.sqrt(3.0)
MA_TIP = TIP_LU / CS
STEPS_PER_REV = int(2 * np.pi / OMEGA_LU)

assert TAU > 0.5, f"UNSTABLE: tau = {TAU:.8f}"
assert MA_TIP < 0.3, f"Ma_tip = {MA_TIP:.3f} > 0.3"

# =============================================================================
# S3. DOMAIN (L0): Up=3D, Lat=2.5D, Down=7D
# =============================================================================
Nx = 10 * D                                      # 400
Ny = 5 * D                                       # 200
Nz = 5 * D                                       # 200

HUB_X = 3 * D                                    # 120 (3D from inlet)
HUB_Y = Ny // 2                                  # 100
HUB_Z = Nz // 2                                  # 100

print(f"  [APC 9x4.5MR HOVERING]")
print(f"  Re_75={RE:,}, D={D}, {N_BLADES} blades, RPM={RPM}")
print(f"  J=0 (hover), U_inf=0")
print(f"  tip_speed={TIP_SPEED:.2f} m/s, Ma_tip={MA_TIP:.4f}")
print(f"  tau={TAU:.8f}, nu_lu={NU_LU:.6e}")
print(f"  steps/rev={STEPS_PER_REV}")

# =============================================================================
# S4. MLG -- 3 Levels
# =============================================================================
OVERLAP_WIDTH = 4
INTERP_SCHEME = "cubic"
FILTER_LEVEL = 1

# L1: upstream 1D, downstream 5D, lateral +/-1.5D
L1_X_MIN = HUB_X - 0.8 * D
L1_X_MAX = HUB_X + 4 * D
L1_Y_MIN = HUB_Y - int(1.25 * D)
L1_Y_MAX = HUB_Y + int(1.25 * D)
L1_Z_MIN = HUB_Z - int(1.25 * D)
L1_Z_MAX = HUB_Z + int(1.25 * D)

# L2: upstream 0.5D, downstream 2D, lateral +/-0.625D
L2_X_MIN = HUB_X - int(D * 0.4)
L2_X_MAX = HUB_X + 2 * D
L2_Y_MIN = HUB_Y - int(0.625 * D)
L2_Y_MAX = HUB_Y + int(0.625 * D)
L2_Z_MIN = HUB_Z - int(0.625 * D)
L2_Z_MAX = HUB_Z + int(0.625 * D)

# =============================================================================
# S5. BLADE GEOMETRY (from 9x45MR-PERF.PE0)
# =============================================================================
# PE0 section data: (station [in], chord [in], twist [deg], airfoil, active)
#
# Airfoil sections (PE0):
#   AIRFOIL1: r=2.57", E63         (Transition Start)
#   AIRFOIL2: r=4.18", APC12=NACA4412  (Transition End)
#   Switchover at midpoint: r=3.375" (r/R=0.75)
#
# Hub transition: r=0.77" (r/R=0.171)
#   Sections below hub transition are inactive.
#
_SECTIONS_PE0 = [
    # --- Hub (inactive) ---
    (0.0000, 0.6500,  0.0000, "e63",      False),
    # --- Active blade from hub transition ---
    (0.7700, 0.6979, 37.7207, "e63",      True),
    (0.8491, 0.7511, 37.7829, "e63",      True),
    (0.9652, 0.8200, 36.3016, "e63",      True),
    (1.0834, 0.8788, 33.4669, "e63",      True),
    (1.2016, 0.9262, 30.7960, "e63",      True),
    (1.3198, 0.9624, 28.4862, "e63",      True),
    (1.4380, 0.9874, 26.4751, "e63",      True),
    (1.5562, 1.0013, 24.7123, "e63",      True),
    (1.6744, 1.0043, 23.1574, "e63",      True),
    (1.7927, 0.9973, 21.7776, "e63",      True),
    (1.9109, 0.9866, 20.5462, "e63",      True),
    (2.0291, 0.9734, 19.4414, "e63",      True),
    (2.1473, 0.9578, 18.4455, "e63",      True),
    (2.2655, 0.9398, 17.5435, "e63",      True),
    (2.3837, 0.9196, 16.7233, "e63",      True),
    (2.5019, 0.8972, 15.9744, "e63",      True),
    # --- Transition zone (E63 side, r < 3.375") ---
    (2.6201, 0.8728, 15.2882, "e63",      True),
    (2.7383, 0.8464, 14.6572, "e63",      True),
    (2.8565, 0.8181, 14.0753, "e63",      True),
    (2.9747, 0.7880, 13.5369, "e63",      True),
    (3.0929, 0.7561, 13.0376, "e63",      True),
    (3.2111, 0.7227, 12.5732, "e63",      True),
    (3.3293, 0.6877, 12.1403, "e63",      True),
    # --- Transition zone (NACA4412 side, r >= 3.375") ---
    (3.4476, 0.6513, 11.7357, "naca4412", True),
    (3.5658, 0.6136, 11.3570, "naca4412", True),
    (3.6840, 0.5746, 11.0016, "naca4412", True),
    (3.8022, 0.5344, 10.6675, "naca4412", True),
    (3.9204, 0.4931, 10.3530, "naca4412", True),
    (4.0386, 0.4508, 10.0562, "naca4412", True),
    # --- Pure NACA4412 (r >= 4.18") ---
    (4.1568, 0.4076,  9.7758, "naca4412", True),
    (4.2700, 0.3650,  9.5214, "naca4412", True),
    (4.2977, 0.3512,  9.4611, "naca4412", True),
    (4.3205, 0.3374,  9.4123, "naca4412", True),
    (4.3432, 0.3211,  9.3639, "naca4412", True),
    (4.3659, 0.3018,  9.3160, "naca4412", True),
    (4.3886, 0.2790,  9.2686, "naca4412", True),
    (4.4114, 0.2514,  9.2217, "naca4412", True),
    (4.4341, 0.2172,  9.1752, "naca4412", True),
    (4.4568, 0.1711,  9.1292, "naca4412", True),
    (4.4795, 0.0946,  9.0837, "naca4412", True),
    (4.5000, 0.0139,  9.0475, "naca4412", True),
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
        "u_ref_lu": TIP_LU,                      # tip speed for normalization
        "L_ref_lu": float(D),
        "initial_flow_velocity": [0.0, 0.0, 0.0],   # quiescent
        "U_inf": U_INF_PHYS,
        "dx_phys": DX_PHYS,
        "dt_phys": DT_PHYS,
    },

    "time": {
        "max_steps": 30 * STEPS_PER_REV,
    },
}

boundaries = {
    "xmin":       {"location": "xmin", "method": "regularized_inlet",
                   "velocity": 0.0, "rho": RHO},
    "outlet":     {"location": "xmax", "method": "sponge",
                   "velocity": [0, 0, 0], "density": RHO,
                   "thickness": 20, "sigma_max": 0.1},
    "ymin":       {"location": "ymin", "method": "regularized_inlet",
                   "velocity": 0.0, "rho": RHO},
    "ymax":       {"location": "ymax", "method": "regularized_inlet",
                   "velocity": 0.0, "rho": RHO},
    "zmin":       {"location": "zmin", "method": "regularized_inlet",
                   "velocity": 0.0, "rho": RHO},
    "zmax":       {"location": "zmax", "method": "regularized_inlet",
                   "velocity": 0.0, "rho": RHO},
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
        },
        "naca4412": {
            "method": "neuralfoil",
            "airfoil_name": "naca4412",
            "Re_target": RE,
            "Re_min": 10000,
            "Re_max": 100000,
            "Re_steps": 10,
            "mode": "asb",
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
                {"r": float(r_in * IN_TO_M),
                 "chord": float(c_in * IN_TO_M),
                 "twist": float(tw),
                 "airfoil": af,
                 "active": act}
                for r_in, c_in, tw, af, act in _SECTIONS_PE0
            ],
        },
        "grid": {
            "n_radial": N_RADIAL,
        },
    },
    "gaussian_cutoff": 3.0,
    "rho_ref": 1.0,
    "coeff_mode": "rotorcraft",
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
    "output": STEPS_PER_REV // 18,
    "log": STEPS_PER_REV // 360,
    "checkpoint": STEPS_PER_REV * 5,
    "conservation": STEPS_PER_REV // 2,
}

conservation = {
    "enabled": True,
    "verbose": 0,
    "log_to_csv": True,
}

convergence = {"enabled": False}
force_calculation = {"enabled": False}

_folder = f"result_apc9x45mr_hover_Re{RE // 1000}k_mlg3"

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
