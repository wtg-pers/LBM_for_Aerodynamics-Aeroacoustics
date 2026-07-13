"""
ERC Propeller -- Cruise @ 3672 RPM, V_inf = 23.15 m/s,
    Cumulant + ALM + MLG (3-Level) + WALE SGS.

3-blade propeller, R = 0.269 m, D = 0.538 m.
Geometry: 87 spanwise sections at r/R = 0.135 to 0.995 (interval midpoints).
Polars: 87 CSV files in input_files/erc_prop/{14..100}section.csv.

Coordinate convention: freestream along +x, thrust along -x (matches BEMT
propeller convention: positive thrust opposes the freestream).

Usage (run from project root that contains main.py, configs/, input_files/):
    python main.py --config configs/alm/alm_erc_cruise_3672rpm_mlg3.py
"""

import numpy as np

# =============================================================================
# S1. PROPELLER & FLOW PARAMETERS (physical units)
# =============================================================================
R_PHYS       = 0.269                                # [m] rotor radius
D_PHYS       = 2.0 * R_PHYS                         # [m] diameter
RPM          = 3672
N_REV_S      = RPM / 60.0
OMEGA_PHYS   = RPM * 2.0 * np.pi / 60.0             # [rad/s]
TIP_SPEED    = OMEGA_PHYS * R_PHYS                  # [m/s]
N_BLADES     = 3
N_RADIAL     = 80                                    # ALM marker count per blade

RHO_PHYS     = 1.225                                # [kg/m^3]
MU_PHYS      = 1.81e-5                              # [Pa.s] (matches BEMT)
NU_PHYS      = MU_PHYS / RHO_PHYS                   # [m^2/s]

U_INF_PHYS   = 23.15                                 # [m/s] axial cruise (BEMT base)

# Reference chord/Re at r/R = 0.75 (relative velocity = rotational + axial)
CHORD_REF    = 0.0274                                # [m] tip-region chord
V_75_ROT     = 0.75 * TIP_SPEED                      # [m/s] rotational only
V_75         = float(np.sqrt(V_75_ROT**2 + U_INF_PHYS**2))   # [m/s] relative
RE           = int(round(V_75 * CHORD_REF / NU_PHYS))

# =============================================================================
# S2. LATTICE / GRID PARAMETERS
# =============================================================================
D            = 40                                    # cells per L_char (diameter) on L0
R_LU         = D // 2
U_MAX_LU     = 0.15
RHO_LU       = 1.0

# tip_speed_lu = u_max -> steps_per_rev = pi * D / u_max
STEPS_PER_REV = int(np.pi * D / U_MAX_LU)

# =============================================================================
# S3. DOMAIN (L0): Up 3D, Lat +/-2.5D, Down 7D (same template as APC ALM)
# =============================================================================
Nx           = 10 * D
Ny           = 5 * D
Nz           = 5 * D

HUB_X        = 3 * D
HUB_Y        = Ny // 2
HUB_Z        = Nz // 2

print(f"  [ERC prop Cruise RPM={RPM}, V_inf={U_INF_PHYS} m/s, MLG 3-Level, WALE]")
print(f"  R={R_PHYS} m, D={D_PHYS} m, {N_BLADES} blades")
print(f"  Re_75={RE:,}, tip_speed={TIP_SPEED:.2f} m/s, u_max_lu={U_MAX_LU}")
print(f"  steps/rev~{STEPS_PER_REV}, n_radial={N_RADIAL}")

# =============================================================================
# S4. MLG -- 3 Levels
# =============================================================================
OVERLAP_WIDTH = 4
INTERP_SCHEME = "cubic"
FILTER_LEVEL  = 1

# L1: upstream 0.8D, downstream 4D, lateral +/-1.25D
L1_X_MIN = HUB_X - int(0.8 * D)
L1_X_MAX = HUB_X + 4 * D
L1_Y_MIN = HUB_Y - int(1.25 * D)
L1_Y_MAX = HUB_Y + int(1.25 * D)
L1_Z_MIN = HUB_Z - int(1.25 * D)
L1_Z_MAX = HUB_Z + int(1.25 * D)

# L2: upstream 0.4D, downstream 2D, lateral +/-0.625D
L2_X_MIN = HUB_X - int(0.4 * D)
L2_X_MAX = HUB_X + 2 * D
L2_Y_MIN = HUB_Y - int(0.625 * D)
L2_Y_MAX = HUB_Y + int(0.625 * D)
L2_Z_MIN = HUB_Z - int(0.625 * D)
L2_Z_MAX = HUB_Z + int(0.625 * D)

# =============================================================================
# S5. ERC BLADE GEOMETRY (BEMT 88-station array; chord [m], twist [deg])
# =============================================================================
# r/R = 0.13 + 0.01*i for i = 0..87 (88 stations).
# We use 87 sections (interval midpoints i_mid = 0..86 -> r/R 0.135..0.995),
# matching the BEMT mapping  "interval i -> (i+14)section.csv".
_R_OVER_R_STATION = [round(0.13 + 0.01 * i, 2) for i in range(88)]
_CHORD_M = [  # [m] absolute chord (BEMT 'CHORD_OVER_R' is spanwise chord, not c/R)
    0.038, 0.03755, 0.03709, 0.03664, 0.03618, 0.03586, 0.03563, 0.0354, 0.03516,
    0.03493, 0.0347, 0.03448, 0.03425, 0.03402, 0.03379, 0.03356, 0.03333, 0.03309,
    0.03286, 0.03264, 0.03241, 0.03218, 0.03195, 0.03172, 0.03149, 0.03126, 0.03102,
    0.031, 0.031, 0.031, 0.031, 0.03094, 0.03083, 0.03071, 0.03059, 0.03048, 0.03036,
    0.03025, 0.03014, 0.03002, 0.02991, 0.02979, 0.02967, 0.02956, 0.02944, 0.02933,
    0.02922, 0.0291, 0.02899, 0.02887, 0.02876, 0.02864, 0.02852, 0.02841, 0.0283,
    0.02818, 0.02807, 0.02794, 0.0278, 0.02767, 0.02753, 0.0274, 0.0274, 0.0274,
    0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274,
    0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274,
    0.0274, 0.0274, 0.0274, 0.0274,
]
_TWIST_DEG = [  # [deg] absolute pitch
    24.9, 24.7409, 24.5818, 24.4227, 24.2636, 24.1023, 23.9395, 23.7767, 23.614,
    23.4523, 23.2932, 23.1341, 22.975, 22.8159, 22.664, 22.5128, 22.3616, 22.2105,
    22.0614, 21.9136, 21.7659, 21.6182, 21.4721, 21.3326, 21.193, 21.0535, 20.914,
    20.7773, 20.6409, 20.5045, 20.3682, 20.236, 20.1081, 19.9802, 19.8523, 19.725,
    19.6, 19.475, 19.35, 19.225, 19.0977, 18.9698, 18.8419, 18.714, 18.5932,
    18.4795, 18.3659, 18.2523, 18.1384, 18.0221, 17.9058, 17.7895, 17.6733, 17.5682,
    17.4659, 17.3636, 17.2614, 17.1581, 17.0535, 16.9488, 16.8442, 16.7398, 16.6375,
    16.5352, 16.433, 16.3307, 16.2349, 16.1419, 16.0488, 15.9558, 15.8636, 15.7727,
    15.6818, 15.5909, 15.5, 15.4186, 15.3372, 15.2558, 15.1744, 15.0943, 15.0148,
    14.9352, 14.8557, 14.7791, 14.7093, 14.6395, 14.5698, 14.5,
]
assert len(_R_OVER_R_STATION) == 88
assert len(_CHORD_M) == 88
assert len(_TWIST_DEG) == 88

# 87 interval midpoints. chord/twist at midpoint = mean of bordering stations.
_N_SECTIONS = 87
_BLADE_SECTIONS = []
for i in range(_N_SECTIONS):
    rR_mid = 0.5 * (_R_OVER_R_STATION[i] + _R_OVER_R_STATION[i + 1])
    chord  = 0.5 * (_CHORD_M[i]          + _CHORD_M[i + 1])
    twist  = 0.5 * (_TWIST_DEG[i]        + _TWIST_DEG[i + 1])
    csv_k  = i + 14                                    # 14..100
    _BLADE_SECTIONS.append(
        dict(r=float(rR_mid * R_PHYS),
             chord=float(chord),
             twist=float(twist),
             airfoil=f"sec_{csv_k:03d}",
             active=True)
    )

# 87 CSV polar registrations (one per blade section).
_POLAR_DIR = "input_files/erc_prop"                    # relative to working dir
_AIRFOIL_DB = {
    f"sec_{k:03d}": {
        "method":   "csv",
        "csv_path": f"{_POLAR_DIR}/{k}section.csv",
        "alpha_col": "AoA(deg)",
        "Re_col":    "Re",
        "CL_col":    "cl",
        "CD_col":    "cd",
    }
    for k in range(14, 101)
}

# =============================================================================
# S6. CONFIG BLOCKS (latest format)
# =============================================================================
simulation = {
    "device_mode":     "gpu",
    "precision":       "float32",
    "dimension":       3,
    "lattice_model":   "D3Q27",
    "collision_model": "cumulant",
}

physics = {
    "rho":            RHO_PHYS,
    "U_inf":          U_INF_PHYS,       # 0 (hover)
    "Re":             RE,
    "nu": ((U_INF_PHYS) * (D_PHYS) / (RE)),  # [m^2/s] auto-migrated (nu-only policy; Re key now ignored)
    "L_char":         D_PHYS,
    "flow_direction": [1.0, 0.0, 0.0],
}

grid = {
    "Nx":         Nx,
    "Ny":         Ny,
    "Nz":         Nz,
    "resolution": D,
}

numerics = {
    "u_max":     U_MAX_LU,
    "collision": "cumulant",
}

# Axial cruise: uniform freestream along +x, sponge at xmax.
# Lateral/spanwise far-field carries the same freestream.
# (Domain BC policy: eq / neumann / sponge only.)
boundaries = {
    "xmin": {"location": "xmin", "method": "eq",
             "velocity": [U_INF_PHYS, 0.0, 0.0]},
    "xmax": {"location": "xmax", "method": "sponge",
             "velocity": [U_INF_PHYS, 0.0, 0.0], "density": RHO_LU,
             "thickness": 20, "strength": 0.5},
    "ymin": {"location": "ymin", "method": "eq",
             "velocity": [U_INF_PHYS, 0.0, 0.0]},
    "ymax": {"location": "ymax", "method": "eq",
             "velocity": [U_INF_PHYS, 0.0, 0.0]},
    "zmin": {"location": "zmin", "method": "eq",
             "velocity": [U_INF_PHYS, 0.0, 0.0]},
    "zmax": {"location": "zmax", "method": "eq",
             "velocity": [U_INF_PHYS, 0.0, 0.0]},
}

internal_geometry = {"type": "none"}

airfoil_polar = {
    "method":   "multi",
    "default":  "sec_057",       # r/R ~ 0.565 representative section
    "airfoils": _AIRFOIL_DB,
}

actuator_line = {
    "enabled": True,
    "rotor": {
        "rpm":              RPM,
        "radius":           R_PHYS,
        "omega":            OMEGA_PHYS,
        "n_blades":         N_BLADES,
        "hub_center":       [HUB_X, HUB_Y, HUB_Z],
        "rotation_axis":    [1, 0, 0],
        "thrust_direction": [-1, 0, 0],
        "theta_0":          0.0,
        "blade": {"sections": _BLADE_SECTIONS},
        "grid":  {"n_radial": N_RADIAL},
    },
    "gaussian_cutoff": 3.0,
    "rho_ref":         1.0,
    "coeff_mode":      "auto",       # hover -> rotor convention auto-selected
    "ramp_steps":      5000,
    "prandtl_loss":    True,
}

mlg = {
    "enabled":       True,
    "num_levels":    3,
    "overlap_width": OVERLAP_WIDTH,
    "interpolation": INTERP_SCHEME,
    "filter_level":  FILTER_LEVEL,
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

time = {
    "max_steps":             40 * STEPS_PER_REV,
    "output_interval":       STEPS_PER_REV // 18,
    "logging_interval":      max(1, STEPS_PER_REV // 360),
    "checkpoint_interval":   10 * STEPS_PER_REV,
    "conservation_interval": STEPS_PER_REV // 2,
}

# Wall-aware SGS: WALE (Nicoud & Ducros 1999, Cw=0.5 standard).
sgs = {"enabled": True, "model": "wale", "Cw": 0.5}

conservation      = {"enabled": True, "verbose": 0, "log_to_csv": True}
convergence       = {"enabled": False}
force_calculation = {"enabled": False}

_folder = f"result_erc_cruise_{RPM}rpm_mlg3"
output = {
    "output_dir":     f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir":        f"./{_folder}/csv",
    "clear_previous": True,
    "vtk":            {"enabled": True, "precision": "float32",
                       "variables": ["density", "pressure", "velocity",
                                     "velocity_magnitude", "nu_t"]},
    "checkpoint":     {"enabled": True, "keep_last_n": 2},
}

config = {
    "simulation":        simulation,
    "physics":           physics,
    "grid":              grid,
    "numerics":          numerics,
    "boundaries":        boundaries,
    "internal_geometry": internal_geometry,
    "mlg":               mlg,
    "sgs":               sgs,
    "airfoil_polar":     airfoil_polar,
    "actuator_line":     actuator_line,
    "conservation":      conservation,
    "convergence":       convergence,
    "force_calculation": force_calculation,
    "output":            output,
    "time":              time,
}

# =============================================================================
# S7. STANDALONE -- geometry + grid preview
# =============================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from src.grid.overlap_manager import OverlapManager, IndexBox
    from src.grid.level_scaling import LevelScaler

    print(f"\n  Sections: {len(_BLADE_SECTIONS)}  "
          f"(r/R {_BLADE_SECTIONS[0]['r']/R_PHYS:.3f} ~ "
          f"{_BLADE_SECTIONS[-1]['r']/R_PHYS:.3f})")
    print(f"  Polar DB: {len(_AIRFOIL_DB)} CSVs in '{_POLAR_DIR}'")

    num_levels = mlg["num_levels"]
    levels_config = mlg["levels"]
    _nu_lu_est = U_MAX_LU * D / RE
    _tau_est = 0.5 + 3.0 * _nu_lu_est
    scaler = LevelScaler(tau_0=_tau_est, num_levels=num_levels)
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
        dx_mm = (D_PHYS / D) * 1000 / (2**k)
        print(f"  L{k}: {fs[0]}x{fs[1]}x{fs[2]} = {nodes:,}  (dx={dx_mm:.2f}mm)")
        coarse_shape = fs

    print(f"  Total: {total_nodes:,} nodes, "
          f"{total_nodes * 27 * 4 * 2 / 1e9:.1f} GB")
    print(f"  Output: ./{_folder}/")
