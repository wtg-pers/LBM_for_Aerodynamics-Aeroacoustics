"""
APC 9x4.5MR Propeller -- Axial Flight J=0.129, Cumulant + ALM + MLG (3-Level)

UIUC blade geometry + dual airfoil (E63 + NACA4412).
Target (propeller convention): CT, CP from UIUC database

Grid (D = 40, 3-level MLG):
  - L0: 400 x 200 x 200 (10D x 5D x 5D)
  - L1: refined near rotor + near wake
  - L2: finest level around rotor disk

Usage:
    python main.py --config configs/alm/alm_9x45e_J0129_mlg3.py
"""

import numpy as np

# =============================================================================
# S1. PROPELLER & FLOW PARAMETERS (physical units)
# =============================================================================
R_IN       = 4.50                                  # [in] prop radius
IN_TO_M    = 0.0254                                # [m/in]
R_PHYS     = R_IN * IN_TO_M                        # [m] = 0.1143
D_PHYS     = 2 * R_PHYS                            # [m] = 0.2286
RPM        = 5000
N_REV_S    = RPM / 60.0                            # [rev/s]
OMEGA_PHYS = RPM * 2.0 * np.pi / 60.0              # [rad/s]
TIP_SPEED  = OMEGA_PHYS * R_PHYS                   # [m/s]
N_BLADES   = 2
N_RADIAL   = 40

# Advance ratio
J          = 0.193
U_INF_PHYS = J * N_REV_S * D_PHYS                  # [m/s]

RHO_PHYS   = 1.225                                 # [kg/m^3] air STP
MU_PHYS    = 1.78943e-5                            # [Pa·s] air STP
NU_PHYS    = MU_PHYS / RHO_PHYS                    # [m^2/s]

# Reference chord at r/R = 0.75 (UIUC: c/R = 0.150)
CHORD_REF  = 0.150 * R_PHYS                        # [m]
V_75       = 0.75 * TIP_SPEED                      # [m/s]
RE         = int(round(V_75 * CHORD_REF / NU_PHYS))

# =============================================================================
# S2. LATTICE / GRID PARAMETERS
# =============================================================================
D          = 40                                     # prop diameter [cells on L0]
R_LU       = D // 2
U_MAX_LU   = 0.15                                   # tip speed target in LU
RHO_LU     = 1.0

# STEPS_PER_REV derived from u_max = tip_speed_LU:
#   omega_lu = u_max / R_lu  ->  steps_per_rev = 2*pi*R_lu / u_max = pi*D / u_max
STEPS_PER_REV = int(np.pi * D / U_MAX_LU)

# =============================================================================
# S3. DOMAIN (L0): Up = 3D, Lat = 2.5D each side, Down = 7D
# =============================================================================
Nx = 10 * D                                         # 400
Ny = 5 * D                                          # 200
Nz = 5 * D                                          # 200

HUB_X = 3 * D                                       # 120 (3D from inlet)
HUB_Y = Ny // 2                                     # 100
HUB_Z = Nz // 2                                     # 100

print(f"  [APC 9x4.5MR Axial J={J}, RPM={RPM}, MLG 3-Level]")
print(f"  Re_75={RE:,}, D={D}, {N_BLADES} blades")
print(f"  U_inf={U_INF_PHYS:.2f} m/s, tip_speed={TIP_SPEED:.2f} m/s, u_max_lu={U_MAX_LU}")
print(f"  steps/rev~{STEPS_PER_REV}")

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
# S5. BLADE GEOMETRY (UIUC data for APC 9x4.5MR)
# =============================================================================
_SECTIONS_UIUC = [
    # r/R    c/R     beta    airfoil       active
    (0.15,  0.157,  34.80,  "e63",         False),
    (0.20,  0.163,  36.50,  "e63",         False),
    (0.25,  0.187,  34.26,  "e63",         True),
    (0.30,  0.206,  29.64,  "e63",         True),
    (0.35,  0.217,  25.62,  "e63",         True),
    (0.40,  0.222,  22.53,  "e63",         True),
    (0.45,  0.222,  20.25,  "e63",         True),
    (0.50,  0.217,  18.37,  "e63",         True),
    (0.55,  0.209,  16.83,  "e63",         True),
    (0.60,  0.197,  15.51,  "e63",         True),
    (0.65,  0.183,  14.38,  "e63",         True),
    (0.70,  0.167,  13.45,  "e63",         True),
    (0.75,  0.150,  12.56,  "naca4412",    True),
    (0.80,  0.133,  12.09,  "naca4412",    True),
    (0.85,  0.116,  11.25,  "naca4412",    True),
    (0.90,  0.099,  10.46,  "naca4412",    True),
    (0.95,  0.074,   9.68,  "naca4412",    True),
    (1.00,  0.049,   8.90,  "naca4412",    True),
]

# =============================================================================
# S6. CONFIG BLOCKS (latest format: physics / grid / numerics separated)
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
    "U_inf":          U_INF_PHYS,
    "Re":             RE,
    "nu": ((U_INF_PHYS) * (D_PHYS) / (RE)),  # [m^2/s] auto-migrated (nu-only policy; Re key now ignored)
    "L_char":         D_PHYS,
    "flow_direction": [1.0, 0.0, 0.0],
}

grid = {
    "Nx":         Nx,
    "Ny":         Ny,
    "Nz":         Nz,
    "resolution": D,                     # cells per L_char (diameter)
}

numerics = {
    "u_max":     U_MAX_LU,
    "collision": "cumulant",
}

# Axial flight: uniform freestream U_inf along +x.
# eq inlet + sponge outlet, lateral/far-field eq with same freestream.
# (Domain BC policy: eq / neumann / sponge only, no regularized_*.)
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

# Dual airfoil polar: E63 (inner) + NACA4412 (outer)
airfoil_polar = {
    "method":  "multi",
    "default": "e63",
    "airfoils": {
        "e63": {
            "method":       "neuralfoil",
            "airfoil_name": "e63",
            "Re_target":    RE,
            "Re_min":       10000,
            "Re_max":       100000,
            "Re_steps":     10,
            "mode":         "asb",
            "ncrit":        9.0,
        },
        "naca4412": {
            "method":       "neuralfoil",
            "airfoil_name": "naca4412",
            "Re_target":    RE,
            "Re_min":       10000,
            "Re_max":       100000,
            "Re_steps":     10,
            "mode":         "asb",
            "ncrit":        9.0,
        },
    },
}

actuator_line = {
    "enabled": True,
    "rotor": {
        # rpm + radius are read by UnitConverter (tip_speed = omega*R -> dt_phys).
        "rpm":              RPM,
        "radius":           R_PHYS,
        # omega is read by Rotor.from_config (physical rad/s).
        "omega":            OMEGA_PHYS,
        "n_blades":         N_BLADES,
        "hub_center":       [HUB_X, HUB_Y, HUB_Z],
        "rotation_axis":    [1, 0, 0],
        "thrust_direction": [-1, 0, 0],
        "theta_0":          0.0,
        "blade": {
            "sections": [
                {"r":       float(rR * R_PHYS),
                 "chord":   float(cR * R_PHYS),
                 "twist":   float(tw),
                 "airfoil": af,
                 "active":  act}
                for rR, cR, tw, af, act in _SECTIONS_UIUC
            ],
        },
        "grid": {"n_radial": N_RADIAL},
    },
    "gaussian_cutoff": 3.0,
    "rho_ref":         1.0,
    "coeff_mode":      "auto",
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
    "max_steps":             20 * STEPS_PER_REV,
    "output_interval":       STEPS_PER_REV,
    "logging_interval":      max(1, STEPS_PER_REV // 10),
    "checkpoint_interval":   5 * STEPS_PER_REV,
    "conservation_interval": STEPS_PER_REV // 2,
}

# Wall-aware SGS: WALE (Nicoud & Ducros 1999, Cw=0.5 standard).
sgs = {"enabled": True, "model": "wale", "Cw": 0.5}

conservation = {"enabled": True, "verbose": 0, "log_to_csv": True}
convergence       = {"enabled": False}
force_calculation = {"enabled": False}

_folder = f"result_apc9x45e_J{J:.3f}_Re{RE // 1000}k_mlg3"

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
# S7. STANDALONE -- grid preview
# =============================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from src.grid.overlap_manager import OverlapManager, IndexBox
    from src.grid.level_scaling import LevelScaler

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
