"""
APC 9x4.5MR Propeller -- Axial Flight J=0.608, Cumulant + ALM + MLG (4-Level)

UIUC blade geometry + dual airfoil (E63 + NACA4412).
Target (propeller convention): CT, CP from UIUC database.

Grid (D=40 on L0, refine 2x per level -> L3 has 320 cells/D, dx_L3=0.714 mm):
  - L0: 7D x 5D x 5D = 280 x 200 x 200,  hub at 20% Nx (x=1.4D)
  - L1: ±0.5D up / 2.5D down / ±1.0D lat  (80 c/D)
  - L2: ±0.2D up / 0.8D down / ±0.65D lat (160 c/D)
  - L3: ±0.1D up / 0.2D down / ±0.6D lat  (320 c/D, rotor disk + Gaussian eps)

Tip resolution at L3 (r/R=1.00, chord_tip=5.6mm):
  chord_tip = 7.84 L3 cells, eps_tip = 2.0 L3 cells (floor: 2·Δx == chord/4
  within 2%). Only the very tip marker sits at the floor; r/R<=0.95 is chord/4
  dominated (>=2.96 cells), so floor risk is localized to a single marker.

Total ~38.3M cells (~8 GB distribution, ~16 GB with buffers).

Sponge: thickness=20 lu (=0.5D in L0), strength=0.1 (mild absorption).

Usage:
    python main.py --config configs/alm/alm_9x45e_J0608_mlg4_test.py

NOTE: 4-level downgrade of alm_9x45e_J0608_mlg5.py for cheaper Step 1
isolation runs. L4 region dropped; disk plane now sits at L3 resolution
(640 → 320 cells/D). Same L3 region (rotor disk + Gaussian cutoff) retained.
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
J          = 0.608
U_INF_PHYS = J * N_REV_S * D_PHYS                  # [m/s]

RHO_PHYS   = 1.225                                 # [kg/m^3] air STP
NU_PHYS    = 1.4e-5                                # [m^2/s] kinematic viscosity (air STP)

# Reference chord at r/R = 0.75 (UIUC: c/R = 0.150)
CHORD_REF  = 0.150 * R_PHYS                        # [m]
V_75       = 0.75 * TIP_SPEED                      # [m/s]
RE         = int(round(V_75 * CHORD_REF / NU_PHYS))

# =============================================================================
# S2. LATTICE / GRID PARAMETERS
# =============================================================================
D          = 40                                     # prop diameter [cells on L0]
R_LU       = D // 2
RHO_LU     = 1.0

# u_max derived from acoustic_scaling (c_s_phys=340) in numerics block:
#   u_max = U_max_phys / (c_s_phys * sqrt(3))
# For 5000 RPM, R=0.1143 m -> tip=59.85 m/s -> u_max ~ 0.1016.
U_MAX_LU_EST = TIP_SPEED / (340.0 * np.sqrt(3.0))    # estimate for STEPS_PER_REV
STEPS_PER_REV = int(np.pi * D / U_MAX_LU_EST)

# =============================================================================
# S3. DOMAIN (L0): 7D x 5D x 5D, hub at 20% of Nx (1.4D from inlet)
# =============================================================================
Nx = 7 * D                                          # 280
Ny = 5 * D                                          # 200
Nz = 5 * D                                          # 200

HUB_X = int(0.2 * Nx)                               # 56 = 1.4D from inlet
HUB_Y = Ny // 2                                     # 100
HUB_Z = Nz // 2                                     # 100

print(f"  [APC 9x4.5MR Axial J={J}, RPM={RPM}, MLG 4-Level]")
print(f"  Re_75={RE:,}, D={D}, {N_BLADES} blades")
print(f"  U_inf={U_INF_PHYS:.2f} m/s, tip_speed={TIP_SPEED:.2f} m/s,"
      f" u_max_lu~{U_MAX_LU_EST:.4f} (acoustic-derived)")
print(f"  steps/rev~{STEPS_PER_REV}")
print(f"  Domain (in D): inlet->hub=1.4, hub->outlet=5.6, lat=+-2.5")

# =============================================================================
# S4. MLG -- 4 Levels (L0..L3)
# =============================================================================
OVERLAP_WIDTH = 2
INTERP_SCHEME = "cubic"
FILTER_LEVEL  = 1

# L1: hub +/-(0.5D up, 2.5D down) x +/-1.0D lateral  (80 cells/D)
L1_X_MIN = HUB_X - int(0.5 * D)                     # 56-20 = 36
L1_X_MAX = HUB_X + int(2.5 * D)                     # 56+100 = 156
L1_Y_MIN = HUB_Y - int(1.0 * D)                     # 60
L1_Y_MAX = HUB_Y + int(1.0 * D)                     # 140
L1_Z_MIN = HUB_Z - int(1.0 * D)                     # 60
L1_Z_MAX = HUB_Z + int(1.0 * D)                     # 140

# L2: hub +/-(0.2D up, 0.8D down) x +/-0.65D lateral  (160 cells/D)
L2_X_MIN = HUB_X - int(0.2 * D)                     # 56-8 = 48
L2_X_MAX = HUB_X + int(0.8 * D)                     # 56+32 = 88
L2_Y_MIN = HUB_Y - int(0.65 * D)                    # 100-26 = 74
L2_Y_MAX = HUB_Y + int(0.65 * D)                    # 126
L2_Z_MIN = HUB_Z - int(0.65 * D)                    # 74
L2_Z_MAX = HUB_Z + int(0.65 * D)                    # 126

# L3: hub +/-(0.1D up, 0.2D down) x +/-0.6D lateral  (320 cells/D)
# This is the finest level in MLG4 -- contains the rotor disk + Gaussian
# spread (3*eps_tip ~= 0.019D axial, well within +/-0.1D padding).
L3_X_MIN = HUB_X - int(0.1 * D)                     # 56-4 = 52
L3_X_MAX = HUB_X + int(0.2 * D)                     # 56+8 = 64
L3_Y_MIN = HUB_Y - int(0.6 * D)                     # 100-24 = 76
L3_Y_MAX = HUB_Y + int(0.6 * D)                     # 124
L3_Z_MIN = HUB_Z - int(0.6 * D)                     # 76
L3_Z_MAX = HUB_Z + int(0.6 * D)                     # 124

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
    "nu":             NU_PHYS,        # 1.4e-5 m^2/s; Re derived inside UnitConverter
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
    # Acoustic scaling: UnitConverter derives u_max so LBM's effective c_s
    # equals c_s_phys (340 m/s, air STP). Required for aeroacoustic accuracy.
    "acoustic_scaling": True,
    "c_s_phys":         340.0,
    "collision":        "cumulant",
}

# Axial flight: uniform freestream U_inf along +x.
# eq inlet + sponge outlet, lateral/far-field eq with same freestream.
# (Domain BC policy: eq / neumann / sponge only, no regularized_*.)
boundaries = {
    "xmin": {"location": "xmin", "method": "eq",
             "velocity": [U_INF_PHYS, 0.0, 0.0]},
    "xmax": {"location": "xmax", "method": "sponge",
             "velocity": [U_INF_PHYS, 0.0, 0.0], "density": RHO_LU,
             "thickness": 20, "strength": 0.1},
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
            "ncrit":        6.0,
        },
        "naca4412": {
            "method":       "neuralfoil",
            "airfoil_name": "naca4412",
            "Re_target":    RE,
            "Re_min":       10000,
            "Re_max":       100000,
            "Re_steps":     10,
            "mode":         "asb",
            "ncrit":        6.0,
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
    "num_levels":    4,
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
        {"region": {"x_min": L3_X_MIN, "x_max": L3_X_MAX,
                    "y_min": L3_Y_MIN, "y_max": L3_Y_MAX,
                    "z_min": L3_Z_MIN, "z_max": L3_Z_MAX}},
    ],
}

time = {
    "max_steps":             20 * STEPS_PER_REV,
    "output_interval":       STEPS_PER_REV,
    "logging_interval":      max(1, STEPS_PER_REV // 10),
    "checkpoint_interval":   5 * STEPS_PER_REV,
    "conservation_interval": STEPS_PER_REV // 2,
}

# SGS off for controlled Step 1 isolation (matches mlg5 baseline that produced
# the negative-CT observation). MLG4 has ~25% less memory than MLG5, so SGS
# can safely be re-enabled later for production runs if desired.
sgs = {"enabled": False, "model": "off"}

conservation = {"enabled": True, "verbose": 0, "log_to_csv": True}
convergence       = {"enabled": False}
force_calculation = {"enabled": False}

_folder = f"result_apc9x45e_J{J:.3f}_Re{RE // 1000}k_mlg4"

output = {
    "output_dir":     f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir":        f"./{_folder}/csv",
    "clear_previous": True,
    "vtk":            {"enabled": True, "precision": "float32",
                       "variables": ["density", "pressure", "velocity",
                                     "velocity_magnitude"]},
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
    _nu_lu_est = U_MAX_LU_EST * D / RE
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
