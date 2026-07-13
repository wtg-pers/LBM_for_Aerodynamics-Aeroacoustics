"""
APC 9x4.5MR Propeller -- Axial Flight J=0.608, Cumulant + ALM + MLG (3-Level)

Reproduction of the prior working old-format setup (configs/alm_9x45e_J02_mlg3.py)
with J changed from 0.2 to 0.608 and ported to the latest config schema.

Match goal: reproduce the old positive-thrust result (CT~0.0066, CP~0.0035 at
J=0.608 in the old-format runs). Every numerical setting is kept identical to
the old J0.2 config aside from:
    - J        : 0.2     -> 0.608        (the physics under test)
    - the config format itself (top-level physics/grid/numerics, time block).

Diff from the latest J0608_mlg5 (this repro reverts):
    - Domain                 : 7D x 5D x 5D, hub at 20%   -> 10D x 5D x 5D, hub at 3D
    - MLG                    : 5-level                    -> 3-level
    - BC (sides + inlet)     : eq                         -> regularized_inlet
    - sponge key             : strength                   -> sigma_max  (same value 0.1)
    - u_max                  : 0.15                       -> 0.10
    - ramp_steps             : 5000                       -> 500
    - SGS                    : WALE                       -> off

Usage:
    python main.py --config configs/alm/alm_9x45e_J0608_repro.py
"""

import numpy as np

# =============================================================================
# S1. PROPELLER & FLOW PARAMETERS (physical units)
# =============================================================================
R_IN       = 4.50                                  # [in]
IN_TO_M    = 0.0254
R_PHYS     = R_IN * IN_TO_M                        # 0.1143 m
D_PHYS     = 2 * R_PHYS                            # 0.2286 m
RPM        = 5000
N_REV_S    = RPM / 60.0
OMEGA_PHYS = RPM * 2.0 * np.pi / 60.0              # 523.6 rad/s
TIP_SPEED  = OMEGA_PHYS * R_PHYS                   # 59.85 m/s
N_BLADES   = 2
N_RADIAL   = 40

J          = 0.608
U_INF_PHYS = J * N_REV_S * D_PHYS                  # 11.58 m/s

# Old config used hardcoded RE = 52238 (J=0.2), but recomputing from same
# air properties at this RPM is consistent with the old chain.
RHO_PHYS   = 1.225
MU_PHYS    = 1.78943e-5                            # air STP
NU_PHYS    = MU_PHYS / RHO_PHYS                    # 1.461e-5 m^2/s

CHORD_REF  = 0.150 * R_PHYS                        # 0.01715 m (c/R=0.150 at r/R=0.75)
V_75       = 0.75 * TIP_SPEED                      # 44.89 m/s
RE         = int(round(V_75 * CHORD_REF / NU_PHYS))

# =============================================================================
# S2. LATTICE / GRID PARAMETERS  (same as old J0.2)
# =============================================================================
D          = 40
R_LU       = D // 2
U_MAX_LU   = 0.10                                  # OLD value (not 0.15)
RHO_LU     = 1.0

STEPS_PER_REV = int(np.pi * D / U_MAX_LU)

# =============================================================================
# S3. DOMAIN (L0): 10D x 5D x 5D, hub at 3D from inlet (OLD layout)
# =============================================================================
Nx = 10 * D                                        # 400
Ny = 5 * D                                         # 200
Nz = 5 * D                                         # 200

HUB_X = 3 * D                                      # 120
HUB_Y = Ny // 2                                    # 100
HUB_Z = Nz // 2                                    # 100

print(f"  [APC 9x4.5MR J={J}, MLG 3-Level REPRO]")
print(f"  Re_75={RE:,}, D={D}, {N_BLADES} blades, RPM={RPM}")
print(f"  U_inf={U_INF_PHYS:.2f} m/s, tip_speed={TIP_SPEED:.2f}, u_max_lu={U_MAX_LU}")
print(f"  steps/rev~{STEPS_PER_REV}")

# =============================================================================
# S4. MLG -- 3 Levels (OLD layout)
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
# S5. BLADE GEOMETRY (UIUC data — same as old J0.2 _SECTIONS_UIUC)
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
# S6. CONFIG BLOCKS (new format)
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
    "resolution": D,
}

numerics = {
    "u_max":     U_MAX_LU,
    "collision": "cumulant",
}

# Old-format BC pattern (regularized_inlet on all sides + xmax sponge),
# but in NEW unit convention: BC velocity is given in physical units [m/s];
# setup.py:526-543 internally converts to LU via UnitConverter.
boundaries = {
    "inlet":  {"location": "xmin", "method": "regularized_inlet",
               "velocity": [U_INF_PHYS, 0.0, 0.0], "rho": RHO_LU},
    "outlet": {"location": "xmax", "method": "sponge",
               "velocity": [U_INF_PHYS, 0.0, 0.0], "density": RHO_LU,
               "thickness": 20, "sigma_max": 0.1},
    "ymin":   {"location": "ymin", "method": "regularized_inlet",
               "velocity": [U_INF_PHYS, 0.0, 0.0], "rho": RHO_LU},
    "ymax":   {"location": "ymax", "method": "regularized_inlet",
               "velocity": [U_INF_PHYS, 0.0, 0.0], "rho": RHO_LU},
    "zmin":   {"location": "zmin", "method": "regularized_inlet",
               "velocity": [U_INF_PHYS, 0.0, 0.0], "rho": RHO_LU},
    "zmax":   {"location": "zmax", "method": "regularized_inlet",
               "velocity": [U_INF_PHYS, 0.0, 0.0], "rho": RHO_LU},
}

internal_geometry = {"type": "none"}

airfoil_polar = {
    "method":  "multi",
    "default": "e63",
    "airfoils": {
        "e63":      {"method": "neuralfoil", "airfoil_name": "e63",
                     "Re_target": RE, "Re_min": 10000, "Re_max": 100000,
                     "Re_steps": 10, "mode": "asb", "ncrit": 9.0},
        "naca4412": {"method": "neuralfoil", "airfoil_name": "naca4412",
                     "Re_target": RE, "Re_min": 10000, "Re_max": 100000,
                     "Re_steps": 10, "mode": "asb", "ncrit": 9.0},
    },
}

actuator_line = {
    "enabled": True,
    "rotor": {
        "rpm":              RPM,
        "radius":           R_PHYS,
        "omega":            OMEGA_PHYS,
        "n_blades":         N_BLADES,
        "hub_center":       [HUB_X, HUB_Y, HUB_Z],   # L0 LU
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
    "ramp_steps":      500,                        # OLD value (not 5000)
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

# SGS off (match the old result chain).
sgs = {"enabled": False, "model": "off"}

conservation      = {"enabled": True, "verbose": 0, "log_to_csv": True}
convergence       = {"enabled": False}
force_calculation = {"enabled": False}

_folder = f"result_apc9x45e_J{J:.3f}_Re{RE // 1000}k_mlg3_repro"

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
