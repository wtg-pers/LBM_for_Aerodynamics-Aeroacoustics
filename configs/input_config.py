import numpy as np
import warnings


L_REF      = 0.025           # [m]      Characteristic length (= blade chord c)
                             #          Used for Re definition: Re = U_ref·L_ref/ν
RHO_REF    = 1.205           # [kg/m³]  Reference density (air at 20°C)

# -----------------------------------------------------------------------------
# Rotor geometry (for ALM)
# -----------------------------------------------------------------------------
R_ROTOR    = 0.125           # [m]      Rotor radius (USER INPUT, independent of L_REF)
D_ROTOR    = 2.0 * R_ROTOR  # [m]      Rotor diameter

# -----------------------------------------------------------------------------
# Rotor operating condition (for ALM)
# -----------------------------------------------------------------------------
RPM        = 1000            # [rpm]    Rotor rotational speed (USER INPUT)
OMEGA_ROTOR = RPM * 2.0 * np.pi / 60.0   # [rad/s]  Angular velocity

# -----------------------------------------------------------------------------
# Reference velocity (for Re calculation) vs Inlet velocity
# -----------------------------------------------------------------------------
U_REF      = OMEGA_ROTOR * (R_ROTOR * 0.75)  # [m/s]    Reference velocity
U_INF      = 2             # [m/s]    Freestream/inlet velocity (0 for hover)

# ─── u_inf_lu 계산 ──────────────────────────────────────────────
# U_inf = 0.1 m/s  (≈ 1% of U_tip = 9.8175 m/s)
# u_inf_lu = U_inf × dt / dx
#          = 0.1 × (u_lu × dx / U_ref) / dx
#          = 0.1 × u_lu / U_ref
#          = 0.1 × 0.1 / 9.8175 ≈ 0.00102 [Δx/Δt]


# -----------------------------------------------------------------------------
# Reynolds number OR viscosity (choose ONE, Re takes priority)
# -----------------------------------------------------------------------------
RE         = 1000             # [-]      Reynolds number (PRIORITY)
NU_PHYS    = None            # [m²/s]   Kinematic viscosity (set if RE=None)

# --- Auto-calculate: Re ↔ ν_phys ---
if RE is not None and NU_PHYS is not None:
    _nu_computed = U_REF * L_REF / RE
    warnings.warn(
        f"Both RE and NU_PHYS defined. Using RE={RE} (priority).\n"
        f"  Computed ν = {_nu_computed:.6e} m²/s\n"
        f"  Provided ν = {NU_PHYS:.6e} m²/s (ignored)"
    )
    NU_PHYS = _nu_computed
elif RE is not None:
    NU_PHYS = U_REF * L_REF / RE
elif NU_PHYS is not None:
    RE = U_REF * L_REF / NU_PHYS
else:
    raise ValueError("Either 'RE' or 'NU_PHYS' must be defined!")

# =============================================================================
# §2. NUMERICAL PARAMETERS (User Input - Discretization)
# =============================================================================

RESOLUTION = 35              # [-]  Grid cells per rotor diameter (D/Δx)
                             #      ALM standard: 20~40 for test, 60+ for production
LATTICE_VELOCITY = 0.1      # [-]  u_lu (controls accuracy, recommend 0.02~0.1)

# =============================================================================
# §3. AUTO-CALCULATED LATTICE PARAMETERS
# =============================================================================
# DO NOT MODIFY - These are derived from user inputs above.

# --- Conversion factors ---
DX_PHYS    = D_ROTOR / RESOLUTION                    # [m/lu]  grid spacing
DT_PHYS    = LATTICE_VELOCITY * DX_PHYS / U_REF      # [s/lt]  timestep

OMEGA_LU = OMEGA_ROTOR * DT_PHYS
STEPS_PER_REV = int(2 * np.pi / OMEGA_LU)

# --- Lattice parameters ---
L_REF_LU   = L_REF / DX_PHYS                         # [lu]  chord in lattice units
D_LU        = RESOLUTION                              # [lu]  rotor diameter = D/Δx
R_LU        = D_LU / 2.0                              # [lu]  rotor radius
U_REF_LU   = LATTICE_VELOCITY                        # [lu/lt]
U_INF_LU   = U_INF / U_REF * LATTICE_VELOCITY        # [lu/lt] inlet velocity in LU
NU_LU      = NU_PHYS * DT_PHYS / (DX_PHYS ** 2)      # [lu²/lt]
TAU        = 0.5 + 3.0 * NU_LU                       # [-]
OMEGA_LBM  = 1.0 / TAU                               # [-] relaxation frequency (NOT rotor omega!)

# --- Lattice Mach number (for reference) ---
CS = 1.0 / np.sqrt(3.0)      # Lattice sound speed ≈ 0.577
MA_LATTICE = LATTICE_VELOCITY / CS

# --- Stability checks ---
_stability_warnings = []
if TAU <= 0.5:
    raise ValueError(f"CRITICAL: τ = {TAU:.4f} ≤ 0.5 → UNSTABLE!")
if TAU < 0.52:
    _stability_warnings.append(f"⚠️  τ = {TAU:.4f} < 0.52 → marginally stable")
if TAU > 2.0:
    _stability_warnings.append(f"⚠️  τ = {TAU:.4f} > 2.0 → accuracy degradation")
if MA_LATTICE > 0.3:
    raise ValueError(f"CRITICAL: Ma_lattice = {MA_LATTICE:.3f} > 0.3 → compressibility error!")
if MA_LATTICE > 0.1:
    _stability_warnings.append(f"⚠️  Ma_lattice = {MA_LATTICE:.3f} > 0.1 → ~{MA_LATTICE**2*100:.1f}% density error")

for _w in _stability_warnings:
    print(_w)

# =============================================================================
# §4. DOMAIN CONFIGURATION
# =============================================================================
# Domain size in lattice units, as multiples of rotor diameter D
# DOMAIN_MULT_X = 5            # [-]  Domain = 5D in x
# DOMAIN_MULT_Y = 5            # [-]  Domain = 5D in y
# DOMAIN_MULT_Z = 6            # [-]  Domain = 8D in z (rotation axis)

# Nx = DOMAIN_MULT_X * RESOLUTION    # [lu]
# Ny = DOMAIN_MULT_Y * RESOLUTION    # [lu]
# Nz = DOMAIN_MULT_Z * RESOLUTION    # [lu]

# Edge flight test quadcopter
Nx = 250
Ny = 200
Nz = 100

# # L+C test
# Nx = 250
# Ny = 100  # 230
# Nz = 100

# =============================================================================
# §4. SIMULATION SETTINGS
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "dimension": 3,
    "lattice_model": "D3Q27",
    
    "domain": {"Nx": Nx, "Ny": Ny, "Nz": Nz,},
    
    "physics": {
        # Physical (SI)
        "L_ref": L_REF,            # [m]
        "U_ref": U_REF,            # [m/s]
        "U_inf": U_INF,            # [m/s]
        "rho_ref": RHO_REF,        # [kg/m³]
        "nu_phys": NU_PHYS,        # [m²/s]
        "Re": RE,                  # [-]
        
        # Lattice (auto-calculated)
        "tau": TAU,
        "omega": OMEGA_LBM,
        "nu_lu": NU_LU,
        "u_ref_lu": U_REF_LU,
        "L_ref_lu": L_REF_LU,

        # Initial flow field
        "initial_flow_velocity": [U_INF_LU, 0.0, 0.0],
        
        # Conversion factors
        "dx": DX_PHYS,             # [m/lu]
        "dt": DT_PHYS,             # [s/lt]
    },
    
    "time": {
        "max_steps": STEPS_PER_REV * 40,
        "output_interval": STEPS_PER_REV // 12,
        "checkpoint_interval": STEPS_PER_REV * 10,
        "probe_interval": 10,
    },
}

# =============================================================================
# §5. BOUNDARY CONDITIONS
# =============================================================================
# Adjust based on your case type

# boundaries = {
#     # Example: External flow (cylinder, airfoil)
#     "inlet":  {"location": "xmin", "method": "equilibrium", 
#                "velocity": U_REF_LU, "density": 1.0},
#     "outlet": {"location": "xmax", "method": "neumann"},
#     "top":    {"location": "ymax", "method": "equilibrium", 
#                "velocity": U_REF_LU, "density": 1.0},
#     "bottom": {"location": "ymin", "method": "equilibrium", 
#                "velocity": U_REF_LU, "density": 1.0},
#     "front":  {"location": "zmin", "method": "periodic"},
#     "back":   {"location": "zmax", "method": "periodic"},
# }

# boundaries = {
#     # "ground": {"location": "zmin", "method": "regularized_wall",},
#     "ground": {"location": "zmin", "method": "regularized_outlet", 
#             "velocity": 0.0, "density": 1.0, "k": 0.1},
#     "top": {"location": "zmax", "method": "regularized_inlet", 
#             'velocity': [0.0, 0.0, -0.001],
#             "density": 1.0, "k": 0.1},

#     "xmin": {"location": "xmin", "method": "regularized_outlet",
#              "velocity": 0.0,"density": 1.0, "k": 0.1},
#     "xmax": {"location": "xmax","method": "regularized_outlet",
#              "velocity": 0.0,"density": 1.0, "k": 0.1},
#     "ymin": {"location": "ymin","method": "regularized_outlet",
#              "velocity": 0.0,"density": 1.0, "k": 0.1},
#     "ymax": {"location": "ymax","method": "regularized_outlet",
#              "velocity": 0.0,"density": 1.0, "k": 0.1},
# }

# boundaries = {
#     "inlet":  {"location": "xmin", "method": "neumann",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0},
#     "outlet": {"location": "xmax", "method": "neumann"},

#     "ymin":   {"location": "ymin", "method": "neumann",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0},
#     "ymax":   {"location": "ymax", "method": "neumann",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0},

#     "top":    {"location": "zmax", "method": "equilibrium",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0},
#     "ground": {"location": "zmin", "method": "neumann",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0},
# }

boundaries = {
    "inlet":  {"location": "xmin", "method": "equilibrium",
               "velocity": [U_INF_LU, 0.0, 0.0], "density": 1.0},
    "outlet": {"location": "xmax", "method": "neumann"},

    "ymin":   {"location": "ymin", "method": "equilibrium",
               "velocity": [U_INF_LU, 0.0, 0.0], "density": 1.0},
    "ymax":   {"location": "ymax", "method": "equilibrium",
               "velocity": [U_INF_LU, 0.0, 0.0], "density": 1.0},

    "top":    {"location": "zmax", "method": "equilibrium",
               "velocity": [U_INF_LU, 0.0, 0.0], "density": 1.0},
    "ground": {"location": "zmin", "method": "equilibrium",
               "velocity": [U_INF_LU, 0.0, 0.0], "density": 1.0},
}

# =============================================================================
# §5. BOUNDARY CONDITIONS — Sponge Layer on lateral + ground
# =============================================================================

# _SPONGE_THICKNESS = 15        # [lu] buffer zone depth (≈ 0.4D ~ 0.5D)
# _SPONGE_STRENGTH  = 0.3       # [-] σ_max (quadratic ramp, 0.2~0.5 recommended)

# boundaries = {
#     # ── Inlet (top): equilibrium — clean freestream ──
#     "top":    {"location": "zmax", "method": "equilibrium",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0},

#     # ── Outlet (ground): sponge — let downwash exit smoothly ──
#     "ground": {"location": "zmin", "method": "sponge",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0,
#                "thickness": _SPONGE_THICKNESS, "sigma_max": _SPONGE_STRENGTH},

#     # ── Lateral faces: sponge — damp induced lateral flow ──
#     "xmin":   {"location": "xmin", "method": "sponge",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0,
#                "thickness": _SPONGE_THICKNESS, "sigma_max": _SPONGE_STRENGTH},
#     "xmax":   {"location": "xmax", "method": "sponge",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0,
#                "thickness": _SPONGE_THICKNESS, "sigma_max": _SPONGE_STRENGTH},
#     "ymin":   {"location": "ymin", "method": "sponge",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0,
#                "thickness": _SPONGE_THICKNESS, "sigma_max": _SPONGE_STRENGTH},
#     "ymax":   {"location": "ymax", "method": "sponge",
#                "velocity": [0.0, 0.0, -U_INF_LU], "density": 1.0,
#                "thickness": _SPONGE_THICKNESS, "sigma_max": _SPONGE_STRENGTH},
# }

# =============================================================================
# §6. INTERNAL GEOMETRY (Optional Module)
# =============================================================================
# Enable/disable objects in the flow domain

internal_geometry = {
    # Example: Cylinder
    # "cylinder": {
    #     "enabled": True,
    #     "center": [Nx // 4, Ny // 2, Nz // 2],  # [lu]
    #     "diameter": RESOLUTION,                   # [lu] = L_REF
    #     "axis": "z",
    # },
    
    # Example: Airfoil
    # "airfoil": {
    #     "enabled": True,
    #     "naca": "0012",
    #     "chord": RESOLUTION,                      # [lu] = L_REF
    #     "center": [Nx // 4, Ny // 2, Nz // 2],
    #     "angle_of_attack": 5.0,                   # [deg]
    # },
}

# =============================================================================
# §7. ACTUATOR LINE MODEL (Optional Module)
# =============================================================================
ALM_ENABLED = True

if ALM_ENABLED:
    # Rotor geometry (R_ROTOR, OMEGA_ROTOR defined in §1)
    CHORD_BLADE = 0.025             # [m]  Blade chord
    PITCH       = 10.0              # [deg] Collective pitch
    N_BLADES    = 2                 # [-]
    ROOT_CUT    = 0.20              # [-]

    # ── Blade section template (재사용) ──
    def _make_blade_sections(R, chord, pitch):
        """공통 blade section 생성기"""
        return [
            {"r": 0.0,                       "chord": chord, 
             "twist": -pitch, "airfoil": "naca0012", "active": False},
            {"r": R * ROOT_CUT,              "chord": chord, 
             "twist": -pitch, "airfoil": "naca0012", "active": False},
            {"r": R * ROOT_CUT + 1e-6,       "chord": chord, 
             "twist": -pitch, "airfoil": "naca0012", "active": True},
            {"r": R,                         "chord": chord, 
             "twist": -pitch, "airfoil": "naca0012", "active": True},
        ]
    
    # ── Rotor 0: Upwind turbine ──
    HUB_0_X_LU = 80
    HUB_0_Y_LU = 80
    HUB_0_Z_LU = Nz * 0.5

    # ── Rotor 1: Downwind turbine ──
    HUB_1_X_LU = 80
    HUB_1_Y_LU = 120
    HUB_1_Z_LU = Nz * 0.5

    # ── Rotor 2: Upwind turbine ──
    HUB_2_X_LU = 120
    HUB_2_Y_LU = 80
    HUB_2_Z_LU = Nz * 0.5

    # ── Rotor 3: Downwind turbine ──
    HUB_3_X_LU = 120
    HUB_3_Y_LU = 120
    HUB_3_Z_LU = Nz * 0.5

    # # ── Rotor 0  ──
    # HUB_0_X_LU = 30
    # HUB_0_Y_LU = 185
    # HUB_0_Z_LU = 50

    # # ── Rotor 1  ──
    # HUB_1_X_LU = 30
    # HUB_1_Y_LU = 145
    # HUB_1_Z_LU = 50

    # # ── Rotor 2  ──
    # HUB_2_X_LU = 30
    # HUB_2_Y_LU = 85
    # HUB_2_Z_LU = 50

    # # ── Rotor 3  ──
    # HUB_3_X_LU = 30
    # HUB_3_Y_LU = 45
    # HUB_3_Z_LU = 50

    # # ── Rotor 4  ──
    # HUB_4_X_LU = 80
    # HUB_4_Y_LU = 185
    # HUB_4_Z_LU = 50

    # # ── Rotor 5  ──
    # HUB_5_X_LU = 80
    # HUB_5_Y_LU = 145
    # HUB_5_Z_LU = 50

    # # ── Rotor 6  ──
    # HUB_6_X_LU = 80
    # HUB_6_Y_LU = 85
    # HUB_6_Z_LU = 50

    # # ── Rotor 7  ──
    # HUB_7_X_LU = 80
    # HUB_7_Y_LU = 45
    # HUB_7_Z_LU = 50

    # # ── Rotor 8 ──
    # HUB_8_X_LU = 30
    # HUB_8_Y_LU = 50
    # HUB_8_Z_LU = 50

    
    # HUB_X_PHYS  = HUB_X_LU * DX_PHYS   # [m]
    # HUB_Y_PHYS  = HUB_Y_LU * DX_PHYS   # [m]
    # HUB_Z_PHYS  = HUB_Z_LU * DX_PHYS   # [m]
    
    # Polar Re range
    RE_POLAR_MIN    = max(1.0, RE * 0.15)
    RE_POLAR_MAX    = RE * 2.5
    RE_POLAR_TARGET = RE

actuator_line = {
    "enabled": ALM_ENABLED,
}

if ALM_ENABLED:
    actuator_line.update({
        # ─── 공유 파라미터 (shared defaults) ───
        "rho_ref": RHO_REF,                # [kg/m³]
        "gaussian_cutoff": 3.0,            # [-]
        "coeff_mode": "rotorcraft",        # 'wind_turbine' | 'rotorcraft' | 'auto'

        # ─── 단위 변환 (shared) ───
        "units": {
            "dx_phys": DX_PHYS,            # [m/lu]
            "dt_phys": DT_PHYS,            # [s/lt]
            "nu_phys": NU_PHYS,            # [m²/s]
        },

        # ─── Multi-rotor 리스트 ───
        "rotors": [
            {
                "name": "rotor_0",
                "rotor": {
                    "n_blades": N_BLADES,
                    "hub_center": [HUB_0_X_LU * DX_PHYS,
                                   HUB_0_Y_LU * DX_PHYS,
                                   HUB_0_Z_LU * DX_PHYS],  # [m]
                    "omega": -OMEGA_ROTOR,           # [rad/s]
                    "theta_0": 0.0,                 # [rad]
                    "rotation_axis": [0, 0, 1],

                    "blade": {
                        "sections": _make_blade_sections(
                            R_ROTOR, CHORD_BLADE, PITCH
                        ),
                    },
                    "grid": {"n_radial": 20},
                },
                # Per-rotor override (optional):
                # "coeff_mode": "wind_turbine",
                # "gaussian_cutoff": 4.0,
            },
            # ────────────── Rotor 1: Downwind ──────────────
            {
                "name": "rotor_1",
                "rotor": {
                    "n_blades": N_BLADES,
                    "hub_center": [HUB_1_X_LU * DX_PHYS,
                                   HUB_1_Y_LU * DX_PHYS,
                                   HUB_1_Z_LU * DX_PHYS],  # [m]
                    "omega": OMEGA_ROTOR,           # [rad/s]
                    "theta_0": 0.0,                 # [rad]
                    "rotation_axis": [0, 0, 1],

                    "blade": {
                        "sections": _make_blade_sections(
                            R_ROTOR, CHORD_BLADE, PITCH
                        ),
                    },
                    "grid": {"n_radial": 20},
                },
            },

            {
                "name": "rotor_2",
                "rotor": {
                    "n_blades": N_BLADES,
                    "hub_center": [HUB_2_X_LU * DX_PHYS,
                                   HUB_2_Y_LU * DX_PHYS,
                                   HUB_2_Z_LU * DX_PHYS],  # [m]
                    "omega": OMEGA_ROTOR,           # [rad/s]
                    "theta_0": 0.0,                 # [rad]
                    "rotation_axis": [0, 0, 1],

                    "blade": {
                        "sections": _make_blade_sections(
                            R_ROTOR, CHORD_BLADE, PITCH
                        ),
                    },
                    "grid": {"n_radial": 20},
                },
                # Per-rotor override (optional):
                # "coeff_mode": "wind_turbine",
                # "gaussian_cutoff": 4.0,
            },

            {
                "name": "rotor_3",
                "rotor": {
                    "n_blades": N_BLADES,
                    "hub_center": [HUB_3_X_LU * DX_PHYS,
                                   HUB_3_Y_LU * DX_PHYS,
                                   HUB_3_Z_LU * DX_PHYS],  # [m]
                    "omega": -OMEGA_ROTOR,           # [rad/s]
                    "theta_0": 0.0,                 # [rad]
                    "rotation_axis": [0, 0, 1],

                    "blade": {
                        "sections": _make_blade_sections(
                            R_ROTOR, CHORD_BLADE, PITCH
                        ),
                    },
                    "grid": {"n_radial": 20},
                },
                # Per-rotor override (optional):
                # "coeff_mode": "wind_turbine",
                # "gaussian_cutoff": 4.0,
            },

            # {
            #     "name": "rotor_4",
            #     "rotor": {
            #         "n_blades": N_BLADES,
            #         "hub_center": [HUB_4_X_LU * DX_PHYS,
            #                        HUB_4_Y_LU * DX_PHYS,
            #                        HUB_4_Z_LU * DX_PHYS],  # [m]
            #         "omega": OMEGA_ROTOR,           # [rad/s]
            #         "theta_0": 0.0,                 # [rad]
            #         "rotation_axis": "hawt_z",

            #         "blade": {
            #             "sections": _make_blade_sections(
            #                 R_ROTOR, CHORD_BLADE, PITCH
            #             ),
            #         },
            #         "grid": {"n_radial": 20},
            #     },
            #     # Per-rotor override (optional):
            #     # "coeff_mode": "wind_turbine",
            #     # "gaussian_cutoff": 4.0,
            # },
            # # ────────────── Rotor 1: Downwind ──────────────
            # {
            #     "name": "rotor_5",
            #     "rotor": {
            #         "n_blades": N_BLADES,
            #         "hub_center": [HUB_5_X_LU * DX_PHYS,
            #                        HUB_5_Y_LU * DX_PHYS,
            #                        HUB_5_Z_LU * DX_PHYS],  # [m]
            #         "omega": -OMEGA_ROTOR,           # [rad/s]
            #         "theta_0": 0.0,                 # [rad]
            #         "rotation_axis": "hawt_z",

            #         "blade": {
            #             "sections": _make_blade_sections(
            #                 R_ROTOR, CHORD_BLADE, PITCH
            #             ),
            #         },
            #         "grid": {"n_radial": 20},
            #     },
            # },

            # {
            #     "name": "rotor_6",
            #     "rotor": {
            #         "n_blades": N_BLADES,
            #         "hub_center": [HUB_6_X_LU * DX_PHYS,
            #                        HUB_6_Y_LU * DX_PHYS,
            #                        HUB_6_Z_LU * DX_PHYS],  # [m]
            #         "omega": OMEGA_ROTOR,           # [rad/s]
            #         "theta_0": 0.0,                 # [rad]
            #         "rotation_axis": "hawt_z",

            #         "blade": {
            #             "sections": _make_blade_sections(
            #                 R_ROTOR, CHORD_BLADE, PITCH
            #             ),
            #         },
            #         "grid": {"n_radial": 20},
            #     },
            #     # Per-rotor override (optional):
            #     # "coeff_mode": "wind_turbine",
            #     # "gaussian_cutoff": 4.0,
            # },

            # {
            #     "name": "rotor_7",
            #     "rotor": {
            #         "n_blades": N_BLADES,
            #         "hub_center": [HUB_7_X_LU * DX_PHYS,
            #                        HUB_7_Y_LU * DX_PHYS,
            #                        HUB_7_Z_LU * DX_PHYS],  # [m]
            #         "omega": -OMEGA_ROTOR,           # [rad/s]
            #         "theta_0": 0.0,                 # [rad]
            #         "rotation_axis": "hawt_z",

            #         "blade": {
            #             "sections": _make_blade_sections(
            #                 R_ROTOR, CHORD_BLADE, PITCH
            #             ),
            #         },
            #         "grid": {"n_radial": 20},
            #     },
            #     # Per-rotor override (optional):
            #     # "coeff_mode": "wind_turbine",
            #     # "gaussian_cutoff": 4.0,
            # },

            # {
            #     "name": "rotor_8",
            #     "rotor": {
            #         "n_blades": 3,
            #         "hub_center": [HUB_8_X_LU * DX_PHYS,
            #                        HUB_8_Y_LU * DX_PHYS,
            #                        HUB_8_Z_LU * DX_PHYS],  # [m]
            #         "omega": -OMEGA_ROTOR,           # [rad/s]
            #         "theta_0": 0.0,                 # [rad]
            #         "rotation_axis": [1, 0, 0],

            #         "blade": {
            #             "sections": _make_blade_sections(
            #                 R_ROTOR, CHORD_BLADE, PITCH
            #             ),
            #         },
            #         "grid": {"n_radial": 20},
            #     },
            #     # Per-rotor override (optional):
            #     # "coeff_mode": "wind_turbine",
            #     # "gaussian_cutoff": 4.0,
            # },
        ]
    })

# =============================================================================
# §9. AIRFOIL POLAR (Only if ALM enabled)
# =============================================================================
# if ALM_ENABLED:
#     airfoil_polar = {
#         "method": "neuralfoil",
#         "airfoil_name": "naca0012",
#         "Re_target": RE_POLAR_TARGET,
#         "Re_min": RE_POLAR_MIN,
#         "Re_max": RE_POLAR_MAX,
#         "mode": "asb",
#         "ncrit": 9.0,
#     }
# else:
#     airfoil_polar = {}
if ALM_ENABLED:
    airfoil_polar = {
        "method": "prescribed",
        "CL_fixed": 1.0,
        "CD_fixed": 0.02,
    }
else:
    airfoil_polar = {}
# =============================================================================
# §9. FORCE CALCULATION (Optional Module)
# =============================================================================
# Enable for drag/lift measurement on internal geometry

force_calculation = {
    "enabled": False,
    # "object": "cylinder",
    # "method": "momentum_exchange",
}

# =============================================================================
# §10. CONSERVATION & CONVERGENCE
# =============================================================================
conservation = {
    "enabled": True,
    "check_interval": 10,
    "verbose": 0,
    "log_to_csv": True,
}
# Convergence Path (자동 선택):
# ────────────────────────────
#   Path A (장애물 없음): avg_energy가 sole criterion
#   Path B (장애물 있음): Cd가 sole criterion, energy/Cl은 monitor only
convergence = {
    "enabled": True,
    "cauchy": {
        "window_size": "auto",
        # --- Auto Window 세부 조절 (window_size가 "auto"일 때만 유효) ---
        # "auto_window": {
        #     "time_coverage": 50.0,       # [-]  T_conv의 몇 배를 window로 쓸지
        #     "min_samples": 100,          # [-]  최소 window sample 수
        # },
        "epsilon": 1e-5,                     # energy
        #  에너지 평균 드리프트 < 0.001%
        #   Path A에서 criterion으로 사용
        #   Path B에서는 monitor only
        "Cd_epsilon": 1e-3,                  # drag coefficient
        #   Cd 평균 드리프트 < 0.1%
        #   Path B에서 sole criterion으로 사용
        #   장애물 없으면 무시됨
        "n_required": 2,                    # [-]  연속 통과 횟수
        #   거짓 수렴 방지. check_interval 간격으로 n_required회 연속
        #   ε_cauchy < threshold 만족 시 최종 수렴 판정.
    },
    # ─── 수렴/발산 시 동작 ────────────────────────────────────────────────
    "on_converged": "checkpoint_and_stop",
    #   "checkpoint_and_stop" : 체크포인트 저장 후 종료 (기본, 권장)
    #   "stop"               : 즉시 종료 (체크포인트 없음)
    #   "continue"           : 수렴해도 계속 진행

    "on_diverged": "stop_with_checkpoint",
    #   "stop_with_checkpoint" : 체크포인트 저장 후 종료 (기본, 권장)
    #   "stop"                 : 즉시 종료

    "on_max_steps": "continue",
    #   "continue" : max_steps 도달해도 수렴 미달이면 계속 (기본)
    #   "warn"     : 경고 출력 후 계속
    #   "stop"     : 종료
}

# =============================================================================
# §11. OUTPUT
# =============================================================================
_case_tag = "ALM" if ALM_ENABLED else "base"
_folder = f"result_{_case_tag}_Re{int(RE)}_L{RESOLUTION}"

output = {
    "output_dir": f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir": f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {
        "enabled": True,
        "precision": "float32",
        "compression_level": 6,
        "variables": ["density", "velocity"],
    },
    "checkpoint": {"enabled": True, "keep_last_n": 3},
}

# =============================================================================
# §12. FINAL CONFIG DICTIONARY
# =============================================================================
config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "actuator_line": actuator_line,
    "airfoil_polar": airfoil_polar,
    "conservation": conservation,
    "convergence": convergence,
    "force_calculation": force_calculation,
    "output": output,
}

# =============================================================================
# §13. SUMMARY
# =============================================================================
def print_summary():
    """Print configuration summary."""
    print()
    print("=" * 70)
    print(" LBM Solver Configuration Summary")
    print("=" * 70)
    print()
    print(" §1. Physical Parameters (User Input):")
    print(f"      L_ref   = {L_REF} m  (= chord, for Re definition)")
    print(f"      R_rotor = {R_ROTOR} m,  D = {D_ROTOR} m")
    print(f"      c/R     = {L_REF/R_ROTOR:.2f}")
    print(f"      σ       = {2*L_REF/(np.pi*R_ROTOR):.4f}  (N_b·c/πR, 2 blades)")
    print(f"      U_ref   = {U_REF:.4f} m/s  (ωR×0.75)")
    print(f"      ρ_ref   = {RHO_REF} kg/m³")
    print(f"      Re      = {RE}")
    print(f"      ν_phys  = {NU_PHYS:.6e} m²/s")
    print()
    print(" §2. Discretization (User Input):")
    print(f"      Resolution       = {RESOLUTION} (D/Δx, rotor diameter basis)")
    print(f"      Lattice velocity = {LATTICE_VELOCITY} (u_lu)")
    print()
    print(" §3. Lattice Parameters (Auto-calculated):")
    print(f"      Δx      = {DX_PHYS:.6e} m  ({DX_PHYS*1000:.3f} mm)")
    print(f"      Δt      = {DT_PHYS:.6e} s")
    print(f"      chord   = {L_REF_LU:.1f} cells")
    print(f"      D       = {D_LU:.0f} cells,  R = {R_LU:.0f} cells")
    print(f"      ν_lu    = {NU_LU:.6f}")
    print(f"      τ       = {TAU:.6f}")
    print(f"      ω_LBM   = {OMEGA_LBM:.6f}")
    print(f"      Ma_lu   = {MA_LATTICE:.4f}")
    print()
    print(" §4. Domain:")
    print(f"      Grid    = {Nx} × {Ny} × {Nz} = {Nx*Ny*Nz:,} cells")
    print(f"      Size    = {Nx*DX_PHYS:.4f} × {Ny*DX_PHYS:.4f} × {Nz*DX_PHYS:.4f} m")
    print(f"      Domain  = {DOMAIN_MULT_X}D × {DOMAIN_MULT_Y}D × {DOMAIN_MULT_Z}D")
    print()
    print(" §5. Time:")
    max_steps = simulation['time']['max_steps']
    print(f"      Max steps      = {max_steps:,}")
    print(f"      Steps/rev      = {STEPS_PER_REV:,}")
    print(f"      Physical time  = {max_steps * DT_PHYS:.4f} s")
    print(f"      Revolutions    = {max_steps / STEPS_PER_REV:.1f}")
    print()
    print(" §6. Modules:")
    print(f"      ALM             = {ALM_ENABLED}")
    print(f"      Force calc      = {force_calculation['enabled']}")
    print()
    print("=" * 70)
    
    # Stability indicator
    if TAU <= 0.5:
        print("❌ UNSTABLE: τ ≤ 0.5")
    elif TAU < 0.52:
        print(f"⚠️  MARGINAL: τ = {TAU:.4f}")
    else:
        print(f"✅ STABLE: τ = {TAU:.4f}, Ma_lu = {MA_LATTICE:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    print_summary()