"""
NTNU Blind Test 1 — Wind Turbine Wake Simulation Config

Reference: Watanabe et al., Comp. & Fluids 305, 106901, 2026
           Krogstad & Eriksen, Renewable Energy 50, 325-333, 2013

Physical Setup:
    - NTNU rotor: D = 0.894 m, 3 × NREL S826 blades, TSR = 6
    - Wind tunnel: 11.175 m × 2.682 m × 1.788 m
    - Inflow: U_∞ = 10 m/s, ρ = 1.205 kg/m³
    - Hub height: 0.817 m from floor
    - Turbine position: 3.66 m from inlet

Grid Resolution: D/32 (coarsest from Watanabe Table 1)
    - Δx = 27.9 mm → Grid: 400 × 96 × 64
    - Δt = 41.9 μs  (Co = u_lu = 0.015)

Reynolds Number Definition:
    ─────────────────────────────────────────────────────────────
    This config uses the STANDARD wind turbine reference Re:

        Re_ref = u_rel,75% × c_75% / ν

    where:
        u_rel,75% = √(U_∞² + (ω·r_75%)²)   [m/s]  relative velocity at 75% span
        c_75%     = chord at 75% span         [m]    interpolated from blade data
        ν         = kinematic viscosity        [m²/s]

    u_init  = freestream velocity in lattice units (for flow initialization)
    u_ref   = reference velocity for Re calculation (u_rel,75% in lattice units)
    char_length = reference chord (c_75% in lattice units)

    main.py computes:  ν_lu = u_ref × char_length / Re

    ┌──────────────────────────────────────────────────────────────┐
    │  NTNU BT1 Reference Conditions (75% span):                  │
    │    r_75%  = 0.3353 m,  c_75%  = 0.0335 m                    │
    │    V_tan  = ω·r_75%  = 134.23 × 0.3353 = 45.0 m/s          │
    │    u_rel  = √(10² + 45²) = 46.1 m/s                         │
    │    Real Re = 46.1 × 0.0335 / 1.512e-5 ≈ 102,100             │
    ├──────────────────────────────────────────────────────────────┤
    │  Testing:  Re_ref = 100 → ν_lu = 8.28e-4, τ = 0.5025  ⚠️   │
    │  Fallback: Re_ref =  50 → ν_lu = 1.66e-3, τ = 0.505   ✅   │
    │                                                              │
    │  WARNING: Testing Re is far below physical Re ≈ 102,100.    │
    │  For production: MRT/Cumulant + Smagorinsky LES required.    │
    └──────────────────────────────────────────────────────────────┘
"""

import numpy as np


# =============================================================================
# Physical Constants
# =============================================================================
D_ROTOR = 0.894             # [m] Rotor diameter
R_ROTOR = D_ROTOR / 2.0     # [m] Rotor radius = 0.447 m
U_INF = 10.0                # [m/s] Freestream velocity
RHO_PHYS = 1.205            # [kg/m³] Air density at 20°C
NU_PHYS = 1.512e-5          # [m²/s] Kinematic viscosity at 20°C
TSR = 6.0                   # [dimensionless] Tip speed ratio (design)

# Hub position in physical coordinates
HUB_X = 3.66                # [m] Distance from inlet
HUB_Y = 1.341               # [m] Spanwise center (≈ tunnel width / 2)
HUB_Z = 0.817               # [m] Hub height from floor

# =============================================================================
# Grid Resolution — Watanabe Table 1, D/32 case
# =============================================================================
RESOLUTION = 32              # [dimensionless] D/Δx
DX_PHYS = D_ROTOR / RESOLUTION   # [m/lu] = 0.02794 m

# Lattice Mach / Courant number (Watanabe: Co = 0.015)
MACH_INLET = 0.015          # [dimensionless] u_lu = U_∞ × Δt / Δx
DT_PHYS = MACH_INLET * DX_PHYS / U_INF   # [s/lt] ≈ 4.19e-5 s

# Domain size (Watanabe Fig. 2)
#   L_x = 11.175 m = 12.5D → 400 lu
#   L_y =  2.682 m =  3.0D →  96 lu
#   L_z =  1.788 m =  2.0D →  64 lu
Nx = 400
Ny = 96
Nz = 64

# Hub in lattice units
hub_x_lu = int(round(HUB_X / DX_PHYS))    # ≈ 131
hub_y_lu = int(round(HUB_Y / DX_PHYS))    # ≈ 48 (= Ny/2)
hub_z_lu = int(round(HUB_Z / DX_PHYS))    # ≈ 29

# Characteristic length in lattice units (for display only)
D_LU = RESOLUTION            # [lu] = 32

# =============================================================================
# Reference Re — Chord-Based at 75% Span
# =============================================================================
#
# Step 1: Angular velocity from TSR
#   ω = TSR × U_∞ / R = 6 × 10 / 0.447 = 134.23 [rad/s]
#
OMEGA = TSR * U_INF / R_ROTOR   # [rad/s] = 134.23

# Step 2: 75% span position and tangential velocity
#   r_75%  = 0.75 × R = 0.75 × 0.447 = 0.3353 [m]
#   V_tan  = ω × r_75% = 134.23 × 0.3353 = 45.0 [m/s]
#
R_75 = 0.75 * R_ROTOR              # [m] = 0.3353
V_TAN_75 = OMEGA * R_75            # [m/s] = 45.0

# Step 3: Relative velocity at 75% span
#   u_rel = √(U_∞² + V_tan²) = √(100 + 2025) = 46.1 [m/s]
#
U_REL_75 = np.sqrt(U_INF**2 + V_TAN_75**2)   # [m/s] = 46.1

# Step 4: Chord at 75% span (interpolated from blade section data)
#   Blade data (Krogstad & Eriksen 2013, Table C.1):
#     r = 0.3188 m → c = 0.0351 m
#     r = 0.3404 m → c = 0.0330 m
#   Linear interpolation at r = 0.3353 m:
#     f = (0.3353 - 0.3188) / (0.3404 - 0.3188) = 0.764
#     c_75% = 0.0351 + 0.764 × (0.0330 - 0.0351) = 0.0335 [m]
#
C_75_PHYS = 0.0335                  # [m] chord at 75% span

# Step 5: Convert to lattice units
#   u_ref_lu  = u_rel × Δt / Δx = 46.1 × 4.19e-5 / 0.02794 = 0.0691 [lu/lt]
#   c_75_lu   = c_75% / Δx = 0.0335 / 0.02794 = 1.199 [lu]
#
U_REF_LU = U_REL_75 * DT_PHYS / DX_PHYS    # [lu/lt] = 0.0691
C_75_LU = C_75_PHYS / DX_PHYS               # [lu] = 1.199

# Step 6: Target Re and resulting ν, τ
#   ν_lu = u_ref_lu × c_75_lu / Re_ref
#   τ    = 3·ν_lu + 0.5
#
#   Re_ref │  ν_lu    │  τ       │ BGK stability
#   ───────┼──────────┼──────────┼──────────────
#     500  │ 1.66e-4  │ 0.5005   │ ❌ UNSTABLE
#     200  │ 4.14e-4  │ 0.5012   │ ⚠️ RISKY
# >>> 100  │ 8.28e-4  │ 0.5025   │ ⚠️ MARGINAL (try first)
#      50  │ 1.66e-3  │ 0.505    │ ✅ SAFE (fallback)
#      20  │ 4.14e-3  │ 0.512    │ ✅ VERY SAFE
#
RE_REF = 100                 # [dimensionless] chord-based Re at 75% span
RHO = 1.0                   # [dimensionless] Reference density in lattice

# Verification printout
_nu_lu = U_REF_LU * C_75_LU / RE_REF
_tau = 3.0 * _nu_lu + 0.5
print(f"  [Config] Re_ref = {RE_REF} (chord-based, 75% span)")
print(f"  [Config] u_ref  = {U_REF_LU:.4f} [lu/lt]  (u_rel at 75% span)")
print(f"  [Config] c_ref  = {C_75_LU:.3f} [lu]  (chord at 75% span)")
print(f"  [Config] ν_lu   = {_nu_lu:.6f},  τ = {_tau:.4f}")


# =============================================================================
# Simulation Parameters
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "dimension": 2,
    "lattice_model": "D2Q9",
    "domain": {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
    },
    "physics": {
        "Re": RE_REF,                        # [dimensionless] Re at 75% span
        "u_init": MACH_INLET,                # [lu/lt] freestream (flow init)
        "u_ref": U_REF_LU,                   # [lu/lt] reference velocity (Re calc)
        "characteristic_length": C_75_LU,    # [lu] chord at 75% span
    },
    "time": {
        "max_steps": 50000,
        "output_interval": 100,
        "checkpoint_interval": 10000,
        "probe_interval": 50,
    }
}


# =============================================================================
# Boundary Conditions
# =============================================================================
# Watanabe: Uniform inflow, outflow BC behind turbine,
#           no-slip walls on all other boundaries.
boundaries = {
    "inlet": {
        "location": "xmin",
        "method": "regularized_inlet",
        "velocity": MACH_INLET,
        "rho": 1.0,
    },
    "outlet": {
        "location": "xmax",
        "method": "regularized_outlet",
        # "velocity": MACH_INLET,      # U_c = 0.015 [lu/lt]
        "rho": RHO,
        "k": 0.1,
    },
    #       Using pressure_relaxation as open-boundary approximation.
    "ymin": {
        "location": "ymin",
        "method": "regularized_wall",
    },
    "ymax": {
        "location": "ymax",
        "method": "regularized_outlet", 
               "rho": RHO, 
               "k":0.1},
    # "floor": {
    #     "location": "zmin",
    #     "method": "pressure_relaxation", 
    #            "rho": RHO, "k":0.1},
    # "ceiling": {
    #     "location": "zmax",
    #     "method": "pressure_relaxation", 
    #            "rho": RHO, "k":0.1},
}


# =============================================================================
# Internal Geometry (Nacelle/Tower — optional, AL only for blades)
# =============================================================================
# Watanabe models nacelle as a cylinder (D=90mm) and tower.
# For initial testing: no solid obstacles (pure AL).
# TODO: Add nacelle/tower IBB geometry for full reproduction.
internal_geometry = {}


# =============================================================================
# Actuator Line Configuration
# =============================================================================
actuator_line = {
    "enabled": False,

    # --- Rotor ---
    "rotor": {
        "preset": "ntnu_bt1",
        "tsr": TSR,                     # [dimensionless]
        "u_inf": U_INF,                 # [m/s] for ω calculation
        "hub_center": [HUB_X, HUB_Y, HUB_Z],  # [m] physical coordinates
        "resolution": RESOLUTION,       # [dimensionless] D/Δx
        "theta_0": 0.0,                # [rad] initial azimuth
    },

    # --- Physical → Lattice unit conversion ---
    "units": {
        "dx_phys": DX_PHYS,            # [m/lu]
        "dt_phys": DT_PHYS,            # [s/lt]
    },

    # --- Gaussian regularization (Eq. 13) ---
    #   ε = max(c_a/4, 2Δx), cutoff at n_cut × ε
    "gaussian_cutoff": 3.0,         # [dimensionless]

    # Reference density for force non-dimensionalization
    "rho_ref": RHO,                 # [dimensionless]
}


# =============================================================================
# Conservation Check Configuration
# =============================================================================
conservation = {
    "enabled": True,
    "check_interval": 100,
    "verbose": 1,
    "log_to_csv": True,
}


# =============================================================================
# Convergence Monitor
# =============================================================================
convergence = {
    "enabled": True,
    "monitor": {
        "energy": {
            "enabled": True,
            "threshold": 1e-5,
            "window": 500,
        },
    },
    "on_converged": "checkpoint_and_stop",
    "on_diverged": "stop_with_checkpoint",
    "on_max_steps": "warn",
}


# =============================================================================
# Force Calculation (disabled — AL computes rotor forces internally)
# =============================================================================
force_calculation = {
    "enabled": False,
}


# =============================================================================
# Output Configuration
# =============================================================================
folder_name = f"result_NTNU_BT1_D{RESOLUTION}_Re{int(RE_REF)}"

output = {
    "output_dir": f"./{folder_name}/vtk",
    "checkpoint_dir": f"./{folder_name}/checkpoints",
    "csv_dir": f"./{folder_name}/csv",

    "clear_previous": True,

    "vtk": {
        "enabled": True,
        "precision": "float32",
        "compression_level": 6,
        "variables": ["density", "velocity", "solid_mask"],
    },
    "checkpoint": {
        "enabled": True,
        "keep_last_n": 3,
    },
}


# =============================================================================
# Final Config Dictionary  (REQUIRED by ConfigLoader)
# =============================================================================
config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "actuator_line": actuator_line,
    "conservation": conservation,
    "convergence": convergence,
    "force_calculation": force_calculation,
    "output": output,
}