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

Stability Note:
    Real Re = U·D/ν ≈ 591,000 → τ ≈ 0.5000 (unstable for BGK)
    Watanabe uses cumulant collision (implicit LES).
    For BGK testing: RE_TGT = 200 → τ ≈ 0.507 (stable)
    → Switch to MRT/Cumulant + LES for production runs.
"""

import numpy as np

# =============================================================================
# Physical Constants
# =============================================================================
D_ROTOR = 0.894             # [m] Rotor diameter
R_ROTOR = D_ROTOR / 2.0     # [m] Rotor radius
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
MACH_INLET = 0.015          # [dimensionless] u_lu = U_inf * Δt / Δx
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

# Characteristic length in lattice units
D_LU = RESOLUTION            # [lu] = 32

# =============================================================================
# Reynolds Number — Stability Consideration
# =============================================================================
# Real:     Re = U·D/ν = 10 × 0.894 / 1.512e-5 ≈ 591,270
#           → ν_lu = 8.1e-7,  τ = 0.500002 (UNSTABLE for BGK!)
#
# BGK test: Use artificially reduced Re for stability
#           Re = 200 → ν_lu = 0.0024, τ = 0.5072
#
# TODO: Replace with MRT/Cumulant + Smagorinsky LES for production.
RE_TGT = 200.0              # [dimensionless] BGK-safe test value
RHO = 1.0                   # [dimensionless] Reference density in lattice


# =============================================================================
# Simulation Parameters
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "dimension": 3,
    "lattice_model": "D3Q27",
    "domain": {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
    },
    "physics": {
        "Re": RE_TGT,
        "u_init": MACH_INLET,
        "characteristic_length": D_LU,
    },
    "time": {
        "max_steps": 50000,
        "output_interval": 500,
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
        "method": "non_equilibrium",
        "velocity": MACH_INLET,
    },
    "outlet": {
        "location": "xmax",
        "method": "pressure_relaxation",
        "rho": RHO,
        "k": 0.1,
    },
    # Tunnel walls
    # Note: bounce_back via DomainWallManager requires 'dim' attribute.
    #       Using pressure_relaxation as open-boundary approximation
    #       until DomainWallManager is extended for 3D walls.
    # TODO: Switch to bounce_back when DomainWallManager supports 3D.
    "ymin": {
        "location": "ymin",
        "method": "pressure_relaxation",
        "rho": RHO,
        "k": 0.1,
    },
    "ymax": {
        "location": "ymax",
        "method": "pressure_relaxation",
        "rho": RHO,
        "k": 0.1,
    },
    "floor": {
        "location": "zmin",
        "method": "pressure_relaxation",
        "rho": RHO,
        "k": 0.1,
    },
    "ceiling": {
        "location": "zmax",
        "method": "pressure_relaxation",
        "rho": RHO,
        "k": 0.1,
    },
}


# =============================================================================
# Internal Geometry (Nacelle/Tower — optional, AL only for blades)
# =============================================================================
# Watanabe models nacelle as a cylinder (D=90mm) and tower.
# For initial testing: no solid obstacles (pure AL).
# TODO: Add nacelle/tower IBB geometry for full reproduction.
internal_geometry = {
    # "nacelle": {
    #     "enabled": True,
    #     "type": "cylinder",
    #     "radius": int(round(0.045 / DX_PHYS)),   # 90mm diameter
    #     "center": (hub_x_lu, hub_y_lu, hub_z_lu),
    #     "axis": "x",
    #     "length": int(round(0.20 / DX_PHYS)),     # ~200mm nacelle
    # },
}


# =============================================================================
# Actuator Line Configuration  (NEW — wind turbine specific)
# =============================================================================
actuator_line = {
    "enabled": True,

    # Rotor definition — uses Rotor.from_config() → from_ntnu_bt1()
    "rotor": {
        "preset": "ntnu_bt1",
        "tsr": TSR,                              # [dimensionless]
        "u_inf": U_INF,                          # [m/s]
        "hub_center": [HUB_X, HUB_Y, HUB_Z],   # [m] physical coordinates
        "resolution": RESOLUTION,                 # [dimensionless] D/Δx
    },

    # Unit conversion (physical ↔ lattice)
    "units": {
        "dx_phys": DX_PHYS,        # [m/lu]
        "dt_phys": DT_PHYS,        # [s/lt]
    },

    # Gaussian spreading (Watanabe Eq. 13)
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
folder_name = f"result_NTNU_BT1_D{RESOLUTION}_Re{int(RE_TGT)}"

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