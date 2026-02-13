# =============================================================================
# LBM Solver — Hover Rotor Configuration (NTNU BT1)
# =============================================================================
#
# Actuator Line Model test case: hovering rotor in quiescent air.
# Ground effect included (Zmin = wall).
#
# Usage:
#   python main.py --config configs/hover_config.py
#   python main.py --config configs/hover_config.py --max-steps 5000
#
# =============================================================================
# Parameter Derivation
# =============================================================================
#
# Physical:
#   Rotor D = 0.894 m,  R = 0.447 m,  3 × NREL S826 blades
#   c_75 = 0.035 m            (75% span chord, BEM Re characteristic length)
#   ω = 134.23 rad/s          (TSR=6 × U=10 / R)
#   V_tip = 60.00 m/s
#   ν_air = 1.5e-5 m²/s
#
# Grid (D/Δx = 40):
#   Δx = D / 40 = 0.02235 m
#   c_75/Δx = 1.57 lu         (chord not resolved — expected for ALM)
#   R/Δx = 20.0 lu
#
# Lattice (Ma_tip = 0.05):
#   V_tip_lu = 0.05 Δx/Δt
#   Δt = V_tip_lu × Δx / V_tip = 1.8625e-05 s
#   ω_lu = ω × Δt = 0.002500 rad/lt
#   1 revolution = 2π / ω_lu ≈ 2513 steps
#
# Viscosity (τ = 0.55):
#   ν_lu = (τ − 0.5) / 3 = 0.01667 lu²/lt     → wake diffusion (Grid Re)
#   Re_D,grid = V_tip_lu × D_lu / ν_lu = 120   → wake retained ~2D downstream
#   Note: BEM Re uses ν_air = 1.5e-5 separately → Re_c ≈ 105,000
#
# Domain (3D × 3D × 4D):
#   120 × 120 × 160 = 2,304,000 cells (2.3M)
#   Hub at 65% height → 2.6D below (wake + ground), 1.4D above (clearance)
#
# Force Spreading:
#   ε = max(c/4, 2Δx) = 2.0 lu for all markers (lower bound dominates)
#
# =============================================================================

import numpy as np

# =============================================================================
# §1. Physical Parameters
# =============================================================================
D_PHYS    = 0.894          # [m]     rotor diameter
R_PHYS    = D_PHYS / 2     # [m]     rotor radius
C_75_PHYS = 0.035          # [m]     75% span chord (char. length for BEM Re)
OMEGA     = 134.23         # [rad/s] angular velocity (= TSR × U∞ / R)
V_TIP     = OMEGA * R_PHYS # [m/s]   tip speed = 60.00 m/s
NU_AIR    = 1.5e-5         # [m²/s]  kinematic viscosity of air at ~20°C
U_INF = 10.0                # [m/s] Freestream velocity
RHO_PHYS = 1.205            # [kg/m³] Air density at 20°C
NU_PHYS = 1.512e-5          # [m²/s] Kinematic viscosity at 20°C
TSR = 6.0                   # [dimensionless] Tip speed ratio (design)

# =============================================================================
# §2. Grid Resolution
# =============================================================================
D_PER_DX  = 40             # [dimensionless] D / Δx
DX_PHYS   = D_PHYS / D_PER_DX  # [m/lu] = 0.02235 m

# =============================================================================
# §3. Lattice Scaling
# =============================================================================
MA_TIP    = 0.05                           # [dimensionless] lattice Mach at tip
V_TIP_LU  = MA_TIP                         # [Δx/Δt]
DT_PHYS   = V_TIP_LU * DX_PHYS / V_TIP    # [s/lt] = 1.8625e-05 s

# Relaxation time → grid viscosity (wake diffusion control)
TAU       = 0.55                           # [dimensionless]
NU_LU     = (TAU - 0.5) / 3               # [lu²/lt] = 0.01667

# =============================================================================
# §4. Domain
# =============================================================================
Nx = 4 * D_PER_DX    # 120 [lu]  — 3D streamwise (X-Y plane for hover)
Ny = 3 * D_PER_DX    # 120 [lu]  — 3D crosswise
Nz = 3 * D_PER_DX    # 160 [lu]  — 4D vertical (wake + ground)

# Hub position: centered in X-Y, 65% height in Z
HUB_X = int(Nx * 0.65)      # 60 [lu]
HUB_Y = Ny // 2      # 60 [lu]
HUB_Z = Nz // 2  # 104 [lu] — 2.6D below (ground + wake), 1.4D above

# Physical hub (for Rotor factory)
HUB_PHYS = (HUB_X * DX_PHYS, HUB_Y * DX_PHYS, HUB_Z * DX_PHYS)  # [m]

# =============================================================================
# §5. Derived Quantities (for reference / assertions)
# =============================================================================
_OMEGA_LU      = OMEGA * DT_PHYS         # [rad/lt] = 0.002500
_STEPS_PER_REV = int(2 * np.pi / _OMEGA_LU)  # ≈ 2513
_RE_D_GRID     = V_TIP_LU * D_PER_DX / NU_LU  # ≈ 120
_RE_C_PHYS     = V_TIP * C_75_PHYS / NU_AIR    # ≈ 140,000

# ── Characteristic length is c_75 in lattice units ──
# Used for Re definition in physics config:
#   Re = u_init × CHAR_LENGTH / ν → ν = u_init × CHAR_LENGTH / Re
# But for hover, this is secondary — τ directly controls ν_lu.
CHAR_LENGTH_LU = C_75_PHYS / DX_PHYS     # [lu] ≈ 1.57

# Re defined so that ν_lu = (τ-0.5)/3 is consistent:
#   ν_lu = u_init × CHAR_LENGTH / Re  →  Re = u_init × CHAR_LENGTH / ν_lu
# Here u_init is the reference velocity for ν derivation.
# For hover: use V_tip_lu as reference velocity.
RE = V_TIP_LU * CHAR_LENGTH_LU / NU_LU   # ≈ 4.7 (Grid Re based on c_75)
# Note: This is the "grid Re" on c_75 scale. The BEM Re (~105k) is separate.

# =============================================================================
# §6. Config Assembly
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
        "Re": RE,                           # [dimensionless] grid Re (→ ν_lu → τ)
        "u_init": V_TIP_LU,                # [Δx/Δt] reference velocity for ν calc
        "initial_flow_velocity": 0.0,       # [Δx/Δt] IC = quiescent air (hover)
        "characteristic_length": CHAR_LENGTH_LU,  # [lu] c_75 in lattice
    },
    "time": {
        "max_steps": _STEPS_PER_REV * 40,   # 5 revolutions ≈ 12,565 steps
        "output_interval": _STEPS_PER_REV // 360,   # ~5 outputs per rev
        "checkpoint_interval": _STEPS_PER_REV * 2, # every 2 revs
        "probe_interval": 10,               # [steps] force/probe sampling
    }
}


# =============================================================================
# Boundary Conditions
# =============================================================================
# Hover rotor:
#   Zmin = ground wall (ground effect)
#   Zmax = open (pressure relaxation, quiescent ρ=1.0)
#   X/Y faces = open (pressure relaxation, quiescent ρ=1.0)
#
# No freestream velocity — rotor induces all flow.
boundaries = {
    "ground":  {"location": "xmax", "method": "regularized_wall"},
    "top":     {"location": "xmin", "method": "regularized_outlet",
                "density": 1.0},
    "zmin":    {"location": "zmin", "method": "regularized_outlet",
                "density": 1.0},
    "zmax":    {"location": "zmax", "method": "regularized_outlet",
                "density": 1.0},
    "ymin":    {"location": "ymin", "method": "regularized_outlet",
                "density": 1.0},
    "ymax":    {"location": "ymax", "method": "regularized_outlet",
                "density": 1.0},
}


# =============================================================================
# Internal Geometry (none for hover — body force only)
# =============================================================================
internal_geometry = {}


# =============================================================================
# Actuator Line Model
# =============================================================================
actuator_line = {
    "enabled": True,
    "rotor": {
        "preset": "ntnu_bt1",
        "tsr": TSR,                     # [dimensionless]
        "u_inf": U_INF,                 # [m/s] for ω calculation
        "hub_center": [HUB_X, HUB_Y, HUB_Z],  # [m] physical coordinates
        "resolution": 40,       # [dimensionless] D/Δx
        "theta_0": 0.0,                # [rad] initial azimuth
        "rotation_axis": "hawt_x",
    },

    # --- Physical → Lattice unit conversion ---
    "units": {
        "dx_phys": DX_PHYS,            # [m/lu]
        "dt_phys": DT_PHYS,            # [s/lt]
        "nu_phys": NU_PHYS,
    },

    # --- Gaussian regularization (Eq. 13) ---
    #   ε = max(c_a/4, 2Δx), cutoff at n_cut × ε
    "gaussian_cutoff": 3.0,         # [dimensionless]

    # Reference density for force non-dimensionalization
    "rho_ref": 1.0,                 # [dimensionless]
}


# =============================================================================
# Conservation & Convergence
# =============================================================================
conservation = {
    "enabled": True,
    "check_interval": _STEPS_PER_REV // 2,  # twice per revolution
    "verbose": 1,
    "log_to_csv": True,
    "tolerance": {
        "mass_drift_percent": 2.0,           # [%] hover may have larger drift
        "warn_on_exceed": True,
    },
    "domain": {"enabled": True},
    "control_volumes": [],                   # no obstacle-based CV
}

convergence = {
    "enabled": True,
    "cauchy": {
        "window_size": _STEPS_PER_REV * 2,  # 2 revolutions
        "epsilon": 1e-4,                     # [dimensionless] avg kinetic energy
        "Cd_epsilon": 1e-2,                  # relaxed (no Cd in hover)
        "n_required": 2,
        "auto_window": {
            "time_coverage": 2.0,            # [T_conv] (revolutions)
            "min_samples": 50,
        },
    },
    "on_converged": "checkpoint_and_stop",
    "on_max_steps": "warn",
    "on_diverged": "stop_with_checkpoint",
}


# =============================================================================
# Force Calculation (disabled — no solid body)
# =============================================================================
force_calculation = {
    "enabled": False,
}


# =============================================================================
# Output
# =============================================================================
_folder = f"result_hover_ALM_D{D_PER_DX}_tau{TAU:.2f}"

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
    "checkpoint": {
        "enabled": True,
        "keep_last_n": 3,
    },
}


# =============================================================================
# Final Config Dictionary
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