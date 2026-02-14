"""
2D Poiseuille Flow — Validation Config

Physical Setup:
    - 2D channel: Nx × Ny
    - Walls at y=0 and y=Ny-1 (no-slip)
    - Velocity inlet at x=0, pressure outlet at x=Nx-1
    - Pressure-driven parabolic profile

Re Definition (Poiseuille):
    Re = u_mean × H / ν

    where H = Ny - 1 (channel height for half-way bounce-back)

    ┌────────────────────────────────────────────────────────┐
    │  Poiseuille Reference Conditions:                     │
    │                                                        │
    │    Re   │  ν_lu     │  τ       │ Status                │
    │  ──────┼───────────┼──────────┼─────────────────────── │
    │    10   │ 0.01200   │ 0.5360   │ ✅ very stable        │
    │    50   │ 0.00240   │ 0.5072   │ ✅ stable             │
    │   100   │ 0.00120   │ 0.5036   │ ✅ OK for BGK         │
    │   200   │ 0.00060   │ 0.5018   │ ⚠️ borderline         │
    │   500   │ 0.00024   │ 0.5007   │ ❌ needs MRT/LES      │
    │                                                        │
    │  (u_inlet = 0.05, H = Ny - 1 = 24)                   │
    └────────────────────────────────────────────────────────┘

Author: LBM Development Team
Date: 2026-02
"""

# =============================================================================
# Domain
# =============================================================================
Nx = 200            # [lu] channel length (> 0.05 × Re × H for full development)
Ny = 25             # [lu] channel height (including wall nodes)

# =============================================================================
# Flow Parameters
# =============================================================================
U_INLET = 0.05      # [lu/lt] inlet velocity (Ma ≈ 0.087, well below 0.1)
H = Ny - 1          # [lu] effective channel height for half-way BB  [Δx]
RE = 50.0            # [dimensionless] Re = u_inlet × H / ν
RHO = 1.0            # [dimensionless] reference density

# Entry length check:
#   L_entry ≈ 0.05 × Re × H = 0.05 × 50 × 24 = 60 lu
#   Nx = 200 >> 60 ✓  (fully developed at mid-channel)

# =============================================================================
# Derived: Viscosity and Relaxation
# =============================================================================
NU_LU = U_INLET * H / RE        # [lu²/lt] = 0.05 × 24 / 50 = 0.024
TAU = 3.0 * NU_LU + 0.5         # [lt] = 3 × 0.024 + 0.5 = 0.572  ✅ 안정

print(f"  [Config] Poiseuille 2D Validation")
print(f"  [Config] Nx={Nx}, Ny={Ny}, H={H}")
print(f"  [Config] Re = {RE}, u_inlet = {U_INLET}")
print(f"  [Config] ν_lu = {NU_LU:.6f}, τ = {TAU:.4f}")
print(f"  [Config] Entry length ≈ {0.05 * RE * H:.0f} lu (Nx={Nx})")


# =============================================================================
# Simulation Parameters
# =============================================================================
simulation = {
    "device_mode": "cpu",           # CPU for validation (small domain)
    "dimension": 2,
    "lattice_model": "D2Q9",
    "domain": {
        "Nx": Nx,
        "Ny": Ny,
    },
    "physics": {
        "Re": RE,                           # [dimensionless]
        "u_init": U_INLET,                  # [lu/lt] for initialization & Re calc
        "characteristic_length": H,         # [lu] channel height  ← Poiseuille 정의!
    },
    "time": {
        "max_steps": 30000,
        "output_interval": 1000,
        "checkpoint_interval": 10000,
        "probe_interval": 100,
    }
}


# =============================================================================
# Boundary Conditions
# =============================================================================
# Poiseuille: inlet + outlet + 양쪽 벽 (대칭!)
#
#     ymax ──────────────── wall (no-slip)
#           │              │
#    inlet  │    channel   │  outlet
#     x=0   │              │  x=Nx-1
#           │              │
#     ymin ──────────────── wall (no-slip)
#
boundaries = {
    "inlet": {
        "location": "xmin",
        "method": "regularized_inlet",
        "velocity": U_INLET,
        "rho": RHO,
    },
    "outlet": {
        "location": "xmax",
        "method": "regularized_outlet",
        "rho": RHO,
        "k": 0.1,
    },
    # *** 양쪽 다 벽! (Poiseuille의 핵심) ***
    "wall_south": {
        "location": "ymin",
        "method": "regularized_wall",
    },
    "wall_north": {
        "location": "ymax",
        "method": "regularized_wall",       # ← outlet이 아니라 wall!
    },
}


# =============================================================================
# Internal Geometry (Poiseuille에서는 없음)
# =============================================================================
internal_geometry = {}


# =============================================================================
# Conservation / Convergence
# =============================================================================
conservation = {
    "enabled": True,
    "check_interval": 100,
    "verbose": 1,
    "log_to_csv": True,
}

convergence = {
    "enabled": True,
    "monitor": {
        "energy": {
            "enabled": True,
            "threshold": 1e-6,
            "window": 500,
        },
    },
    "on_converged": "checkpoint_and_stop",
    "on_diverged": "stop_with_checkpoint",
    "on_max_steps": "warn",
}

force_calculation = {"enabled": False}
actuator_line = {"enabled": False}


# =============================================================================
# Output
# =============================================================================
folder_name = f"result_Poiseuille2D_Ny{Ny}_Re{int(RE)}"

output = {
    "output_dir": f"./{folder_name}/vtk",
    "checkpoint_dir": f"./{folder_name}/checkpoints",
    "csv_dir": f"./{folder_name}/csv",
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
# Final Config
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