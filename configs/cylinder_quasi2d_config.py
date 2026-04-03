"""
Quasi-2D Cylinder Flow — BGK vs Cumulant Comparison

Purpose:
    Compare drag coefficient (Cd) between BGK and Cumulant collision
    models on a well-validated 2D cylinder benchmark using D3Q27.

Physical Setup:
    - 3D domain with Nz=3 (quasi-2D, z-periodic via streaming)
    - Cylinder diameter D = 40 [lu], axis along z
    - Uniform inlet velocity u = 0.1 [Δx/Δt]
    - Re = 20, 40, 100 (change RE_TGT below)

    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │  inlet ──→   ●(cylinder)   ──→ outlet           │
    │  u=0.1       D=40              Neumann          │
    │                                                 │
    └─────────────────────────────────────────────────┘

    z-direction: Nz=3, periodic (no BC specified → streaming wraps)

Lattice Parameters (auto-calculated):
    Re=20:  ν=0.200, τ=1.100  ✅ very stable
    Re=40:  ν=0.100, τ=0.800  ✅ stable
    Re=100: ν=0.040, τ=0.620  ✅ stable

Reference Cd Values (2D cylinder, Henderson 1995):
    Re=20:  Cd ≈ 2.00
    Re=40:  Cd ≈ 1.50
    Re=100: Cd ≈ 1.33 (with vortex shedding, St ≈ 0.164)

Collision Model:
    Change COLLISION_MODEL to "bgk" or "cumulant" to switch.

Author: LBM Development Team
Date: 2026-03
"""

import numpy as np

# =============================================================================
# §1. USER PARAMETERS — CHANGE THESE
# =============================================================================

RE_TGT = 10000                    # [-]  Target Reynolds number (20, 40, or 100)
COLLISION_MODEL = "cumulant"    # "bgk" or "cumulant"

# =============================================================================
# §2. CYLINDER & DOMAIN
# =============================================================================

diameter = 100                    # [lu]  Cylinder diameter D
U_INLET = 0.1                   # [Δx/Δt]  Inlet velocity (Ma ≈ 0.17)
RHO = 1.0                       # [dimensionless]  Reference density

# Domain sizing (multiples of D)
# Nx = diameter * 25               # [lu]  = 1000 (streamwise)
# Ny = diameter * 20               # [lu]  = 600  (cross-stream)
# Nz = 3                           # [lu]  quasi-2D slab (z-periodic)
Nx = 1000               # [lu]  = 1000 (streamwise)
Ny = 600               # [lu]  = 600  (cross-stream)
Nz = 3                           # [lu]  quasi-2D slab (z-periodic)

# Cylinder center
center_x = Nx // 6              # [lu]  25% from inlet = 200
center_y = Ny // 2              # [lu]  centered = 300
center_z = Nz // 2              # [lu]  centered = 1

# =============================================================================
# §3. DERIVED LATTICE PARAMETERS
# =============================================================================

NU_LU = U_INLET * diameter / RE_TGT   # [lu²/lt]  ν = uD/Re
TAU = 0.5 + 3.0 * NU_LU              # [-]  τ = 0.5 + 3ν
OMEGA_LBM = 1.0 / TAU                 # [-]  ω = 1/τ

CS = 1.0 / np.sqrt(3.0)               # [Δx/Δt]  lattice speed of sound
MA = U_INLET / CS                      # [-]  Mach number

# Reference quantities (setup.py _extract_physics requires these keys)
U_REF_LU = U_INLET                     # [Δx/Δt]  reference velocity = inlet
L_REF_LU = float(diameter)             # [lu]  reference length = D

# --- Stability checks ---
assert TAU > 0.5, f"UNSTABLE: τ = {TAU:.4f} ≤ 0.5"
if MA > 0.3:
    raise ValueError(f"Ma = {MA:.3f} > 0.3: compressibility error!")

print(f"  Quasi-2D Cylinder: Re={RE_TGT}, D={diameter}, Nz={Nz}")
print(f"  ν_lu = {NU_LU:.6f}, τ = {TAU:.4f}, Ma = {MA:.4f}")
print(f"  Collision model: {COLLISION_MODEL}")
print(f"  Domain: {Nx} × {Ny} × {Nz} = {Nx*Ny*Nz:,} cells")

# =============================================================================
# §4. SIMULATION CONFIG
# =============================================================================

simulation = {
    "device_mode": "gpu",
    "dimension": 3,
    "lattice_model": "D3Q27",

    # ★ Collision model selection (Phase C2) ★
    "collision_model": COLLISION_MODEL,     # "bgk" or "cumulant"
    # "omega_bulk": 1.0,                    # Optional: bulk viscosity rate
    # "omega_high": 1.0,                    # Optional: high-order rates

    "domain": {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
    },

    "physics": {
        # --- Core parameters ---
        "Re": RE_TGT,                       # [-]  Reynolds number

        # --- Pre-calculated lattice parameters ---
        # (required by setup.py _extract_physics)
        "tau": TAU,                          # [-]  relaxation time
        "omega": OMEGA_LBM,                 # [-]  relaxation frequency
        "nu_lu": NU_LU,                     # [lu²/lt]  lattice viscosity
        "u_ref_lu": U_REF_LU,              # [Δx/Δt]  reference velocity
        "L_ref_lu": L_REF_LU,              # [lu]  reference length (= D)

        # --- Initial flow field ---
        "initial_flow_velocity": [U_INLET, 0.0, 0.0],  # [Δx/Δt]
    },

    "time": {
        "max_steps": 100000,                 # Enough for convergence
        "output_interval": 500,
        "checkpoint_interval": 20000,
        "probe_interval": 10,
    },
}

# =============================================================================
# §5. BOUNDARY CONDITIONS
# =============================================================================
# x: inlet (equilibrium) → outlet (Neumann)
# y: far-field (equilibrium) top/bottom
# z: NO BC → periodic via streaming (quasi-2D)

# boundaries = {
#     "inlet": {
#         "location": "xmin",
#         "method": "equilibrium",
#         "velocity": U_INLET,
#     },
#     "outlet": {
#         "location": "xmax",
#         "method": "neumann",
#     },
#     "south": {
#         "location": "ymin",
#         "method": "equilibrium",
#         "velocity": U_INLET,
#     },
#     "north": {
#         "location": "ymax",
#         "method": "equilibrium",
#         "velocity": U_INLET,
#     },
#     # z-direction: periodic (implicit, no BC needed)
# }

boundaries = {
    "inlet": {
        "location": "xmin",
        "method": "reg_inlet",
        "velocity": U_INLET,
        "rho": 1.0,
    },
    "outlet": {
        "location": "xmax",
        "method": "reg_inlet",
        "velocity": U_INLET,
        "rho": 1.0,
    },
    "south": {
        "location": "ymin",
        "method": "reg_inlet",
        "velocity": U_INLET,
        "rho": 1.0,
    },
    "north": {
        "location": "ymax",
        "method": "sponge",
        "velocity": U_INLET,        # U∞  [Δx/Δt]
        "rho": 1.0,             # ρ∞  [dimensionless]
        "thickness": 10,        # L   [lattice units]
        "strength": 0.5         # σ_max [dimensionless]
    },
}

# =============================================================================
# §6. INTERNAL GEOMETRY — 3D Cylinder (z-axis)
# =============================================================================

internal_geometry = {
    "cylinder": {
        "enabled": True,
        "center": [center_x, center_y],     # [lu] (x, y center)
        "radius": diameter // 2,             # [lu] = 20
        "axis": "z",                         # extrude direction
    },
}

# =============================================================================
# §7. FORCE CALCULATION (MEM for Cd/Cl)
# =============================================================================

force_calculation = {
    "enabled": True,
    "interval": 10,
    "start_step": 100,

    "reference": {
        "rho": RHO,                          # [dimensionless]
        "velocity": U_INLET,                 # [Δx/Δt]
        "char_length": diameter,             # [lu] D
        "span_length": Nz,                   # [lu] for 3D: Nz
    },

    "log": {
        "enabled": True,
        "filename": "force_history",
    },
}

# =============================================================================
# §8. CONSERVATION & CONVERGENCE
# =============================================================================

conservation = {
    "enabled": True,
    "check_interval": 100,
    "verbose": 0,
    "log_to_csv": True,
}

convergence = {
    "enabled": True,
    "cauchy": {
        "window_size": "auto",
        "epsilon": 1e-5,                     # energy
        "Cd_epsilon": 3e-4,                  # drag coefficient
        "n_required": 3,
    },
    "on_converged": "checkpoint_and_stop",
    "on_diverged": "stop_with_checkpoint",
    "on_max_steps": "continue",
}

# =============================================================================
# §9. OUTPUT
# =============================================================================

_tag = COLLISION_MODEL.upper()
_folder = f"result_cylinder_Q2D_{_tag}_Re{int(RE_TGT)}_D{diameter}"

output = {
    "output_dir": f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir": f"./{_folder}/csv",
    "clear_previous": True,

    "vtk": {
        "enabled": True,
        "precision": "float32",
        "compression_level": 6,
        "variables": ["density", "pressure", "velocity",
                      "velocity_magnitude", "solid_mask"],
    },
    "checkpoint": {
        "enabled": True,
        "keep_last_n": 3,
    },
}

# =============================================================================
# §10. FINAL CONFIG
# =============================================================================

config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "conservation": conservation,
    "convergence": convergence,
    "force_calculation": force_calculation,
    "output": output,
}

# =============================================================================
# §11. SUMMARY
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print(f" Quasi-2D Cylinder — {COLLISION_MODEL.upper()}")
    print("=" * 60)
    print(f"  Re = {RE_TGT},  D = {diameter} lu")
    print(f"  Domain = {Nx} × {Ny} × {Nz} = {Nx*Ny*Nz:,} cells")
    print(f"  ν_lu = {NU_LU:.6f},  τ = {TAU:.4f},  Ma = {MA:.4f}")
    print(f"  Collision: {COLLISION_MODEL}")
    print(f"  Center: ({center_x}, {center_y})")
    print(f"  Output: {_folder}/")
    print("=" * 60)

    # Reference Cd values
    ref_cd = {20: 2.00, 40: 1.50, 100: 1.33}
    if RE_TGT in ref_cd:
        print(f"  Reference Cd ≈ {ref_cd[RE_TGT]:.2f} (Henderson 1995)")
    print("=" * 60)