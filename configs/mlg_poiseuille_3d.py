"""
MLG Poiseuille 3D — Multi-Level Grid Validation Config

2-level grid refinement test for a 3D Poiseuille channel flow.
Level 0 (coarse) covers the full domain, Level 1 (fine, 2× resolution)
covers the central region where velocity gradients are moderate.

Physical Setup:
    - 3D channel with no-slip walls (y-direction)
    - Pressure-driven via inlet/outlet (x-direction)
    - Periodic or wall in z-direction

    y=Ny ──────────── wall (no-slip)
          │          │
    inlet │ channel  │ outlet
    x=0   │          │ x=Nx
          │          │
    y=0   ──────────── wall (no-slip)

Validation Targets:
    1. Velocity profile matches Poiseuille analytic solution
    2. Smooth velocity/density transition at grid interface
    3. Mass conservation within 0.1%

Usage:
    python main.py --config configs/mlg_poiseuille_3d.py

Author: LBM Development Team
Date: 2026-04
"""

import numpy as np

# =============================================================================
# §1. PHYSICAL PARAMETERS
# =============================================================================
RE = 20                 # [-]  Low Re for laminar validation
U_INLET = 0.02          # [Δx/Δt]  Inlet velocity (keep Ma < 0.1)
RHO = 1.0               # [-]  Reference density

# Channel geometry
H = 30                  # [lu]  Channel height (y-direction)
Nx = 120                # [lu]  Streamwise length
Ny = H                  # [lu]  Channel height
Nz = 10                 # [lu]  Spanwise (thin for quasi-3D)

# =============================================================================
# §2. DERIVED LATTICE PARAMETERS
# =============================================================================
NU_LU = U_INLET * H / RE                    # [lu²/lt]
TAU = 0.5 + 3.0 * NU_LU                     # [-]
OMEGA_LBM = 1.0 / TAU                       # [-]

CS = 1.0 / np.sqrt(3.0)
MA = U_INLET / CS

U_REF_LU = U_INLET
L_REF_LU = float(H)

# Stability checks
assert TAU > 0.5, f"UNSTABLE: τ = {TAU:.4f}"
assert MA < 0.3, f"Ma = {MA:.3f} > 0.3"

print(f"  [MLG Poiseuille 3D] Re={RE}, H={H}, τ={TAU:.4f}, Ma={MA:.4f}")

# =============================================================================
# §3. MLG CONFIGURATION
# =============================================================================
# Fine region: central portion of the channel (away from inlet/outlet)
# Placed away from all walls so fine BC = coupling only (no physical wall BC)
FINE_X_MIN = 30          # [coarse lu]  Start of fine region
FINE_X_MAX = 90          # [coarse lu]  End of fine region
FINE_Y_MIN = 3           # [coarse lu]  Away from ymin wall
FINE_Y_MAX = Ny - 4      # [coarse lu]  Away from ymax wall
FINE_Z_MIN = 2           # [coarse lu]  Away from zmin
FINE_Z_MAX = Nz - 3      # [coarse lu]  Away from zmax

OVERLAP_WIDTH = 2        # [coarse cells]
INTERP_SCHEME = "cubic"  # "cubic" or "compact_second_order"
FILTER_LEVEL = 1         # F→C filter depth

# Fine grid dimensions (auto-calculated)
_fdc_xmin = max(0, FINE_X_MIN - OVERLAP_WIDTH)
_fdc_xmax = min(Nx - 1, FINE_X_MAX + OVERLAP_WIDTH)
_fdc_ymin = max(0, FINE_Y_MIN - OVERLAP_WIDTH)
_fdc_ymax = min(Ny - 1, FINE_Y_MAX + OVERLAP_WIDTH)
_fdc_zmin = max(0, FINE_Z_MIN - OVERLAP_WIDTH)
_fdc_zmax = min(Nz - 1, FINE_Z_MAX + OVERLAP_WIDTH)
_fine_nx = (_fdc_xmax - _fdc_xmin) * 2 + 1
_fine_ny = (_fdc_ymax - _fdc_ymin) * 2 + 1
_fine_nz = (_fdc_zmax - _fdc_zmin) * 2 + 1

print(f"  Fine region: x[{FINE_X_MIN},{FINE_X_MAX}] "
      f"y[{FINE_Y_MIN},{FINE_Y_MAX}] z[{FINE_Z_MIN},{FINE_Z_MAX}]")
print(f"  Fine grid: {_fine_nx} × {_fine_ny} × {_fine_nz} "
      f"= {_fine_nx*_fine_ny*_fine_nz:,} nodes")
print(f"  Coarse grid: {Nx} × {Ny} × {Nz} = {Nx*Ny*Nz:,} nodes")

# =============================================================================
# §4. SIMULATION SETTINGS
# =============================================================================
simulation = {
    "device_mode": "gpu",
    "dimension": 3,
    "lattice_model": "D3Q27",
    "collision_model": "bgk",

    "domain": {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
    },

    "physics": {
        "Re": RE,
        "tau": TAU,
        "omega": OMEGA_LBM,
        "nu_lu": NU_LU,
        "u_ref_lu": U_REF_LU,
        "L_ref_lu": L_REF_LU,
        "initial_flow_velocity": [U_INLET, 0.0, 0.0],
    },

    "time": {
        "max_steps": 30000,
        "output_interval": 1000,
        "checkpoint_interval": 10000,
        "probe_interval": 100,
    },
}

# =============================================================================
# §5. BOUNDARY CONDITIONS
# =============================================================================
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
    "wall_south": {
        "location": "ymin",
        "method": "regularized_wall",
    },
    "wall_north": {
        "location": "ymax",
        "method": "regularized_wall",
    },
    "wall_zmin": {
        "location": "zmin",
        "method": "regularized_wall",
    },
    "wall_zmax": {
        "location": "zmax",
        "method": "regularized_wall",
    },
}

# =============================================================================
# §6. NO INTERNAL OBSTACLE
# =============================================================================
internal_geometry = {
    "type": "none",
}

# =============================================================================
# §7. MLG SECTION
# =============================================================================
mlg = {
    "enabled": True,
    "num_levels": 2,
    "overlap_width": OVERLAP_WIDTH,
    "interpolation": INTERP_SCHEME,
    "filter_level": FILTER_LEVEL,
    "levels": [
        {},  # Level 0 = simulation.domain (full domain, uses existing config)
        {
            "region": {
                "x_min": FINE_X_MIN,
                "x_max": FINE_X_MAX,
                "y_min": FINE_Y_MIN,
                "y_max": FINE_Y_MAX,
                "z_min": FINE_Z_MIN,
                "z_max": FINE_Z_MAX,
            },
        },
    ],
}

# =============================================================================
# §8. MONITORS & OUTPUT
# =============================================================================
conservation = {
    "enabled": True,
    "interval": 1000,
}

convergence = {
    "enabled": True,
    "method": "drag_coefficient",
    "window": 2000,
    "threshold": 1e-4,
}

force_calculation = {
    "enabled": False,
}

output = {
    "vtk": {
        "enabled": True,
    },
    "checkpoint": {
        "enabled": False,
    },
}

# =============================================================================
# §9. FINAL CONFIG
# =============================================================================
config = {
    "simulation": simulation,
    "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "mlg": mlg,
    "conservation": conservation,
    "convergence": convergence,
    "force_calculation": force_calculation,
    "output": output,
}