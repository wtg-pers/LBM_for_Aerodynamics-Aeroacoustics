"""
3D Cylinder Re=3900 — IBB + padded MLG-3, D=20  (v2 + Smagorinsky SGS)

Same domain & MLG layout as cyl_Re3900_ibb_mlg3_3d_v2.py.
Only difference: SGS Smagorinsky (Cs = 0.18) is enabled.

Goal: bring Cd_mean from cumulant-baseline (~1.22) toward 3D LES range
(Rajani Run 6/7: 1.01-1.05; experiment 0.98 +- 0.05).

Open issue: MLG f^neq rescaling currently uses molecular per-level tau
(see src/grid/level_scaling.py), not the SGS-modified s_ii values that
Stiebler 2011 prescribes (Eq. 15-17). Stress continuity at the L0/L1 and
L1/L2 interfaces is therefore approximate when SGS is active. Magnitude
of the artefact will be assessed from this run; Stiebler-correct
rescaling (M-2) is the next refinement if Cd remains > 1.10.

Reference 3D LES (Rajani et al. 2016, Run 6/7):
    Cd_mean = 1.01-1.05
    -Cpb    = 0.90-0.93
    St      = 0.21
    theta_sep = 87.5 deg
    L_w/D   = 1.20
    Measurement: Cd = 0.98 +- 0.05, St = 0.215, theta_sep = 86 deg
"""

import numpy as np

WALL_BC    = "ibb"
NUM_LEVELS = 3
REFINE     = 2

D_LU       = 20
RE         = 3900.0
U_MAX_LU   = 0.05

D_PHYS     = 1.0
U_INF_PHYS = 1.0
RHO_PHYS   = 1.0
NU_PHYS    = U_INF_PHYS * D_PHYS / RE

UPSTREAM_D   = 10.0
DOWNSTREAM_D = 19.0
LATERAL_D    = 15.0
SPAN_D       = np.pi

L1_UP_D    = 2.0
L1_DOWN_D  = 10.0
L1_LAT_D   = 2.0

L2_UP_D    = 1.0
L2_DOWN_D  = 4.0
L2_LAT_D   = 1.0

CPC_L0     = D_LU
CPC_FINEST = CPC_L0 * (REFINE ** (NUM_LEVELS - 1))

Nx = int(round((UPSTREAM_D + 1.0 + DOWNSTREAM_D) * CPC_L0))
Ny = int(round(2.0 * LATERAL_D * CPC_L0))
Nz = int(round(SPAN_D * CPC_L0))

CYL_CX = int(round(UPSTREAM_D * CPC_L0)) + CPC_L0 // 2
CYL_CY = Ny // 2

L1_XMIN = CYL_CX - int(round(L1_UP_D   * CPC_L0))
L1_XMAX = CYL_CX + int(round(L1_DOWN_D * CPC_L0))
L1_YMIN = CYL_CY - int(round(L1_LAT_D  * CPC_L0))
L1_YMAX = CYL_CY + int(round(L1_LAT_D  * CPC_L0))
L1_ZMIN = 0
L1_ZMAX = Nz - 1

L2_XMIN = CYL_CX - int(round(L2_UP_D   * CPC_L0))
L2_XMAX = CYL_CX + int(round(L2_DOWN_D * CPC_L0))
L2_YMIN = CYL_CY - int(round(L2_LAT_D  * CPC_L0))
L2_YMAX = CYL_CY + int(round(L2_LAT_D  * CPC_L0))
L2_ZMIN = 0
L2_ZMAX = Nz - 1

NU_LU = U_MAX_LU * D_LU / RE
TAU   = 0.5 + 3.0 * NU_LU
CS    = 1.0 / np.sqrt(3.0)
MA_LU = U_MAX_LU / CS

assert TAU > 0.5
assert MA_LU < 0.3
assert (L1_XMIN >= 0 and L1_XMAX <= Nx - 1 and
        L1_YMIN >= 0 and L1_YMAX <= Ny - 1)
assert (L2_XMIN >= L1_XMIN and L2_XMAX <= L1_XMAX and
        L2_YMIN >= L1_YMIN and L2_YMAX <= L1_YMAX)

nodes_L0 = Nx * Ny * Nz
nodes_L1 = (L1_XMAX - L1_XMIN + 4) * (L1_YMAX - L1_YMIN + 4) * Nz * 8
nodes_L2 = (L2_XMAX - L2_XMIN + 4) * (L2_YMAX - L2_YMIN + 4) * Nz * 64
tot_nodes = nodes_L0 + nodes_L1 + nodes_L2
mem_gb = tot_nodes * 250 / 1e9

print(f"  [cyl Re={RE:.0f}, 3D MLG-{NUM_LEVELS} v2+SMAG (D_fine={CPC_FINEST}), D={D_LU} L0, {WALL_BC}]")
print(f"  L0 {Nx}x{Ny}x{Nz} = {nodes_L0/1e6:.2f}M")
print(f"  L1 ~{nodes_L1/1e6:.1f}M, L2 ~{nodes_L2/1e6:.1f}M")
print(f"  tau_L0={TAU:.4f}, Ma={MA_LU:.4f}, Total ~{tot_nodes/1e6:.1f}M cells, ~{mem_gb:.1f} GB")
print(f"  SGS: Smagorinsky Cs=0.18 enabled")

simulation = {"device_mode": "gpu", "precision": "float32", "dimension": 3,
              "lattice_model": "D3Q27", "collision_model": "cumulant",
              "omega_3": 0.6, "omega_4": 1.4}
physics    = {"rho": RHO_PHYS, "U_inf": U_INF_PHYS, "Re": RE,
 "nu": ((U_INF_PHYS) * (D_PHYS) / (RE)),  # [m^2/s] auto-migrated (nu-only policy; Re key now ignored)
              "L_char": D_PHYS, "flow_direction": [1.0, 0.0, 0.0]}
grid       = {"Nx": Nx, "Ny": Ny, "Nz": Nz, "resolution": CPC_L0}
numerics   = {"u_max": U_MAX_LU, "collision": "cumulant"}
boundaries = {
    "inlet":  {"location": "xmin", "method": "eq",
               "velocity": [U_INF_PHYS, 0.0, 0.0]},
    "outlet": {"location": "xmax", "method": "sponge",
               "velocity": [U_INF_PHYS, 0.0, 0.0], "density": 1.0,
               "thickness": 50, "strength": 0.5},
    "ymin":   {"location": "ymin", "method": "slip"},
    "ymax":   {"location": "ymax", "method": "slip"},
    "zmin":   {"location": "zmin", "method": "periodic"},
    "zmax":   {"location": "zmax", "method": "periodic"},
}
internal_geometry = {
    "cylinder": {
        "enabled": True,
        "center":  (CYL_CX, CYL_CY),
        "radius":  D_LU / 2.0,
        "axis":    "z",
        "wall_bc": WALL_BC,
    },
}
mlg = {
    "enabled": True, "num_levels": NUM_LEVELS, "overlap_width": 2,
    "interpolation": "cubic", "filter_level": 1,
    "levels": [
        {},
        {"region": {"x_min": L1_XMIN, "x_max": L1_XMAX,
                    "y_min": L1_YMIN, "y_max": L1_YMAX,
                    "z_min": L1_ZMIN, "z_max": L1_ZMAX}},
        {"region": {"x_min": L2_XMIN, "x_max": L2_XMAX,
                    "y_min": L2_YMIN, "y_max": L2_YMAX,
                    "z_min": L2_ZMIN, "z_max": L2_ZMAX}},
    ],
}

# === SGS Smagorinsky ====================================================
sgs = {"enabled": True, "model": "smagorinsky", "Cs": 0.18}

force_calculation = {
    "enabled": True, "interval": 10, "start_step": 1000,
    "save_link_forces": False,
    "reference": {"rho": 1.0, "velocity": U_MAX_LU,
                  "char_length": float(D_LU), "span_length": Nz},
    "log": {"enabled": True, "filename": "force_history"},
}
time = {"max_steps": 200_000, "output_interval": 50_000,
        "logging_interval": 1_000, "checkpoint_interval": 100_000,
        "conservation_interval": 5_000}
conservation = {"enabled": True, "verbose": 0, "log_to_csv": True}
convergence = {
    "enabled": True,
    "cauchy": {"window_size": "auto", "epsilon": 1e-5,
               "Cd_epsilon": 3e-4, "n_required": 3},
    "on_converged": "checkpoint_and_stop",
    "on_diverged":  "stop_with_checkpoint",
    "on_max_steps": "continue",
}

_folder = "result_cyl_Re3900_ibb_mlg3_3d_v2_smag"
output = {
    "output_dir":     f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir":        f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": True, "precision": "float32",
            "variables": ["density", "pressure", "velocity",
                          "velocity_magnitude", "solid_mask", "nu_t"]},
    "checkpoint": {"enabled": True, "keep_last_n": 2},
}

config = {
    "simulation": simulation, "physics": physics, "grid": grid,
    "numerics": numerics, "boundaries": boundaries,
    "internal_geometry": internal_geometry, "mlg": mlg,
    "sgs": sgs,
    "conservation": conservation, "convergence": convergence,
    "force_calculation": force_calculation, "output": output, "time": time,
}
