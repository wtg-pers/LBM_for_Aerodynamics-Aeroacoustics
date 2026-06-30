"""
2D Cylinder Re=800, Henderson 60D×40D — IBB + padded MLG-3 (auto-gen)

Re sweep member. Setup follows Task #2 padded-mlg3 rule
(L2 region keeps ≥ 0.5D padding from cylinder surface, otherwise
BL coupling artifact shifts Cd ~3%).

Layers (in L0 lu, D_LU=20):
    L0:  full 60D × 40D domain   (D=20)
    L1:  ±2D up / +4D down / ±2D lat   (D=40)
    L2:  ±1D up / +2D down / ±1D lat   (D=80)
        — L2 user edges sit 10 lu (= 0.5D) from cylinder surface

Reference Re-Cd table assembled in summarize_cyl_re_sweep.py.
"""

import numpy as np

WALL_BC    = "ibb"
NUM_LEVELS = 3
REFINE     = 2

D_LU       = 20
RE         = 800.0
U_MAX_LU   = 0.05

D_PHYS     = 1.0
U_INF_PHYS = 1.0
RHO_PHYS   = 1.0
NU_PHYS    = U_INF_PHYS * D_PHYS / RE

UPSTREAM_D   = 20.0
DOWNSTREAM_D = 39.0
LATERAL_D    = 20.0

L1_UP_D    = 2.0
L1_DOWN_D  = 4.0
L1_LAT_D   = 2.0

L2_UP_D    = 1.0
L2_DOWN_D  = 2.0
L2_LAT_D   = 1.0

CPC_L0     = D_LU
CPC_FINEST = CPC_L0 * (REFINE ** (NUM_LEVELS - 1))

Nx = int(round((UPSTREAM_D + 1.0 + DOWNSTREAM_D) * CPC_L0))
Ny = int(round(2.0 * LATERAL_D * CPC_L0))
CYL_CX = int(round(UPSTREAM_D * CPC_L0)) + CPC_L0 // 2
CYL_CY = Ny // 2

L1_XMIN = CYL_CX - int(round(L1_UP_D   * CPC_L0))
L1_XMAX = CYL_CX + int(round(L1_DOWN_D * CPC_L0))
L1_YMIN = CYL_CY - int(round(L1_LAT_D  * CPC_L0))
L1_YMAX = CYL_CY + int(round(L1_LAT_D  * CPC_L0))

L2_XMIN = CYL_CX - int(round(L2_UP_D   * CPC_L0))
L2_XMAX = CYL_CX + int(round(L2_DOWN_D * CPC_L0))
L2_YMIN = CYL_CY - int(round(L2_LAT_D  * CPC_L0))
L2_YMAX = CYL_CY + int(round(L2_LAT_D  * CPC_L0))

NU_LU = U_MAX_LU * D_LU / RE
TAU   = 0.5 + 3.0 * NU_LU
CS    = 1.0 / np.sqrt(3.0)
MA_LU = U_MAX_LU / CS

assert TAU > 0.5, f"UNSTABLE: tau={TAU}"
assert MA_LU < 0.3
assert (L1_XMIN >= 0 and L1_XMAX <= Nx - 1 and
        L1_YMIN >= 0 and L1_YMAX <= Ny - 1), "L1 outside L0"
assert (L2_XMIN >= L1_XMIN and L2_XMAX <= L1_XMAX and
        L2_YMIN >= L1_YMIN and L2_YMAX <= L1_YMAX), "L2 outside L1"

print(f"  [cyl Re=800, D_fine={CPC_FINEST}, padded MLG-3 IBB]")
print(f"  L0 {Nx}x{Ny}={Nx*Ny/1e6:.2f}M, cyl=({CYL_CX},{CYL_CY})  "
      f"tau_L0={TAU:.4f}, Ma={MA_LU:.4f}")

simulation = {"device_mode": "gpu", "precision": "float32", "dimension": 2,
              "lattice_model": "D2Q9", "collision_model": "cumulant",
              "omega_3": 0.6, "omega_4": 1.4}
physics    = {"rho": RHO_PHYS, "U_inf": U_INF_PHYS, "Re": RE,
              "L_char": D_PHYS, "flow_direction": [1.0, 0.0]}
grid       = {"Nx": Nx, "Ny": Ny, "resolution": CPC_L0}
numerics   = {"u_max": U_MAX_LU, "collision": "cumulant"}
boundaries = {
    "inlet":  {"location": "xmin", "method": "eq",
               "velocity": [U_INF_PHYS, 0.0]},
    "outlet": {"location": "xmax", "method": "sponge",
               "velocity": [U_INF_PHYS, 0.0], "density": 1.0,
               "thickness": 30, "strength": 0.5},
    "ymin":   {"location": "ymin", "method": "slip"},
    "ymax":   {"location": "ymax", "method": "slip"},
}
internal_geometry = {
    "circle": {"enabled": True, "center": (CYL_CX, CYL_CY),
               "radius": D_LU / 2.0, "wall_bc": WALL_BC},
}
mlg = {
    "enabled": True, "num_levels": NUM_LEVELS, "overlap_width": 2,
    "interpolation": "cubic", "filter_level": 1,
    "levels": [
        {},
        {"region": {"x_min": L1_XMIN, "x_max": L1_XMAX,
                    "y_min": L1_YMIN, "y_max": L1_YMAX}},
        {"region": {"x_min": L2_XMIN, "x_max": L2_XMAX,
                    "y_min": L2_YMIN, "y_max": L2_YMAX}},
    ],
}
force_calculation = {
    "enabled": True, "interval": 10, "start_step": 1000,
    "save_link_forces": False,
    "reference": {"rho": 1.0, "velocity": U_MAX_LU,
                  "char_length": float(D_LU), "span_length": 1},
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

_folder = f"result_cyl_Re800_ibb_mlg3"
output = {
    "output_dir":     f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir":        f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": True, "precision": "float32",
            "variables": ["density", "pressure", "velocity",
                          "velocity_magnitude", "solid_mask"]},
    "checkpoint": {"enabled": True, "keep_last_n": 2},
}

config = {
    "simulation": simulation, "physics": physics, "grid": grid,
    "numerics": numerics, "boundaries": boundaries,
    "internal_geometry": internal_geometry, "mlg": mlg,
    "conservation": conservation, "convergence": convergence,
    "force_calculation": force_calculation, "output": output, "time": time,
}
