"""2D end-to-end regression case — cyl Re=100 IBB + padded MLG-3, SHRUNK.

Gate companion of configs/cylinders/cyl_Re100_ibb_mlg3.py (review #3, R3-3):
the 2D path was a gate-ladder blind spot (the fixed-order fallback range(3)
regression broke every D2Q9 run and only a manual run caught it). This case
drives the full production 2D chain — D2Q9 cumulant + IBB + 3-level MLG +
MEM force + convergence machinery — in ~1 minute.

Shrunk vs the physics config (25D x 16D instead of 60D x 40D, D_LU=16,
12k steps): blockage ~6% biases Cd slightly high vs unbounded literature
(~1.35 Henderson), so the gate asserts the RECORDED band of this exact
config, with literature only as a sanity anchor. Padded-MLG3 rule kept
(L2 edge >= 0.5D from the surface).

Run (the gate does this):  python main.py --config configs/hpc_bench/cyl_re100_2d_gate.py
"""

import numpy as np

WALL_BC    = "ibb"
NUM_LEVELS = 3
REFINE     = 2

D_LU       = 16
RE         = 100.0
U_MAX_LU   = 0.05

D_PHYS     = 1.0
U_INF_PHYS = 1.0
RHO_PHYS   = 1.0

UPSTREAM_D   = 8.0
DOWNSTREAM_D = 16.0
LATERAL_D    = 8.0

L1_UP_D    = 2.0
L1_DOWN_D  = 4.0
L1_LAT_D   = 2.0

L2_UP_D    = 1.0
L2_DOWN_D  = 2.0
L2_LAT_D   = 1.0

CPC_L0     = D_LU

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
MA_LU = U_MAX_LU * np.sqrt(3.0)

assert TAU > 0.5 and MA_LU < 0.3
assert L1_XMIN >= 0 and L1_XMAX <= Nx - 1 and L1_YMIN >= 0 and L1_YMAX <= Ny - 1
assert (L2_XMIN >= L1_XMIN and L2_XMAX <= L1_XMAX and
        L2_YMIN >= L1_YMIN and L2_YMAX <= L1_YMAX)

print(f"  [cyl Re=100 2D GATE, D_fine={D_LU * REFINE ** (NUM_LEVELS - 1)}]")
print(f"  L0 {Nx}x{Ny}, cyl=({CYL_CX},{CYL_CY})  tau_L0={TAU:.4f}")

simulation = {"device_mode": "gpu", "precision": "float32", "dimension": 2,
              "lattice_model": "D2Q9", "collision_model": "cumulant",
              "omega_3": 0.6, "omega_4": 1.4}
physics    = {"rho": RHO_PHYS, "U_inf": U_INF_PHYS,
              "nu": U_INF_PHYS * D_PHYS / RE,
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
    "enabled": True, "interval": 10, "start_step": 400,
    "save_link_forces": False,
    "reference": {"rho": 1.0, "velocity": U_MAX_LU,
                  "char_length": float(D_LU), "span_length": 1},
    "log": {"enabled": True, "filename": "force_history"},
}
time = {"max_steps": 12_000, "output_interval": 50_000,
        "logging_interval": 2_000, "checkpoint_interval": 50_000,
        "conservation_interval": 5_000}
conservation = {"enabled": False}
convergence = {"enabled": False}

_folder = "result_cyl_re100_2d_gate"
output = {
    "output_dir":     f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir":        f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": False},
    "checkpoint": {"enabled": False},
}

config = {
    "simulation": simulation, "physics": physics, "grid": grid,
    "numerics": numerics, "boundaries": boundaries,
    "internal_geometry": internal_geometry, "mlg": mlg,
    "conservation": conservation, "convergence": convergence,
    "force_calculation": force_calculation, "output": output, "time": time,
}
