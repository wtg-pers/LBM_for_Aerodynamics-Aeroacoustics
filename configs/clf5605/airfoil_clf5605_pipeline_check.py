"""CLF5605 pipeline check — Re=500, low enough to be robustly stable
at CPC=40 (τ=0.512), so the post-processor's upper/lower Cp(x/c) split
can be verified without chasing NaN instabilities.

NOT physics-accurate. Purpose: confirm that
    result_clf5605_pipeline/csv/surface_cp.dat
has an "upper" and a "lower" zone, Cp_LE_stag is positive, and the
density-based path reports `Cp mode: density`.

Usage:
    python main.py --config configs/airfoil_clf5605_pipeline_check.py
    python -m src.utilities.surface_distribution \
        result_clf5605_pipeline configs/airfoil_clf5605_pipeline_check.py \
        --window-start 500
"""

import os
import numpy as np

AOA_DEG    = 2.0
RE         = 500.0          # ← low Re for rock-solid stability
RHO_PHYS   = 1.0
U_INF_PHYS = 1.0            # (arbitrary; pure lattice-unit operation)
CHORD_PHYS = 1.0

CPC        = 40
UPSTREAM_C, DOWNSTREAM_C = 3.0, 6.0
LATERAL_C  = 3.0
U_MAX_LU   = 0.05

HERE = os.path.dirname(os.path.abspath(__file__))
SELIG_FILE = os.path.abspath(os.path.join(HERE, '..', 'to_claude', 'clf5605.dat'))

DX_PHYS = CHORD_PHYS / CPC
DT_PHYS = U_MAX_LU * DX_PHYS / U_INF_PHYS
U_INF_LU = U_MAX_LU
CHORD_LU = float(CPC)
Nx = int(round((UPSTREAM_C + 1.0 + DOWNSTREAM_C) * CPC))
Ny = int(round(2.0 * LATERAL_C * CPC))
AIRFOIL_CENTER = (int(round((UPSTREAM_C + 0.5) * CPC)), Ny // 2)
NU_LU = U_INF_LU * CHORD_LU / RE        # = 0.004
TAU   = 0.5 + 3.0 * NU_LU               # = 0.512 (safe)

print(f"  [CLF5605 pipeline α={AOA_DEG}° Re={RE:.0f}]  "
      f"{Nx}×{Ny} = {Nx*Ny/1e6:.2f} M cells, τ={TAU:.4f}")

simulation = {
    "device_mode": "gpu", "precision": "float32",
    "dimension": 2, "lattice_model": "D2Q9",
    "collision_model": "cumulant",
}
physics = {
    "rho": RHO_PHYS, "U_inf": U_INF_PHYS, "Re": RE,
    "L_char": CHORD_PHYS, "flow_direction": [1.0, 0.0],
}
grid = {"Nx": Nx, "Ny": Ny, "resolution": CPC}
numerics = {"u_max": U_MAX_LU, "collision": "cumulant"}
boundaries = {
    "inlet":  {"location": "xmin", "method": "eq",
               "velocity": [U_INF_PHYS, 0.0], "rho": 1.0},
    "outlet": {"location": "xmax", "method": "sponge",
               "velocity": [U_INF_PHYS, 0.0], "density": 1.0,
               "thickness": 10, "strength": 0.5},
    "ymin":   {"location": "ymin", "method": "neumann"},
    "ymax":   {"location": "ymax", "method": "neumann"},
}
internal_geometry = {
    "airfoil": {
        "enabled": True, "selig_file": SELIG_FILE,
        "chord": CHORD_LU, "center": AIRFOIL_CENTER,
        "angle_of_attack": AOA_DEG,
        "wall_bc": "ibb",          # test IBB path on thin airfoil
    },
}
force_calculation = {
    "enabled": True, "interval": 10, "start_step": 100,
    "reference": {"rho": 1.0, "velocity": U_INF_LU,
                  "char_length": CHORD_LU, "span_length": 1},
    "log": {"enabled": True, "filename": "force_history"},
}
mlg = {"enabled": False}
time = {
    "max_steps": 3000,
    "output_interval": 500,
    "logging_interval": 50,
    "checkpoint_interval": 2000,
    "conservation_interval": 500,
}
conservation = {"enabled": True, "verbose": 0, "log_to_csv": True}
convergence = {"enabled": False}

_folder = "result_clf5605_pipeline"
output = {
    "output_dir":     f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir":        f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": True, "precision": "float32",
            "variables": ["density", "velocity", "solid_mask"]},
    "checkpoint": {"enabled": True, "keep_last_n": 1},
}
config = {
    "simulation": simulation, "physics": physics, "grid": grid,
    "numerics": numerics, "boundaries": boundaries,
    "internal_geometry": internal_geometry, "mlg": mlg,
    "conservation": conservation,
    "convergence": convergence, "force_calculation": force_calculation,
    "output": output,
    "time":       time,
}
