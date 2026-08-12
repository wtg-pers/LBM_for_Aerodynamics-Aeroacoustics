"""Tiny 3D Re=20 cylinder smoke test for IBB-3D + WALE path verification.

Same domain as cyl_Re20_3d_ibb_smoke.py. Purpose: verify end-to-end that
    - WALE 3-pass (macro + wale + cumulant_collide_wale) dispatches
    - nu_t buffer populates and is non-zero where strain has rotation
    - simulation runs to completion without crash
"""

import numpy as np

D_LU = 10
RE   = 20.0
U_MAX_LU = 0.05

UPSTREAM_D, DOWNSTREAM_D, LATERAL_D = 1.5, 2.5, 2.0
SPAN_LU = 8

CPC_L0 = D_LU
Nx = int(round((UPSTREAM_D + 1.0 + DOWNSTREAM_D) * CPC_L0))
Ny = int(round(2.0 * LATERAL_D * CPC_L0))
Nz = SPAN_LU

CYL_CX = int(round(UPSTREAM_D * CPC_L0)) + CPC_L0 // 2
CYL_CY = Ny // 2

NU_LU = U_MAX_LU * D_LU / RE
TAU = 0.5 + 3.0 * NU_LU

print(f"  [smoke 3D IBB+WALE Re={RE}, D={D_LU}] Nx={Nx} Ny={Ny} Nz={Nz}, tau={TAU:.4f}")

simulation = {"device_mode": "gpu", "precision": "float32", "dimension": 3,
              "lattice_model": "D3Q27", "collision_model": "cumulant",
              "omega_3": 0.6, "omega_4": 1.4}
physics = {"rho": 1.0, "U_inf": 1.0, "nu": ((1.0) * (1.0) / (RE)),  # [m^2/s] auto-migrated (nu-only policy; Re key now ignored)
           "L_char": 1.0, "flow_direction": [1.0, 0.0, 0.0]}
grid = {"Nx": Nx, "Ny": Ny, "Nz": Nz, "resolution": CPC_L0}
numerics = {"u_max": U_MAX_LU, "collision": "cumulant"}
boundaries = {
    "inlet":  {"location": "xmin", "method": "eq",
               "velocity": [1.0, 0.0, 0.0]},
    "outlet": {"location": "xmax", "method": "neumann"},
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
        "wall_bc": "ibb",
    },
}
sgs = {"enabled": True, "model": "wale", "Cw": 0.5}
force_calculation = {
    "enabled": True, "interval": 10, "start_step": 50,
    "save_link_forces": False,
    "reference": {"rho": 1.0, "velocity": U_MAX_LU,
                  "char_length": float(D_LU), "span_length": Nz},
    "log": {"enabled": True, "filename": "force_history"},
}
time = {"max_steps": 500, "output_interval": 250,
        "logging_interval": 100, "checkpoint_interval": 1_000_000,
        "conservation_interval": 250}
conservation = {"enabled": False}
convergence  = {"enabled": False}

_folder = "result_3d_ibb_wale_smoke"
output = {
    "output_dir":     f"./_sgs_test/{_folder}/vtk",
    "checkpoint_dir": f"./_sgs_test/{_folder}/checkpoints",
    "csv_dir":        f"./_sgs_test/{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": True, "precision": "float32",
            "variables": ["density", "velocity", "solid_mask", "nu_t"]},
    "checkpoint": {"enabled": False},
}

config = {
    "simulation": simulation, "physics": physics, "grid": grid,
    "numerics": numerics, "boundaries": boundaries,
    "internal_geometry": internal_geometry,
    "sgs": sgs,
    "conservation": conservation, "convergence": convergence,
    "force_calculation": force_calculation, "output": output, "time": time,
}
