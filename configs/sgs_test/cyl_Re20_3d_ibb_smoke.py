"""Tiny 3D Re=20 cylinder smoke test for IBB-3D + SGS path verification.

Domain: 5D x 4D x 1D, D=10 lu, single-grid (no MLG).
Total cells: ~6k. Local-friendly. Purpose: verify end-to-end that
    - 3D cylinder geometry mask + q-fraction work
    - InterpolatedBounceBack 3D path constructs without errors
    - D3Q27 IBB kernel applies cleanly each step
    - SGS Smagorinsky on top runs without crash
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

print(f"  [smoke 3D IBB Re={RE}, D={D_LU}] Nx={Nx} Ny={Ny} Nz={Nz}, tau={TAU:.4f}")

simulation = {"device_mode": "gpu", "precision": "float32", "dimension": 3,
              "lattice_model": "D3Q27", "collision_model": "cumulant",
              "omega_3": 0.6, "omega_4": 1.4}
physics = {"rho": 1.0, "U_inf": 1.0, "Re": RE,
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
sgs = {"enabled": True, "model": "smagorinsky", "Cs": 0.18}
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

_folder = "result_3d_ibb_smag_smoke"
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
