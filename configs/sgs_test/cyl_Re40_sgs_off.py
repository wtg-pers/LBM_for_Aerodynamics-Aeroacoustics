"""SGS regression test config — SGS off, Re=40, single-grid, D=20."""

import numpy as np

D_LU       = 20
RE         = 40.0
U_MAX_LU   = 0.05

D_PHYS     = 1.0
U_INF_PHYS = 1.0
RHO_PHYS   = 1.0

UPSTREAM_D, DOWNSTREAM_D, LATERAL_D = 10.0, 19.0, 10.0
CPC_L0 = D_LU

Nx = int(round((UPSTREAM_D + 1.0 + DOWNSTREAM_D) * CPC_L0))
Ny = int(round(2.0 * LATERAL_D * CPC_L0))
CYL_CX = int(round(UPSTREAM_D * CPC_L0)) + CPC_L0 // 2
CYL_CY = Ny // 2

NU_LU = U_MAX_LU * D_LU / RE
TAU   = 0.5 + 3.0 * NU_LU
print(f"  [SGS regression: Re={RE}, D={D_LU}, single grid, tau={TAU:.4f}]")
print(f"  Nx={Nx}, Ny={Ny}")

simulation = {"device_mode": "gpu", "precision": "float32", "dimension": 2,
              "lattice_model": "D2Q9", "collision_model": "cumulant",
              "omega_3": 0.6, "omega_4": 1.4}
physics    = {"rho": RHO_PHYS, "U_inf": U_INF_PHYS, "nu": ((U_INF_PHYS) * (D_PHYS) / (RE)),  # [m^2/s] auto-migrated (nu-only policy; Re key now ignored)
              "L_char": D_PHYS, "flow_direction": [1.0, 0.0]}
grid       = {"Nx": Nx, "Ny": Ny, "resolution": CPC_L0}
numerics   = {"u_max": U_MAX_LU, "collision": "cumulant"}
boundaries = {
    "inlet":  {"location": "xmin", "method": "eq",
               "velocity": [U_INF_PHYS, 0.0]},
    "outlet": {"location": "xmax", "method": "neumann"},
    "ymin":   {"location": "ymin", "method": "slip"},
    "ymax":   {"location": "ymax", "method": "slip"},
}
internal_geometry = {
    "circle": {"enabled": True, "center": (CYL_CX, CYL_CY),
               "radius": D_LU / 2.0, "wall_bc": "ibb"},
}

# SGS DISABLED for this regression baseline
sgs = {"enabled": False, "model": "off"}

force_calculation = {
    "enabled": True, "interval": 10, "start_step": 100,
    "save_link_forces": False,
    "reference": {"rho": 1.0, "velocity": U_MAX_LU,
                  "char_length": float(D_LU), "span_length": 1},
    "log": {"enabled": True, "filename": "force_history"},
}
time = {"max_steps": 5000, "output_interval": 100000,
        "logging_interval": 1000, "checkpoint_interval": 100000,
        "conservation_interval": 5000}
conservation = {"enabled": False}
convergence  = {"enabled": False}

_folder = "result_sgs_off"
output = {
    "output_dir":     f"./_sgs_test/{_folder}/vtk",
    "checkpoint_dir": f"./_sgs_test/{_folder}/checkpoints",
    "csv_dir":        f"./_sgs_test/{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": False},
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
