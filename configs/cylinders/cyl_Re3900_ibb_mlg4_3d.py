"""
3D Cylinder Re=3900 — IBB + padded MLG-4, D=20

Same compact 3D domain as `_mlg3_3d.py` plus an additional L3 finest
level to better resolve the BL at high Re:
    L0  D=20  (full 30D x 20D x πD/2 domain)
    L1  D=40  ±3D up / +6D down / ±3D lat   (larger outer)
    L2  D=80  ±2D up / +4D down / ±2D lat   (sweep mlg2 region)
    L3  D=160 ±1D up / +2D down / ±1D lat   (cylinder + 0.5D padding;
                                              same as mlg3 padded L2)

Estimated cells: **~70M** (DOMINATED BY L3 = ~40M because refine ×8
combined with full spanwise extent). This **exceeds the user 20M budget**.
A100 80GB / H100 80GB single GPU can handle it (~17 GB), or:
  - reduce span (πD/2 → πD/3 or D) — will lose Mode A capture
  - shrink L3 region (less padding from cylinder) — risks BL artifact
  - drop L3 entirely → use mlg3_3d sibling instead

BL at Re=3900 ≈ 0.016D = 0.32 lu (L0). In L3 fine: 2.56 lu — marginal
but better than mlg3 (1.28 lu in L2).
"""

import numpy as np

WALL_BC    = "ibb"
NUM_LEVELS = 4
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
LATERAL_D    = 10.0
SPAN_D       = np.pi / 2

L1_UP_D    = 3.0
L1_DOWN_D  = 6.0
L1_LAT_D   = 3.0

L2_UP_D    = 2.0
L2_DOWN_D  = 4.0
L2_LAT_D   = 2.0

L3_UP_D    = 1.0
L3_DOWN_D  = 2.0
L3_LAT_D   = 1.0

CPC_L0     = D_LU
CPC_FINEST = CPC_L0 * (REFINE ** (NUM_LEVELS - 1))   # 160

Nx = int(round((UPSTREAM_D + 1.0 + DOWNSTREAM_D) * CPC_L0))
Ny = int(round(2.0 * LATERAL_D * CPC_L0))
Nz = int(round(SPAN_D * CPC_L0))

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

L3_XMIN = CYL_CX - int(round(L3_UP_D   * CPC_L0))
L3_XMAX = CYL_CX + int(round(L3_DOWN_D * CPC_L0))
L3_YMIN = CYL_CY - int(round(L3_LAT_D  * CPC_L0))
L3_YMAX = CYL_CY + int(round(L3_LAT_D  * CPC_L0))

NU_LU = U_MAX_LU * D_LU / RE
TAU   = 0.5 + 3.0 * NU_LU
CS    = 1.0 / np.sqrt(3.0)
MA_LU = U_MAX_LU / CS

assert TAU > 0.5
assert MA_LU < 0.3
assert (L1_XMIN >= 0 and L1_XMAX <= Nx - 1 and
        L1_YMIN >= 0 and L1_YMAX <= Ny - 1), "L1 outside L0"
assert (L2_XMIN >= L1_XMIN and L2_XMAX <= L1_XMAX and
        L2_YMIN >= L1_YMIN and L2_YMAX <= L1_YMAX), "L2 outside L1"
assert (L3_XMIN >= L2_XMIN and L3_XMAX <= L2_XMAX and
        L3_YMIN >= L2_YMIN and L3_YMAX <= L2_YMAX), "L3 outside L2"

nodes_L0 = Nx * Ny * Nz
nodes_L1 = (L1_XMAX - L1_XMIN + 4) * (L1_YMAX - L1_YMIN + 4) * Nz * 8
nodes_L2 = (L2_XMAX - L2_XMIN + 4) * (L2_YMAX - L2_YMIN + 4) * Nz * 64
nodes_L3 = (L3_XMAX - L3_XMIN + 4) * (L3_YMAX - L3_YMIN + 4) * Nz * 512
tot_nodes = nodes_L0 + nodes_L1 + nodes_L2 + nodes_L3
mem_gb = tot_nodes * 250 / 1e9

print(f"  [cyl Re={RE:.0f}, 3D MLG-{NUM_LEVELS} (D_fine={CPC_FINEST}), D={D_LU} L0, {WALL_BC}]")
print(f"  L0 {Nx}x{Ny}x{Nz} = {nodes_L0/1e6:.2f}M, span={SPAN_D:.3f}D = {Nz} lu")
print(f"  L1 box xy: x[{L1_XMIN},{L1_XMAX}] y[{L1_YMIN},{L1_YMAX}], z full → ~{nodes_L1/1e6:.1f}M")
print(f"  L2 box xy: x[{L2_XMIN},{L2_XMAX}] y[{L2_YMIN},{L2_YMAX}], z full → ~{nodes_L2/1e6:.1f}M")
print(f"  L3 box xy: x[{L3_XMIN},{L3_XMAX}] y[{L3_YMIN},{L3_YMAX}], z full → ~{nodes_L3/1e6:.1f}M")
print(f"  cyl center=({CYL_CX},{CYL_CY}), axis=z, L3 padding=0.5D")
print(f"  τ_L0={TAU:.4f}, Ma={MA_LU:.4f}")
print(f"  Total ~{tot_nodes/1e6:.1f}M cells, est ~{mem_gb:.1f} GB (NEEDS A100/H100 80GB)")

simulation = {"device_mode": "gpu", "precision": "float32", "dimension": 3,
              "lattice_model": "D3Q27", "collision_model": "cumulant",
              "omega_3": 0.6, "omega_4": 1.4}
physics    = {"rho": RHO_PHYS, "U_inf": U_INF_PHYS, "Re": RE,
              "L_char": D_PHYS, "flow_direction": [1.0, 0.0, 0.0]}
grid       = {"Nx": Nx, "Ny": Ny, "Nz": Nz, "resolution": CPC_L0}
numerics   = {"u_max": U_MAX_LU, "collision": "cumulant"}
boundaries = {
    "inlet":  {"location": "xmin", "method": "eq",
               "velocity": [U_INF_PHYS, 0.0, 0.0]},
    "outlet": {"location": "xmax", "method": "sponge",
               "velocity": [U_INF_PHYS, 0.0, 0.0], "density": 1.0,
               "thickness": 30, "strength": 0.5},
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
                    "z_min": 0, "z_max": Nz - 1}},
        {"region": {"x_min": L2_XMIN, "x_max": L2_XMAX,
                    "y_min": L2_YMIN, "y_max": L2_YMAX,
                    "z_min": 0, "z_max": Nz - 1}},
        {"region": {"x_min": L3_XMIN, "x_max": L3_XMAX,
                    "y_min": L3_YMIN, "y_max": L3_YMAX,
                    "z_min": 0, "z_max": Nz - 1}},
    ],
}
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

_folder = f"result_cyl_Re3900_ibb_mlg4_3d"
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
