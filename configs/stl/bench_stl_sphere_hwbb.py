"""bench_sphere_hwbb STL twin — analytic sphere -> icosphere STL (HWBB).

Same bench5 5-level topology, same D = 2 L0-lu body at (16.5, 24, 24);
the mask now comes from input_files/geom/icosphere_r25_s4.stl
(R = 25 file units x scale 0.04, chord error ~3e-4 lu = 0.005 L4 cells).
Acceptance vs bench_sphere_hwbb: Cd plateau within the chord-band cell
difference (L4 solid 17,072 vs 17,077), stability, rho drift < 0.1%.

REPLICATED INIT ONLY (--dist-init fail-fasts on obstacles):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \
        --config configs/stl/bench_stl_sphere_hwbb.py --steps 32 --log-every 8
    python main.py --config configs/stl/bench_stl_sphere_hwbb.py --steps 32
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(os.path.dirname(_here))
_base = os.path.join(_repo, "configs", "hpc_bench", "bench_sphere_hwbb.py")
_spec = importlib.util.spec_from_file_location("bench_stl_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m.config

config["internal_geometry"] = {
    "stl": {
        "enabled": True,
        "file": os.path.join(_repo, "input_files", "geom",
                             "icosphere_r25_s4.stl"),
        "scale_to_lu": 1.0 / 25.0,
        "center_lu": (16.5, 24.0, 24.0),
        "wall_bc": "hwbb",
    },
}

# ── S4 fix: BC velocities are PHYSICAL [m/s] in the current schema ──────
# (setup._setup_boundaries converts via phys_to_lu_velocity = * dt/dx).
# The parent bench_sphere_hwbb predates the acoustic-scaling migration and
# feeds LATTICE 0.05 into that field -> u_bc = 2.2e-5 lu: six near-wall
# faces around a 0.05-lu uniform IC, i.e. a domain-wide deceleration
# compression wave from step 1 (observed in the first VTK). The parent is
# an HPC bit-anchor (gu1 baselines) and stays untouched; the S4 flow
# clones override with proper physical values. U_PHYS is derived through
# the SAME UnitConverter recipe setup uses (rotor tip speed sets dx/dt
# even with ALM disabled), so it round-trips to exactly 0.05 lu.
import sys as _sys
if _repo not in _sys.path:
    _sys.path.insert(0, _repo)
from src.solver.unit_converter import UnitConverter as _UC

U_LU = 0.05
_g = config["grid"]
_gc = {"Nx": _g["Nx"], "Ny": _g["Ny"], "resolution":
       (config.get("numerics", {}).get("resolution") or _g.get("resolution"))}
if _g.get("Nz") is not None:
    _gc["Nz"] = _g["Nz"]
_uc = _UC(
    physics=config["physics"], grid=_gc, numerics=config.get("numerics", {}),
    actuator_line=(config["actuator_line"]
                   if config.get("actuator_line", {}).get("rotor") else None),
)
U_PHYS = float(_uc.lu_to_phys_velocity(U_LU))   # 110.5975 m/s for bench5
for _face, _bc in config["boundaries"].items():
    _bc["velocity"] = [U_PHYS, 0.0, 0.0]
config["physics"]["initial_flow_velocity"] = [U_LU, 0.0, 0.0]  # [lu]

_folder = "results_bench_stl_sphere_hwbb"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
