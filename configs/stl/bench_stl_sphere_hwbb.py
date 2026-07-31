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

_folder = "results_bench_stl_sphere_hwbb"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
