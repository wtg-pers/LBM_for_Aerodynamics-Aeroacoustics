"""bench_stl_sphere_hwbb twin with Bouzidi IBB (ray-triangle q, S3).

Standard (non-esoteric) path only until S5 lands the eso IBB kernel:
    python main.py --config configs/stl/bench_stl_sphere_ibb.py --steps 32
Sanity twin: LBM_FORCE_Q_HALF=1 must reproduce the HWBB run bit-for-bit.
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "bench_stl_sphere_hwbb.py")
_spec = importlib.util.spec_from_file_location("bench_stl_ibb_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m.config

config["internal_geometry"]["stl"]["wall_bc"] = "ibb"

_folder = "results_bench_stl_sphere_ibb"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
