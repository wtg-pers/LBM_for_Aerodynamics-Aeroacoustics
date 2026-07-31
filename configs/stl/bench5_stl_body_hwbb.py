"""bench5_stl_body_ibb twin with HWBB walls (S7 variant).

Same rotor/ALM/body placement; only internal_geometry.stl.wall_bc and the
results folder differ. Sanity twin: LBM_FORCE_Q_HALF=1 on the ibb config
must reproduce this run bit-for-bit (S3/S5 invariant).

    python main.py --config configs/stl/bench5_stl_body_hwbb.py
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "bench5_stl_body_ibb.py")
_spec = importlib.util.spec_from_file_location("bench5_stl_body_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m.config

config["internal_geometry"]["stl"]["wall_bc"] = "hwbb"

_folder = "results_bench5_stl_body_hwbb"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
