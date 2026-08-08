"""NACA0012 a=10, FULL grid (c=100, Nz=16) — wall-damped SGS final.

Production-grid re-measurement with the wall-SGS model chosen by the
nz4 A/B (run ONLY the winning variant of {this, _sgsoff}). See
naca0012_a10_nz4_walldamp2.py for the wall_damp_cells rationale.

    LBM_ESOTERIC=1 mpirun -n 2 python main.py \
        --config configs/stl/naca0012_a10_walldamp2.py \
        --gpu 2,3 --dist-init --max-steps 40000
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_wd2f_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100)

config["sgs"]["wall_damp_cells"] = 2

_folder = "results_naca0012_a10_walldamp2"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
