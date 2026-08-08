"""NACA0012 a=10, FULL grid (c=100, Nz=16) — SGS-off (ILES) final.

Production-grid re-measurement with the wall-SGS model chosen by the
nz4 A/B (run ONLY the winning variant of {this, _walldamp2}). Pure
ILES: cumulant limiter only, matching the HVAB production philosophy.

    LBM_ESOTERIC=1 mpirun -n 2 python main.py \
        --config configs/stl/naca0012_a10_sgsoff.py \
        --gpu 2,3 --dist-init --max-steps 40000
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_soff_f_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100)

config["sgs"] = {"enabled": False}

_folder = "results_naca0012_a10_sgsoff"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
