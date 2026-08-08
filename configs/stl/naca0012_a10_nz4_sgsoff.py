"""NACA0012 a=10, Nz=4, STANDARD path — wall-SGS A/B leg (c): SGS off.

Identical to naca0012_a10_std_nz4.py except sgs disabled: pure ILES
(cumulant limiter provides the subgrid dissipation) — the same
philosophy as the confirmed HVAB production baseline (mlg4 + K17L +
SGS off). Removes the wall nu_t pathology by removing the model
everywhere; the A/B against leg (b) prices what the wake/shear-layer
SGS was actually contributing.

    python main.py --config configs/stl/naca0012_a10_nz4_sgsoff.py \
        --gpu 3 --max-steps 30000
(STANDARD path — do not set LBM_ESOTERIC.)
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_soff_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100, nz_frac=0.04)          # Nz = 4

config["sgs"] = {"enabled": False}

_folder = "results_naca0012_a10_nz4_sgsoff"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
