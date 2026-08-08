"""NACA0012 3D thin slab (Nz=4) a=10 — Re 5e4, STANDARD path, SGS off.

Fundamentals-ladder rung (see naca0012_a10_nz4_re2e4.py): identical to
the validated Re-2e4 twin except physics.nu. Maps where the alpha=10
solution family degenerates (Cd sign / Cl collapse) on the way to the
acoustic-scaled Re 6e6 regime. Protocol held fixed: SGS off, IBB,
c=100, Nz=4.

    python main.py --config configs/stl/naca0012_a10_nz4_re5e4.py \
        --gpu 2 --max-steps 20000
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca_lad_5e4", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100, nz_frac=0.04)

RE_TARGET = float("5e4")
config["physics"]["nu"] = _m.U_INF * _m.L_CHAR / RE_TARGET
config["sgs"] = {"enabled": False}

_folder = "results_naca0012_a10_nz4_re5e4"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
