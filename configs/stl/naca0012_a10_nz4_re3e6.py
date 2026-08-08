"""NACA0012 3D thin slab (Nz=4) a=10 — Re 3e6, STANDARD path, SGS off.

Re-ladder bisection rung: the Cd-sign boundary is bracketed in
(1e6, 6e6) — healthy +0.032 at 1e6 (tau_L0-0.5 = 2.6e-5), collapsed
-0.27 at 6e6 (4.3e-6). This rung (8.7e-6) splits the bracket.
Protocol held fixed: SGS off, IBB, c=100, Nz=4.

    python main.py --config configs/stl/naca0012_a10_nz4_re3e6.py \
        --gpu 2 --max-steps 20000
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca_lad_3e6", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100, nz_frac=0.04)

RE_TARGET = float("3e6")
config["physics"]["nu"] = _m.U_INF * _m.L_CHAR / RE_TARGET
config["sgs"] = {"enabled": False}

_folder = "results_naca0012_a10_nz4_re3e6"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
