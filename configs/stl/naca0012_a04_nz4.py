"""NACA0012 a=4, Nz=4, STANDARD path — attached-regime validation leg.

Purpose (wall-SGS A/B follow-up): separate SOLVER correctness from the
thin-slab high-alpha REGIME question. At a=10 the quasi-2D slab shows a
collapse into LEV/dynamic-stall-like states (cluster: steady Cd -0.27 /
Cl 0.51 at Nz=4; large-amplitude Cl cycle 0.34..1.06 at Nz=16) whose
onset tracks effective dissipation, not the force path (paths agree).
At a=4 the flow is firmly attached and quasi-2D is defensible: a sane
Cl (~0.40-0.48, thin-airfoil 2*pi*a with LE-resolution deficit) and a
small positive Cd validate the repaired machinery independently of the
high-alpha regime debate.

Rz(+6) cancels 6 deg of the baked +10 pitch -> alpha = 4 (same
convention as the a=0 twin's Rz(+10)).

    python main.py --config configs/stl/naca0012_a04_nz4.py \
        --gpu 4 --max-steps 30000
(STANDARD path, current dyn_smag — do not set LBM_ESOTERIC.)
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_a04_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100, nz_frac=0.04)          # Nz = 4

config["internal_geometry"]["stl"]["rotation_deg"] = (-90.0, 0.0, 6.0)

_folder = "results_naca0012_a04_nz4"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
