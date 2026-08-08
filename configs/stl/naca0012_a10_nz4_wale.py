"""NACA0012 a=10, Nz=4, STANDARD path — wall-SGS A/B leg (d): WALE.

WALE's S_d operator vanishes for pure shear, so it is designed to be
silent at (smooth, resolved) walls without damping functions. The f1g
probe on the staircase wing confirms it is MOSTLY silent where dyn_smag
blanketed the wall (L3 shell nu_t/nu_mol median 0.1 vs 4.4, p95 23 vs
66) but keeps a local heavy tail (~960x max) at staircase corners and
the LE stagnation region (known WALE false-positive in irrotational
strain). This leg prices whether that residual tail matters; if it
does, sgs.wall_damp_cells composes with WALE (acts on nu_t_in after
the model kernel) as a one-line follow-up.

    python main.py --config configs/stl/naca0012_a10_nz4_wale.py \
        --gpu 5 --max-steps 30000
(STANDARD path — do not set LBM_ESOTERIC.)
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_wale_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100, nz_frac=0.04)          # Nz = 4

config["sgs"]["model"] = "wale"

_folder = "results_naca0012_a10_nz4_wale"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
