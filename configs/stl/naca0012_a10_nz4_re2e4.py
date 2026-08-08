"""NACA0012 3D thin slab (Nz=4) a=10 — Re 2e4, STANDARD path.

Fundamentals-ladder twin of the 2D Re-2e4 anchor
(naca0012_2d_a10_re2e4.py: Cd +0.086 +- 0.011, Cl +/-0.789 mirror):
same chord resolution, same Re (tau_L0 ~= 0.5013), same SGS-off model,
same IBB wall — the only added machinery is the 3D slab (STL mask,
span-through z-periodic, D3Q27, 3D MLG).

Judgment: tail Cd/Cl within the 2D anchor band => the ENTIRE 3D body
machinery is validated at a Re with a working reference, and the
alpha=10 Cd<0 problem at Re 6e6 is isolated to the high-Re
acoustic-scaled regime (tau -> 0.5, unresolved BL), not to the slab /
z-periodicity / mask / force chain. Disagreement convicts the slab
machinery — then bisect (Nz, span BC) at this comfortable Re.

    python main.py --config configs/stl/naca0012_a10_nz4_re2e4.py \
        --gpu 2 --max-steps 15000
(STANDARD path — do not set LBM_ESOTERIC.)
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca_nz4_re2e4_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100, nz_frac=0.04)          # Nz = 4

RE_TARGET = 2.0e4
config["physics"]["nu"] = _m.U_INF * _m.L_CHAR / RE_TARGET
config["sgs"] = {"enabled": False}      # match the 2D anchor protocol

_folder = "results_naca0012_a10_nz4_re2e4"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
