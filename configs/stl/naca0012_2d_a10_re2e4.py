"""NACA0012 2D (D2Q9) a=10 — Re 2e4 anchor config (fundamentals ladder).

The VALIDATED reduced-Re anchor of the 2D cross-check twin (measured
2026-07-31, ad-hoc nu override then): Cd = +0.086 +- 0.011, Cl = +/-0.789
exact mirror under AoA sign flip, converging windows. This config makes
that recipe reproducible: same 2D twin, physics.nu set for Re = 2e4
(tau_L0 ~= 0.5013 — inside D2Q9's robust band), SGS off.

Role in the ladder: leg 2D-anchor. Its 3D thin-slab twin at the SAME Re
(naca0012_a10_nz4_re2e4.py) isolates the slab/z variables: agreement
validates the whole 3D body machinery at a Re with a working reference,
pinning the alpha=10 Cd<0 problem to the high-Re acoustic-scaled regime.

    python main.py --config configs/stl/naca0012_2d_a10_re2e4.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_2d_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca2d_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m.config

RE_TARGET = 2.0e4
config["physics"]["nu"] = _m.U_INF * _m.L_CHAR / RE_TARGET
config["sgs"] = {"enabled": False}      # anchor protocol (measured leg)

_folder = "results_naca0012_2d_a10_re2e4"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
