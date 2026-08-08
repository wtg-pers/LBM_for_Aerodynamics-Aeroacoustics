"""NACA0012 3D thin slab (Nz=4) a=10 — Re 6e6 + constant-Cs Smagorinsky.

Ladder closure run: the constant-Cs dissipation floor cured the
tau->0.5 wall-oscillation momentum pumping at Re 3e6 (three-instrument
agreement restored: MEM +0.049 / wake +0.07 / pressure +0.063 — see
runbook section 4f). This run states the original benchmark Re with the
same brake. Expectation: positive, stable Cd (sign/stability claim);
quantitative Cl/Cd remain off Ladson (unresolved BL + global brake
penalty) — that is the wall-model track's territory, documented.

    python main.py --config configs/stl/naca0012_a10_nz4_re6e6_smag.py \
        --gpu 2 --max-steps 20000
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca_6e6smag", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100, nz_frac=0.04)          # Re 6e6 = base physics

config["sgs"] = {"enabled": True, "model": "smagorinsky", "Cs": 0.17}

_folder = "results_naca0012_a10_nz4_re6e6_smag"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
