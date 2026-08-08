"""NACA0012 3D thin slab (Nz=4) a=10 — Re 3e6 + constant-Cs Smagorinsky.

Fix-candidate rung for the tau->0.5 wall instability (runbook section 4;
mechanism: below nu_lu ~ 2e-5 the staircase wall grows weakly-damped
cell-scale oscillations whose rectified momentum pumping shows as
spurious thrust in the — exactly correct — MEM force, while the smooth
hydrodynamic fields keep normal drag: wake deficit +0.10 vs MEM -0.39
at 3e6; the F1 accounting identity holds at tau=0.500009, so this is
real discrete wall dynamics, not an instrument error).

Constant-Cs is the one model measured to survive this regime in 2D
(dyn_smag/WALE/no-SGS all NaN at the LE; the constant floor damps the
LE strain spike — config comment of the 2D twin, 2026-07-31): a
dissipation floor that never switches off, unlike dyn_smag's dynamic
constant. Judgment: MEM Cd back to agreement with the wake integral
(+0.03..+0.10 class) => wall floor confirmed as the remedy; then try
Re 6e6 with the same model.

    python main.py --config configs/stl/naca0012_a10_nz4_re3e6_smag.py \
        --gpu 2 --max-steps 20000
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca_3e6smag", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100, nz_frac=0.04)

RE_TARGET = 3.0e6
config["physics"]["nu"] = _m.U_INF * _m.L_CHAR / RE_TARGET
config["sgs"] = {"enabled": True, "model": "smagorinsky", "Cs": 0.17}

_folder = "results_naca0012_a10_nz4_re3e6_smag"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
