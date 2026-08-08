"""NACA0012 a=10, Nz=4, STANDARD path — wall-SGS A/B leg (b): wall-damped.

Identical to naca0012_a10_std_nz4.py except sgs.wall_damp_cells = 2:
nu_t is forced to 0 within 2 cells of the body before it enters
collision. Rationale (patch 12 follow-up, f1g probe): the dyn_smag
gradient/test-filter stencils read the staircase no-slip jump at
wall-adjacent cells as resolved strain, driving nu_t to O(100-900) x
nu_mol at tau -> 0.5 — a hyper-viscous wall layer that killed the LE
suction at a=10 (path-consistent Cl ~ 0.51 / Cd ~ -0.27). Physically,
near-wall eddy viscosity must vanish at the wall anyway (van Driest /
WALE y^3 scaling); the damp radius 2 = the test-filter reach. SGS stays
active outside (wake, shear layers) — that is the difference vs the
sgs-off leg (c).

    python main.py --config configs/stl/naca0012_a10_nz4_walldamp2.py \
        --gpu 2 --max-steps 30000
(STANDARD path — do not set LBM_ESOTERIC: direct twin of the leg-(a)
data already collected as 1c.)
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_wd2_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
config = _m._build(100, nz_frac=0.04)          # Nz = 4

config["sgs"]["wall_damp_cells"] = 2

_folder = "results_naca0012_a10_nz4_walldamp2"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
