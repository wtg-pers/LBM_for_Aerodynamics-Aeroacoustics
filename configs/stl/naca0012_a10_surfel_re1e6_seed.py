"""NACA0012 a=10 Re 1e6 + solenoidal IC seed — seed-discrimination arm.

The registered experiment of patch_notes/surfel/57 sec. 4: identical to
naca0012_a10_surfel_re1e6 (full stack) except a reproducible
divergence-free velocity perturbation planted in the INITIAL CONDITION
over the suction-side BL region (runtime untouched). Three-way verdict
vs the unseeded 1e6 run (patch 57):
  (1) turbulence SUSTAINS and separation closes -> engineering a
      sustained trip is worth it (and the 0.04c span can hold it);
  (2) seed decays back to the patch-57 solution -> span widening first;
  (3) turbulence lives but separation stays -> the supply
      self-reference hole (sigma ~ tau_w) owns the残 gap.
Read: force history unsteadiness spectrum + the 4-item reader vs 57.

Run (cluster, single GPU):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    python main.py --config configs/stl/naca0012_a10_surfel_re1e6_seed.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "re1e6_base", os.path.join(_here, "naca0012_a10_surfel_re1e6.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.config

# suction-side BL box (L0 lu): wing x 300.7~399.2, surface y<=~315,
# delta(L3) ~ 2.6 lu; z = full span (periodic, no taper)
config["initial_perturbation"] = {
    "enabled": True,
    "sigma_u": 0.00433,            # 0.05 * U_lu
    "box_lu": [303.0, 345.0, 298.0, 320.0, 0.0, 4.0],
    "lambda_lu": [0.5, 2.0],       # 4~16 fine cells at L3
    "n_modes": 64,
    "seed": 7,
}

_folder = "results_naca0012_a10_surfel_re1e6_seed"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
