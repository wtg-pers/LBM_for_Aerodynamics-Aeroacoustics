"""span16 + sustained trip strip (patch 66) — forced-transition arm.

Identical to naca0012_a10_surfel_re1e6_span16_seed (5.76 delta span,
solenoidal IC seed) plus the SUSTAINED numerical trip: a divergence-
free body-force fluctuation band on the suction side at x/c 0.05-0.15,
applied every substep (src/utilities/trip_forcing.py). Verdict
discriminators are pre-registered in patch_notes/surfel/66 sec. 4:
F0 P0@x/c=0.30 >= 1%, F1 separation onset 0.53 -> >= 0.70, F2 Cd down
from 0.0469, F3 Cl >= 0.958 held.

Strip geometry measured on the span16 L3 snapshot (66 sec. 2):
suction wall y over the strip = 309.62..310.00 L0 lu; the box brackets
it [wall-0.25, wall_max+0.50] — body spill lands on dead cells (zero
advect sources), the live band is ~4-7 L3 cells above the local wall.

Amplitude: A = 1e-3 [L0 lu accel] default; the sweep clones only need
TRIP_AMP changed (66 sec. 4 registers a weak/strong pair).

Run (cluster, ONE node with 2x24 GiB) — --mca pml ucx REQUIRED
(64 sec. 19c):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    LBM_ESOTERIC=1 mpirun --mca pml ucx -n 2 python main.py --mpi \\
        --config configs/stl/naca0012_a10_surfel_re1e6_span16_trip.py \\
        --axis z --ghost 4 --cuda-aware 1 --gpu 0,1
"""
import importlib.util
import os

TRIP_AMP = 1e-3            # [L0 lu] acceleration amplitude (sweep knob)

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "span16_seed_base",
    os.path.join(_here, "naca0012_a10_surfel_re1e6_span16_seed.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = dict(_m.config)
_folder = "results_naca0012_a10_surfel_re1e6_span16_trip"
config["output"] = dict(config["output"],
                        output_dir=f"./{_folder}/vtk",
                        checkpoint_dir=f"./{_folder}/checkpoints",
                        csv_dir=f"./{_folder}/csv")
config["trip_forcing"] = {
    "enabled": True,
    "amp_lu": TRIP_AMP,
    "box_lu": [305.75, 315.50, 309.38, 310.50],   # global L0 lu, x/y
    "taper_lu": 0.5,          # thin y-window (1.12 lu) keeps a plateau
    "lambda_lu": [1.5, 4.0],
    "n_modes": 16,
    "seed": 20260822,
    "u_ref_lu": 0.0866025,
    "omega_scale": 1.0,
}
