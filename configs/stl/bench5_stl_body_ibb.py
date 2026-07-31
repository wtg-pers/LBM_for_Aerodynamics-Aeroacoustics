"""S7 — ALM + STL body coexistence (final stage of the STL body track).

bench5_baseline's EXACT rotor/grid/ALM build (kleine free wake, NASA
OVERFLOW decks, 5-level slab topology) plus an STL icosphere body on the
wake axis 1R below (downstream, +x) the rotor disk:

  body   D = 4 L0 lu (icosphere_r25_s4.stl x 0.08), center (26.5, 24, 24)
  level  resident on L2 (dx = 1/4 -> 16 cells across); bbox [24.5, 28.5]
         does NOT intersect L3 (x<=21) or the ALM L4 slab (x[15,18]) --
         ALM stays on the deepest level by construction (PLAN S7: any
         violation is a config error, never a code generalisation).
  pads   >= 2.5 lu to every fine-region face = 0.625*L_body (rule: 0.5).
  MPI    the n=2 rank cut (x=28) passes through the body -> solid cells,
         IBB links and the owned-clip force straddle the boundary.

Gates (patch_notes/stl_body/08):
  1. CT/CP rev-2 tail within bench5_baseline's noise band (thrust CV).
  2. body CD finite & stable. Anchor normalisation, NOT physics: U_ref =
     momentum-theory far-wake w = sqrt(2*CT)*V_tip ~= sqrt(0.0196)*0.1
     = 0.014 lu (CT ~= 0.0098). reference lengths are L0 lu -- setup
     rescales *2^k to the body's force level (L2: D_fine=16).
  3. 1 <-> 2-rank equivalence (esoteric path).

    python main.py --config configs/stl/bench5_stl_body_ibb.py
    LBM_ESOTERIC=1 mpirun -n 2 python main.py \
        --config configs/stl/bench5_stl_body_ibb.py --gpu 0,1
"""
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, os.path.join(_repo, "configs", "hvab"))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_stl_body_ibb",
)

config["internal_geometry"] = {
    "stl": {
        "enabled": True,
        "file": os.path.join(_repo, "input_files", "geom",
                             "icosphere_r25_s4.stl"),
        "scale_to_lu": 2.0 / 25.0,          # R=25 file units -> R=2 L0 lu
        "center_lu": (26.5, 24.0, 24.0),    # 1R below the disk, on axis
        "wall_bc": "ibb",
    },
}

U_REF = 0.014                               # far-wake w [lu] (see docstring)
config["force_calculation"] = {
    "enabled": True,
    # reference lengths in L0 lu (setup multiplies by 2^k for the MLG
    # force level): D=4 -> 16 L2 cells, A_ref = pi/4 D^2 via span.
    "reference": {"rho": 1.0, "velocity": U_REF,
                  "char_length": 4.0, "span_length": math.pi / 4.0 * 4.0},
}

_folder = "results_bench5_stl_body_ibb"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
