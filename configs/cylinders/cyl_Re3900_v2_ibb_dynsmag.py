"""3D Cylinder Re=3900, v2 domain — IBB wall + Dynamic Smagorinsky.

Comparison-set (full SGS sweep):
    (1) v2_ibb         : IBB + no-SGS    (wall BC alone)
    (2) v2_ibb_smag    : IBB + Smag Cs=0.17  (over-dissipative)
    (3) v2_ibb_wale    : IBB + WALE Cw=0.5   (wall-aware)
    (4) THIS           : IBB + Dyn-Smag       (self-tuning Cs)

Hypothesis: Dynamic Smag locally adapts Cs² to the flow → less
over-dissipation than fixed-Cs Smag, expected to track WALE behaviour.

Output folder: result_cyl_Re3900_v2_ibb_dynsmag/.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cyl_Re3900_ibb_mlg3_3d_v2 import config

# --- Wall BC : IBB ----------------------------------------------------
for _k, _v in config["internal_geometry"].items():
    if isinstance(_v, dict) and _v.get("enabled", False):
        _v["wall_bc"] = "ibb"

# --- SGS : Dynamic Smagorinsky -----------------------------------------
config["sgs"] = {"enabled": True, "model": "dyn_smag",
                  "Cs_max": 0.5, "alpha_sq": 3.0}

# --- Output folder ----------------------------------------------------
_folder = "result_cyl_Re3900_v2_ibb_dynsmag"
config["output"] = {
    "output_dir":     f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir":        f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": True, "precision": "float32",
            "variables": ["density", "pressure", "velocity",
                          "velocity_magnitude", "solid_mask", "nu_t"]},
    "checkpoint": {"enabled": True, "keep_last_n": 2},
}

print("  >> [comparison] IBB on, SGS=Dynamic Smagorinsky (Cs_max=0.5)")
