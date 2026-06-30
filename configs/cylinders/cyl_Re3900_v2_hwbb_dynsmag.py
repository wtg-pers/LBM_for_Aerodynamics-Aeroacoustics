"""3D Cylinder Re=3900, v2 domain — HWBB wall + Dynamic Smagorinsky.

Comparison-set: HWBB-side SGS sweep, Dynamic completing the trio.

Output folder: result_cyl_Re3900_v2_hwbb_dynsmag/.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cyl_Re3900_ibb_mlg3_3d_v2 import config

# --- Wall BC : HWBB --------------------------------------------------
for _k, _v in config["internal_geometry"].items():
    if isinstance(_v, dict) and _v.get("enabled", False):
        _v["wall_bc"] = "hwbb"

# --- SGS : Dynamic Smagorinsky ---------------------------------------
config["sgs"] = {"enabled": True, "model": "dyn_smag",
                  "Cs_max": 0.5, "alpha_sq": 3.0}

# --- Output folder ---------------------------------------------------
_folder = "result_cyl_Re3900_v2_hwbb_dynsmag"
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

print("  >> [comparison] HWBB on, SGS=Dynamic Smagorinsky (Cs_max=0.5)")
