"""3D Cylinder Re=3900, v2 domain — HWBB wall + WALE SGS.

Comparison-set: HWBB-side SGS sweep
    (1) v2_hwbb        : HWBB + no-SGS  (already run; baseline 1.22)
    (2) v2_hwbb_smag   : HWBB + Smag    (already run)
    (3) THIS           : HWBB + WALE
    (4) v2_hwbb_dynsmag: HWBB + Dyn-Smag

Output folder: result_cyl_Re3900_v2_hwbb_wale/.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cyl_Re3900_ibb_mlg3_3d_v2 import config

# --- Wall BC : HWBB --------------------------------------------------
for _k, _v in config["internal_geometry"].items():
    if isinstance(_v, dict) and _v.get("enabled", False):
        _v["wall_bc"] = "hwbb"

# --- SGS : WALE Cw=0.5 ------------------------------------------------
config["sgs"] = {"enabled": True, "model": "wale", "Cw": 0.5}

# --- Output folder ---------------------------------------------------
_folder = "result_cyl_Re3900_v2_hwbb_wale"
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

print("  >> [comparison] HWBB on, SGS=WALE Cw=0.5")
