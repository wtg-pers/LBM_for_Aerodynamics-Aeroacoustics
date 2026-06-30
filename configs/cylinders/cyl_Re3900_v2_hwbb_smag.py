"""
3D Cylinder Re=3900, v2 domain — HWBB wall + Smagorinsky SGS.

Comparison-set member 3 of 3:
    (1) v2_ibb:         IBB + no-SGS    -> wall BC contribution alone
    (2) v2_ibb_smag:    IBB + Smag      -> combined effect
    (3) THIS:           HWBB + Smag     -> SGS contribution alone

Cross-check vs v2 baseline (HWBB + no-SGS, Cd_mean = 1.2222) isolates
the SGS effect. With (1) we then isolate the wall-BC effect; (2)
gives the combined target.

Output folder: result_cyl_Re3900_v2_hwbb_smag/.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cyl_Re3900_ibb_mlg3_3d_v2 import config

# --- Wall BC : HWBB --------------------------------------------------
for _k, _v in config["internal_geometry"].items():
    if isinstance(_v, dict) and _v.get("enabled", False):
        _v["wall_bc"] = "hwbb"

# --- SGS : Smagorinsky Cs=0.17 ---------------------------------------
config["sgs"] = {"enabled": True, "model": "smagorinsky", "Cs": 0.17}

# --- Output folder ---------------------------------------------------
_folder = "result_cyl_Re3900_v2_hwbb_smag"
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

print("  >> [comparison] HWBB on, SGS=Smagorinsky Cs=0.17")
