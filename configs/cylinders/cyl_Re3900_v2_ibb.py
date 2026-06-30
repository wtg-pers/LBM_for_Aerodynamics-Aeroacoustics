"""
3D Cylinder Re=3900, v2 domain — IBB wall, SGS OFF.

Comparison-set member 1 of 3:
    (1) THIS:           IBB + no-SGS    -> wall BC contribution alone
    (2) v2_ibb_smag:    IBB + Smag      -> combined effect
    (3) v2_hwbb_smag:   HWBB + Smag     -> SGS contribution alone

Baseline (v2): HWBB + no-SGS, Cd_mean = 1.2222 +- 0.0485 (200k step run).
Note: previous v2 effectively ran HWBB because IBB fell back to HWBB in
3D before #24 fix. With the IBB-D3Q27 kernel landed, this config now
produces actually-IBB results.

Reuses the v2 grid/MLG/forcing layout from cyl_Re3900_ibb_mlg3_3d_v2.py.
Output folder: result_cyl_Re3900_v2_ibb/.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cyl_Re3900_ibb_mlg3_3d_v2 import config

# --- Wall BC : IBB ----------------------------------------------------
# Already 'ibb' in v2 base; with the D3Q27 IBB kernel in place this is
# now actually IBB (was silently HWBB before).
for _k, _v in config["internal_geometry"].items():
    if isinstance(_v, dict) and _v.get("enabled", False):
        _v["wall_bc"] = "ibb"

# --- SGS : OFF --------------------------------------------------------
config["sgs"] = {"enabled": False, "model": "off"}

# --- Output folder ----------------------------------------------------
_folder = "result_cyl_Re3900_v2_ibb"
config["output"] = {
    "output_dir":     f"./{_folder}/vtk",
    "checkpoint_dir": f"./{_folder}/checkpoints",
    "csv_dir":        f"./{_folder}/csv",
    "clear_previous": True,
    "vtk": {"enabled": True, "precision": "float32",
            "variables": ["density", "pressure", "velocity",
                          "velocity_magnitude", "solid_mask"]},
    "checkpoint": {"enabled": True, "keep_last_n": 2},
}

print("  >> [comparison] IBB on, SGS off")
