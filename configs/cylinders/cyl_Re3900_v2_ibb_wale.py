"""3D Cylinder Re=3900, v2 domain — IBB wall + WALE SGS (Cw=0.5).

Comparison-set extension to study wall-bounded SGS:
    (1) v2_ibb         : IBB + no-SGS    (wall BC alone)
    (2) v2_ibb_smag    : IBB + Smag Cs=0.17 (over-dissipative)
    (3) v2_hwbb_smag   : HWBB + Smag        (SGS alone)
    (4) THIS           : IBB + WALE Cw=0.5  (wall-aware SGS)

Hypothesis: WALE auto-damps near the wall (S^d-tensor → 0 at the wall),
giving less over-dissipation than fixed-Cs Smagorinsky. Expected result vs
v2_ibb_smag: shorter L_r/D (closer to Parnaudeau PIV 1.51), Cl_rms a bit
larger, Cd similar.

Output folder: result_cyl_Re3900_v2_ibb_wale/.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cyl_Re3900_ibb_mlg3_3d_v2 import config

# --- Wall BC : IBB ----------------------------------------------------
for _k, _v in config["internal_geometry"].items():
    if isinstance(_v, dict) and _v.get("enabled", False):
        _v["wall_bc"] = "ibb"

# --- SGS : WALE Cw=0.5 (Nicoud & Ducros 1999 standard) ----------------
config["sgs"] = {"enabled": True, "model": "wale", "Cw": 0.5}

# --- Output folder ----------------------------------------------------
_folder = "result_cyl_Re3900_v2_ibb_wale"
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

print("  >> [comparison] IBB on, SGS=WALE Cw=0.5")
