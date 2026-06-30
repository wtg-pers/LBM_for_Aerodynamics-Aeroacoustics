"""
Caradonna-Tung hover — collective 8 deg, M_tip = 0.877, with ε tip-taper (Stage B).

A/B partner of `ct_hover_t08_m088.py` (the baseline that produced C_T ≈ 0.00553).
Identical grid/preset/SGS/Prandtl — the ONLY change is the variable-ε tip taper
(epsilon_mode="tip_taper"), which narrows the projection Gaussian beyond r/R=0.7
toward the LBM floor (2·dx) at the tip to reduce tip-vortex over-smearing.

Output goes to a distinct `..._taper/` folder so it does NOT overwrite the baseline.

Compare against baseline:
  C_T vs exp 0.00473  ·  tip inflow-angle φ recovery  ·  ε(r/R) profile
  via:  python -m src.utilities.spanwise_post <result_dir> --exp <ref>   (plots eps_lu)

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_taper.py

Note: Prandtl tip-loss stays ON (matches the baseline deck). A narrower tip ε raises
R_tip_eff → slightly less Prandtl deduction; this is the intended, self-consistent
coupling (not double-counting). For a Prandtl-isolated study, add
config["actuator_line"]["prandtl_loss"] = False below.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877)

# --- Stage B: variable-ε tip taper (the only change vs baseline) ---
config["actuator_line"]["epsilon_mode"] = "tip_taper"
config["actuator_line"]["epsilon_tip_factor"] = 1.0   # tip ε target = max(1.0·2·dx, 2·dx)
config["actuator_line"]["epsilon_taper_start"] = 0.7   # r/R where taper begins

# --- Distinct output folder so the baseline result is preserved for A/B ---
_base = config["output"]["output_dir"].rsplit("/vtk", 1)[0] + "_taper"
config["output"]["output_dir"] = _base + "/vtk"
config["output"]["checkpoint_dir"] = _base + "/checkpoints"
config["output"]["csv_dir"] = _base + "/csv"
