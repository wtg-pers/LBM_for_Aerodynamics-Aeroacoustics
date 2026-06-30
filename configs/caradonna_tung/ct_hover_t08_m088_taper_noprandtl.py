"""
Caradonna-Tung hover — collective 8 deg, M_tip = 0.877.
Prandtl 격리 실험: ε 팁 테이퍼(epsilon_mode=tip_taper) + **Prandtl tip-loss OFF**.

`ct_hover_t08_m088_noprandtl.py`(baseline, Prandtl OFF)와 짝.
두 런의 ΔC_T = ε-테이퍼의 **순수** 효과 (Prandtl 커플링 제거됨).

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_taper_noprandtl.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877)

config["actuator_line"]["epsilon_mode"] = "tip_taper"
config["actuator_line"]["epsilon_tip_factor"] = 1.0
config["actuator_line"]["epsilon_taper_start"] = 0.7
config["actuator_line"]["prandtl_loss"] = False        # ← 격리

_base = config["output"]["output_dir"].rsplit("/vtk", 1)[0] + "_taper_noprandtl"
config["output"]["output_dir"] = _base + "/vtk"
config["output"]["checkpoint_dir"] = _base + "/checkpoints"
config["output"]["csv_dir"] = _base + "/csv"
