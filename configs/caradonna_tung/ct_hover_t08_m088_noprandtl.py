"""
Caradonna-Tung hover — collective 8 deg, M_tip = 0.877.
Prandtl 격리 실험: baseline ε (epsilon_mode=default) + **Prandtl tip-loss OFF**.

`ct_hover_t08_m088_taper_noprandtl.py`와 짝. 둘 다 Prandtl을 끈 상태로 비교하면,
ε-테이퍼의 순수 효과(유도속도 변화)를 Prandtl 커플링(R_tip_eff 변화로 인한 tip-loss
차감량 변화)과 분리할 수 있다. (기본 A/B는 Prandtl ON이라 두 효과가 섞임.)

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_noprandtl.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877)

config["actuator_line"]["epsilon_mode"] = "default"
config["actuator_line"]["prandtl_loss"] = False        # ← 격리

_base = config["output"]["output_dir"].rsplit("/vtk", 1)[0] + "_noprandtl"
config["output"]["output_dir"] = _base + "/vtk"
config["output"]["checkpoint_dir"] = _base + "/checkpoints"
config["output"]["csv_dir"] = _base + "/csv"
