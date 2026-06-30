"""
Caradonna-Tung hover — θ=8°, M_tip=0.877.
표준 Prandtl(R_tip_eff = R) + ε 팁 테이퍼.

목적: Prandtl을 ε에서 디커플(표준 R)한 상태에서 테이퍼를 얹으면, 지난번 light+taper의
R_tip_eff 커플링 artifact(외측 마커 un-zero로 추력↑) 없이 **테이퍼의 순수 −효과가
Prandtl 위에 건설적으로 stack**되는지 확인.

비교:
  - vs `ct_hover_t08_m088_prtipR` (표준 Prandtl, no taper) → 테이퍼의 깨끗한 ΔC_T(<0 기대)
  - 지난번 light+taper(0.00573, 비표준 R−ε라 +)와 대조

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_prtipR_taper.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877)

config["actuator_line"]["prandtl_loss"] = {"enabled": True, "eps_offset": False}
config["actuator_line"]["epsilon_mode"] = "tip_taper"
config["actuator_line"]["epsilon_tip_factor"] = 1.0
config["actuator_line"]["epsilon_taper_start"] = 0.7

_base = config["output"]["output_dir"].rsplit("/vtk", 1)[0] + "_prtipR_taper"
config["output"]["output_dir"] = _base + "/vtk"
config["output"]["checkpoint_dir"] = _base + "/checkpoints"
config["output"]["csv_dir"] = _base + "/csv"
