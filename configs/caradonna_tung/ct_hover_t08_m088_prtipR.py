"""
Caradonna-Tung hover — θ=8°, M_tip=0.877.
Prandtl 표준화 테스트: tip-loss를 **표준 R_tip_eff = R_tip** 로 (ε 차감 제거).

기존 light(0.00553)는 비표준 R_tip_eff = R − ε_tip 을 써서 외측 4.2% 스팬을 hard-zero
하던 것 → 여기서는 교과서/BEMT 일치 형태로 부드러운 roll-off. taper와도 디커플됨.
no-taper(default ε)라 ε 효과는 없고, **순수하게 Prandtl 표준화의 baseline 영향**만 본다.

비교: result_ct_t08.0_M877_mlg4_D32_light (0.00553, 비표준 R−ε) vs 이 런(표준 R).
baseline C_T가 올라갈 것으로 예상(덜 공격적 팁제거) — 더 물리적으로 방어 가능한 기준.

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_prtipR.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877)

# 표준 Prandtl: R_tip_eff = R (ε 차감 제거)
config["actuator_line"]["prandtl_loss"] = {"enabled": True, "eps_offset": False}

_base = config["output"]["output_dir"].rsplit("/vtk", 1)[0] + "_prtipR"
config["output"]["output_dir"] = _base + "/vtk"
config["output"]["checkpoint_dir"] = _base + "/checkpoints"
config["output"]["csv_dir"] = _base + "/csv"
