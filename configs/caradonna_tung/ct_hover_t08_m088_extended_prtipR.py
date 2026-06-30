"""
Caradonna-Tung hover — θ=8°, M_tip=0.877. EXTENDED fine region + 표준 Prandtl(R).

production 설정(표준 Prandtl)에서 wake extent 효과를 보는 후속 확인용.
※ 권장 실행 순서: 먼저 ct_hover_t08_m088_extended_noprandtl.py (Prandtl 마스킹 없는
  가장 깨끗한 extent 진단). 그게 extent 효과를 확인한 뒤에 이 stdR 버전으로
  "쓸 모델에서도 extent가 돕는가"를 확인. extent가 안 도우면 이 런은 불필요.

비교 기준: result_ct_t08.0_M877_mlg4_D32_light_prtipR (표준 Prandtl, light, 0.00639).
주의(경미): light_prtipR는 SGS ON, 이 런은 SGS OFF(다른 extended 런들과 일관). 격자수렴
연구상 SGS ±0.2%로 extent 효과 대비 무시 가능.

~80M cells, ~33 GB → DGX Spark.

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_extended_prtipR.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877, preset="extended", use_sgs=False)
config["actuator_line"]["prandtl_loss"] = {"enabled": True, "eps_offset": False}  # 표준 R

_base = config["output"]["output_dir"].rsplit("/vtk", 1)[0] + "_prtipR"
config["output"]["output_dir"] = _base + "/vtk"
config["output"]["checkpoint_dir"] = _base + "/checkpoints"
config["output"]["csv_dir"] = _base + "/csv"
