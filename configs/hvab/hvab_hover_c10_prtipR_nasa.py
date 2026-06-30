"""HVAB hover c10 — 표준 Prandtl(R_tip_eff=R), **NASA OVERFLOW C81 덱**, 25-rev.

기존 prtipR(`hvab_hover_c10_prtipR.py`, NeuralFoil, 폴더 ..._light_prtipR)와 동일 물리,
덱만 NASA OVERFLOW로 교체. 별도 폴더(..._light_prtipR_nasa). 표준 Prandtl은 CV가 가장
낮아(0.3%) 덱 효과를 가장 깨끗하게 비교할 수 있는 기준점.
    python main.py --config configs/hvab/hvab_hover_c10_prtipR_nasa.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    prandtl_loss={"enabled": True, "eps_offset": False},   # 표준 R_tip_eff = R
    sampling={"mode": "gaussian"},
    polar_source="nasa_overflow", n_rev=25, run_tag="prtipR_nasa",
)
