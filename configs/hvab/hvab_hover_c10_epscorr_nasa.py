"""HVAB hover c10 — Dağ viscous-core 보정, **NASA OVERFLOW C81 덱**, 25-rev.

기존 epscorr(`hvab_hover_c10_epscorr.py`, NeuralFoil, 폴더 ..._light_epscorr)와 동일 물리
(prandtl OFF + Dağ inviscid), 덱만 NASA OVERFLOW로 교체. 별도 폴더(..._light_epscorr_nasa).
물리 기반 팁 보정 위에서 덱 교체가 팁 하중/CT에 주는 영향 비교용.
    python main.py --config configs/hvab/hvab_hover_c10_epscorr_nasa.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "target": "inviscid"},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    polar_source="nasa_overflow", n_rev=25, run_tag="epscorr_nasa",
)
