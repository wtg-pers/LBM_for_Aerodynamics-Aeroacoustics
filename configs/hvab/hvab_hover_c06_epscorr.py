"""HVAB hover, collective 6.0 deg, M_tip=0.65 — viscous-core smearing 보정 ON.

A/B 쌍: hvab_hover_c06.py (보정 OFF + Prandtl ON, 기존 baseline) vs 이 파일.
보정 = non-iterative viscous-core de-induction (Dağ&Sørensen / JFM-2019).
target="inviscid" + prandtl_loss=False (물리 보정이 팁 de-loading을 직접 해상,
경험적 Prandtl과 중복 방지). 목적: CT/CP 팁 과예측 완화 → FM peak 회복 여부 확인.
    python main.py --config configs/hvab/hvab_hover_c06_epscorr.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=6.0, mtip=0.65,
    eps_correction={"enabled": True, "target": "inviscid"},
    prandtl_loss=False, run_tag="epscorr",
)
