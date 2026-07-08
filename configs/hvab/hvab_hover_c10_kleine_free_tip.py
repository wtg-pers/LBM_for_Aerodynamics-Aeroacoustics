"""HVAB c10 — kleine free-wake, TIP-ONLY shedding (Q2).

A/B 짝: hvab_hover_c10_kleine_free.py (wake 전스팬 shed) vs 이 파일
(**wake_markers="tip"** → 최외곽 활성 마커에서만 shed = 단일 팁-와류 필라멘트).
free-wake의 애초 목적(팁 rollup)에 맞춘 최소 모델. 나머지 셋업 동일(pure ALM, light, n_w=50).
관찰: 팁 φ/α 회복이 전스팬 shed 대비 어떤지 + fallback 빈도 + walltime(필라멘트 1점이라 ↓).
    python main.py --config configs/hvab/hvab_hover_c10_kleine_free_tip.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "n_w": 50, "wake_markers": "tip"},
    run_tag="kleine_free_tip",
)
