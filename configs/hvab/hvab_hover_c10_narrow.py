"""HVAB hover 10° — fine-region margin 진단 A/B (narrow, preset=light).

pure ALM (eps_correction OFF + prandtl_loss OFF) → 팁 φ/u_n 회복이 순수히 격자
효과인지 확인. narrow(light, 팁여유 0.09D) vs wide(light_wide, 0.28D) 비교.
가설: wide에서 팁 u_n이 0으로 안 무너지고 φ가 매끄럽게 양수 유지되면
→ CP/FM 붕괴의 주범 = fine-region 과소(격자) 확정.
    python main.py --config configs/hvab/hvab_hover_c10_narrow.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(collective_deg=10.0, mtip=0.65, preset="light",
                      eps_correction=None, prandtl_loss=False, run_tag="pureALM")
