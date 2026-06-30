"""HVAB Task 3 (Hover FM) production — collective 10.0 deg, M_tip=0.65.

AIAA HPW Task 3 제출용 baseline = **legacy eps (no taper) + Prandtl tip-loss + eps_corr OFF**.
진단 결과(보고서 HVAB_ALM_results_report_kr.md) ALM 고유 팁 유입 결손 → Prandtl이 미포착
팁 유입의 surrogate로 최선. legacy eps = Prandtl R_tip_eff=R-ε_tip (eps_offset=True).
preset=light(D32, ~12.5GB, RTX4090). 결과 폴더 run_tag="task3"로 분리(덮어쓰기 방지).
    python main.py --config configs/hvab/task3_c10.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65,
    prandtl_loss={"enabled": True, "tip": True, "root": True, "eps_offset": True},  # legacy R-ε
    eps_correction=None,        # OFF (Dağ 불충분 확인)
    tip_sweep=True,          # 팁 30° sweep simple-sweep 보정 (실제 geometry)
    run_tag="task3",
)
