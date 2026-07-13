"""HVAB hover c10 — UNIFORM grid (MLG OFF, 단일레벨), D=20. 격자수렴 스윕 1/5.

목적: "LBM 격자가 CT·팁하중에 정말 유의한가"를 MLG 아티팩트 없이 D만 스윕해 검토.
  uniform이라 D = 로터 지름 격자수 = 최종 해상도(로터 근처도 D). 셀=80·D³.
  D20 → 0.64M 셀(~0.3GB), STEPS_REV≈628. (가장 거친 케이스, 빠름)

물리/모델은 slab5 pure-ALM과 동일: collective 10°, M_tip 0.65, NASA OVERFLOW 덱,
  gaussian 샘플링, prandtl OFF, eps_correction 無(순수 ALM), uniform 마커.
격자만 uniform(MLG off)로 교체. 판정 = 이 스윕(D20→80) 내부의 CT·팁 M²cₙ 수렴 추세.

스윕: unif_d20 / d32 / d40 / d60 / d80 _c10.py (preset unif20..unif80).
⚠ 절대값은 도메인이 MLG(slab5/light)와 달라 직접비교 불가 — 추세로 판정.
   5-level 아님이라 nesting 크래시 무관하나, 첫 런 smoke(nan_trap) 권장.

    python main.py --config configs/0708_grid_convergence/unif_d20_c10.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="unif20",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="unif20",
)
