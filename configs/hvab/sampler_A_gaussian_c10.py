"""HVAB hover 10° — 속도 샘플러 A/B/C 실험: **A = gaussian (baseline)**.

P1 결정 실험 (patch_notes/almlbm_sampler_ab/, docs/almlbm_paper_analysis_kr.md §2①).
팁 유입 결손(φ→0)의 "샘플링 기여(b)"를 격리하기 위해 A·B-iii·B-i 3런을 동일 조건에서
돌리고 팁 φ·α·u_n 스팬분포·C_T·FM을 비교한다.

이 런 = **현행 ±3ε Gaussian 샘플러**(우리만 쓰는 적분 샘플; off-disk 진단 29.8%).
세 런 모두 **pure ALM**: prandtl_loss=False + eps_correction=None →
팁 φ를 가리는/보정하는 요소를 모두 끄고 *샘플러만* 비교.
preset=light(DEFAULT, 24GB) — off-disk 진단·기존 c10 spanwise와 동일 격자.

대조: A−B-iii = off-disk(b1) 기여,  B-iii−B-i = 반경 smoothing(b2) 기여.

    python main.py --config configs/hvab/sampler_A_gaussian_c10.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False,        # pure ALM (샘플러만 격리)
    sampling={"mode": "gaussian"},                  # A — 현행 baseline
    run_tag="sampA_gauss",
)
