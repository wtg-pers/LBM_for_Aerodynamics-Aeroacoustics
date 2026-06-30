"""HVAB hover 10° — 속도 샘플러 A/B/C 실험: **B-i = point/trilinear**.

공간 필터를 완전히 끄고 8노드 trilinear로 "점의 속도"만 읽음 (참고논문 5편과 동일).
off-disk(b1) + 반경 smoothing(b2) + 코드·축 필터까지 모두 제거.
B-iii(mask_disk)와의 차이 = **b2(반경 smoothing) 기여**. smoke 합성장 팁 100% 회복.

조건은 A와 동일(pure ALM, preset=light, c10) — 샘플러 mode만 다름.

    python main.py --config configs/hvab/sampler_B1_point_c10.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False,        # pure ALM (샘플러만 격리)
    sampling={"mode": "point"},                     # B-i — 필터 없음(trilinear)
    run_tag="sampB1_point",
)
