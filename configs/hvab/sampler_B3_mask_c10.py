"""HVAB hover 10° — 속도 샘플러 A/B/C 실험: **B-iii = mask_disk**.

현행 등방 ±3ε Gaussian에서 **디스크 밖(cyl radius > R) 노드를 가중에서 제외**하고
재정규화 → off-disk 혼입(b1)만 제거, 반경 smoothing(b2)은 유지.
A(gaussian)와의 차이 = **b1(off-disk) 기여**. smoke 합성장에서 팁 다운워시 100% 회복 확인.

조건은 A와 동일(pure ALM, preset=light, c10) — 샘플러 mode만 다름.

    python main.py --config configs/hvab/sampler_B3_mask_c10.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False,        # pure ALM (샘플러만 격리)
    sampling={"mode": "mask_disk"},                 # B-iii — off-disk 제거
    run_tag="sampB3_mask",
)
