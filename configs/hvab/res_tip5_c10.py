"""HVAB c10 — 해상도 검증 B: **light_tip5 (팁 국소 +1 level)**. fine보다 저렴한 중간 단계.

light_tip5(D32, 3 fine + 팁 disk-plane 슬랩 +1): 팁 chord≈12.6, ε=0.25c(floor 탈출), ε/Δx≈3.15,
cells/R 256. 전체 fine을 못 돌릴 때 "팁만 국소 고해상"으로 Merabet 가설을 부분 검증.
ABC-A와 동일 설정(gaussian·pure ALM), 해상도만 다름.

주의: box refinement라 슬랩이 disk-plane 전체(팁 annulus 포함 inboard까지)를 덮어 비용이 큼.
★ ~82M cells, ~34 GB → **A100-40GB+ 필요** (24/32GB 아님; full fine보다는 ~2× 저렴).
축방향 슬랩을 더 얇게(down↓) 하면 더 저렴(팁와류 wake 커버리지와 trade-off).
    python main.py --config configs/hvab/res_tip5_c10.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light_tip5",
    eps_correction=None, prandtl_loss=False,        # pure ALM (ABC-A와 동일)
    sampling={"mode": "gaussian"},                  # gaussian (ABC-A와 동일)
    run_tag="res_tip5",
)
