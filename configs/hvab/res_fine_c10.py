"""HVAB c10 — 해상도 검증 A: **fine preset (DGX)**. Merabet 가설 테스트.

fine(D40, 5 level): 팁 chord≈16, ε=0.25c(2Δx floor 탈출), ε/Δx≈3.9, cells/R 320.
ABC-A(`sampler_A_gaussian_c10.py` = light·gaussian·pure ALM)와 **동일 설정·다른 해상도**.
목적: "해상도만으로 팁 φ가 운동량 기대(~3°) 쪽으로 회복되는가?"
  → 회복되면 Merabet 레시피(ε=0.25c+integral 샘플+tip vortex 해상) 확정, 보정 불요/최소.
  → 안 되면 잔차에 Kleine 2022 비반복 보정.

★ ~170–183M cells, ~75 GB → **DGX(128GB) 필요.** (24/32GB GPU 불가.)
    python main.py --config configs/hvab/res_fine_c10.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="fine",
    eps_correction=None, prandtl_loss=False,        # pure ALM (ABC-A와 동일)
    sampling={"mode": "gaussian"},                  # gaussian (ABC-A와 동일)
    run_tag="res_fine",
)
