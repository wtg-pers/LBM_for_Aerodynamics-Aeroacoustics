"""격리 테스트 1 — medium(D40, **4-level**) pure-ALM. fine NaN 원인 좁히기.

fine(D40,5-level)이 step5 NaN으로 죽음. medium은 같은 D40·같은 cumulant인데 **4-level**.
  → medium이 돌면: 원인 = **5번째 레벨**(deep MLG coupling 또는 level-4 fine-ALM 스케일).
  → medium도 죽으면: 원인 = D40 L0 / cumulant 일반(레벨수 무관).
(~60M, ~25GB → 32GB+ GPU. DGX에서 빠른 테스트용. 몇 스텝만 봐도 판정 가능.)
    python main.py --config configs/hvab/res_medium_c10.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="medium",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    run_tag="iso_medium",
)
config["time"]["max_steps"] = 60          # 격리: 50+ step만 안정하면 OK 판정
