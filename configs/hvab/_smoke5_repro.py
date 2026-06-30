"""5-level MLG 안정성 재현 — DGX fine NaN(step5, |u|=4.1) 로컬 CPU 재현용.
smoke(D16) + smoke_levels=5 → light(4-level, production OK)와 같은 finest res(D/256),
단 nesting 깊이만 5-level. 5-level 자체가 불안정이면 여기서도 NaN 떠야 함.
    python main.py --config configs/hvab/_smoke5_repro.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, smoke=True, smoke_levels=5,
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    run_tag="smoke5",
)
config["time"]["max_steps"] = 12          # step5 근처에서 터지는지
config["time"]["output_interval"] = 12
