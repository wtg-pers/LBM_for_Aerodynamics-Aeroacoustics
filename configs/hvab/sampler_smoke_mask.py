"""HVAB sampler smoke — mask_disk 경로가 실제 MLG+fine-ALM 파이프라인에서 도는지 확인.
smoke=True (D=16, 2-level, CPU). 검증용; 물리 의미 없음.
    python main.py --config configs/hvab/sampler_smoke_mask.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, smoke=True,
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "mask_disk"}, run_tag="smoke_mask",
)
config["time"]["max_steps"] = 6          # smoke: 6 스텝만 (샘플러 dispatch 확인)
config["time"]["output_interval"] = 6
