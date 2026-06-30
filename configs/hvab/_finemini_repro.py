"""DEBUG repro: fine preset(5-level) 크래시를 D16 축소판으로 로컬(24GB) 재현.
fine과 동일 nesting 비율 + ALM-on-finest + NASA 덱. production 아님.
    CUDA_LAUNCH_BLOCKING=1 python main.py --config configs/hvab/_finemini_repro.py --max-steps 3 --no-vtk --clear
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="fine_mini",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    polar_source="nasa_overflow", n_rev=1, run_tag="finemini_repro",
)
