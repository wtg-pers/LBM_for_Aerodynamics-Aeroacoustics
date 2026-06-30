"""DEBUG: int32 인덱스 천장(>79.5M셀/레벨) before/after 검증.
단일레벨 D76 = 87.78M 셀(>79.5M) → 수정 전 CUDA_ILLEGAL_ADDRESS, 후 통과.
SGS off로 24GB 적합. production 아님.
    CUDA_LAUNCH_BLOCKING=1 python main.py --config configs/hvab/_ovf87_repro.py --max-steps 2 --no-vtk --clear
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="ovf87", use_sgs=False,
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    polar_source="nasa_overflow", n_rev=1, run_tag="ovf87_repro",
)
