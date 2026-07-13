"""HVAB hover c10 — UNIFORM grid (MLG OFF, 단일레벨), D=60. 격자수렴 스윕 4/5.

셀=80·60³=17.3M(~7.1GB, 24GB OK), STEPS_REV≈1885. 물리/모델 slab5 pure-ALM과 동일
(collective 10°, NASA덱, gaussian, prandtl OFF, 순수 ALM, uniform 마커).
상세: unif_d20_c10.py 참조.

    python main.py --config configs/0708_grid_convergence/unif_d60_c10.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="unif60",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="unif60",
)
