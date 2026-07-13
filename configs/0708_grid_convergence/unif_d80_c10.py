"""HVAB hover c10 — UNIFORM grid (MLG OFF, 단일레벨), D=80. 격자수렴 스윕 5/5 (최정밀).

셀=80·80³=40.96M(~16.8GB, 24GB OK), STEPS_REV≈2513. 15rev≈37.7k step — 가장 무거움.
⚠ 40.96M 셀은 (옛) int32 커널 천장 79.5M 아래이나 fine 근처 — 64-bit 인덱싱 적용본
   기준 정상([[project_int32_kernel_ceiling]]). 물리/모델 slab5 pure-ALM과 동일
   (collective 10°, NASA덱, gaussian, prandtl OFF, 순수 ALM, uniform 마커).
상세: unif_d20_c10.py 참조.

    python main.py --config configs/0708_grid_convergence/unif_d80_c10.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="unif80",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="unif80",
)
