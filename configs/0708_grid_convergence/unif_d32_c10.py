"""HVAB hover c10 — UNIFORM grid (MLG OFF, 단일레벨), D=32. 격자수렴 스윕 2/5.

D=32는 우리 MLG 케이스들의 L0(coarse) 지름과 동일 — 단, 여기선 uniform이라 로터가
실제로 32셀만 span(MLG는 최종레벨서 256~512). 셀=80·32³=2.62M(~1.1GB), STEPS_REV≈1005.
그 외는 slab5 pure-ALM과 동일(collective 10°, NASA덱, gaussian, prandtl OFF, 순수 ALM,
uniform 마커). 상세: unif_d20_c10.py 참조.

    python main.py --config configs/0708_grid_convergence/unif_d32_c10.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="unif32",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="unif32",
)
