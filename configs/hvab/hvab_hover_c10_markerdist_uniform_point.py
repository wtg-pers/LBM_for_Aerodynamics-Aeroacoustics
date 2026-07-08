"""Case 1 — marker dist comparison. uniform + POINT sampler (pure ALM).

baseline(uniform+gauss)=hvab_hover_c10_pureALM_nasa.py (결과 보유). 이 파일은
같은 셋업서 **sampler만 point(trilinear, 인접 8노드)** 로 교체. 5편 LBM-ALM 논문이
쓰는 표준 샘플러. uniform 분포 유지. spanwise 분포(φ·α·u_n·CL·CD·F_n vs r/R) 비교용.
    python main.py --config configs/hvab/hvab_hover_c10_markerdist_uniform_point.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "point"},
    marker_distribution="uniform",
    n_rev=25, polar_source="nasa_overflow", run_tag="md_uniform_point",
)
