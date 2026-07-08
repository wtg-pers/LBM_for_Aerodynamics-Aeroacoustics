"""HVAB hover c10, M_tip=0.65 — marker distribution A/B: ENDPOINT + trapezoid.

순수 ALM(보정 OFF) + NASA 덱 + light + 25rev 로 Task 2 pureALM_nasa와 동일 셋업,
**marker_distribution=endpoint**: 마커를 root-cut·tip 끝점에 배치(끝점 포함) + 사다리꼴
적분(끝점 ½ 가중). 팁 마커가 r/R=1.0(시위 3.27in)에 정확히 놓임. uniform(셀중심,
끝점 없음) 대비 팁/루트 거동 차이 비교.
    python main.py --config configs/hvab/hvab_hover_c10_markerdist_endpoint.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="endpoint",
    n_rev=25, polar_source="nasa_overflow", run_tag="markerdist_endpoint",
)
