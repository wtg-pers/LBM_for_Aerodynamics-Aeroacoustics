"""Case 1 — marker dist comparison. endpoint + POINT sampler (pure ALM).

endpoint(끝점포함+사다리꼴) 분포 + point(trilinear) 샘플러. 짝:
hvab_hover_c10_markerdist_endpoint.py (endpoint+gauss). 팁 마커가 r/R=1.0(시위 3.27in)에
정확히 놓인 상태에서 point 샘플 효과. spanwise 분포 비교용.
    python main.py --config configs/hvab/hvab_hover_c10_markerdist_endpoint_point.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "point"},
    marker_distribution="endpoint",
    n_rev=25, polar_source="nasa_overflow", run_tag="md_endpoint_point",
)
