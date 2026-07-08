"""Case 1: endpoint(끝점포함+사다리꼴) + point(trilinear). 팁 마커 r/R=1.0.
    python main.py --config configs/0630_markers_test/md_endpoint_point.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "point"},
    marker_distribution="endpoint",
    n_rev=25, polar_source="nasa_overflow", run_tag="md_endpoint_point",
)
