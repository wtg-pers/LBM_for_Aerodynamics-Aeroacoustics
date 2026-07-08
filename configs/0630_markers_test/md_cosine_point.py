"""Case 1: cosine(양끝 조밀) + point(trilinear).
    python main.py --config configs/0630_markers_test/md_cosine_point.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "point"},
    marker_distribution="cosine", cosine_side="both",
    n_rev=25, polar_source="nasa_overflow", run_tag="md_cosine_point",
)
