"""Case 1: uniform(셀중심) + point(trilinear). sampler만 baseline서 교체.
    python main.py --config configs/0630_markers_test/md_uniform_point.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "point"},
    marker_distribution="uniform",
    n_rev=25, polar_source="nasa_overflow", run_tag="md_uniform_point",
)
