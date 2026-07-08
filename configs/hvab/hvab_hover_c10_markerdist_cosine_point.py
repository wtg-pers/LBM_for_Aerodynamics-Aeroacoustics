"""Case 1 — marker dist comparison. cosine(both) + POINT sampler (pure ALM).

cosine 양끝 조밀 분포 + point(trilinear) 샘플러. 짝: hvab_hover_c10_markerdist_cosine.py
(cosine+gauss). spanwise 분포 비교용 (CT/FM 적분값보다 마커별 r/R 분포가 핵심).
    python main.py --config configs/hvab/hvab_hover_c10_markerdist_cosine_point.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "point"},
    marker_distribution="cosine", cosine_side="both",
    n_rev=25, polar_source="nasa_overflow", run_tag="md_cosine_point",
)
