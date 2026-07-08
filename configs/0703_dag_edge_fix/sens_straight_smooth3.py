"""SENSITIVITY — Dağ straight, Γ smoothing smooth=3 (15 rev).

One leg of the non-literature-component sensitivity matrix (docs/
alm_fundamental_issues_audit_kr.md "문헌 대비 편차 총괄표"). See
sens_straight_smooth1.py for the matrix rationale. smooth=3 (Nyquist gain 0.66)
over-damps relative to the smooth=2 baseline — if CT/spanwise barely move from
smooth=2 while smooth=1 also agrees, the answer is filter-insensitive and the
regularization is safe to report as numerics-only.

    python main.py --config configs/0703_dag_edge_fix/sens_straight_smooth3.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "dag", "target": "inviscid",
                    "relax": 0.5, "smooth": 3, "wake": "straight"},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="sens_straight_smooth3",
)
