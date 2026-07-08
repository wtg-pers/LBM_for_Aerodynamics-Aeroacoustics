"""SENSITIVITY — Dağ straight, Γ smoothing smooth=1 (15 rev).

One leg of the non-literature-component sensitivity matrix (docs/
alm_fundamental_issues_audit_kr.md "문헌 대비 편차 총괄표"): the spanwise Γ filter
is OUR regularization (not in Dağ/Kleine), so its effect on CT/spanwise must be
reported. Legs: smooth=0 (paper-literal, DIVERGES to sawtooth — reference only),
1 (marginal, Nyquist gain 0.89), 2 (baseline testA_dag_straight, gain 0.73),
3 (sens_straight_smooth3, gain 0.66). Everything else identical to the baseline.

Judge: CT(last-rev) and outer-span alpha/F_n vs smooth=2 — if the answer moves
materially with the filter strength, the filter is doing physics, not just
stabilizing, and must be treated as a model parameter.

    python main.py --config configs/0703_dag_edge_fix/sens_straight_smooth1.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "dag", "target": "inviscid",
                    "relax": 0.5, "smooth": 1, "wake": "straight"},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="sens_straight_smooth1",
)
