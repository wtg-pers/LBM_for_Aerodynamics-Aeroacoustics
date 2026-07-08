"""Dağ edge-fix re-run: cosine(both) markers + gauss + FIXED viscous-core correction.
See dagfix_endpoint_gauss.py for the fix/relax rationale + patch_notes/alm_dag_edge_fix/.

    python main.py --config configs/0703_dag_edge_fix/dagfix_cosine_gauss.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "dag",
                    "target": "inviscid", "relax": 0.5},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="cosine", cosine_side="both",
    n_rev=25, polar_source="nasa_overflow", run_tag="dagfix_cosine_gauss",
)
