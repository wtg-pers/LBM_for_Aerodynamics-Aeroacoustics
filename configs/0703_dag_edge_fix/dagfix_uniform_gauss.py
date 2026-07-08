"""Dağ edge-fix re-run: uniform markers + gauss + FIXED viscous-core correction.
CONTROL — compare directly against the OLD (broken) dag run (260630 dag_csv, same
uniform+gauss) to isolate the CODE-FIX effect from the marker-distribution effect.
See dagfix_endpoint_gauss.py + patch_notes/alm_dag_edge_fix/.

    python main.py --config configs/0703_dag_edge_fix/dagfix_uniform_gauss.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "dag",
                    "target": "inviscid", "relax": 0.5},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=25, polar_source="nasa_overflow", run_tag="dagfix_uniform_gauss",
)
