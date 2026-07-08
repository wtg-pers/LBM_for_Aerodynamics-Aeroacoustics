"""TEST A — Dağ prescribed HELIX, TIP marker only, n_w=2 (15 rev).

Same as testA_dag_helix.py but the prescribed helix trails ONLY from the tip-vortex
closure edge (wake_markers="tip") and is just n_w=2 rings long — the minimum to form
a single filament (2 nodes = 1 segment). Isolates the tip-vortex de-induction and
gives the cleanest helix wake_*.vtp filament to inspect in ParaView (a single spiral
strand). Contrast with the kleine tip/n_w=2 (convected) filament.

Downwash-sign fix (descent+projection = −thrust_axis) and the sawtooth-stabilising
Γ smoothing (smooth=2) are both applied.

    python main.py --config configs/0703_dag_edge_fix/testA_dag_helix_tip_nw2.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "dag", "target": "inviscid",
                    "relax": 0.5, "smooth": 2, "wake": "prescribed_helix",
                    "wake_markers": "tip", "n_w": 2},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testA_dag_helix_tip_nw2",
)
