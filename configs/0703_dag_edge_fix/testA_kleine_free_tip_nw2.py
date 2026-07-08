"""TEST A — Kleine + FREE wake, TIP marker only, n_w=2 (15 rev).

Same as testA_kleine_free.py but the free wake is shed ONLY from the tip-vortex
closure edge (wake_markers="tip") and is just n_w=2 rings long — the minimum to
form a single filament (2 nodes = 1 segment). Isolates the tip-vortex de-induction
and gives the cleanest wake_*.vtp filament to inspect in ParaView.

Downwash-sign fix (−thrust_axis projection) and the sawtooth-stabilising Γ
smoothing (smooth=2) are both applied.

    python main.py --config configs/0703_dag_edge_fix/testA_kleine_free_tip_nw2.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "kleine", "wake": "free",
                    "rebuild_every": 5, "wake_markers": "tip", "n_w": 2,
                    "target": "inviscid", "smooth": 2},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testA_kleine_free_tip_nw2",
)
