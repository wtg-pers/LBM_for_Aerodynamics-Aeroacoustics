"""Kleine free-wake, TIP-ONLY shedding, n_w=2 + endpoint + gauss (edge-fixed).

Minimal free-wake tip model: shed the free wake ONLY from the tip closure edge
(wake_markers="tip" → edge index N, which carries the tip vortex −Γ_tip in the
edge-based operator), with just n_w=2 wake rings (short near-wake + semi-infinite
tail). Isolates the tip vortex's convected de-induction.

rebuild_every=1 (exact) — the wake is tiny (1 filament × 2 rings) so rebuilding
every step is cheap. See patch_notes/alm_dag_edge_fix/.

    python main.py --config configs/0703_dag_edge_fix/kleine_free_tip_nw2_endpoint.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "n_w": 2, "rebuild_every": 1,
                    "wake_markers": "tip", "target": "inviscid"},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="endpoint",
    n_rev=25, polar_source="nasa_overflow", run_tag="kleine_free_tip_nw2_endpoint",
)
