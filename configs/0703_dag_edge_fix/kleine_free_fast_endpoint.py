"""Kleine free-wake (FAST) + endpoint + gauss, with the edge-fixed correction.

Kleine non-iterative solve (method="kleine") with a convected FREE wake
(wake="free"). The trailed-vorticity operator is now edge-based (Dağ Eq.17-18):
the free wake sheds from the N+1 panel EDGES incl. the tip vortex −Γ_tip, and
A = B_edge @ E. See patch_notes/alm_dag_edge_fix/.

"fast" = rebuild the free-wake influence matrix every 5 steps (rebuild_every=5)
instead of every step — the wake convects slowly so this is a good, much cheaper
approximation. Wake length is the Dağ default (2 revolutions shed at 2° azimuthal
intervals ≈ 360 rings); the ring COUNT is auto (n_w omitted). rebuild_every=1
would be the exact (controlled) baseline; 5 is the fast approximation.

Kleine solves the Γ–w coupling self-consistently (ρ(diag(b)A)≈7 → naive iteration
diverges, the linear solve handles it), so NO under-relaxation is needed (relax is
Dağ-only). A bad/stiff solve falls back to the (edge-fixed) Dağ single pass.

    python main.py --config configs/0703_dag_edge_fix/kleine_free_fast_endpoint.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "rebuild_every": 5,  # n_w auto = 2 rev @ 2°
                    "wake_markers": "all", "target": "inviscid"},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="endpoint",
    n_rev=25, polar_source="nasa_overflow", run_tag="kleine_free_fast_endpoint",
)
