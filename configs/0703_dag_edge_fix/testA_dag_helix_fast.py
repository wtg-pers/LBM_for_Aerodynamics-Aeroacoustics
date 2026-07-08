"""TEST A — Dağ prescribed HELIX, FAST (rebuild_every=5), full span (15 rev).

Identical to testA_dag_helix.py except the helix (B@E) operator is rebuilt every 5
steps instead of every step (rebuild_every=5). The prescribed helix geometry/pitch
drifts slowly with the loading, so a 5-step-stale operator is a good approximation;
the circulation Γ is still applied fresh every step (w = M_cached @ Γ_current).

Compare against testA_dag_helix.py (rebuild_every=1, exact) to confirm the fast
approximation is acceptable (loading / w_corr / CT within tolerance).

Sign fix (descent+projection = −thrust_axis) + sawtooth smoothing (smooth=2) applied,
same as the exact.

    python main.py --config configs/0703_dag_edge_fix/testA_dag_helix_fast.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "dag", "target": "inviscid",
                    "relax": 0.5, "smooth": 2, "wake": "prescribed_helix",
                    "rebuild_every": 5},                  # FAST (exact = testA_dag_helix.py)
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testA_dag_helix_fast",
)
