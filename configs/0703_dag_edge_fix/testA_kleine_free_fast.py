"""TEST A — Kleine + FREE wake, FAST (rebuild_every=5), full span (15 rev).

Identical to testA_kleine_free.py except the free-wake influence matrix is rebuilt
every 5 steps instead of every step (rebuild_every=5). The wake convects every step
regardless; only the (expensive) Biot-Savart influence-matrix rebuild is throttled,
and the geometry drifts slowly so a 5-step-stale matrix is a good approximation.

Compare against testA_kleine_free.py (rebuild_every=1, exact) to confirm the fast
approximation is acceptable (loading / w_corr / CT within tolerance).

Sign fix (−thrust_axis) + sawtooth smoothing (smooth=2) applied, same as the exact.

    python main.py --config configs/0703_dag_edge_fix/testA_kleine_free_fast.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "rebuild_every": 5,   # FAST; n_w auto = 2 rev @ 2°
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testA_kleine_free_fast",
)
