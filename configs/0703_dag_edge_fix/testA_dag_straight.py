"""TEST A — Dağ straight-kernel CONTROL (15 rev).

Reference for the helix sign check (testA_dag_helix.py): the straight Eq.17 kernel
is verified to give w_corr > 0 (added downwash / de-induction) at the tip. Compare
the tip `w_corr` sign in the marker VTP against the helix run — they must match.

The straight CORRECTION is unchanged by the 2026-07-04 downwash-sign fix (it is the
analytic radial kernel, no 3-D projection). Re-run only (a) adds the straight-wake
VISUALIZATION (vtk/markers/wake/wake_*.vtp: straight semi-infinite filaments from
each edge in the downwash direction — VIZ only, does not change w_corr), and (b)
matches the 15-rev / same-code state of the helix & kleine_free runs.

    python main.py --config configs/0703_dag_edge_fix/testA_dag_straight.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "dag", "target": "inviscid",
                    "relax": 0.5, "smooth": 2, "wake": "straight"},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testA_dag_straight",
)
