"""TEST A — Dağ prescribed-helix SIGN check (smoke, 3 rev).

Confirms the prescribed-helix w_corr has the correct sign in the REAL solver
(local synthetic test was inconclusive on sign, see patch_notes/alm_dag_edge_fix/).

2026-07-04 fix applied: the wake DESCENT and the induced-velocity PROJECTION now
use the downwash direction (−thrust_axis). The first cut used −sign(ω)·axis, which
for HVAB (thrust=−axis) is the THRUST side → the wake climbed the wrong way and the
tip w_corr came out −(upwash). Post-fix, validated (synthetic): helix tip w_corr =
+ = straight, corr +1.00.

How to read the result (15 rev, ~converged):
  - Open vtk/markers/markers_*.vtp in ParaView, colour by `w_corr`.
  - At the TIP (r/R→1): w_corr > 0 = added downwash = de-inducing (CORRECT, same
    as testA_dag_straight). Confirm helix now matches straight's + sign.
  - Open vtk/markers/wake/wake_*.vtp: the helix should now DESCEND on the DOWNWASH
    side (+x for HVAB) and wrap ~2 rev behind the blade.

    python main.py --config configs/0703_dag_edge_fix/testA_dag_helix.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "dag", "target": "inviscid",
                    "relax": 0.5, "smooth": 2, "wake": "prescribed_helix"},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testA_dag_helix",
)
