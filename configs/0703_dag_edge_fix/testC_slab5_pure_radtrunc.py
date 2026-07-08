"""TEST C — slab5 5-level, PURE ALM + tip/root RADIAL TRUNCATION, 15 rev.

Clean ε=0.25c confirmation of the (B) radial-truncation hypothesis (slab5 tip
chord 13.6 cells → 2Δx floor OFF → ε=0.25c, matching Merabet). A/B against the
"off" baseline (identical but radial_truncation=False):
  off = aeromechanics_workshop/HVAB/260706_5level_test/slab5_pure_alm_csv
Judge: M^2*Cn tip + azimuthally-averaged field tip inflow (fieldviz_A/B).

Run testC_light_pure_radtrunc first (cheap); promote here if the tip overload
drops. Smoke first (~300 steps, nan_trap) — 5-level grid.

    python main.py --config configs/0703_dag_edge_fix/testC_slab5_pure_radtrunc.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light_slab5",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform", radial_truncation=True,
    n_rev=15, polar_source="nasa_overflow", run_tag="testC_slab5_pure_radtrunc",
)
