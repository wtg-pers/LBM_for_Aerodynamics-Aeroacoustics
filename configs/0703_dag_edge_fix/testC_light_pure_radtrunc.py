"""TEST C — light 4-level, PURE ALM + tip/root RADIAL TRUNCATION, 25 rev.

(B)-hypothesis test: Merabet & Laurendeau (2021) radially truncate the Gaussian
source at the geometric blade limits [r_root, r_tip] and renormalise so the
tip marker's force is deposited INBOARD of r/R=1 instead of ~30% leaking to
r/R>1. Fast FIRST test on the cheap light grid (tip overload present there too,
per the 3-way field comparison).

A/B against the "off" baseline (identical but radial_truncation=False):
  off = aeromechanics_workshop/HVAB/260630/260630_results_nasa_c81/pureALM_csv
Judge: M^2*Cn tip (should DROP toward EXP 0.146 if the hypothesis holds) +
azimuthally-averaged field tip inflow (should recover downwash).

NOTE: light tip chord ~6.3 cells → ε at 2Δx floor (ε>0.25c), so the outboard
leak is even larger here than on slab5 — the effect may be exaggerated. Confirm
on slab5 (testC_slab5_pure_radtrunc) at the clean ε=0.25c condition.

    python main.py --config configs/0703_dag_edge_fix/testC_light_pure_radtrunc.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform", radial_truncation=True,
    n_rev=25, polar_source="nasa_overflow", run_tag="testC_light_pure_radtrunc",
)
# eps_correction omitted -> pure ALM; only the radial truncation differs from baseline.
