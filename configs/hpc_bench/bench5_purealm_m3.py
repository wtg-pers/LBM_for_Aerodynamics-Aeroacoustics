"""bench5 + pure ALM (corrections OFF) — multi-GPU M3 gate config.

Same 5-level topology as bench5_baseline but eps_correction=None (no kleine,
no free-wake) so the distributed-ALM gate isolates the sampling/spreading
protocol. Not a physics case.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "gaussian"}, marker_distribution="uniform",
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_purealm_m3",
)
