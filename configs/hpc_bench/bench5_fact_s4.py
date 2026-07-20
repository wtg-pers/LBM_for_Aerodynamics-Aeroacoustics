"""bench5 pure ALM + aniso sampling + aniso spreading — factorial stage-4 e2e smoke."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "aniso", "c": 1.0, "t": 0.4, "r": 0.5, "r_ref": "spacing"},
    anisotropic={"c": 1.0, "t": 0.4, "r": 0.5, "r_ref": "spacing"},
    marker_distribution="uniform",
    n_rev=1, polar_source="nasa_overflow", run_tag="bench5_fact_s4",
)
