"""bench5 pure ALM + Shen tip-loss (g=0.3) — end-to-end smoke for patch 09."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction=None,
    prandtl_loss={"enabled": True, "model": "shen", "g": 0.3,
                  "tip": True, "root": False, "eps_offset": False},
    sampling={"mode": "gaussian"}, marker_distribution="uniform",
    n_rev=1, polar_source="nasa_overflow", run_tag="bench5_shen",
)
