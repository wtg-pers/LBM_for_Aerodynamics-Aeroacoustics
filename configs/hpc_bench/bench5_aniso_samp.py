"""bench5 pure ALM + anisotropic 3-axis sampling (physical widths) — e2e/MPI smoke.
eps_c=0.25c (c=1), eps_t=0.1c (t=0.4), eps_r=0.5*dr (r=0.5, spacing)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "aniso", "c": 1.0, "t": 0.4, "r": 0.5, "r_ref": "spacing"},
    marker_distribution="uniform",
    n_rev=1, polar_source="nasa_overflow", run_tag="bench5_aniso_samp",
)
