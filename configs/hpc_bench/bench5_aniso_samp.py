"""bench5 pure ALM + anisotropic 3-axis velocity sampling — MPI/e2e smoke."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "aniso", "c": 1.0, "t": 0.5, "r": 1.0},
    marker_distribution="uniform",
    n_rev=1, polar_source="nasa_overflow", run_tag="bench5_aniso_samp",
)
