"""bench5 + archB (radial truncation + kleine STRAIGHT) — multi-GPU M3 gate.

Exercises the distributed paths of the production cases 3/4/6: replicated
kleine-straight solve + global radial-truncation scales + local spreading.
Not a physics case.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "straight", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss=False, radial_truncation=True,
    sampling={"mode": "gaussian"}, marker_distribution="uniform",
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_archb_m3",
)
