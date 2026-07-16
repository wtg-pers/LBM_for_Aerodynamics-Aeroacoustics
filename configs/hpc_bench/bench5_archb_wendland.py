"""bench5 + archB (radial trunc + kleine STRAIGHT), kernel = WENDLAND —
G-beta3 A/B member (vs archb_m3).

Exercises the FULL beta chain of production case-4': Wendland spreading +
sampling + radial-truncation renormalization (kernel-consistent scales) +
kleine-straight solve whose influence matrix uses the DERIVED Wendland
deficit K (03_correction_derivation.md). Not a physics case.
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
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_archb_wendland",
)
config["actuator_line"]["kernel"] = {"type": "wendland"}
