"""bench5 archB WENDLAND + KSAS(psu) deck — D40-defect discriminator smoke.

Deck swap of bench5_archb_wendland (NASA -> ksas_psu): after the axis-y
rank geometry, marker spacing, and kernel activity were all matched clean
in pairs A/B, the polar deck is the only remaining *physics-config*
difference to the broken D40 case-4' (the rest: cuda-aware UCX transport,
absolute scale). Not a physics case.
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
    n_rev=2, polar_source="ksas_psu", run_tag="bench5_archb_ksas_wendland",
)
config["actuator_line"]["kernel"] = {"type": "wendland"}
