"""bench5 + pure ALM + BGK collision + dyn_smag — BGK/SGS/force smoke.

Regression anchor for the esoteric-BGK extension (body force + local-omega
SGS, patch 21): same bench5 topology as bench5_purealm_m3 but
collision_model=bgk. Judge: stable N steps (BGK at tau~0.5 is the risk),
CT finite/sane, nu_t written. Not a physics case.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "gaussian"}, marker_distribution="uniform",
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_bgk_sgs",
)
# numerics.collision takes precedence over simulation.collision_model in
# setup._create_lbm_components — override BOTH (the first masked the smoke).
config["numerics"]["collision"] = "bgk"
config["simulation"]["collision_model"] = "bgk"
