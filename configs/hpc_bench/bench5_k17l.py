"""bench5 + pure ALM + K17L (Geier 2017 I) — omega_345=0 + limiter smoke.

Third-order cumulant rates at the Geier fourth-order-accurate limit for
omega_1 -> 2 (omega_3=omega_4=omega_5=0) with the eq.(116) limiter
(lambda=0.01, the canonical NC0.01 / 2021-TGV K17L choice). SGS off.
Judge: stability (the optimized point is less stable than all-one without
the limiter) + CT differential vs bench5 all-one. Not a physics case.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5", use_sgs=False,
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "gaussian"}, marker_distribution="uniform",
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_k17l",
)
config["simulation"]["omega_3"] = 0.0
config["simulation"]["omega_4"] = 0.0
config["simulation"]["omega_5"] = 0.0
config["simulation"]["cumulant_limiter"] = 0.01
