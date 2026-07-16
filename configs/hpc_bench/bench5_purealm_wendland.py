"""bench5 + pure ALM, kernel = WENDLAND — G-beta3 A/B member (vs purealm_m3).

Identical to bench5_purealm_m3 except actuator_line.kernel: the Wendland C2
family (01_design.md #11) with eps-equivalent support R_s = sqrt(7.5) eps
(support factor replaces gaussian_cutoff automatically). Isolates the
KERNEL-FAMILY effect on spreading + sampling (no correction, no radial
truncation). Not a physics case.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "gaussian"}, marker_distribution="uniform",
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_purealm_wendland",
)
config["actuator_line"]["kernel"] = {"type": "wendland"}
