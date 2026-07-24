"""HVAB hover c10 M0.65 — mlg4 + omega_345=0.5 (sweep midpoint, limiter on).

Optional second sweep point between all-one (mlg4_nosgs, 263 deg) and the
Geier limit (mlg4_k17l): omega_3/4/5 = 0.5, lambda = 0.01. Run if k17l
shows a large effect and the response shape (linear vs threshold) matters.
Same frozen setup as mlg4_k17l otherwise.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_mlg4",
    use_sgs=False,
    eps_correction=None, prandtl_loss=False,
    sampling=None,
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=20,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_mlg4_w3mid",
)
config["simulation"]["omega_3"] = 0.5
config["simulation"]["omega_4"] = 0.5
config["simulation"]["omega_5"] = 0.5
config["simulation"]["cumulant_limiter"] = 0.01
