"""bench5 + K17L + SGS off + LineAverage sampling — new-formulation smoke.

Production combo of the 2026-07-22 formulation revision: iso variable-eps
spreading (unchanged) + coupled LineAverage sampling (Melani 2024:
constant rs=0.25c, N_s=80) + EMA time filter 6 deg. Judge: stability,
midspan alpha == gaussian twin (bench5_k17l), tip alpha lower ~1 deg,
1-rank == 2-rank (dist partial-sum path). Not a physics case.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5", use_sgs=False,
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "lineavg", "profile": "constant",
              "n_points": 80, "rs_chord_factor": 0.25,
              "time_filter_deg": 6.0},
    marker_distribution="uniform",
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_lineavg",
)
config["simulation"]["omega_3"] = 0.0
config["simulation"]["omega_4"] = 0.0
config["simulation"]["omega_5"] = 0.0
config["simulation"]["cumulant_limiter"] = 0.01
