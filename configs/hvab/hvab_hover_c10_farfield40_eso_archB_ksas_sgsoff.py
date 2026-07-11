"""HVAB hover c10 M0.65 — far-field D40 esoteric, CASE 6: archB + KSAS, SGS OFF.

A/B against CASE 4 (identical except use_sgs=False): the decisive run — does
removing the explicit SGS dissipation (leaving cumulant implicit LES only,
omega_high=1.0) close the remaining +14.8% span-wide offset of the best
configuration? See CASE 5 docstring for the mechanism.

Run: LBM_ESOTERIC=1 python main.py --config configs/hvab/hvab_hover_c10_farfield40_eso_archB_ksas_sgsoff.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    use_sgs=False,                # <- ONLY change vs CASE 4
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "straight", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss=False,
    radial_truncation=True,
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_case6_archB_ksas_sgsoff",
)
