"""HVAB hover c10 M0.65 — far-field D40 esoteric, CASE 4: archB + KSAS(psu) deck.

CASE 3 (archB: radial truncation + FLLC-straight, Prandtl off) with the
KSAS/psu C81 deck (RC6-08T -> RC68_psu fallback). Completes the 2x2 matrix
{pure ALM / archB} x {NASA / KSAS} at D40 (cells/R 320), n_radial=64.

Run (esoteric REQUIRED for 24GB; see patch_notes/hpc_upgrade/15):
    LBM_ESOTERIC=1 python main.py --config configs/hvab/hvab_hover_c10_farfield40_eso_archB_ksas.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
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
    run_tag="farfield40_eso_case4_archB_ksas",
)
