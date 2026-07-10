"""HVAB hover c10 M0.65 — far-field D40 esoteric, CASE 2: pure ALM + KSAS(psu) deck.

Identical to CASE 1 (hvab_hover_c10_farfield40_eso.py) except the C81 deck:
polar_source="ksas_psu" (data/airfoils/RC{412,410,68}_psu.c81; RC6-08T falls
back to RC68_psu — psu has no tab variant, noted for comparison caveats).
Isolates the DECK contribution to the tip-load overprediction vs CASE 1.

Run (esoteric REQUIRED for 24GB; see patch_notes/hpc_upgrade/15):
    LBM_ESOTERIC=1 python main.py --config configs/hvab/hvab_hover_c10_farfield40_eso_ksas.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    eps_correction=None,          # corrections OFF (pure-ALM baseline)
    prandtl_loss=False,
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_case2_ksas",
)
