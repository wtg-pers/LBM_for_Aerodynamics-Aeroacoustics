"""HVAB hover c10 M0.65 — far-field D40, CASE 4': archB + KSAS, WENDLAND.

beta-kernel D40 matrix member (patch_notes/alm_beta_kernel/01_design #14.5):
CASE 4 (radial truncation + kleine-straight + KSAS deck) with the full
Wendland chain — spreading/sampling + kernel-consistent radial-truncation
scales + the DERIVED Wendland deficit K in the straight-wake solve
(03_correction_derivation.md). Success criteria vs the case-4 gaussian
archive: peak-region +13~17% -> single digits, CT +15% -> within +5%,
tip dip 0.215 preserved (00_handoff #5.4).

Run (esoteric REQUIRED for 24GB; 4-rank: wrap like the mpi variants):
    LBM_ESOTERIC=1 python main.py \
        --config configs/hvab/hvab_hover_c10_farfield40_eso_archB_ksas_wendland.py
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
    run_tag="farfield40_eso_case4w_archB_ksas_wendland",
)
config["actuator_line"]["kernel"] = {"type": "wendland"}
