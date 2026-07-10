"""HVAB hover c10 M0.65 — far-field D40 esoteric, CASE 3: archB + NASA deck.

archB (handoff §1/§5): isotropic Gaussian sampling + RADIAL TRUNCATION with
renormalization (tip-blob fix) + FLLC-straight tip-deficit correction
(kleine, wake="straight" = Dag influence-matrix semi-infinite filaments;
free wake retired: ~1/12 tip under-correction). Prandtl OFF (the correction
resolves the physical tip de-loading directly; double-counting otherwise).
Goal: pull the tip M^2 c_n roll-off onto KSAS/EXP.

Run (esoteric REQUIRED for 24GB; see patch_notes/hpc_upgrade/15):
    LBM_ESOTERIC=1 python main.py --config configs/hvab/hvab_hover_c10_farfield40_eso_archB.py
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
    radial_truncation=True,       # tip Gaussian radial truncation + renorm
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="nasa_overflow",
    run_tag="farfield40_eso_case3_archB",
)
