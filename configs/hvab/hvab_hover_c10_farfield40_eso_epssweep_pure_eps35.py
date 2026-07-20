"""HVAB hover c10 M0.65 — far-field D40, eps-sweep: pure ALM, eps=0.35c.

Finite-eps correction-convergence test (patch_notes/alm_beta_kernel/08):
FIXED grid (D40 farfield40, n_radial=64, gaussian kernel, KSAS deck) with
ONLY epsilon_chord_factor swept. archB curves collapsing across eps =
correction converged (eps-independent) -> residual +13% is the ALM answer,
finite-eps excluded. Fanning -> correction incomplete. pure runs give the
uncorrected eps-sensitivity reference (the slope archB must flatten).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    prandtl_loss=False,
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=15,
    polar_source="ksas_psu",
    epsilon_chord_factor=0.35,
    run_tag="farfield40_eso_case1_ksas_eps35",
)
