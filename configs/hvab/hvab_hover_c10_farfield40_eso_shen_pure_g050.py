"""HVAB hover c10 M0.65 — far-field D40: PURE ALM + Shen tip-loss (g=0.50).

Path A (patch_notes/alm_beta_kernel/09): wake/FLLC correction OFF entirely
(eps_correction=None), tip relief supplied by an explicit Shen 2005 tip-loss
load factor F=(2/π)arccos(exp(-g·f)), f=(B/2)(R-r)/(r sinφ). g=1.0 = Prandtl;
g<1 broadens the roll-off inboard. Shen's hover limit g=0.1 is very aggressive
(cuts mid-span) -> g needs calibration, hence this sweep. gaussian kernel,
isotropic spreading, gaussian sampling, KSAS deck, D40, n_radial=64.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    eps_correction=None,
    prandtl_loss={"enabled": True, "model": "shen", "g": 0.50,
                  "tip": True, "root": False, "eps_offset": False},
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=15,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_shen_pure_g050",
)
