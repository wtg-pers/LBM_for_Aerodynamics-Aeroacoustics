"""HVAB hover c10 M0.65 — far-field D40, pure-ALM 2x2 factorial: anisotropic sampling + isotropic spreading.

Sampling x spreading anisotropy factorial (patch_notes/alm_natelson_aniso_gaussian
/SAMPLING_3AXIS.md). Aniso widths (both sampling & spreading, per user):
eps_c=0.25c (c=1), eps_t=0.1c (t=0.4), eps_r=0.5*dr (r=0.5, spacing).
PURE ALM (no wake correction, no tip-loss) so the raw anisotropy effect on
the tip is isolated. gaussian kernel, KSAS deck, D40, n_radial=64, 20 rev.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "aniso", "c": 1.0, "t": 0.4, "r": 0.5, "r_ref": "spacing"},
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=20,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_fact_s2_anisosamp",
)
