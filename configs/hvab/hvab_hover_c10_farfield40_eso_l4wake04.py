"""HVAB hover c10 M0.65 — L4-slab wake extension A/B, case 04: z/R −0.4125.

Deeper variant of l4wake03: L4 down 0.20625D (edge z/R −0.4125, core stays
in L4 past age 360 deg) + L3 down 0.25D -> 0.3125D to keep the 0.08D-guide
axial nesting margin (0.10625D). Everything else frozen at fact_s1_isoiso
(pure ALM iso/iso, gaussian, KSAS deck, D40, n_radial=64, 20 rev).
If the trajectory/core recovery saturates between 03 and 04, the L4->L3
interface is confirmed as the dominant wake-dissipation source.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_l4w04",
    eps_correction=None, prandtl_loss=False,
    sampling=None,
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=20,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_l4wake04",
)
