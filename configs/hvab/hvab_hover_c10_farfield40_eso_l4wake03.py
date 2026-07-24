"""HVAB hover c10 M0.65 — L4-slab wake extension A/B, case 03: z/R −0.3125.

Shadowgraphy trajectory validation showed the tip-vortex core radius doubling
at the L4->L3 interface (z/R −0.11, wake age ~130 deg) = grid-dissipation
dominated. This case extends ONLY the L4 slab downstream (down 0.0625D ->
0.15625D, edge z/R −0.3125) so the core stays in L4 to age ~300 deg;
everything else is frozen at fact_s1_isoiso (pure ALM iso/iso, gaussian,
KSAS deck, D40, n_radial=64, 20 rev). Compare wake trajectory + core
metrics (extract_tip_vortex_trajectory.py) against s1 and EXP.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_l4w03",
    eps_correction=None, prandtl_loss=False,
    sampling=None,
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=20,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_l4wake03",
)
