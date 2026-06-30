"""
Caradonna-Tung hover — θ=8°, M_tip=0.877, BALANCED grid (D=40), SGS OFF.

Two-in-one test for the diagnosed TIP OVER-LOADING (downwash deficit):
  (1) finer dx (D=40 vs light D=32) -> better trailed tip-vortex resolution
  (2) SGS off -> removes dyn_smag eddy viscosity that may over-diffuse the
      coherent tip vortex (cumulant's low numerical dissipation preserves it)
Both should strengthen the tip induced downwash -> tip inflow angle recovers
(phi: 0 -> positive) -> C_T drops toward the experimental 0.00473.

SGS off also makes D=40 FIT a ~24 GB GPU (~17 GB vs ~26 GB with dyn_smag).

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_balanced.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877, preset="balanced",
                      use_sgs=False)
