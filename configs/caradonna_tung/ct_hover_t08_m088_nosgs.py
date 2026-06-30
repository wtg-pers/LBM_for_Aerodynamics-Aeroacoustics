"""
Caradonna-Tung hover — θ=8°, M_tip=0.877, LIGHT grid (D=32), SGS OFF.

Isolates the SGS variable: same grid as the original light run (which used
dyn_smag, C_T=0.00553, +17%) but with SGS OFF.  If C_T drops here, dyn_smag was
over-diffusing the tip vortex; if not, the deficit is purely grid resolution.
Cheap (~8 GB).

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_nosgs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877, preset="light",
                      use_sgs=False)
