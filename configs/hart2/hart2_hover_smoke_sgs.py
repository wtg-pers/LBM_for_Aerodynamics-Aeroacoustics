"""
HART-II hover SMOKE TEST (CPU, small grid, Dynamic Smagorinsky SGS ON).
Phase E step 2: verify dyn_smag SGS integration with ALM + MLG.

    python main.py --config configs/hart2/hart2_hover_smoke_sgs.py --max-steps 10 --no-vtk --clear
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _hart2_hover_base import build_config

config = build_config(collective_deg=8.0, smoke=True, smoke_sgs=True)
