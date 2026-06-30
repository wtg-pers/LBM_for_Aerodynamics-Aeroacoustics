"""
HART-II hover SMOKE TEST (CPU, small grid, SGS OFF).
Phase E step 1: verify the core pipeline (Cumulant + ALM + MLG + C81 deck).

    python main.py --config configs/hart2/hart2_hover_smoke.py --max-steps 10 --no-vtk --clear
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _hart2_hover_base import build_config

config = build_config(collective_deg=8.0, smoke=True)
