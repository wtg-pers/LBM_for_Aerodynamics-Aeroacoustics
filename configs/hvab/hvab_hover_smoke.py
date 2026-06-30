"""HVAB hover smoke (CPU, D=16) — Mach-pass + multi-airfoil(4 RC) + taper 배관 검증.
    python main.py --config configs/hvab/hvab_hover_smoke.py --max-steps 6 --no-vtk --clear
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(collective_deg=8.0, mtip=0.65, smoke=True)
