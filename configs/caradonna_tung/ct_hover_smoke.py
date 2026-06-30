"""CT rotor smoke test (CPU, small grid) — verify Yee deck + 2-blade pipeline.

    python main.py --config configs/caradonna_tung/ct_hover_smoke.py --max-steps 6 --no-vtk --clear
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.439, smoke=True)
