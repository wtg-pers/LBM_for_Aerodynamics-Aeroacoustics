"""
Caradonna-Tung hover — collective 8 deg, M_tip = 0.877 (RPM ~2500).
High-tip-Mach pressure-survey case: stresses the compressible (Mach-indexed)
Yee NACA0012 deck where transonic effects are strong near the tip.

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877)
