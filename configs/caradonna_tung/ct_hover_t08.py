"""
Caradonna-Tung hover — collective 8 deg, M_tip = 0.439 (RPM ~1250).
Primary validation point (most-cited pressure-survey case).

    python main.py --config configs/caradonna_tung/ct_hover_t08.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.439)
