"""
HART-II hover, collective = 12.0 deg @ 0.75R (production, Task 3 sweep point).

    python main.py --config configs/hart2/hart2_hover_c12.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _hart2_hover_base import build_config

config = build_config(collective_deg=12.0)
