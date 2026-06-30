"""HVAB hover, collective 8.0 deg, M_tip=0.65. 클러스터 production (preset=medium 기본).
preset 바꾸려면: build_config(collective_deg=8.0, preset="fine")  # tip chord ~16 cell
    python main.py --config configs/hvab/hvab_hover_c08.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(collective_deg=8.0, mtip=0.65)
