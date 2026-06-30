"""
Caradonna-Tung hover — θ=8°, M_tip=0.877, MEDIUM grid (D=36), dyn_smag.

Grid-convergence point: the FINEST uniform grid that fits a ~24 GB GPU
(~43M cells, ~18 GB).  Compare C_T and tip inflow angle vs the light (D=32)
run to see whether the diagnosed tip over-loading is grid-sensitive.  SGS kept
ON (matches the light run) — SGS is not the memory driver and gives a clean
grid-only comparison.  (D=40 balanced OOMs on 24 GB regardless of SGS.)

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_D36.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877, preset="medium")
