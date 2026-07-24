"""HART-II hover collective 3.0 deg — archB + Shen g=0.3 formulation, mlg4.

Workshop Task 3 (FM vs CT/sigma) sweep point, 2026-07-23 insurance set.
Formulation = HVAB archB g030 mlg4 verbatim: all-one cumulant + dyn_smag
SGS, iso gaussian sampling/spreading + Merabet radial truncation + Kleine
straight correction (smooth=2) + Shen g=0.3 tip loss, uniform n64, 25 rev,
NACA23012 C81 deck (Mach-interpolated). Grid = farfield40_mlg4 (HVAB
campaign D-relative geometry, ALM on L3, ~65M cells, 1257 steps/rev).
Sweep: c03/c06/c09/c12/c15 — driver run_hart2_archb_sweep.sh.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hart2_hover_base import build_config

config = build_config(
    collective_deg=3.0, preset="farfield40_mlg4",
    use_sgs=True,
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "straight", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss={"enabled": True, "model": "shen", "g": 0.3,
                  "tip": True, "root": False, "eps_offset": False},
    radial_truncation=True,
    sampling={"mode": "gaussian"},
    n_radial=64,
    n_rev=25,
    run_tag="archB_shen_g030",
)
