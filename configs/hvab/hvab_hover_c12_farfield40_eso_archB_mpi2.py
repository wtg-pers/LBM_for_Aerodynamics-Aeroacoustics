"""HVAB hover collective 12 deg — archB sweep member (NASA deck).

archB = radial truncation + Kleine non-iterative correction (straight
wake), SAME NASA deck as the pure sweep -> the pure-archB delta isolates
the correction effect vs loading (single-variable discipline). n_rev=25
keeps the ramp identical; run --steps 27654 (22 rev).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=12.0, mtip=0.65, preset="farfield40",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "straight", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid",
                    "smooth": 2},
    radial_truncation=True,
    prandtl_loss=False,
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="nasa_overflow",
    run_tag="farfield40_eso_archB_c12_mpi2",
)
