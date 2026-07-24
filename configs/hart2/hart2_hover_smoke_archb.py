"""HART-II CPU pipeline smoke of the archB+Shen stack (3 steps, tiny grid).

Verifies the full config schema + loader path (C81 deck, Kleine straight,
Shen dict, radial truncation, gaussian sampling) on the constant-chord
HART-II blade before a cluster launch. Not a physics run.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hart2_hover_base import build_config

config = build_config(
    collective_deg=6.0, smoke=True, smoke_sgs=True,
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "straight", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss={"enabled": True, "model": "shen", "g": 0.3,
                  "tip": True, "root": False, "eps_offset": False},
    radial_truncation=True,
    sampling={"mode": "gaussian"},
    n_radial=64,
)
config["time"]["max_steps"] = 3
config["output"]["vtk"]["enabled"] = False
