"""HVAB hover collective 8 deg — far-field D40, collective-sweep member.

Physics identical to the 10-deg case-1 twin (pure ALM, isotropic Gaussian,
N=64, corrections OFF, NASA OVERFLOW deck) except collective_deg=8.
n_rev kept at 25 so ramp/derived parameters match the 10-deg archive
EXACTLY (comparability); the sweep runs --steps 27654 (22 rev).

Launched by scripts: run_collective_sweep.sh (2-rank, GPUs 0,1).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=8.0, mtip=0.65, preset="farfield40",
    eps_correction=None,
    prandtl_loss=False,
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="nasa_overflow",
    run_tag="farfield40_eso_c8_mpi2",
)
