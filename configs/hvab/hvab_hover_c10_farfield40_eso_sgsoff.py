"""HVAB hover c10 M0.65 — far-field D40 esoteric, CASE 5: pure ALM + NASA, SGS OFF.

A/B against CASE 1 (identical except use_sgs=False): isolates the effect of the
explicit dyn_smag eddy viscosity at production scale. Motivation (memory item 5,
2026-07-12): omega_high=1.0 (max implicit/ILES damping of high-order cumulants)
PLUS dyn_smag may double-dissipate the tip vortices -> weakened BVI downwash at
the tip AND weakened wake circulation -> span-wide induced-inflow deficit =
the remaining +15% span-wide offset. Merabet 2021 (S-76, k-w SST, integral
sampling, NO tip correction) matched tips because the previous blade's tip
vortex SURVIVED to strike the next blade.

Run: LBM_ESOTERIC=1 python main.py --config configs/hvab/hvab_hover_c10_farfield40_eso_sgsoff.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    use_sgs=False,                # <- ONLY change vs CASE 1 (pure ILES)
    eps_correction=None,
    prandtl_loss=False,
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="nasa_overflow",
    run_tag="farfield40_eso_case5_sgsoff",
)
