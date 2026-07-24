"""HVAB hover c10 M0.65 — aniso re-tune (afix): both aniso, legal widths.

Re-run of fact_s4 (aniso sampling + aniso spreading) with
discretization-legal widths (EPS_FLOORS patch): n_radial 128
(dr=1.85dx), r-factor 1.0, every axis floored at 2dx ->
eps_c=0.25c / eps_t=max(0.1c,2dx) / eps_r=max(dr,2dx)=2.0 lu on both
sampling and spreading. Same absolute radial sharpness as fact_s4
(spreading eps_r was already floored to 2.0; sampling was 1.87 unfloored),
but overlap ratio 0.5->1.08 kills the Poisson beading and the sampling
filter is grid-legal. See afix_s2 docstring for the full rationale.
Judge vs s1/s2'/s4: does the s4 tip roll-off survive without the ripple,
and do CT/CP stop inflating?
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "aniso", "c": 1.0, "t": 0.4, "r": 1.0, "r_ref": "spacing"},
    anisotropic={"c": 1.0, "t": 0.4, "r": 1.0, "r_ref": "spacing"},
    marker_distribution="uniform",
    n_radial=128,
    n_rev=20,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_afix_s4_bothaniso",
)
