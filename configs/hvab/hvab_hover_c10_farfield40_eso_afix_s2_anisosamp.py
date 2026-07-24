"""HVAB hover c10 M0.65 — aniso re-tune (afix): anisotropic SAMPLING only, legal widths.

Re-run of fact_s2 with discretization-legal widths (EPS_FLOORS patch):
the factorial's sampling widths (eps_t=0.1c=1.67dx, eps_r=0.5*dr,
eps_r/dr=0.5) violated both the grid-aliasing floor (2dx) and the
marker-overlap bound (Poisson beading 2*exp(-pi^2 eps^2/dr^2) -> +-17%
line ripple), producing the observed inboard Gamma ripple (s2 worst:
4.2%/11.2% in the 1-3.5/4-8 marker bands vs s1 0.7%/2.2%).

Fix: n_radial 64->128 halves dr to 1.85dx, r-factor 0.5->1.0, and the
code now floors every axis at 2dx -> eps_r = max(dr, 2dx) = 2.0 lu:
same absolute radial sharpness as before (1.87), overlap ratio 0.5->1.08.
eps_t floored 1.67->2.0 at the tip (inboard 0.1c > 2dx untouched).
Marker-count change is non-confounding (2026-07-09 crossover study:
marker count does not shift M2cn). Judge: ripple gone AND does the s4-style
tip roll-off survive legal coefficients?
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "aniso", "c": 1.0, "t": 0.4, "r": 1.0, "r_ref": "spacing"},
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=128,
    n_rev=20,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_afix_s2_anisosamp",
)
