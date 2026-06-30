"""Turbulence modeling: SGS (LES) and wall models.

Public API surface mirrors the directory layout:
    sgs           — Smagorinsky, WALE eddy-viscosity formulas (CPU reference)
    strain_rate   — Pi^neq -> S_alpha_beta, |S|, Q extraction
    tau_total     — implicit quadratic tau_total solver (Stiebler 2011)
    wall_model    — placeholder; populated in next phase
"""

from . import sgs, strain_rate, tau_total, wall_model  # noqa: F401
