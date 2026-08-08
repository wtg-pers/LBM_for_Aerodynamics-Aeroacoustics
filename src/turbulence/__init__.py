"""Turbulence modeling: SGS (LES) closures.

Public API surface mirrors the directory layout:
    sgs           — Smagorinsky, WALE eddy-viscosity formulas (CPU reference)
    strain_rate   — Pi^neq -> S_alpha_beta, |S|, Q extraction
    tau_total     — implicit quadratic tau_total solver (Stiebler 2011)
    near_wall_sgs — wall-model-consistent near-wall nu_t (CAMWA Eq.39)

The wall-function closure moved to src/boundary/wall_function.py (user
decision 2026-08-01: a wall model is a boundary-condition concept).
near_wall_sgs stays here: it produces an eddy viscosity (a turbulence
closure), it is merely DRIVEN by u_tau from the wall model.
"""

from . import near_wall_sgs, sgs, strain_rate, tau_total  # noqa: F401
