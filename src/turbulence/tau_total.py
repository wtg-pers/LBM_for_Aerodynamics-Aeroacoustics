"""Implicit quadratic solver for the SGS-augmented relaxation time.

Stiebler 2011 derives the closed-form solution by combining
    tau_total = 3 nu_total / c^2 + 0.5 Delta_t,
    nu_total  = nu_0 + nu_T,
    nu_T (Smagorinsky) = (Cs Delta)^2 |S|,
    |S|      = (3 / (rho c^2 tau_total)) * sqrt(Pi:Pi / 2).

In lattice units (c = 1, Delta_t = 1, Delta_x = 1 per level):

    tau_total = (tau_0 + sqrt(tau_0^2 + 18 Cs^2 Q)) / 2,
    Q         = sqrt(2 Pi:Pi),

with tau_0 = 3 nu_0 + 0.5 the molecular relaxation time. The expression is
applied per node every step, and per level (Delta_x_l = 1 in the local lu of
that level, so the (Cs Delta)^2 factor is identical across levels in lu —
the level-dependent eddy viscosity emerges naturally from per-level nu_0
and per-level tau_0 via hyperbolic scaling).

For WALE the eddy viscosity is purely explicit (no tau coupling on the rhs
through |S|), so we just compute nu_T from the gradient tensor and form

    tau_total = tau_0 + 3 nu_T.

This module exposes both paths as numpy reference functions for tests.
"""

from __future__ import annotations

import numpy as np


def tau_total_smagorinsky(tau_0: float | np.ndarray,
                          Q: np.ndarray,
                          Cs: float) -> np.ndarray:
    """Implicit quadratic solution (Stiebler 2011, Eq. 12) in lattice units.

    Args:
        tau_0: molecular relaxation time = 3 nu_0 + 0.5 (per level).
        Q:     sqrt(2 Pi^neq : Pi^neq) per node.
        Cs:    Smagorinsky constant.

    Returns:
        tau_total per node, satisfying
            tau_total = (tau_0 + sqrt(tau_0^2 + 18 Cs^2 Q)) / 2.
    """
    return 0.5 * (tau_0 + np.sqrt(tau_0 * tau_0 + 18.0 * Cs * Cs * Q))


def tau_total_wale(tau_0: float | np.ndarray, nu_t: np.ndarray) -> np.ndarray:
    """Explicit closure: tau_total = tau_0 + 3 nu_T (lattice units, c = 1)."""
    return tau_0 + 3.0 * nu_t


def omega_from_tau(tau: np.ndarray | float) -> np.ndarray | float:
    """omega = 1 / tau."""
    return 1.0 / tau
