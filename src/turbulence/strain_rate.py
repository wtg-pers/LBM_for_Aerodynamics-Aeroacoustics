"""Strain rate extraction from non-equilibrium distribution moments.

The strain-rate tensor is recovered from the LBM second-moment non-equilibrium
part directly, avoiding finite differences on velocity:

    S_ab = -3 / (2 rho c^2 tau) * Pi^neq_ab,
    Pi^neq_ab = sum_i e_ia e_ib (f_i - f_i^eq).

For a D2Q9 / D3Q27 lattice with cs^2 = 1/3 in lattice units (c = 1):

    Pi^neq_ab = rho * (K_ab - cs^2 * delta_ab)        [a == b]
    Pi^neq_ab = rho * K_ab                            [a != b]

where K_ab is the rho-normalised central second moment as computed inside the
cumulant collision (e.g. K20, K02, K11 in the D2Q9 kernel).

This module provides numpy-side reference implementations for unit tests; the
production path inlines the same algebra inside the cumulant kernel string.
"""

from __future__ import annotations

import numpy as np

CS2 = 1.0 / 3.0


def pi_neq_from_central_moments_2d(rho: np.ndarray,
                                   K20: np.ndarray,
                                   K02: np.ndarray,
                                   K11: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Pi_xx, Pi_yy, Pi_xy) from rho-normalised central moments."""
    pi_xx = rho * (K20 - CS2)
    pi_yy = rho * (K02 - CS2)
    pi_xy = rho * K11
    return pi_xx, pi_yy, pi_xy


def pi_neq_from_central_moments_3d(rho: np.ndarray,
                                   K200: np.ndarray, K020: np.ndarray, K002: np.ndarray,
                                   K110: np.ndarray, K101: np.ndarray, K011: np.ndarray
                                   ) -> tuple[np.ndarray, ...]:
    """Return (Pi_xx, Pi_yy, Pi_zz, Pi_xy, Pi_xz, Pi_yz)."""
    pi_xx = rho * (K200 - CS2)
    pi_yy = rho * (K020 - CS2)
    pi_zz = rho * (K002 - CS2)
    pi_xy = rho * K110
    pi_xz = rho * K101
    pi_yz = rho * K011
    return pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz


def strain_rate_from_pi_neq_2d(rho: np.ndarray, tau: np.ndarray | float,
                               pi_xx: np.ndarray, pi_yy: np.ndarray, pi_xy: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (S_xx, S_yy, S_xy) using S_ab = -3 / (2 rho tau) * Pi_ab. (c=1)."""
    pref = -1.5 / (rho * tau)
    return pref * pi_xx, pref * pi_yy, pref * pi_xy


def s_norm_2d(s_xx: np.ndarray, s_yy: np.ndarray, s_xy: np.ndarray) -> np.ndarray:
    """|S| = sqrt(2 S:S). 2D: S:S = S_xx^2 + S_yy^2 + 2 S_xy^2."""
    return np.sqrt(2.0 * (s_xx * s_xx + s_yy * s_yy + 2.0 * s_xy * s_xy))


def s_norm_3d(s_xx, s_yy, s_zz, s_xy, s_xz, s_yz) -> np.ndarray:
    """|S| = sqrt(2 S:S). 3D: S:S = sum_diag^2 + 2 sum_off^2."""
    return np.sqrt(2.0 * (s_xx * s_xx + s_yy * s_yy + s_zz * s_zz
                          + 2.0 * (s_xy * s_xy + s_xz * s_xz + s_yz * s_yz)))


def pi_norm_2d(pi_xx: np.ndarray, pi_yy: np.ndarray, pi_xy: np.ndarray) -> np.ndarray:
    """Q = sqrt(2 Pi:Pi). Stiebler 2011 Eq. 12 input."""
    return np.sqrt(2.0 * (pi_xx * pi_xx + pi_yy * pi_yy + 2.0 * pi_xy * pi_xy))


def pi_norm_3d(pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz) -> np.ndarray:
    """Q = sqrt(2 Pi:Pi). 3D version."""
    return np.sqrt(2.0 * (pi_xx * pi_xx + pi_yy * pi_yy + pi_zz * pi_zz
                          + 2.0 * (pi_xy * pi_xy + pi_xz * pi_xz + pi_yz * pi_yz)))
