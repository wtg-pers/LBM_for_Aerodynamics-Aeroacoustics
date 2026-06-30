"""Lift-curve slope ∂C_l/∂α for the Kleine (2022) non-iterative smearing correction.

The non-iterative vortex smearing correction (Kleine, Hanifi & Henningson 2022,
arXiv:2206.05448; see patch_notes/kleine_smearing_correction/00_design.md)
linearizes the lifting line around the previous time step's solution. The
sensitivity coefficients (Kleine Eq. A5/A6, 5.14) require the **lift-curve slope**
∂C_l/∂α at the linearization point.

We obtain the slope by **central finite difference** on the existing polar query
(C81 / csv / flat_plate alike), so no change to the polar loaders is needed.
`delta_deg` (default 1°) smooths over local kinks of a bilinearly-interpolated
C81 deck — Kleine §6.1 recommends a smooth (PCHIP-like) slope; a 1° central
difference averages across the local bilinear segments to that end.

Slope is returned in **per-radian** units (Kleine's α is in radians).

Phase 0 of the Kleine implementation plan. Pure functions, no model state.
"""
from typing import Callable, List, Optional

import numpy as np

try:
    import numpy.typing as npt  # noqa: F401
except Exception:  # pragma: no cover
    pass

_DEG2RAD = np.pi / 180.0


def _cl_at(
    polar_query: Callable,
    alpha_deg: float,
    Re: float,
    name: Optional[str],
    mach: Optional[float],
) -> float:
    """Evaluate C_l only, mirroring actuator_line._lookup_cl_cd dispatch.

    Handles the four polar_query signatures used in the codebase:
        (α, Re) | (α, Re, name) | (α, Re, mach=) | (α, Re, name, mach=)
    """
    if name is not None:
        if mach is not None:
            cl, _ = polar_query(alpha_deg, Re, name, mach=mach)
        else:
            cl, _ = polar_query(alpha_deg, Re, name)
    else:
        if mach is not None:
            cl, _ = polar_query(alpha_deg, Re, mach=mach)
        else:
            cl, _ = polar_query(alpha_deg, Re)
    return float(cl)


def lift_curve_slope(
    polar_query: Callable,
    alpha_deg: float,
    Re: float,
    name: Optional[str] = None,
    mach: Optional[float] = None,
    delta_deg: float = 1.0,
) -> float:
    """Central-difference ∂C_l/∂α  [per radian]  at one (α, Re[, name][, mach]).

    Args:
        polar_query: Callable returning (C_l, C_d).
        alpha_deg: Angle of attack  [deg].
        Re: Reynolds number.
        name: Airfoil name (multi-airfoil polars) or None.
        mach: Section Mach (Mach-indexed polars) or None.
        delta_deg: Central-difference half-step  [deg].

    Returns:
        dC_l/dα  [1/rad].
    """
    cl_p = _cl_at(polar_query, alpha_deg + delta_deg, Re, name, mach)
    cl_m = _cl_at(polar_query, alpha_deg - delta_deg, Re, name, mach)
    return (cl_p - cl_m) / (2.0 * delta_deg * _DEG2RAD)   # [1/rad]


def lift_curve_slope_batch(
    polar_query: Callable,
    alpha_deg: 'npt.NDArray',
    Re: 'npt.NDArray',
    active: 'npt.NDArray',
    multi_airfoil: bool = False,
    marker_airfoil: Optional[List[str]] = None,
    mach: Optional['npt.NDArray'] = None,
    delta_deg: float = 1.0,
) -> 'npt.NDArray':
    """Per-marker ∂C_l/∂α  [1/rad], mirroring _lookup_cl_cd's marker loop.

    Inactive markers (and u_rel≈0 handled by caller) return 0. Args parallel
    actuator_line._lookup_cl_cd so the Kleine correction can reuse the same
    polar_query / multi-airfoil / Mach-pass wiring.

    Args:
        alpha_deg, Re: (N,) per-marker  [deg], [-].
        active: (N,) bool — aerodynamically active markers.
        multi_airfoil: True → pass marker_airfoil[j] as name.
        marker_airfoil: list of airfoil names per marker (multi only).
        mach: (N,) per-marker section Mach, or None (Re-only polars).
        delta_deg: central-difference half-step  [deg].

    Returns:
        (N,) dC_l/dα  [1/rad]  (0 where inactive).
    """
    n = len(alpha_deg)
    out = np.zeros(n, dtype=np.float64)
    for j in range(n):
        if not active[j]:
            continue
        name = marker_airfoil[j] if multi_airfoil and marker_airfoil is not None else None
        m = float(mach[j]) if mach is not None else None
        out[j] = lift_curve_slope(
            polar_query, float(alpha_deg[j]), float(Re[j]),
            name=name, mach=m, delta_deg=delta_deg,
        )
    return out
