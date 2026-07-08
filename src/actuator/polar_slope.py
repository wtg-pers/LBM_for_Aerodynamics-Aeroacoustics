"""Lift-curve slope ∂C_l/∂α for the Kleine (2022) non-iterative smearing correction.

The non-iterative vortex smearing correction (Kleine, Hanifi & Henningson 2022,
arXiv:2206.05448; see patch_notes/kleine_smearing_correction/00_design.md)
linearizes the lifting line around the previous time step's solution. The
sensitivity coefficients (Kleine Eq. A5/A6, 5.14) require the **lift-curve slope**
∂C_l/∂α at the linearization point.

We obtain the slope from a local **third-order (cubic) polynomial fit** of C_l(α)
on the existing polar query (C81 / csv / flat_plate alike), matching Kleine §6.1
("a third-order polynomial interpolation of C_l is used to avoid discontinuities
in the lift coefficient slope" of a bilinearly-interpolated deck). Four C_l
samples spanning ±1.5·`delta_deg` about α are fit with a cubic and its analytic
derivative is evaluated at α. (Pre-2026-07 this was a 2-point central difference;
the cubic is the paper's choice.)

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
    """Third-order-polynomial ∂C_l/∂α  [per radian] at one (α, Re[, name][, mach]).

    Kleine §6.1: a cubic interpolation of C_l avoids the slope discontinuities of
    a bilinearly-interpolated deck. Four C_l samples at α + δ·{−1.5,−0.5,0.5,1.5}
    (δ = `delta_deg`) are fit with a cubic p(x), x = (α′−α) in radians; the slope
    is p′(0) = the linear coefficient.

    Args:
        polar_query: Callable returning (C_l, C_d).
        alpha_deg: Angle of attack  [deg].
        Re: Reynolds number.
        name: Airfoil name (multi-airfoil polars) or None.
        mach: Section Mach (Mach-indexed polars) or None.
        delta_deg: Fit-stencil spacing  [deg].

    Returns:
        dC_l/dα  [1/rad].
    """
    offs = delta_deg * np.array([-1.5, -0.5, 0.5, 1.5])       # [deg], centred at α
    cls = np.array([_cl_at(polar_query, alpha_deg + o, Re, name, mach)
                    for o in offs])
    xs = offs * _DEG2RAD                                       # [rad], x=0 at α
    coeffs = np.polyfit(xs, cls, 3)         # [a,b,c,d]: a x³ + b x² + c x + d
    return float(coeffs[2])                 # p′(0) = c  [1/rad]


def cubic_slope_4pt(cls, delta_deg: float = 1.0, xp=np):
    """∂C_l/∂α [1/rad] from 4 C_l samples at α + δ·{−1.5,−0.5,0.5,1.5} [deg].

    Closed-form derivative of the SAME cubic that lift_curve_slope fits with
    np.polyfit (Kleine §6.1) — the exact 4-point centred Fornberg weights for
    p′(0) on the equally-spaced stencil (spacing δ):

        p′(0) = (c₋₃ − 27·c₋₁ + 27·c₊₁ − c₊₃) / (24·δ_rad)

    Since 4 points determine the cubic uniquely, this is identical to
    np.polyfit(xs, cls, 3)[2] up to FP round-off (verified <1e-12). Needed
    because cupy has no polyfit → the GPU-resident BEM path uses this.

    Args:
        cls: (4, ...) samples in the order [−1.5δ, −0.5δ, +0.5δ, +1.5δ].
        delta_deg: stencil spacing δ [deg].
        xp: numpy or cupy (array module of `cls`).

    Returns:
        dC_l/dα [1/rad], shape cls.shape[1:].
    """
    d_rad = delta_deg * _DEG2RAD
    c = cls
    return (c[0] - 27.0 * c[1] + 27.0 * c[2] - c[3]) / (24.0 * d_rad)


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
