"""q-fraction computation for Bouzidi interpolated bounce-back (IBB).

For each fluid-solid boundary link, we need the fluid-side distance fraction
    q = |fluid_node → wall| / |fluid_node → solid_node|,  q ∈ (0, 1]
indexed by (direction i, fluid_node). This module provides:

    compute_needs_bounce()            generic link mask (2D/3D)
    compute_q_fraction_circle()       analytic q for a 2D circle
    compute_q_fraction_polyline()     ray-segment intersection for a 2D polyline
                                      (airfoil from Selig coords, etc.)

All q-fraction arrays default to 0.5 at links without a specific value.
Choosing q = 0.5 as the sentinel makes the Bouzidi linear formula
(Bouzidi, Firdaouss, Lallemand, Phys. Fluids 13, 3452 (2001), Eqs. 5a-5b)
degenerate exactly to half-way bounce-back — a safe fallback.

References:
    - Bouzidi et al., Phys. Fluids 13, 3452 (2001).
    - Palabos: src/offLattice/bouzidiOffLatticeModel3D.hh
    - OpenLB:  src/boundary/setBouzidiBoundary.h
    - walberla/lbmpy: NoSlipLinearBouzidi

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


def compute_needs_bounce(
    xp: "ModuleType",
    lattice,
    solid_mask: "npt.NDArray",
) -> "npt.NDArray":
    """Boundary-link mask needs_bounce[i, x, y(, z)].

    True iff the node is fluid AND its direction-i neighbor is solid.
    Matches HalfwayBounceBack / MomentumExchangeForce convention.
    """
    Q = lattice.Q
    dim = lattice.dim
    shape = solid_mask.shape
    c = xp.asarray(lattice.c)
    solid = xp.asarray(solid_mask, dtype=bool)

    nb = xp.zeros((Q,) + shape, dtype=bool)
    for i in range(Q):
        if i == 0:
            continue
        if dim == 2:
            cx, cy = int(c[0, i]), int(c[1, i])
            shifted = xp.roll(xp.roll(solid, -cx, axis=0), -cy, axis=1)
        else:
            cx, cy, cz = int(c[0, i]), int(c[1, i]), int(c[2, i])
            shifted = xp.roll(
                xp.roll(xp.roll(solid, -cx, axis=0), -cy, axis=1),
                -cz, axis=2,
            )
        nb[i] = (~solid) & shifted
    return nb


def _to_numpy(arr) -> np.ndarray:
    """Pull a (possibly CuPy) array to host numpy."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def _to_xp(xp, arr: np.ndarray):
    """Push a host numpy array to the target array module."""
    if xp.__name__ == "cupy":
        import cupy as cp
        return cp.asarray(arr)
    return arr


def compute_q_fraction_circle(
    xp: "ModuleType",
    lattice,
    solid_mask: "npt.NDArray",
    needs_bounce: "npt.NDArray",
    center: Tuple[float, float],
    radius: float,
) -> "npt.NDArray":
    """Analytic q-fraction for a 2D circle obstacle.

    Physical process:
        For a boundary link at fluid node x_f with direction c_i pointing
        into solid, the wall point lies at x_f + t * c_i on the unit circle
        of radius R centered at C. Solving
            |x_f + t c_i - C|^2 = R^2
        yields a quadratic in t whose smallest positive root t* gives
        q = t*. We take t* ∈ (0, 1] (the link segment).

    Fallback: if the quadratic has no such root (numerical degeneracy),
    q = 0.5 is kept — this reduces Bouzidi to HWBB for that link.

    Args:
        xp:             Array module (numpy or cupy).
        lattice:        Lattice model (must have dim == 2).
        solid_mask:     Boolean solid mask  (Nx, Ny).
        needs_bounce:   Link mask (Q, Nx, Ny) from compute_needs_bounce.
        center:         (cx, cy) circle center  [lattice units].
        radius:         Circle radius  [lattice units].

    Returns:
        q_fraction:     (Q, Nx, Ny) float32 array, 0.5 default, q_true at links.
    """
    if lattice.dim != 2:
        raise ValueError("compute_q_fraction_circle is 2D-only.")

    Q = lattice.Q
    shape = tuple(solid_mask.shape)
    c = _to_numpy(lattice.c).astype(np.float64)
    nb_np = _to_numpy(needs_bounce)

    q_out = np.full((Q,) + shape, 0.5, dtype=np.float32)
    cx_c, cy_c = float(center[0]), float(center[1])
    R = float(radius)

    for i in range(1, Q):
        cx, cy = float(c[0, i]), float(c[1, i])
        if cx == 0.0 and cy == 0.0:
            continue

        idx = np.argwhere(nb_np[i])  # (N_links, 2)
        if idx.size == 0:
            continue

        xf = idx[:, 0].astype(np.float64)
        yf = idx[:, 1].astype(np.float64)
        dx = xf - cx_c
        dy = yf - cy_c

        a = cx * cx + cy * cy
        b = 2.0 * (dx * cx + dy * cy)
        k = dx * dx + dy * dy - R * R
        disc = b * b - 4.0 * a * k

        valid = disc >= 0.0
        sqrt_disc = np.sqrt(np.where(valid, disc, 0.0))
        t1 = np.where(valid, (-b - sqrt_disc) / (2.0 * a), np.inf)
        t2 = np.where(valid, (-b + sqrt_disc) / (2.0 * a), np.inf)

        # Smallest positive root in (0, 1]
        t1 = np.where((t1 > 1e-10) & (t1 <= 1.0), t1, np.inf)
        t2 = np.where((t2 > 1e-10) & (t2 <= 1.0), t2, np.inf)
        q_i = np.minimum(t1, t2)

        good = np.isfinite(q_i)
        if not np.any(good):
            continue
        q_out[i, idx[good, 0], idx[good, 1]] = q_i[good].astype(np.float32)

    return _to_xp(xp, q_out)


def compute_q_fraction_cylinder_axis(
    xp: "ModuleType",
    lattice,
    solid_mask: "npt.NDArray",
    needs_bounce: "npt.NDArray",
    center: Tuple[float, float],
    radius: float,
    axis: str = "z",
) -> "npt.NDArray":
    """Analytic q-fraction for a 3D axis-aligned cylinder obstacle.

    For an axis-aligned cylinder (axis = 'x', 'y' or 'z'), the wall surface
    in the cross-section perpendicular to the axis is a circle. A boundary
    link with direction c_i intersects the wall at:

        |proj(x_f + t c_i - C)|^2 = R^2

    where proj() drops the axis component. The leading t^2 coefficient
    `a = c_perp_x^2 + c_perp_y^2` is zero only for links lying purely along
    the cylinder axis -- these never cross the side surface and keep the
    q = 0.5 sentinel.

    Args:
        xp:           Array module (numpy or cupy).
        lattice:      Lattice (must be 3D).
        solid_mask:   Boolean solid mask (Nx, Ny, Nz).
        needs_bounce: Link mask (Q, Nx, Ny, Nz) from compute_needs_bounce.
        center:       (C_a, C_b) cylinder centre in the cross-section
                      [lattice units]. The two coordinates are the ones
                      perpendicular to `axis`.
        radius:       Cylinder radius [lattice units].
        axis:         'x', 'y', or 'z' (cylinder axis direction).

    Returns:
        q_fraction (Q, Nx, Ny, Nz) float32, 0.5 default elsewhere.
    """
    if lattice.dim != 3:
        raise ValueError("compute_q_fraction_cylinder_axis is 3D-only.")
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of 'x','y','z'; got '{axis}'")

    Q = lattice.Q
    shape = tuple(solid_mask.shape)
    c = _to_numpy(lattice.c).astype(np.float64)
    nb_np = _to_numpy(needs_bounce)

    # Index of the perpendicular coordinates and the axis coordinate.
    perp = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[axis]
    p0_idx, p1_idx = perp

    q_out = np.full((Q,) + shape, 0.5, dtype=np.float32)
    cx_c, cy_c = float(center[0]), float(center[1])
    R = float(radius)

    for i in range(1, Q):
        ci = c[:, i]
        cp0 = float(ci[p0_idx])
        cp1 = float(ci[p1_idx])
        # Pure axis link cannot cross the side surface.
        a = cp0 * cp0 + cp1 * cp1
        if a == 0.0:
            continue

        idx = np.argwhere(nb_np[i])  # (N_links, 3)
        if idx.size == 0:
            continue

        x0 = idx[:, p0_idx].astype(np.float64)
        x1 = idx[:, p1_idx].astype(np.float64)
        d0 = x0 - cx_c
        d1 = x1 - cy_c

        b = 2.0 * (d0 * cp0 + d1 * cp1)
        k = d0 * d0 + d1 * d1 - R * R
        disc = b * b - 4.0 * a * k

        valid = disc >= 0.0
        sqrt_disc = np.sqrt(np.where(valid, disc, 0.0))
        t1 = np.where(valid, (-b - sqrt_disc) / (2.0 * a), np.inf)
        t2 = np.where(valid, (-b + sqrt_disc) / (2.0 * a), np.inf)
        t1 = np.where((t1 > 1e-10) & (t1 <= 1.0), t1, np.inf)
        t2 = np.where((t2 > 1e-10) & (t2 <= 1.0), t2, np.inf)
        q_i = np.minimum(t1, t2)

        good = np.isfinite(q_i)
        if not np.any(good):
            continue
        q_out[i, idx[good, 0], idx[good, 1], idx[good, 2]] = (
            q_i[good].astype(np.float32)
        )

    return _to_xp(xp, q_out)


def compute_q_fraction_sphere(
    xp: "ModuleType",
    lattice,
    solid_mask: "npt.NDArray",
    needs_bounce: "npt.NDArray",
    center: Tuple[float, float, float],
    radius: float,
) -> "npt.NDArray":
    """Analytic q-fraction for a 3D sphere obstacle.

    3D twin of compute_q_fraction_circle: for a boundary link at fluid
    node x_f with direction c_i, solve |x_f + t c_i - C|^2 = R^2 and take
    the smallest root t* in (0, 1]; q = t*. Numerical degeneracy keeps the
    0.5 sentinel (HWBB fallback) for that link.

    Serves as the validation reference for the STL ray-triangle q
    (icosphere parity, STL track S3) and closes the historical gap where a
    3D sphere with wall_bc='ibb' fell to the q=0.5 sentinel.

    Args:
        xp:             Array module (numpy or cupy).
        lattice:        Lattice model (must have dim == 3).
        solid_mask:     Boolean solid mask  (Nx, Ny, Nz).
        needs_bounce:   Link mask (Q, Nx, Ny, Nz) from compute_needs_bounce.
        center:         (cx, cy, cz) sphere center  [lattice units].
        radius:         Sphere radius  [lattice units].

    Returns:
        q_fraction:     (Q, Nx, Ny, Nz) float32, 0.5 default, q_true at links.
    """
    if lattice.dim != 3:
        raise ValueError("compute_q_fraction_sphere is 3D-only.")

    Q = lattice.Q
    shape = tuple(solid_mask.shape)
    c = _to_numpy(lattice.c).astype(np.float64)
    nb_np = _to_numpy(needs_bounce)

    q_out = np.full((Q,) + shape, 0.5, dtype=np.float32)
    c0 = np.asarray(center, dtype=np.float64)
    R = float(radius)

    for i in range(1, Q):
        ci = c[:, i]
        idx = np.argwhere(nb_np[i])  # (N_links, 3)
        if idx.size == 0:
            continue

        d = idx.astype(np.float64) - c0[None, :]
        a = float(ci @ ci)
        b = 2.0 * (d @ ci)
        k = np.einsum('ij,ij->i', d, d) - R * R
        disc = b * b - 4.0 * a * k

        valid = disc >= 0.0
        sqrt_disc = np.sqrt(np.where(valid, disc, 0.0))
        t1 = np.where(valid, (-b - sqrt_disc) / (2.0 * a), np.inf)
        t2 = np.where(valid, (-b + sqrt_disc) / (2.0 * a), np.inf)

        # Smallest root in (0, 1]
        t1 = np.where((t1 > 1e-10) & (t1 <= 1.0), t1, np.inf)
        t2 = np.where((t2 > 1e-10) & (t2 <= 1.0), t2, np.inf)
        q_i = np.minimum(t1, t2)

        good = np.isfinite(q_i)
        if not np.any(good):
            continue
        q_out[i, idx[good, 0], idx[good, 1], idx[good, 2]] = \
            q_i[good].astype(np.float32)

    return _to_xp(xp, q_out)


def compute_q_fraction_polyline(
    xp: "ModuleType",
    lattice,
    solid_mask: "npt.NDArray",
    needs_bounce: "npt.NDArray",
    x_poly: "npt.NDArray",
    y_poly: "npt.NDArray",
) -> "npt.NDArray":
    """q-fraction from closed-polyline intersection (2D arbitrary geometry).

    Physical process:
        For each boundary link segment [x_f, x_f + c_i], we intersect with
        every polygon edge E_j = [P_j, P_{j+1}]. A 2x2 linear system gives
        (t, s); an intersection is valid when t ∈ (0, 1] and s ∈ [0, 1].
        q = smallest valid t across all edges.

    Fallback: links with no valid intersection (usually numerical edge
    cases where the polygon is coarser than the lattice) keep q = 0.5
    ⇒ HWBB behaviour.

    Args:
        xp:             Array module.
        lattice:        Lattice model (dim == 2).
        solid_mask:     Boolean solid mask (Nx, Ny).
        needs_bounce:   Link mask (Q, Nx, Ny).
        x_poly, y_poly: Polygon vertices in lattice units. Open contour
                        is closed automatically.

    Returns:
        q_fraction: (Q, Nx, Ny) float32.
    """
    if lattice.dim != 2:
        raise ValueError("compute_q_fraction_polyline is 2D-only.")

    Q = lattice.Q
    shape = tuple(solid_mask.shape)
    c = _to_numpy(lattice.c).astype(np.float64)
    nb_np = _to_numpy(needs_bounce)

    q_out = np.full((Q,) + shape, 0.5, dtype=np.float32)

    xp_poly = _to_numpy(x_poly).astype(np.float64)
    yp_poly = _to_numpy(y_poly).astype(np.float64)
    if not (
        np.isclose(xp_poly[0], xp_poly[-1])
        and np.isclose(yp_poly[0], yp_poly[-1])
    ):
        xp_poly = np.append(xp_poly, xp_poly[0])
        yp_poly = np.append(yp_poly, yp_poly[0])

    E0x = xp_poly[:-1]
    E0y = yp_poly[:-1]
    Ex = xp_poly[1:] - xp_poly[:-1]
    Ey = yp_poly[1:] - yp_poly[:-1]

    for i in range(1, Q):
        cx, cy = float(c[0, i]), float(c[1, i])
        if cx == 0.0 and cy == 0.0:
            continue

        idx = np.argwhere(nb_np[i])
        if idx.size == 0:
            continue

        xf = idx[:, 0].astype(np.float64)
        yf = idx[:, 1].astype(np.float64)

        # [cx  -Ex_j] [t]   [E0x_j - xf]
        # [cy  -Ey_j] [s] = [E0y_j - yf]
        #   det = Ex_j * cy - cx * Ey_j
        det = Ex[None, :] * cy - cx * Ey[None, :]              # (N, M)
        rhs_x = E0x[None, :] - xf[:, None]                     # (N, M)
        rhs_y = E0y[None, :] - yf[:, None]

        safe = np.abs(det) > 1e-12
        det_safe = np.where(safe, det, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (-Ey[None, :] * rhs_x + Ex[None, :] * rhs_y) / det_safe
            s = (-cy * rhs_x + cx * rhs_y) / det_safe

        valid = (
            safe
            & (t > 1e-10) & (t <= 1.0)
            & (s >= -1e-12) & (s <= 1.0 + 1e-12)
        )
        t_valid = np.where(valid, t, np.inf)
        q_i = t_valid.min(axis=1)                              # (N,)

        good = np.isfinite(q_i)
        if not np.any(good):
            continue
        q_out[i, idx[good, 0], idx[good, 1]] = q_i[good].astype(np.float32)

    return _to_xp(xp, q_out)
