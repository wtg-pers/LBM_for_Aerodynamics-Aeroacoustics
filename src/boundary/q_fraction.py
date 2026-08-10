"""q-fraction computation for Bouzidi interpolated bounce-back (IBB).

For each fluid-solid boundary link, we need the fluid-side distance fraction
    q = |fluid_node → wall| / |fluid_node → solid_node|,  q ∈ (0, 1]
indexed by (direction i, fluid_node). This module provides:

    compute_needs_bounce()            generic link mask (2D/3D), (Q,)+shape
    compute_boundary_links()          the SAME links, enumerated sparsely
    compute_q_fraction_circle()       analytic q for a 2D circle
    compute_q_fraction_polyline()     ray-segment intersection for a 2D polyline
                                      (airfoil from Selig coords, etc.)

Dense vs sparse
---------------
`needs_bounce` and `q_fraction` are (Q,)+shape arrays: 27 B/cell and
108 B/cell in 3D, for a payload of ~10 links per 1000 cells. At production
grid sizes that is the dominant build-time allocation (measured 248 B/cell
device peak for one level's IBB setup), so the 3D path enumerates links
directly instead:

    link_cell (n_links,) int32   flat C-order cell index of the fluid node
    link_dir  (n_links,) int8    direction i in 1..Q-1
    link_q    (n_links,) float32 Bouzidi q

`compute_boundary_links` produces (link_cell, link_dir) in EXACTLY the order
`where(compute_needs_bounce(...).reshape(Q, N))` does — direction-major,
then ascending flat cell index — so the two representations are
interchangeable and the dense functions below are thin wrappers over the
link cores (one implementation, no drift).

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


def compute_boundary_links(
    xp: "ModuleType",
    lattice,
    solid_mask: "npt.NDArray",
) -> Tuple["npt.NDArray", "npt.NDArray"]:
    """Sparse boundary-link enumeration — needs_bounce without the (Q,)+shape.

    Returns (link_cell, link_dir) listing every (fluid node, direction into
    solid) pair, in the same order as

        link_dir, link_cell = xp.where(compute_needs_bounce(...).reshape(Q, N))

    i.e. direction-major (i ascending), then ascending flat C-order cell
    index within each direction. The dense mask is never allocated: the
    loop holds one rolled copy and one per-direction link mask at a time
    (~4 B/cell transient instead of 27 B/cell standing).

    Args:
        xp:         Array module (numpy or cupy).
        lattice:    Lattice (2D or 3D).
        solid_mask: Boolean mask, True = solid.

    Returns:
        link_cell (n_links,) int32 — flat index into solid_mask.ravel()
        link_dir  (n_links,) int8  — direction index i
    """
    Q = lattice.Q
    dim = lattice.dim
    c = xp.asarray(lattice.c)
    solid = xp.asarray(solid_mask, dtype=bool)
    N = int(solid.size)
    fluid = ~solid

    cells, dirs = [], []
    for i in range(1, Q):
        if dim == 2:
            cx, cy = int(c[0, i]), int(c[1, i])
            shifted = xp.roll(xp.roll(solid, -cx, axis=0), -cy, axis=1)
        else:
            cx, cy, cz = int(c[0, i]), int(c[1, i]), int(c[2, i])
            shifted = xp.roll(
                xp.roll(xp.roll(solid, -cx, axis=0), -cy, axis=1),
                -cz, axis=2,
            )
        # in-place: no third full-size array. roll always copies (checked on
        # both numpy and cupy, incl. shift=0), so this never touches `solid`.
        shifted &= fluid
        idx = xp.where(shifted.reshape(N))[0]
        del shifted
        if idx.size == 0:
            continue
        cells.append(idx.astype(xp.int32))
        dirs.append(xp.full(int(idx.size), i, dtype=xp.int8))
        del idx

    if not cells:
        return (xp.zeros(0, dtype=xp.int32), xp.zeros(0, dtype=xp.int8))
    return xp.concatenate(cells), xp.concatenate(dirs)


def links_from_needs_bounce(
    xp: "ModuleType",
    needs_bounce: "npt.NDArray",
) -> Tuple["npt.NDArray", "npt.NDArray"]:
    """(link_cell, link_dir) from an already-built dense needs_bounce."""
    Q = int(needs_bounce.shape[0])
    N = 1
    for s in needs_bounce.shape[1:]:
        N *= int(s)
    link_dir, link_cell = xp.where(needs_bounce.reshape(Q, N))
    return link_cell.astype(xp.int32), link_dir.astype(xp.int8)


def count_seam_links(
    xp: "ModuleType",
    lattice,
    shape: Tuple[int, ...],
    link_cell: "npt.NDArray",
    link_dir: "npt.NDArray",
    periodic_axes: Tuple[int, ...] = (),
) -> int:
    """Links whose solid neighbour is only a neighbour through the wrap.

    `compute_needs_bounce` / `compute_boundary_links` find the neighbour with
    `roll`, i.e. they treat the box as a TORUS. On L0 that is intended (the
    domain may genuinely be periodic, and `validate_geometry_config` refuses
    a body touching a domain face precisely to keep it honest). On an MLG
    fine block it is not: every face is a coupling boundary, so when the body
    is cut by one face the OPPOSITE face's outermost cell layer acquires
    links to cells a whole box away.

    Those links are what a fine block's "q=0.5 sentinel" count is usually
    made of — the ray from x_f toward x_f+c_i crosses nothing, so the q
    builder reports a mesh miss. Counting them separates "my mesh is not
    watertight" from "my block face cuts the body".

    `periodic_axes` lists axes whose wrap is BY CONTRACT (a span_through
    prism pierces both faces of its axis, and the wrap there is the physics);
    links wrapping only on those axes are not counted.

    Returns the number of such links (0 when nothing wraps).
    """
    n = int(xp.asarray(link_cell).size)
    if n == 0:
        return 0
    dims = np.asarray([int(s) for s in shape], dtype=np.int64)
    c = _to_numpy(lattice.c).astype(np.int64)          # (dim, Q)
    g = _link_coords(_to_numpy(link_cell), tuple(int(s) for s in shape))
    nb = g + c[:, _to_numpy(link_dir).astype(np.int64)].T
    out = (nb < 0) | (nb >= dims[None, :])
    for ax in periodic_axes:
        out[:, int(ax)] = False
    return int(out.any(axis=1).sum())


def _dir_ranges(link_dir_np: np.ndarray, Q: int) -> np.ndarray:
    """(Q, 2) [start, stop) row per direction of a direction-major link_dir."""
    i = np.arange(Q, dtype=np.int16)
    d = link_dir_np.astype(np.int16)
    return np.stack([np.searchsorted(d, i, side="left"),
                     np.searchsorted(d, i, side="right")], axis=1)


def _link_coords(link_cell_np: np.ndarray,
                 shape: Tuple[int, ...]) -> np.ndarray:
    """(n_links, dim) int64 lattice coords from flat C-order cell indices."""
    return np.stack(np.unravel_index(link_cell_np.astype(np.int64), shape),
                    axis=1)


def _scatter_dense(xp, lattice, shape, link_cell, link_dir, link_q):
    """(Q,)+shape float32 with 0.5 default and link_q written at the links.

    The dense-array adapter for callers that still need the (Q,)+shape form
    (2D CUDA kernels, gates). 3D production paths use the links directly.
    """
    Q = lattice.Q
    N = 1
    for s in shape:
        N *= int(s)
    q_out = xp.full((Q, N), xp.float32(0.5), dtype=xp.float32)
    if int(link_cell.size):
        q_out[xp.asarray(link_dir).astype(xp.int64),
              xp.asarray(link_cell).astype(xp.int64)] = xp.asarray(link_q)
    return q_out.reshape((Q,) + tuple(int(s) for s in shape))


def broadcast_links_mid_slice(
    xp: "ModuleType",
    shape: Tuple[int, int, int],
    link_cell: "npt.NDArray",
    link_dir: "npt.NDArray",
    link_q: "npt.NDArray",
    axis: int,
) -> "npt.NDArray":
    """Link-form of the span-through prism's mid-slice q broadcast.

    Each link takes the q of the link with the same direction and the same
    perpendicular coordinates on the mid slice of `axis` — which IS the ideal
    prism's q, whereas the raw per-link q wobbles by the chordal sagitta of
    an unstructured side tessellation and breaks slice invariance at the wall
    (see setup.py's span_through branch).

    A link with no mid-slice twin falls back to the 0.5 sentinel, matching
    the dense `broadcast_to(q[..., mid:mid+1])` this replaces (there the
    mid-slice cell simply holds the 0.5 default). For a genuinely
    axis-invariant mask — the contract this is used under — every link has
    a twin and the fallback never fires.

    Returns (n_links,) float32.
    """
    dims = tuple(int(s) for s in shape)
    lc = _to_numpy(link_cell).astype(np.int64)
    ld = _to_numpy(link_dir).astype(np.int64)
    q = _to_numpy(link_q).astype(np.float32).copy()
    if lc.size == 0:
        return _to_xp(xp, q)

    g = np.stack(np.unravel_index(lc, dims), axis=1)
    perp = [a for a in range(3) if a != axis]
    key = ((ld * dims[perp[0]] + g[:, perp[0]]) * dims[perp[1]]
           + g[:, perp[1]])

    on_mid = g[:, axis] == dims[axis] // 2
    key_mid, q_mid = key[on_mid], q[on_mid]
    order = np.argsort(key_mid, kind="stable")
    key_mid, q_mid = key_mid[order], q_mid[order]

    out = np.full(q.shape, 0.5, dtype=np.float32)
    if key_mid.size:
        pos = np.searchsorted(key_mid, key)
        hit = pos < key_mid.size
        hit[hit] = key_mid[pos[hit]] == key[hit]
        out[hit] = q_mid[pos[hit]]
    return _to_xp(xp, out)


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
    link_cell, link_dir = links_from_needs_bounce(xp, needs_bounce)
    link_q = compute_q_fraction_cylinder_axis_links(
        xp, lattice, tuple(solid_mask.shape), link_cell, link_dir,
        center=center, radius=radius, axis=axis,
    )
    return _scatter_dense(xp, lattice, tuple(solid_mask.shape),
                          link_cell, link_dir, link_q)


def compute_q_fraction_cylinder_axis_links(
    xp: "ModuleType",
    lattice,
    shape: Tuple[int, int, int],
    link_cell: "npt.NDArray",
    link_dir: "npt.NDArray",
    center: Tuple[float, float],
    radius: float,
    axis: str = "z",
) -> "npt.NDArray":
    """Per-link q for an axis-aligned cylinder (sparse core; see the dense
    twin `compute_q_fraction_cylinder_axis` for the geometry).

    Returns (n_links,) float32, 0.5 where no root lies in (0, 1] — including
    pure-axis links, which never cross the side surface.
    """
    if lattice.dim != 3:
        raise ValueError("compute_q_fraction_cylinder_axis is 3D-only.")
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of 'x','y','z'; got '{axis}'")

    Q = lattice.Q
    c = _to_numpy(lattice.c).astype(np.float64)
    lc = _to_numpy(link_cell)
    ld = _to_numpy(link_dir)

    q_link = np.full(int(lc.size), 0.5, dtype=np.float32)
    if lc.size == 0:
        return _to_xp(xp, q_link)

    perp = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[axis]
    p0_idx, p1_idx = perp
    coords = _link_coords(lc, tuple(int(s) for s in shape))
    rng = _dir_ranges(ld, Q)
    cx_c, cy_c = float(center[0]), float(center[1])
    R = float(radius)

    for i in range(1, Q):
        lo, hi = int(rng[i, 0]), int(rng[i, 1])
        if hi <= lo:
            continue
        ci = c[:, i]
        cp0 = float(ci[p0_idx])
        cp1 = float(ci[p1_idx])
        # Pure axis link cannot cross the side surface.
        a = cp0 * cp0 + cp1 * cp1
        if a == 0.0:
            continue

        idx = coords[lo:hi]
        d0 = idx[:, p0_idx].astype(np.float64) - cx_c
        d1 = idx[:, p1_idx].astype(np.float64) - cy_c

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
        q_link[lo:hi][good] = q_i[good].astype(np.float32)

    return _to_xp(xp, q_link)


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
    link_cell, link_dir = links_from_needs_bounce(xp, needs_bounce)
    link_q = compute_q_fraction_sphere_links(
        xp, lattice, tuple(solid_mask.shape), link_cell, link_dir,
        center=center, radius=radius,
    )
    return _scatter_dense(xp, lattice, tuple(solid_mask.shape),
                          link_cell, link_dir, link_q)


def compute_q_fraction_sphere_links(
    xp: "ModuleType",
    lattice,
    shape: Tuple[int, int, int],
    link_cell: "npt.NDArray",
    link_dir: "npt.NDArray",
    center: Tuple[float, float, float],
    radius: float,
) -> "npt.NDArray":
    """Per-link q for a sphere (sparse core; see the dense twin
    `compute_q_fraction_sphere` for the geometry).

    Returns (n_links,) float32, 0.5 where no root lies in (0, 1].
    """
    if lattice.dim != 3:
        raise ValueError("compute_q_fraction_sphere is 3D-only.")

    Q = lattice.Q
    c = _to_numpy(lattice.c).astype(np.float64)
    lc = _to_numpy(link_cell)
    ld = _to_numpy(link_dir)

    q_link = np.full(int(lc.size), 0.5, dtype=np.float32)
    if lc.size == 0:
        return _to_xp(xp, q_link)

    coords = _link_coords(lc, tuple(int(s) for s in shape))
    rng = _dir_ranges(ld, Q)
    c0 = np.asarray(center, dtype=np.float64)
    R = float(radius)

    for i in range(1, Q):
        lo, hi = int(rng[i, 0]), int(rng[i, 1])
        if hi <= lo:
            continue
        ci = c[:, i]
        idx = coords[lo:hi]

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
        q_link[lo:hi][good] = q_i[good].astype(np.float32)

    return _to_xp(xp, q_link)


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
