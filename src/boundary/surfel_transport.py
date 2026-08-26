"""Swept-volume (parallelepiped) transport tables — surfel track S2.

Implements the geometric weights of the Chen-Teixeira-Molvig volumetric
advection ([C1] = Chen, Teixeira & Molvig, Int. J. Mod. Phys. C 9(8) (1998)
1281-1292):

    N_i(x + c_i dt, t+dt) = [1 - P_i^S(x)] N'_i(x,t) + Q_i(x + c_i dt, t)  (3,4)
    P_i^S(x)  = sum_a V_i^a(x) / dV(x)                                      (5)
    Q_i(x,t)  = sum_a [V_i^a(x) / V_i^a] Gamma_i^{out,a}(t)                 (6)
    Gamma_i^{in,a}(t) = sum_x P_i^a(x) N'_i(x,t)                            (8)

where V_i^a is the volume of the parallelepiped G_i^a extruded from facet a
along the direction that points INTO the fluid, and V_i^a(x) is its overlap
with cell x.

TWO STRUCTURAL FACTS THIS MODULE IS BUILT ON
--------------------------------------------
1. **One prism per (facet, direction PAIR), not per direction.** For facet a
   and lattice direction i the extrusion direction is d = sign(c_i . n) c_i,
   so i and its opposite i* give the SAME region. 27 directions therefore
   need 13 prisms per facet, and V_i^a = V_{i*}^a = |c_i . n| A.

2. **The overlap integral is computed EXACTLY, in closed form.** Write it
   as an integral over the facet polygon,

       V_i^a(x) = |d . n| * integral_polygon  l(y, x) dA,

   where l(y,x) is the length (in s within [0,1]) that the segment
   y + s d spends in cell x. Because every |c_i| component is 0 or +-1,
   the segment crosses at most one boundary per axis:

       s_a(y) = 1/2 + d_a (x_a - y_a)  in [0, 1],   a with d_a != 0

   -- an AFFINE function of y, needing no clipping. The four sub-interval
   lengths are differences of the SORTED s_a, and the destination cells
   depend only on the sort order. So l(., x) is affine on each region where
   the order is fixed, and those regions are bounded by the planes
   s_a = s_b: at most three, and CONCURRENT (s_a = s_b and s_b = s_c imply
   s_a = s_c). Splitting the polygon by them therefore gives at most 6
   nonempty convex pieces on each of which

       integral_piece l dA = area(piece) * l(centroid(piece))

   is exact. Conservation, sum_x V_i^a(x) = |d.n| A, follows because the
   pieces partition the polygon and sum_x l = 1 pointwise.

   (A midpoint-quadrature variant is kept as `mode="quad"` for the S2 gate's
   comparison: it conserves just as exactly but distributes with an O(h^1.5)
   error, which is far too coarse -- see fact 3.)

THE FLUID-VOLUME IDENTITY (derived here, gate s2 [I])
-----------------------------------------------------
Uniform-state preservation is not automatic: it holds iff the cell fluid
volumes dV and the prism overlaps satisfy, for every cell x and direction i,

    dV(x + c_i) = dV(x) - g_i(x) + h_i(x + c_i)                         (*)
    g_i(x) = sum_{a: c_i.n_a < 0} V_i^a(x)      (leaves x toward facets)
    h_i(y) = sum_{a: c_i.n_a > 0} V_i^a(y)      (arrives at y from facets)

Proof: every point z in the fluid part of cell y = x + c_i either came from
z - c_i (also fluid) or crossed the surface, i.e. lies in an outgoing prism;
and translation by c_i maps D(x) onto D(y) exactly.

Consequence: **the volumetric scheme cannot run at dV == 1.** A first
implementation that ignores partial volumes injects a steady spurious
source of order (1 - dV) at every cut cell (measured: 555x the exact-dV
residual, gate s2 [I]). (*) also gives dV for free: march it along a
lattice axis from a far-field cell where dV = 1.

3. **(*) is why the overlaps must be exact, not merely conservative.** The
   identity couples 26 directions to one dV field, so it can only hold for
   all of them if the overlaps are mutually consistent. With midpoint
   quadrature the residual falls only as ~n^-1.5 (4.6e-3, 9.4e-4, 3.4e-4,
   2.0e-4 for n = 1..4) -- a steady spurious near-wall source. With the
   exact splitting it is at roundoff. This is the S2 answer to open
   question 1 of 00_design.md sec. 8.

Author: LBM Development Team
Date: 2026-08 (surfel track S2)
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.boundary.surfel_geometry import (
    _clip_halfspace, _polygon_area_vector, _polygon_centroid,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

# D3Q27, src/lattice/d3q27.py ordering (mirrored in wfb.py / the gates)
_CX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0,
                1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int64)
_CY = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1,
                1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int64)
_CZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1,
                1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int64)
C27 = np.stack([_CX, _CY, _CZ], axis=1)
OPP27 = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17,
                  16, 15, 26, 25, 24, 23, 22, 21, 20, 19], dtype=np.int64)

#: canonical representative of each +- direction pair (13 of them)
PAIR_DIR = C27[np.array([i for i in range(1, 27) if OPP27[i] > i])]
#: pair index of every lattice direction (rest = -1)
PAIR_OF = np.full(27, -1, dtype=np.int64)
for _p, _i in enumerate(i for i in range(1, 27) if OPP27[i] > i):
    PAIR_OF[_i] = _p
    PAIR_OF[OPP27[_i]] = _p
N_PAIR = PAIR_DIR.shape[0]


def _bary_centroids(n: int) -> np.ndarray:
    """Barycentric coordinates of the n^2 sub-triangle centroids of a
    uniformly subdivided triangle (composite midpoint, equal weights).

    Composite midpoint rather than a high-order symmetric rule on purpose:
    l(y, x) is piecewise linear with kinks along the cell-boundary preimages,
    so polynomial exactness buys nothing and equal-area subdivision gives a
    clean, monotone h-refinement to measure against.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n_quad must be >= 1, got {n}")
    pts = []
    for a in range(n):                    # "upward" sub-triangles
        for b in range(n - a):
            c = n - 1 - a - b
            pts.append(((a + 1 / 3) / n, (b + 1 / 3) / n, (c + 1 / 3) / n))
    for a in range(n - 1):                # "downward" sub-triangles
        for b in range(n - 1 - a):
            c = n - 2 - a - b
            pts.append(((a + 2 / 3) / n, (b + 2 / 3) / n, (c + 2 / 3) / n))
    out = np.asarray(pts, dtype=np.float64)
    assert out.shape == (n * n, 3), out.shape
    return out


def _quadrature_points(surfels: Dict[str, np.ndarray], n_quad: int):
    """Fan-triangulate every polygon and place n_quad^2 points per triangle.

    Returns (sid, pos, wgt): flat arrays of surfel index, position and area
    weight, with sum of wgt over each surfel == its area (to roundoff).
    """
    poly = surfels['poly']
    nvert = surfels['nvert']
    n_s, K, _ = poly.shape
    bary = _bary_centroids(n_quad)                      # (nq, 3)
    nq = bary.shape[0]
    sid_l, pos_l, wgt_l = [], [], []
    v0 = poly[:, 0, :]
    for k in range(1, K - 1):
        act = np.nonzero((k + 1) < nvert)[0]
        if act.size == 0:
            continue
        a = v0[act]
        b = poly[act, k, :]
        c = poly[act, k + 1, :]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        p = (bary[None, :, 0, None] * a[:, None, :]
             + bary[None, :, 1, None] * b[:, None, :]
             + bary[None, :, 2, None] * c[:, None, :])       # (m, nq, 3)
        sid_l.append(np.repeat(act, nq))
        pos_l.append(p.reshape(-1, 3))
        wgt_l.append(np.repeat(area / nq, nq))
    if not sid_l:
        return (np.zeros(0, dtype=np.int64), np.zeros((0, 3)), np.zeros(0))
    return (np.concatenate(sid_l), np.concatenate(pos_l),
            np.concatenate(wgt_l))


#: vertex capacity of a sort-order piece: a 9-gon split by 3 planes
_PIECE_CAP = 14


def _split_pieces(poly, nvert, base, d, axes):
    """Split polygons by the planes s_a = s_b of the active axes.

    On each resulting piece the crossing ORDER is fixed, hence l(., x) is
    affine and one centroid evaluation integrates it exactly.
    phi_ab(y) = s_b(y) - s_a(y) = d_b (x_b - y_b) - d_a (x_a - y_a).
    Returns a list of (poly, nvert) pieces, up to 2^len(planes).
    """
    P, K, _ = poly.shape
    pp = np.zeros((P, _PIECE_CAP, 3))
    pp[:, :K, :] = poly
    pieces: List[Tuple[np.ndarray, np.ndarray]] = [(pp, nvert.copy())]
    for ai in range(len(axes)):
        for bi in range(ai + 1, len(axes)):
            a, b = int(axes[ai]), int(axes[bi])
            nxt: List[Tuple[np.ndarray, np.ndarray]] = []
            for q, n in pieces:
                phi = (d[:, b:b + 1] * (base[:, b:b + 1] - q[:, :, b])
                       - d[:, a:a + 1] * (base[:, a:a + 1] - q[:, :, a]))
                nxt.append(_clip_halfspace(q, n, phi))
                nxt.append(_clip_halfspace(q, n, -phi))
            pieces = nxt
    return pieces


def _exact_samples(poly, nvert, base, d, axes, area_eps=1e-15):
    """(row, position, weight) of the exact sort-order pieces.

    weight = piece area, position = piece centroid; the pair integrates
    l exactly (module doc, fact 2).
    """
    rows_l, pos_l, wgt_l = [], [], []
    for q, n in _split_pieces(poly, nvert, base, d, axes):
        area = np.linalg.norm(_polygon_area_vector(q, n), axis=1)
        keep = np.nonzero((n >= 3) & (area > area_eps))[0]
        if keep.size == 0:
            continue
        cen = _polygon_centroid(q[keep], n[keep])
        rows_l.append(keep)
        pos_l.append(cen)
        wgt_l.append(area[keep])
    if not rows_l:
        return (np.zeros(0, dtype=np.int64), np.zeros((0, 3)), np.zeros(0))
    return (np.concatenate(rows_l), np.concatenate(pos_l),
            np.concatenate(wgt_l))


def build_prism_tables(
    surfels: Dict[str, np.ndarray],
    shape: Sequence[int],
    *,
    mode: str = "exact",
    n_quad: int = 2,
    periodic: Sequence[bool] = (True, True, True),
    chunk: int = 20000,
    stats: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """Overlap weights V_i^a(x) of every (facet, direction-pair) prism.

    Args:
        surfels: output of `build_surfels(..., return_polygons=True)`.
        shape:   (Nx, Ny, Nz).
        mode:    "exact" (sort-order splitting, closed form -- the
                 production path) or "quad" (midpoint quadrature, kept for
                 the S2 gate's comparison only).
        n_quad:  mode="quad" only: sub-triangles per fan-triangle edge.
        periodic: axes on which a prism that runs off the box WRAPS instead
                 of being dropped. Dropping breaks mass conservation (an S3
                 failure: a facet on a periodic x-boundary lost its prism
                 tail), and the rest of the module already assumes periodic
                 (np.roll in the march / residual), so all-periodic is the
                 coherent default.
        chunk:   surfels processed per chunk (bounds transient memory).
        stats:   optional dict for diagnostics.

    Returns CSR over the (surfel, pair) key s * N_PAIR + p:
        indptr  (n_s * N_PAIR + 1,) int64
        cell    (nnz,) int64   destination/source cell, flat C-order
        weight  (nnz,) float64 overlap volume [lu^3]
        vtot    (n_s, N_PAIR) float64  = |v_p . n| * A   (the exact V_i^a)
        sgn     (n_s, N_PAIR) int8     sign(v_p . n): the prism runs along
                                       sgn * v_p, i.e. into the fluid
    """
    if mode not in ("exact", "quad"):
        raise ValueError(f"mode must be exact|quad, got {mode}")
    nx, ny, nz = (int(s) for s in shape)
    n_s = int(surfels['cell'].size)
    normal = surfels['normal']
    scell = surfels['cell']
    sgn_f = np.sign(normal @ PAIR_DIR.T)                 # (n_s, N_PAIR)
    vtot = np.abs(normal @ PAIR_DIR.T) * surfels['area'][:, None]

    key_l, cell_l, w_l = [], [], []
    n_lost, lost_w = 0, 0.0
    for beg in range(0, n_s, int(chunk)):
        end = min(beg + int(chunk), n_s)
        gid = np.arange(beg, end)
        cx = scell[gid] // (ny * nz)
        cy = (scell[gid] // nz) % ny
        cz = scell[gid] % nz
        base = np.stack([cx, cy, cz], axis=1)
        poly = surfels['poly'][beg:end]
        nvert = surfels['nvert'][beg:end]
        if mode == "quad":
            q_sid, q_pos, q_wgt = _quadrature_points(
                {'poly': poly, 'nvert': nvert}, n_quad)
        for p in range(N_PAIR):
            d = (PAIR_DIR[p][None, :]
                 * sgn_f[gid, p][:, None]).astype(np.int64)   # (m, 3)
            if mode == "exact":
                axes = np.nonzero(PAIR_DIR[p])[0]
                rows, pos, wgt = _exact_samples(poly, nvert, base, d, axes)
            else:
                rows, pos, wgt = q_sid, q_pos, q_wgt
            if rows.size == 0:
                continue
            k, c, w, lost, lw = _march(base[rows], pos, wgt, d[rows],
                                       (nx, ny, nz), periodic)
            if k.size == 0:
                continue
            n_lost += lost
            lost_w += lw
            row = rows[k]
            wt = w * np.abs(normal[gid[row], :] @ PAIR_DIR[p])
            key_l.append(gid[row] * N_PAIR + p)
            cell_l.append(c)
            w_l.append(wt)

    if key_l:
        key = np.concatenate(key_l)
        cell = np.concatenate(cell_l)
        w = np.concatenate(w_l)
        # aggregate duplicate (key, cell) entries
        comb = key * (nx * ny * nz) + cell
        order = np.argsort(comb, kind='stable')
        comb, key, cell, w = comb[order], key[order], cell[order], w[order]
        first = np.concatenate([[True], comb[1:] != comb[:-1]])
        grp = np.cumsum(first) - 1
        w = np.bincount(grp, w)
        key = key[first]
        cell = cell[first]
    else:
        key = np.zeros(0, dtype=np.int64)
        cell = np.zeros(0, dtype=np.int64)
        w = np.zeros(0)

    n_key = n_s * N_PAIR
    counts = np.bincount(key, minlength=n_key)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    if stats is not None:
        stats['nnz'] = int(cell.size)
        stats['mode'] = mode
        stats['entries_per_prism'] = (float(cell.size) / max(n_key, 1))
        stats['n_lost'] = int(n_lost)
        stats['lost_volume'] = float(lost_w)
        stats['vtot_sum'] = float(vtot.sum())

    return {'indptr': indptr, 'cell': cell, 'weight': w,
            'vtot': vtot, 'sgn': sgn_f.astype(np.int8)}


def _march(base, pos, wgt, d, dims, periodic=(True, True, True)):
    """Walk each unit segment pos -> pos + d through the cell grid.

    Every |d| component is 0 or +-1, so the segment crosses at most one
    boundary per axis: <= 3 crossings, <= 4 cells. Returns
    (point_index, flat_cell, length_weight, n_lost, lost_weight); entries
    leaving the box wrap on periodic axes and are dropped-and-counted
    otherwise.
    """
    nx, ny, nz = dims
    m = pos.shape[0]
    big = 2.0
    s = np.full((m, 3), big)
    for a in range(3):
        nz_a = d[:, a] != 0
        if not np.any(nz_a):
            continue
        bnd = base[:, a] + 0.5 * d[:, a]
        s[nz_a, a] = ((bnd[nz_a] - pos[nz_a, a])
                      / d[nz_a, a].astype(np.float64))
    np.clip(s, 0.0, 1.0, out=s, where=(s < big))
    s = np.where(s >= big, 1.0, s)

    order = np.argsort(s, axis=1, kind='stable')
    ss = np.take_along_axis(s, order, axis=1)            # (m, 3) sorted
    edges = np.concatenate([np.zeros((m, 1)), ss, np.ones((m, 1))], axis=1)
    length = np.diff(edges, axis=1)                      # (m, 4)

    rows = np.arange(m)
    off = np.zeros((m, 4, 3), dtype=np.int64)
    cur = np.zeros((m, 3), dtype=np.int64)
    for j in range(3):
        ax = order[:, j]
        cur = cur.copy()
        cur[rows, ax] += d[rows, ax]
        off[:, j + 1, :] = cur

    cells = base[:, None, :] + off                        # (m, 4, 3)
    good = length > 0.0
    for a, n_a in enumerate((nx, ny, nz)):
        if periodic[a]:
            cells[:, :, a] %= n_a
        else:
            good &= (cells[:, :, a] >= 0) & (cells[:, :, a] < n_a)
    oob = (length > 0.0) & ~good
    n_lost = int(oob.sum())
    lost_w = float((wgt[:, None] * length * oob).sum())

    pi, qi = np.nonzero(good)
    flat = ((cells[pi, qi, 0] * ny + cells[pi, qi, 1]) * nz
            + cells[pi, qi, 2])
    return pi, flat, wgt[pi] * length[pi, qi], n_lost, lost_w


# ----------------------------------------------------------------------
def pair_cell_sums(
    tables: Dict[str, np.ndarray],
    surfels: Dict[str, np.ndarray],
    shape: Sequence[int],
    pair: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """(S_minus, S_plus) per-cell prism mass for one direction pair.

    S_minus sums facets with v_p . n < 0, S_plus those with v_p . n > 0.
    For lattice direction i = +v_p these are (g_i, h_i) of identity (*);
    for i = -v_p they swap. Shapes (Nx, Ny, Nz).
    """
    nx, ny, nz = (int(s) for s in shape)
    ncell = nx * ny * nz
    n_s = int(surfels['cell'].size)
    indptr, cell, w = tables['indptr'], tables['cell'], tables['weight']
    sgn = tables['sgn'][:, pair]

    keys = np.arange(n_s) * N_PAIR + pair
    beg, end = indptr[keys], indptr[keys + 1]
    cnt = end - beg
    if int(cnt.sum()) == 0:
        z = np.zeros((nx, ny, nz))
        return z, z.copy()
    sid = np.repeat(np.arange(n_s), cnt)
    idx = (np.arange(int(cnt.sum()))
           - np.repeat(np.cumsum(cnt) - cnt, cnt)
           + np.repeat(beg, cnt))
    cc, ww = cell[idx], w[idx]
    neg = sgn[sid] < 0
    S_minus = np.bincount(cc[neg], ww[neg], ncell).reshape(nx, ny, nz)
    S_plus = np.bincount(cc[~neg], ww[~neg], ncell).reshape(nx, ny, nz)
    return S_minus, S_plus


def fluid_fraction_by_march(
    tables: Dict[str, np.ndarray],
    surfels: Dict[str, np.ndarray],
    shape: Sequence[int],
    axis: int = 0,
) -> np.ndarray:
    """Cell fluid volume dV from identity (*), marched along one axis.

    (*) fixes dV only up to one constant per march LINE. The constant is
    pinned by max(dV) == 1 on each line rather than by seeding the first
    layer: a channel's first z-layer is inside the solid, so seeding it with
    1 shifts the whole line (this was an actual S3 failure).

    PRECONDITION: every line along `axis` must contain at least one fully
    fluid cell -- true unless the body spans the domain in that direction.
    A line with no prism activity at all is taken to be all fluid, which is
    the correct reading of "no surface anywhere on this line" for a body
    that does not span the domain.

    Exact given exact overlaps, so the spread between the three axis
    marches measures the residual geometric error (gate s2 [D]).
    """
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0|1|2, got {axis}")
    unit = np.zeros(3, dtype=np.int64)
    unit[axis] = 1
    pair = int(np.nonzero((PAIR_DIR == unit[None, :]).all(axis=1))[0][0])
    S_minus, S_plus = pair_cell_sums(tables, surfels, shape, pair)
    if not (PAIR_DIR[pair] == unit).all():          # canonical is -e_axis
        S_minus, S_plus = S_plus, S_minus

    g = np.moveaxis(S_minus, axis, 0)
    h = np.moveaxis(S_plus, axis, 0)
    inc = np.zeros_like(g)
    inc[1:] = -g[:-1] + h[1:]
    acc = np.cumsum(inc, axis=0)
    dV = acc + (1.0 - acc.max(axis=0))[None, ...]
    return np.moveaxis(dV, 0, axis)


def uniform_state_residual(
    tables: Dict[str, np.ndarray],
    surfels: Dict[str, np.ndarray],
    shape: Sequence[int],
    dV: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-direction violation of identity (*), and the resulting spurious
    mass source of a uniform resting equilibrium.

        eps_i(y) = [dV(y - c_i) - g_i(y - c_i) + h_i(y)] - dV(y)
        source(y) = sum_i w_i eps_i(y)        (rho factored out)

    A scheme that satisfies (*) exactly has source == 0 everywhere, which is
    the facet-level statement of [C1] sec. 5's "no spurious currents ... for
    zero-velocity flow involving an arbitrarily shaped solid surface".
    """
    from src.boundary.surfel_transport import PAIR_OF as _POF   # self-ref ok
    W = np.array([8 / 27] + [2 / 27] * 6 + [1 / 54] * 12 + [1 / 216] * 8)
    src = np.zeros_like(dV)
    worst = np.zeros(27)
    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for i in range(1, 27):
        p = int(_POF[i])
        if p not in cache:
            cache[p] = pair_cell_sums(tables, surfels, shape, p)
        S_minus, S_plus = cache[p]
        if (C27[i] == PAIR_DIR[p]).all():
            g, h = S_minus, S_plus
        else:
            g, h = S_plus, S_minus
        # dV_up[y] must be dV(y - c_i); np.roll(a, s)[y] = a[y - s]
        sh = tuple(int(v) for v in C27[i])
        dV_up = np.roll(dV, sh, axis=(0, 1, 2))
        g_up = np.roll(g, sh, axis=(0, 1, 2))
        eps = (dV_up - g_up + h) - dV
        src += W[i] * eps
        worst[i] = np.abs(eps).max()
    return {'source': src, 'worst_eps': worst}


def cap_prism_overlap(
    tables: Dict[str, np.ndarray],
    surfels: Dict[str, np.ndarray],
    shape: Sequence[int],
    dV: np.ndarray,
    tol: float = 1e-9,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Per-(cell, direction) renormalisation of the prism weights.

    The advect source of direction i from cell x is (dV(x) - g_i(x)) n_i(x)
    with g_i = sum over facets of the prism overlap volume (Eq. 5). For a
    convex/smooth wall every facet prism claims a distinct part of the
    cell's fluid volume and g_i <= dV holds. At a CONCAVE crease (ROBIN
    body/pylon junction, patch_notes/robin/02 sec. 7) the prisms of the
    two walls overlap inside one cell, g_i exceeds dV (measured up to 17x
    on the pylon trailing edge), the source turns negative and the density
    goes negative within a few substeps. This is the canonical volumetric-
    formulation step (Chen, Teixeira & Molvig 1998: the captured fraction
    P_i^a(x) = V_i^a(x)/V(x) is renormalised so that sum_a P_i^a <= 1):
    for every direction pair and sign group, weights in a cell whose sum S
    exceeds dV are scaled by dV/S. Vsum (per facet) is derived from the
    scaled table downstream, so distribute stays exactly mass-conserving
    (sum_x Q_i == Gamma_out). Cells with dV = 0 (dead) are left untouched.

    tol is RELATIVE (S > dV (1 + tol)): 1e-9 leaves the round-off
    cases of a convex wall untouched (bit-identical weights, measured on
    the ROBIN fuselage) while every real overlap is >= 10 % of dV.

    Returns (tables copy with the scaled 'weight', stats dict).
    """
    nx, ny, nz = (int(s) for s in shape)
    ncell = nx * ny * nz
    n_s = int(surfels['cell'].size)
    indptr, cell = tables['indptr'], tables['cell']
    w = np.array(tables['weight'], dtype=np.float64, copy=True)
    dv_flat = np.asarray(dV, dtype=np.float64).reshape(-1)
    live = dv_flat > 0.0
    n_capped, max_ratio, removed = 0, 0.0, 0.0
    facet_capped = np.zeros(n_s, dtype=bool)
    w_orig_sum = np.bincount(np.repeat(np.arange(n_s * N_PAIR) // N_PAIR, np.diff(indptr)), tables['weight'], n_s)
    for pair in range(N_PAIR):
        sgn = tables['sgn'][:, pair]
        keys = np.arange(n_s) * N_PAIR + pair
        beg, end = indptr[keys], indptr[keys + 1]
        cnt = end - beg
        if int(cnt.sum()) == 0:
            continue
        sid = np.repeat(np.arange(n_s), cnt)
        idx = (np.arange(int(cnt.sum()))
               - np.repeat(np.cumsum(cnt) - cnt, cnt)
               + np.repeat(beg, cnt))
        cc = cell[idx]
        for group in (sgn[sid] < 0, sgn[sid] >= 0):
            ig = idx[group]
            cg = cc[group]
            if ig.size == 0:
                continue
            S = np.bincount(cg, w[ig], ncell)
            over = live & (S > dv_flat * (1.0 + tol))
            if not over.any():
                continue
            scale = np.ones(ncell)
            scale[over] = dv_flat[over] / S[over]
            n_capped += int(over.sum())
            max_ratio = max(max_ratio, float((S[over] / dv_flat[over]).max()))
            before = w[ig].sum()
            sc = scale[cg]
            w[ig] = w[ig] * sc
            facet_capped[np.unique(sid[group][sc < 1.0 - 1e-12])] = True
            removed += float(before - w[ig].sum())
    out = dict(tables)
    out['weight'] = w
    return out, {'n_cells': n_capped, 'max_ratio': max_ratio,
                 'removed': removed, 'total': float(np.sum(tables['weight'])),
                 'facet_capped': facet_capped,
                 # fraction of each facet's prism weight removed by the cap
                 'facet_cap_frac': 1.0 - np.bincount(np.repeat(np.arange(n_s * N_PAIR) // N_PAIR, np.diff(indptr)), w, n_s) / np.maximum(w_orig_sum, 1e-300)}
