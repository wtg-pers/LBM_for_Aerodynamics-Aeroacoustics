"""Surfel (surface element) geometry — surfel track S1.

A *surfel* is the intersection of one STL triangle with one voxel: a convex
planar polygon carrying an exact area, an exact unit normal, and a centroid.
It is the geometric object the Chen-Teixeira-Molvig boundary algorithm
exchanges momentum through:

    H. Chen, C. Teixeira, K. Molvig, "Realization of Fluid Boundary
    Conditions via Discrete Boltzmann Dynamics", Int. J. Mod. Phys. C
    9(8) (1998) 1281-1292.                          [= [C1] in the notes]

Why the exact area/normal are load-bearing rather than cosmetic: the
algorithm's gather weight is the parallelepiped volume V_i = |c_i . n| A dt
(paper sec. 3), and it is that |c_i . n| factor -- not the staircase link
set -- that makes the tangential surface force vanish identically at ANY
orientation (paper Eqs. 18-20). See patch_notes/surfel/00_design.md sec. 2.

FRAME. Everything is in the level-local lattice frame used by
`compute_q_fraction_triangles` / `compute_link_wall_geometry`: cell (i,j,k)
is the unit cube centred on the integer point (i,j,k), so it spans
[i-1/2, i+1/2) x ... . Triangle vertices must already be in that frame
(`geom_info['triangles_lu']` for a global build; caller rebases for MLG
levels). The candidate (triangle, cell) binning is deliberately the same
policy as those two validated functions.

ORIENTATION. Normals point INTO THE FLUID (out of the solid). For a
watertight, consistently wound mesh that is the face normal
(v1-v0) x (v2-v0) when the winding is outward/CCW; `orient="auto"`
detects the winding from the divergence-theorem volume and flips if needed.

Author: LBM Development Team
Date: 2026-08 (surfel track S1)
"""

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

#: Vertex capacity of a clipped polygon. A triangle clipped by the six
#: half-spaces of a box gains at most one vertex per plane -> 3 + 6 = 9.
#: Asserted at run time, so a violation is loud rather than silent.
_MAX_VERTS = 10

_EPS_BIN = 1e-9        # binning slack (same value as the sibling functions)

#: Tie-break nudge, applied to the BINNING bbox only (never to the clipped
#: geometry). A triangle lying exactly ON a voxel face would otherwise be
#: binned into BOTH neighbours and clipped to the full polygon in each,
#: DOUBLING the surface there -- measured as a dead channel when a wall was
#: placed at z = 3.5 (S4 gate [O]). Nudging along the fluid-pointing normal
#: gives the half-open ownership [c-1/2, c+1/2) resolved toward the fluid,
#: which is also the side the facet's prisms extrude into. 100x _EPS_BIN so
#: it breaks exact ties, still geometrically negligible.
_NUDGE_BIN = 1e-7


def build_surfels(
    shape: Sequence[int],
    triangles_lu: Tuple['npt.NDArray', 'npt.NDArray'],
    *,
    orient: str = "auto",
    area_min: float = 1e-12,
    tri_budget: int = 200_000,
    return_polygons: bool = False,
    stats: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """Clip every triangle against every voxel it touches.

    Args:
        shape: (Nx, Ny, Nz) cells of the target box.
        triangles_lu: (vertices (n_v, 3), faces (n_f, 3)) in the level-local
            lattice frame (cell centres at integer coordinates).
        orient: "auto" (flip if the mesh winding is inward, detected by the
            divergence-theorem volume), "as_is", or "flip".
        area_min: drop polygon fragments below this area [lu^2]. Grazing
            triangle/voxel touches produce slivers of ~1e-16 that carry no
            momentum but cost memory.
        tri_budget: max triangles processed per chunk (bounds transient
            memory; does not change results).
        return_polygons: also return the clipped polygon vertices, which
            the S2 swept-volume quadrature integrates over.
        stats: optional dict, filled with build diagnostics.

    Returns:
        dict of per-surfel arrays, surfel-major:
            cell      (n_s,)    int64, flat C-order voxel index
            area      (n_s,)    float64 [lu^2]
            normal    (n_s, 3)  float64, unit, into the fluid
            centroid  (n_s, 3)  float64, level-local lattice coords
            tri_id    (n_s,)    int64, index into `faces`
        and, with return_polygons=True,
            poly      (n_s, 10, 3) float64, first nvert rows valid, CCW
            nvert     (n_s,)    int64, 3..9
        Sorted by (cell, tri_id) so a facet's voxel neighbours are contiguous.
    """
    nx, ny, nz = (int(s) for s in shape)
    if min(nx, ny, nz) <= 0:
        raise ValueError(f"shape must be positive, got {shape}")
    v = np.asarray(triangles_lu[0], dtype=np.float64)
    f = np.asarray(triangles_lu[1], dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (n_v, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (n_f, 3), got {f.shape}")

    if orient not in ("auto", "as_is", "flip"):
        raise ValueError(f"orient must be auto|as_is|flip: {orient}")
    flip = orient == "flip"
    if orient == "auto":
        t = v[f]
        # The winding test below is a divergence-theorem volume, which is
        # only meaningful for a CLOSED surface. On an open mesh (a plane, a
        # patch) its sign is arbitrary, and getting it wrong inverts every
        # normal -- i.e. swaps fluid and solid, silently. Refuse instead.
        av = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        closure = (np.linalg.norm(av.sum(axis=0))
                   / max(np.linalg.norm(av, axis=1).sum(), 1e-300))
        if closure > 1e-9:
            raise ValueError(
                f"orient='auto' needs a closed surface (|sum A n| / sum A = "
                f"{closure:.3e}); this mesh is open, so the winding test is "
                f"meaningless. Pass orient='as_is' or 'flip' explicitly.")
        # divergence-theorem volume; positive for outward (CCW) winding
        vol6 = np.einsum('ij,ij->i', t[:, 0],
                         np.cross(t[:, 1], t[:, 2])).sum()
        flip = vol6 < 0.0

    keys = ["cell", "area", "normal", "centroid", "tri_id"]
    if return_polygons:
        keys += ["poly", "nvert"]
    out: Dict[str, list] = {k: [] for k in keys}
    n_slivers = 0
    n_pairs = 0
    for beg in range(0, f.shape[0], int(tri_budget)):
        end = min(beg + int(tri_budget), f.shape[0])
        chunk = _build_chunk(v, f[beg:end], (nx, ny, nz), flip,
                             area_min, return_polygons)
        n_slivers += chunk.pop('_n_slivers')
        n_pairs += chunk.pop('_n_pairs')
        chunk['tri_id'] += beg
        for k in out:
            out[k].append(chunk[k])

    if out['cell'] and any(a.size for a in out['cell']):
        res = {k: np.concatenate(out[k]) for k in keys}
        order = np.lexsort((res['tri_id'], res['cell']))
        res = {k: v_[order] for k, v_ in res.items()}
    else:
        res = {'cell': np.zeros(0, dtype=np.int64), 'area': np.zeros(0),
               'normal': np.zeros((0, 3)), 'centroid': np.zeros((0, 3)),
               'tri_id': np.zeros(0, dtype=np.int64)}
        if return_polygons:
            res['poly'] = np.zeros((0, _MAX_VERTS, 3))
            res['nvert'] = np.zeros(0, dtype=np.int64)
    cell, area = res['cell'], res['area']

    if stats is not None:
        stats['n_surfels'] = int(cell.size)
        stats['n_pairs'] = int(n_pairs)
        stats['n_slivers'] = int(n_slivers)
        stats['n_cells'] = int(np.unique(cell).size) if cell.size else 0
        stats['area_total'] = float(area.sum())
        stats['flipped'] = bool(flip)

    return res


# ----------------------------------------------------------------------
def _build_chunk(v, faces, dims, flip, area_min, want_poly):
    """One triangle chunk: bin -> clip -> area/normal/centroid."""
    nx, ny, nz = dims
    tri = v[faces]                                   # (T, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    nrm = np.cross(e1, e2)
    if flip:
        nrm = -nrm
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(nlen, 1e-300)

    # ── candidate (triangle, cell) pairs: same binning policy as
    #    compute_q_fraction_triangles / compute_link_wall_geometry,
    #    but clipped to REAL cells (a surfel must live in the box) and with
    #    the face-coincidence tie-break of _NUDGE_BIN.
    tri_bin = tri + _NUDGE_BIN * nrm[:, None, :]
    lo = np.floor(tri_bin.min(axis=1) + 0.5 - _EPS_BIN).astype(np.int64)
    hi = np.floor(tri_bin.max(axis=1) + 0.5 + _EPS_BIN).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array([nx, ny, nz], dtype=np.int64) - 1)
    span = hi - lo + 1
    keep = (span > 0).all(axis=1) & (nlen[:, 0] > 0.0)
    if not np.any(keep):
        return _empty_chunk(want_poly)
    lo, span = lo[keep], span[keep]
    tri_ids = np.nonzero(keep)[0]
    ncell = span.prod(axis=1)
    total = int(ncell.sum())
    cum = np.concatenate([[0], np.cumsum(ncell)])
    rep = np.repeat(np.arange(lo.shape[0]), ncell)
    off = np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], ncell)
    sy, sz = span[rep, 1], span[rep, 2]
    cx = lo[rep, 0] + off // (sy * sz)
    cy = lo[rep, 1] + (off // sz) % sy
    cz = lo[rep, 2] + off % sz
    ctr = np.stack([cx, cy, cz], axis=1).astype(np.float64)
    pair_tri = tri_ids[rep]

    # ── clip, in cell-local coordinates (better conditioning far from 0)
    poly = np.zeros((total, _MAX_VERTS, 3))
    poly[:, :3, :] = tri[pair_tri] - ctr[:, None, :]
    cnt = np.full(total, 3, dtype=np.int64)
    for axis in range(3):
        for sign in (+1.0, -1.0):
            # inside <=> sign * x_axis <= 1/2   ->   d = 1/2 - sign * x
            d = 0.5 - sign * poly[:, :, axis]
            poly, cnt = _clip_halfspace(poly, cnt, d)
            if not np.any(cnt >= 3):
                return _empty_chunk(want_poly)

    area_v = _polygon_area_vector(poly, cnt)
    area = np.linalg.norm(area_v, axis=1)
    good = (cnt >= 3) & (area > float(area_min))
    n_slivers = int(((cnt >= 3) & ~good).sum())
    if not np.any(good):
        c = _empty_chunk(want_poly)
        c['_n_slivers'] = n_slivers
        c['_n_pairs'] = total
        return c

    poly, cnt = poly[good], cnt[good]
    area = area[good]
    pair_tri = pair_tri[good]
    ctr = ctr[good]
    cx, cy, cz = cx[good], cy[good], cz[good]
    cen = _polygon_centroid(poly, cnt) + ctr

    chunk = {
        'cell': (cx * ny + cy) * nz + cz,
        'area': area,
        'normal': np.ascontiguousarray(nrm[pair_tri]),
        'centroid': cen,
        'tri_id': pair_tri,
        '_n_slivers': n_slivers,
        '_n_pairs': total,
    }
    if want_poly:
        # back to absolute coords; unused slots collapse onto the centre
        chunk['poly'] = poly + ctr[:, None, :]
        chunk['nvert'] = cnt
    return chunk


def _empty_chunk(want_poly=False):
    c = {'cell': np.zeros(0, dtype=np.int64), 'area': np.zeros(0),
         'normal': np.zeros((0, 3)), 'centroid': np.zeros((0, 3)),
         'tri_id': np.zeros(0, dtype=np.int64),
         '_n_slivers': 0, '_n_pairs': 0}
    if want_poly:
        c['poly'] = np.zeros((0, _MAX_VERTS, 3))
        c['nvert'] = np.zeros(0, dtype=np.int64)
    return c


def _clip_halfspace(poly, cnt, d):
    """Sutherland-Hodgman clip of convex polygons by one half-space.

    Args:
        poly: (P, K, 3) vertex buffer, first cnt[p] rows valid, CCW.
        cnt:  (P,) vertex counts.
        d:    (P, K) signed distance, inside where d >= 0.

    Emission rule (duplicate-free): emit the current vertex when it is
    inside, and emit the edge intersection only on a STRICT sign change
    (d_cur * d_nxt < 0). A vertex lying exactly on the plane is therefore
    emitted once, not twice -- which is what keeps the 3 + 6 capacity bound
    valid.
    """
    P, K, _ = poly.shape
    # +2 slack columns so an over-capacity polygon trips the explicit
    # assertion below with a useful message instead of an IndexError.
    out = np.zeros((P, K + 2, 3))
    ocnt = np.zeros(P, dtype=np.int64)
    rows = np.arange(P)
    for k in range(K):
        act = k < cnt
        if not act.any():
            continue
        nk = np.where(k + 1 < cnt, k + 1, 0)
        cur = poly[:, k, :]
        nxt = poly[rows, nk, :]
        dc = d[:, k]
        dn = d[rows, nk]

        m = act & (dc >= 0.0)
        if m.any():
            out[rows[m], ocnt[m]] = cur[m]
            ocnt[m] += 1

        m2 = act & (dc * dn < 0.0)
        if m2.any():
            t = dc[m2] / (dc[m2] - dn[m2])
            out[rows[m2], ocnt[m2]] = (cur[m2]
                                       + t[:, None] * (nxt[m2] - cur[m2]))
            ocnt[m2] += 1
    if int(ocnt.max(initial=0)) > K:
        raise AssertionError(
            f"clipped polygon exceeded the {K}-vertex capacity "
            f"(max {int(ocnt.max())}); the emission rule or the capacity "
            f"bound is wrong.")
    return out[:, :K, :], ocnt


def _polygon_area_vector(poly, cnt):
    """(P, 3) area vector 1/2 sum_k v_k x v_{k+1} of each closed loop.

    Translation invariant for a closed loop, so cell-local coordinates are
    safe; |A| is the area and A/|A| the polygon normal.
    """
    P, K, _ = poly.shape
    acc = np.zeros((P, 3))
    rows = np.arange(P)
    for k in range(K):
        act = k < cnt
        if not act.any():
            continue
        nk = np.where(k + 1 < cnt, k + 1, 0)
        cr = np.cross(poly[:, k, :], poly[rows, nk, :])
        acc += np.where(act[:, None], cr, 0.0)
    return 0.5 * acc


def _polygon_centroid(poly, cnt):
    """(P, 3) area centroid by fan decomposition from vertex 0."""
    P, K, _ = poly.shape
    num = np.zeros((P, 3))
    den = np.zeros(P)
    v0 = poly[:, 0, :]
    for k in range(1, K - 1):
        act = (k + 1) < cnt
        if not act.any():
            continue
        a = poly[:, k, :] - v0
        b = poly[:, k + 1, :] - v0
        w = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
        w = np.where(act, w, 0.0)
        num += w[:, None] * (v0 + poly[:, k, :] + poly[:, k + 1, :]) / 3.0
        den += w
    return num / np.maximum(den, 1e-300)[:, None]


# ----------------------------------------------------------------------
def surfel_cell_normals(
    surfels: Dict[str, np.ndarray],
    n_cells: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted mean normal and total area per voxel.

    Returns (uniq_cell, area_sum, mean_normal) for the voxels that own at
    least one surfel. Used for cell-level diagnostics and for the fallback
    no-slip path; the facet algorithm itself works per surfel, so the
    "cell-mean tangent" degeneracy of the link formulation (wall_model
    patch 18) never arises.
    """
    cell = surfels['cell']
    if cell.size == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0),
                np.zeros((0, 3)))
    uniq, inv = np.unique(cell, return_inverse=True)
    area = np.bincount(inv, surfels['area'], uniq.size)
    acc = np.stack([
        np.bincount(inv, surfels['area'] * surfels['normal'][:, a], uniq.size)
        for a in range(3)], axis=1)
    nrm = np.linalg.norm(acc, axis=1, keepdims=True)
    return uniq, area, acc / np.maximum(nrm, 1e-300)
