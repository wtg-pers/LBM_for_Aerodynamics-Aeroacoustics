"""Per-link wall geometry for the WFB (wall-function bounce) track — W0.

For every boundary link (fluid node x_f, direction c_i into the solid)
computes, from the SAME level-local triangle mesh the mask/q were built
from:

    q        first ray-triangle crossing parameter (== Bouzidi q)
    x_w      wall point            x_f + q c_i
    n        unit surface normal of the hit triangle, oriented INTO the
             fluid (n . c_i < 0)
    x_s      wall-model sampling point  x_w + h n   (h in this level's
             cells; the log-layer velocity is sampled here in W1+)
    s_ok     True if the 2x2x2 trilinear neighbourhood of x_s is fully
             inside the domain and fully FLUID (clean sampling)

The link ORDER matches InterpolatedBounceBack._build_sparse_links exactly
(direction-major, then C-order flat cell index): both derive from
where(needs_bounce) — asserted by the W0 gate.

Deliberately a SIBLING of stl_geometry.compute_q_fraction_triangles
(same binning, same Moller-Trumbore, same t-window/slack policy) rather
than a refactor: the production q function is a validated anchor, and
the W0 gate proves sibling-q == production-q, which mutually validates
both implementations.

Author: LBM Development Team
Date: 2026-08 (wall_model track W0)
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


def compute_link_wall_geometry(
    lattice,
    solid_mask_np: np.ndarray,
    needs_bounce_np: np.ndarray,
    triangles_lu: Tuple[np.ndarray, np.ndarray],
    h_cells: float = 3.0,
    slack: float = 1e-12,
    stats: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """Host-numpy per-link wall geometry (init-time; see module doc).

    Returns dict of flat per-link arrays (n_links,):
        link_cell (int64 flat C-order), link_dir (int8),
        q (f32), wall (n_links, 3 f64), normal (n_links, 3 f64),
        sample (n_links, 3 f64), s_ok (bool)
    plus stats {'n_links', 'n_miss', 'n_grazing', 'n_sample_bad'}.
    Missed links (no crossing — expected 0 for a watertight shared-source
    mesh) keep q = 0.5, normal = -c_i/|c_i|, and s_ok = False.
    """
    if lattice.dim != 3:
        raise ValueError("compute_link_wall_geometry is 3D-only.")
    v = np.asarray(triangles_lu[0], dtype=np.float64)
    f = np.asarray(triangles_lu[1], dtype=np.int64)
    shape = tuple(int(s) for s in solid_mask_np.shape)
    nx, ny, nz = shape
    Q = lattice.Q
    c_all = lattice.c
    c_np = (c_all.get() if hasattr(c_all, 'get')
            else np.asarray(c_all)).astype(np.float64)
    nb = np.asarray(needs_bounce_np, dtype=bool)
    solid = np.asarray(solid_mask_np, dtype=bool)

    tri = v[f]
    v0 = tri[:, 0, :]
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    tri_n = np.cross(e1, e2)
    tri_n /= np.maximum(np.linalg.norm(tri_n, axis=1, keepdims=True), 1e-300)

    # ── triangle binning: identical policy to compute_q_fraction_triangles
    eps = 1e-9
    lo = np.floor(tri.min(axis=1) + 0.5 - eps).astype(np.int64)
    hi = np.floor(tri.max(axis=1) + 0.5 + eps).astype(np.int64)
    lo = np.maximum(lo, -1)
    hi = np.minimum(hi, np.array([nx, ny, nz], dtype=np.int64))
    span = hi - lo + 1
    keep = (span > 0).all(axis=1)
    lo, span = lo[keep], span[keep]
    tri_ids = np.nonzero(keep)[0]
    n_cells = span.prod(axis=1)
    total = int(n_cells.sum())
    cum = np.concatenate([[0], np.cumsum(n_cells)])
    rep_tri = np.repeat(np.arange(lo.shape[0]), n_cells)
    off = np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], n_cells)
    sy = span[rep_tri, 1]
    sz = span[rep_tri, 2]
    cxk = lo[rep_tri, 0] + off // (sy * sz)
    cyk = lo[rep_tri, 1] + (off // sz) % sy
    czk = lo[rep_tri, 2] + off % sz
    key_dim_y, key_dim_z = ny + 2, nz + 2
    keys = ((cxk + 1) * key_dim_y + (cyk + 1)) * key_dim_z + (czk + 1)
    order = np.argsort(keys, kind='stable')
    keys_sorted = keys[order]
    tris_sorted = tri_ids[rep_tri[order]]

    out_cell, out_dir = [], []
    out_q, out_wall, out_n, out_hit = [], [], [], []
    n_miss_total = 0
    n_grazing = 0

    for i in range(1, Q):
        idx = np.argwhere(nb[i])
        n = idx.shape[0]
        if n == 0:
            continue
        d = c_np[:, i]
        p0 = idx.astype(np.float64)

        cand_link, cand_t, cand_tid = [], [], []
        ax_offsets = [
            [0] if d[ax] == 0.0 else [0, int(d[ax])] for ax in range(3)
        ]
        for ox in ax_offsets[0]:
            for oy in ax_offsets[1]:
                for oz in ax_offsets[2]:
                    cell = idx + np.array([ox, oy, oz], dtype=np.int64)
                    qkey = (((cell[:, 0] + 1) * key_dim_y
                             + (cell[:, 1] + 1)) * key_dim_z
                            + (cell[:, 2] + 1))
                    beg = np.searchsorted(keys_sorted, qkey, side='left')
                    end = np.searchsorted(keys_sorted, qkey, side='right')
                    cnt = end - beg
                    tot = int(cnt.sum())
                    if tot == 0:
                        continue
                    link_id = np.repeat(np.arange(n), cnt)
                    pos = (np.arange(tot, dtype=np.int64)
                           - np.repeat(np.cumsum(cnt) - cnt, cnt)
                           + np.repeat(beg, cnt))
                    tid = tris_sorted[pos]
                    e1t, e2t = e1[tid], e2[tid]
                    h = np.cross(np.broadcast_to(d, e2t.shape), e2t)
                    a = np.einsum('ij,ij->i', e1t, h)
                    nonpar = np.abs(a) > 1e-14
                    inv = np.where(nonpar, 1.0 / np.where(nonpar, a, 1.0), 0.0)
                    s = p0[link_id] - v0[tid]
                    u = inv * np.einsum('ij,ij->i', s, h)
                    qv = np.cross(s, e1t)
                    w = inv * (qv @ d)
                    t = inv * np.einsum('ij,ij->i', e2t, qv)
                    valid = (nonpar
                             & (u >= -slack) & (w >= -slack)
                             & (u + w <= 1.0 + slack)
                             & (t > 1e-10) & (t <= 1.0))
                    if np.any(valid):
                        cand_link.append(link_id[valid])
                        cand_t.append(t[valid])
                        cand_tid.append(tid[valid])

        q_i = np.full(n, 0.5, dtype=np.float64)
        n_i = np.broadcast_to(-d / np.linalg.norm(d), (n, 3)).copy()
        hit_i = np.zeros(n, dtype=bool)
        if cand_link:
            cl = np.concatenate(cand_link)
            ct = np.concatenate(cand_t)
            ctid = np.concatenate(cand_tid)
            # first crossing per link: lexsort by (link, t), take head
            o = np.lexsort((ct, cl))
            cl, ct, ctid = cl[o], ct[o], ctid[o]
            first = np.concatenate([[True], cl[1:] != cl[:-1]])
            hit_links = cl[first]
            hit_i[hit_links] = True
            q_i[hit_links] = ct[first]
            nrm = tri_n[ctid[first]]
            # orient INTO the fluid: n . c_i < 0
            sgn = np.einsum('ij,j->i', nrm, d)
            n_grazing += int((np.abs(sgn) < 1e-6).sum())
            nrm = np.where((sgn > 0)[:, None], -nrm, nrm)
            n_i[hit_links] = nrm
            n_miss_total += n - hit_links.size
        else:
            n_miss_total += n

        out_cell.append(idx[:, 0] * (ny * nz) + idx[:, 1] * nz + idx[:, 2])
        out_dir.append(np.full(n, i, dtype=np.int8))
        out_q.append(q_i)
        out_wall.append(p0 + q_i[:, None] * d)
        out_n.append(n_i)
        out_hit.append(hit_i)

    link_cell = np.concatenate(out_cell)
    link_dir = np.concatenate(out_dir)
    q = np.concatenate(out_q)
    wall = np.concatenate(out_wall)
    normal = np.concatenate(out_n)
    hit = np.concatenate(out_hit)
    sample = wall + h_cells * normal

    s_ok = _sample_ok(solid, sample, hit)

    if stats is not None:
        stats['n_links'] = int(link_cell.size)
        stats['n_miss'] = int(n_miss_total)
        stats['n_grazing'] = int(n_grazing)
        stats['n_sample_bad'] = int((~s_ok).sum())
    return {
        'link_cell': link_cell, 'link_dir': link_dir,
        'q': q.astype(np.float32), 'wall': wall, 'normal': normal,
        'sample': sample, 's_ok': s_ok, 'hit': hit,
    }


def _sample_ok(solid: np.ndarray, sample: np.ndarray, hit: np.ndarray,
               periodic_axes: Tuple[int, ...] = ()) -> np.ndarray:
    """True where the 2x2x2 trilinear neighbourhood of the sample point is
    fully inside the domain and fully FLUID. Axes listed in periodic_axes
    wrap (span_through / periodic domains) instead of failing the
    inside-the-box test."""
    dims = solid.shape
    base = np.floor(sample).astype(np.int64)
    s_ok = np.ones(sample.shape[0], dtype=bool)
    for off in np.ndindex(2, 2, 2):
        cs = []
        inside = np.ones(sample.shape[0], dtype=bool)
        for a in range(3):
            ca = base[:, a] + off[a]
            if a in periodic_axes:
                ca = ca % dims[a]
            else:
                inside &= (ca >= 0) & (ca < dims[a])
            cs.append(np.clip(ca, 0, dims[a] - 1))
        ok = inside.copy()
        ok[inside] = ~solid[cs[0][inside], cs[1][inside], cs[2][inside]]
        s_ok &= ok
    return s_ok & hit


def canonicalize_span_through(
    geom: Dict[str, np.ndarray],
    axis: int,
    shape: Tuple[int, int, int],
    solid_mask_np: np.ndarray,
    h_cells: float = 3.0,
    periodic_axes: Optional[Tuple[int, ...]] = None,
) -> Dict[str, np.ndarray]:
    """Enforce the declared span_through symmetry on link geometry.

    An infinite-extrusion body (span_through_axis) has, by contract, a
    mask and dynamics invariant along the axis (proven <= 7e-7 for the
    wing) — but a real STL's tessellation/f32 quantisation leaves
    O(1e-2) axis components and cross-slice scatter in the raw hit
    normals. A WFB using them would inject axis-varying wall forces and
    break the contract. Canonicalisation (same principle as the mask
    pipeline's z-invariance): zero the axis component, average the
    normal over the axis for each (perp-cell, dir) link group,
    renormalise, rebuild sample points and their cleanliness flags.
    Modifies a copy; original dict untouched.
    """
    g = {k: v.copy() for k, v in geom.items()}
    nx, ny, nz = shape
    dims = np.array([nx, ny, nz], dtype=np.int64)
    cell = g['link_cell']
    gx = cell // (ny * nz)
    gy = (cell // nz) % ny
    gz = cell % nz
    coords = [gx, gy, gz]
    perp = [a for a in range(3) if a != axis]
    key = ((coords[perp[0]] * dims[perp[1]] + coords[perp[1]]) * 32
           + g['link_dir'].astype(np.int64))

    n = g['normal']
    n[:, axis] = 0.0
    # group-average over the axis (hits only), then renormalise
    order = np.argsort(key, kind='stable')
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    ks = key[order]
    grp_start = np.concatenate([[True], ks[1:] != ks[:-1]])
    gid = np.cumsum(grp_start) - 1
    n_grp = int(gid[-1]) + 1 if gid.size else 0
    hit_o = g['hit'][order].astype(np.float64)
    acc = np.zeros((n_grp, 3))
    cnt = np.zeros(n_grp)
    for a in range(3):
        np.add.at(acc[:, a], gid, n[order][:, a] * hit_o)
    np.add.at(cnt, gid, hit_o)
    mean = acc / np.maximum(cnt, 1.0)[:, None]
    n_new = mean[gid][inv]
    nrm = np.linalg.norm(n_new, axis=1, keepdims=True)
    good = (nrm[:, 0] > 1e-12) & g['hit']
    n = np.where(good[:, None], n_new / np.maximum(nrm, 1e-300), n)
    g['normal'] = n
    g['sample'] = g['wall'] + h_cells * n
    if periodic_axes is None:
        periodic_axes = (axis,)       # span_through axis is periodic
    g['s_ok'] = _sample_ok(np.asarray(solid_mask_np, dtype=bool),
                           g['sample'], g['hit'], periodic_axes)
    return g
