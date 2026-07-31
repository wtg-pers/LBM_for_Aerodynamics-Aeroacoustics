"""Cell-center solid voxelization for STL bodies (STL body track, S2).

Provides:
    create_stl_mask()      column-parity ray casting -> boolean solid mask
    stl_bbox_l0lu()        config helper: transformed body bbox in L0 lu
    suggest_fine_region()  config helper: padded MLG region from the bbox

Algorithm (create_stl_mask) — column-parity ray casting along +z:
    1. Localize vertices p = (V - origin_lu) / dx so cell centers sit at
       integer coordinates of the target (sub-)box.
    2. Normalize every triangle to CCW in xy-projection; drop triangles with
       zero projected area (vertical walls carry no column crossings — for a
       watertight mesh their parity is carried by the adjacent caps).
    3. For each integer column (i, j) inside a triangle's xy-bbox, evaluate
       the three f64 edge functions in CANONICAL per-edge orientation
       (smaller welded vertex id -> larger; only the sign is flipped for
       the triangle's traversal). Shared edges thus see bitwise-identical
       |E| with exactly opposite signs, so ownership of centers on or
       within epsilon of an edge is an exact exclusive-or. Ties (E == 0)
       are owned via a fixed half-plane rule (accept iff dy > 0, or
       dy == 0 and dx < 0 — the raster "top-left" rule up to axis
       orientation). CCW normalization makes silhouette-fold edges count
       0 or 2 (tangent graze) and pass-through edges exactly 1, so
       watertight parity is exact. No jitter is ever applied.
    4. Crossing height z_hit = sum(w_i z_i) / sum(w_i) (barycentric from the
       edge functions); the crossing toggles cells k >= ceil(z_hit). An
       exactly-integer z_hit toggles its own cell, so a cell center lying ON
       the entry surface is solid — matching the analytic `<=` convention
       (create_sphere_mask). Consequence: the overall convention is
       half-open (entry-closed / exit-open); only cell centers exactly on
       the surface are affected, a measure-zero configuration.
    5. np.add.at over a body-bbox-local counts array, cumulative parity
       along z, scatter into the full mask. Cost scales with the body's
       projected area, not the domain size.

Contracts (PLAN.md):
    - Physics core runs in host numpy float64 regardless of xp; the result
      is uploaded once (2D PIP precedent — bitwise numpy/cupy identity).
    - Cell-center point classification ONLY. No VOF, no dilation, no
      connected-component filter (see geometry.py:708-733 design record).
    - (origin_lu, shape, dx) is a rank/level-agnostic sub-box signature:
      integer origins and power-of-two dx localize exactly in f64, so a
      sub-box result equals the slice of the full-box result bitwise
      (MLG fine levels and MPI dist-init share this API).

Author: LBM Development Team
Date: 2026-07
"""

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np

from src.boundary.stl_mesh import load_stl_checked, transform_vertices_to_l0lu

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


def _edge_ok(
    pa: np.ndarray, pb: np.ndarray,
    ia: np.ndarray, ib: np.ndarray,
    px: np.ndarray, py: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tie-exact edge function for directed edge A -> B of a welded mesh.

    The edge function is evaluated in a CANONICAL orientation — from the
    endpoint with the smaller global vertex id to the larger — and only its
    SIGN is flipped for the triangle's actual traversal direction. The two
    triangles sharing an edge therefore see bitwise-identical |E| and
    exactly opposite signs (a sign flip is exact in f64), which makes the
    ownership rule an exact exclusive-or even when floating-point rounding
    puts a cell center within epsilon of the edge. (Evaluating E from each
    triangle's own vertex order instead gives two *different* roundings of
    ~0, which can drop or double-count a column — observed on a box face
    diagonal passing exactly through a cell center.)

    Coverage: E > 0 (strictly left), or E == 0 and the traversal direction
    is "accepting" (dy > 0, or dy == 0 and dx < 0) — for opposite
    traversals exactly one side accepts.

    Args:
        pa, pb: (n, 3) endpoint coords; ia, ib: (n,) global vertex ids.
        px, py: (n,) query points.

    Returns:
        (E_dir, ok) for the directed edge as traversed A -> B.
    """
    swap = ia > ib
    sw = swap[:, None]
    c0 = np.where(sw, pb, pa)
    c1 = np.where(sw, pa, pb)
    dxc = c1[:, 0] - c0[:, 0]
    dyc = c1[:, 1] - c0[:, 1]
    e_canon = dxc * (py - c0[:, 1]) - dyc * (px - c0[:, 0])
    sigma = np.where(swap, -1.0, 1.0)
    e_dir = sigma * e_canon
    dxd = sigma * dxc
    dyd = sigma * dyc
    accept = (dyd > 0.0) | ((dyd == 0.0) & (dxd < 0.0))
    return e_dir, (e_dir > 0.0) | ((e_dir == 0.0) & accept)


def create_stl_mask(
    xp: 'ModuleType',
    shape: Tuple[int, int, int],
    vertices_lu: 'npt.NDArray',
    faces: 'npt.NDArray',
    origin_lu: Sequence[float] = (0.0, 0.0, 0.0),
    dx: float = 1.0,
    pair_budget: int = 5_000_000,
) -> 'npt.NDArray':
    """Solid mask of a watertight triangle mesh on a (sub-)box of cells.

    Args:
        xp: Array module for the RETURNED mask (numpy or cupy). All
            computation is host numpy f64.
        shape: (Nx, Ny, Nz) cells of the target box.
        vertices_lu: (n_v, 3) f64 vertices, in the frame origin_lu/dx refer
            to (L0 lu for global builds; already-local frames pass
            origin_lu=(0,0,0), dx=1).
        faces: (n_f, 3) int vertex indices (winding irrelevant).
        origin_lu: coordinate of cell (0,0,0)'s center in the vertex frame.
        dx: cell spacing in the vertex frame (power of two for exact
            sub-box localization).
        pair_budget: max (column, triangle) pairs processed per chunk —
            bounds transient memory, does not change results.

    Returns:
        Boolean xp array, shape (Nx, Ny, Nz), True = solid.
    """
    if len(shape) != 3:
        raise ValueError(f"create_stl_mask is 3D-only, got shape {shape}")
    nx, ny, nz = (int(s) for s in shape)
    if not (float(dx) > 0.0 and np.isfinite(float(dx))):
        raise ValueError(f"dx must be finite and > 0, got {dx}")

    v = np.asarray(vertices_lu, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    origin = np.asarray(origin_lu, dtype=np.float64)
    mask_np = np.zeros((nx, ny, nz), dtype=bool)

    if v.size == 0 or f.size == 0:
        return xp.asarray(mask_np)

    p = (v - origin) / float(dx)
    tri = p[f]      # (T, 3, 3) corner coords
    fid = f.copy()  # (T, 3) corner global ids (canonical edge orientation)

    # ── CCW normalization in xy; drop zero-projected-area triangles ──
    area2 = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
             - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0]))
    flip = area2 < 0.0
    if np.any(flip):
        tri[flip] = tri[flip][:, [0, 2, 1], :]
        fid[flip] = fid[flip][:, [0, 2, 1]]
    keep2d = area2 != 0.0
    tri = tri[keep2d]
    fid = fid[keep2d]
    if tri.shape[0] == 0:
        return xp.asarray(mask_np)

    # ── Integer column bboxes per triangle, clipped to the box ──────
    ix0 = np.maximum(np.ceil(tri[:, :, 0].min(axis=1)), 0.0).astype(np.int64)
    ix1 = np.minimum(np.floor(tri[:, :, 0].max(axis=1)), nx - 1).astype(np.int64)
    iy0 = np.maximum(np.ceil(tri[:, :, 1].min(axis=1)), 0.0).astype(np.int64)
    iy1 = np.minimum(np.floor(tri[:, :, 1].max(axis=1)), ny - 1).astype(np.int64)
    ncx = ix1 - ix0 + 1
    ncy = iy1 - iy0 + 1
    keep = (ncx > 0) & (ncy > 0)
    if not np.any(keep):
        return xp.asarray(mask_np)
    tri, fid = tri[keep], fid[keep]
    ix0, iy0, ncx, ncy = ix0[keep], iy0[keep], ncx[keep], ncy[keep]

    # ── Body-bbox-local counts array (z spill slot at the top) ───────
    bx0 = int(ix0.min())
    bx1 = int((ix0 + ncx - 1).max())
    by0 = int(iy0.min())
    by1 = int((iy0 + ncy - 1).max())
    bz0 = int(np.clip(np.ceil(tri[:, :, 2].min()), 0, nz - 1))
    bz1 = int(np.clip(np.floor(tri[:, :, 2].max()), 0, nz - 1))
    if bz1 < bz0:
        return xp.asarray(mask_np)  # body crosses no cell-center z-plane
    nbz = bz1 - bz0 + 1
    counts = np.zeros((bx1 - bx0 + 1, by1 - by0 + 1, nbz + 1), dtype=np.int32)

    # ── Chunked (column, triangle) pair expansion ────────────────────
    ncols = ncx * ncy
    cum = np.concatenate([[0], np.cumsum(ncols)])
    total = int(cum[-1])
    budget = max(int(pair_budget), int(ncols.max()))
    start_t = 0
    while start_t < tri.shape[0]:
        end_t = int(np.searchsorted(cum, cum[start_t] + budget, side='left'))
        end_t = min(max(end_t, start_t + 1), tri.shape[0])

        t_sl = slice(start_t, end_t)
        rep = ncols[t_sl]
        n_pairs = int(rep.sum())
        tid = np.repeat(np.arange(start_t, end_t, dtype=np.int64), rep)
        off = (np.arange(n_pairs, dtype=np.int64)
               - np.repeat(cum[start_t:end_t] - cum[start_t], rep))
        ci = ix0[tid] + off // ncy[tid]
        cj = iy0[tid] + off % ncy[tid]
        px = ci.astype(np.float64)
        py = cj.astype(np.float64)

        a, b, c = tri[tid, 0], tri[tid, 1], tri[tid, 2]
        ida, idb, idc = fid[tid, 0], fid[tid, 1], fid[tid, 2]
        e_ab, ok_ab = _edge_ok(a, b, ida, idb, px, py)
        e_bc, ok_bc = _edge_ok(b, c, idb, idc, px, py)
        e_ca, ok_ca = _edge_ok(c, a, idc, ida, px, py)
        owned = ok_ab & ok_bc & ok_ca

        if np.any(owned):
            # Barycentric weights = opposite-vertex edge functions (>= 0).
            w0, w1, w2 = e_bc[owned], e_ca[owned], e_ab[owned]
            z_hit = ((w0 * a[owned, 2] + w1 * b[owned, 2] + w2 * c[owned, 2])
                     / (w0 + w1 + w2))
            k = np.ceil(z_hit).astype(np.int64)
            k = np.clip(k, bz0, bz1 + 1)  # below-box -> toggle all; above -> spill
            np.add.at(
                counts,
                (ci[owned] - bx0, cj[owned] - by0, k - bz0),
                1,
            )
        start_t = end_t
    del tid, off, ci, cj, px, py, e_ab, e_bc, e_ca, ok_ab, ok_bc, ok_ca

    # ── Parity: odd cumulative crossings below-or-at the cell ────────
    parity = (np.cumsum(counts[:, :, :nbz], axis=2) & 1).astype(bool)
    mask_np[bx0:bx1 + 1, by0:by1 + 1, bz0:bz1 + 1] = parity
    return xp.asarray(mask_np)


# =============================================================================
# Config-side helpers (bbox + MLG region suggestion)
# =============================================================================

def stl_bbox_l0lu(stl_cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Transformed body bbox (min, max) in L0 lu from an `stl` config block.

    Uses the cached checked loader and the canonical raw->L0lu transform, so
    the bbox is bitwise-consistent with what the mask/q builders will see.
    """
    mesh = load_stl_checked(stl_cfg['file'])
    v_lu = transform_vertices_to_l0lu(
        mesh,
        scale_to_lu=stl_cfg['scale_to_lu'],
        center_lu=stl_cfg['center_lu'],
        rotation_deg=stl_cfg.get('rotation_deg'),
    )
    return v_lu.min(axis=0), v_lu.max(axis=0)


def suggest_fine_region(
    stl_cfg: Dict,
    pad_factor: float = 0.5,
    domain_shape: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, int]:
    """MLG fine-region box around the body with the 0.5·L_body padding rule.

    Region edge <-> body surface distance >= pad_factor * L_body, where
    L_body is the largest bbox extent (= D for a sphere). Returns the
    region dict used by `mlg.levels[k]['region']` (global L0 lu, ints,
    rounded outward), optionally clipped to `domain_shape`.

    Rule provenance: fine-level region edge closer than 0.5 D to the body
    surface couples the interface into the boundary layer and shifts Cd
    non-physically (MLG region padding record).
    """
    bb_min, bb_max = stl_bbox_l0lu(stl_cfg)
    pad = float(pad_factor) * float(np.max(bb_max - bb_min))
    lo = np.floor(bb_min - pad).astype(np.int64)
    hi = np.ceil(bb_max + pad).astype(np.int64)
    if domain_shape is not None:
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(domain_shape, dtype=np.int64) - 1)
    return {
        'x_min': int(lo[0]), 'x_max': int(hi[0]),
        'y_min': int(lo[1]), 'y_max': int(hi[1]),
        'z_min': int(lo[2]), 'z_max': int(hi[2]),
    }
