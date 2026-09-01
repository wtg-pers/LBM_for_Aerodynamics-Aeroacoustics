"""Per-triangle flow-direction normal curvature + kappa_n*h channel mask.

Runtime home of the robin/11 wall-pressure channel criterion (robin/13b):
the surface writer outputs, per STL triangle,
    p_facet  wall-attached sample (h = sample_h, inside the modeled layer)
    p_state_ph  raw outer sample (h = p_sample_h, outside the layer)
    p_sknh   the SELECTED channel: outer where kappa_n(flow)*h >= kh_star,
             wall-attached elsewhere (robin/11 K4: beats both fixed
             conventions and the ROBIN station rule on all five runs).
kappa_n(flow) = normal curvature along the freestream direction projected
onto the tangent plane (Euler's theorem), from a local quadric fit at the
triangle centres. All geometry here is the WRITER's mesh (already in the
global L0-lu frame, rotation applied), so the freestream is the lab-frame
flow direction (+x for every registered config) and kappa*h is evaluated
in lu x 1/lu = dimensionless, identical to the offline robin/11 value.

The fit REQUIRES the intercept term (robin/11 K0: a point sitting off the
discrete surface aliases 2f/r^2 into the quadratic terms). Fit radius
defaults to 2.2 x the median mesh edge length -- for robin_mod_v1c this is
0.012 R, the radius validated against the analytic genROBIN surface
(|k|max rel err median 1.0 %, p90 7.6 %).

numpy + scipy only (no trimesh in the solver path). Python 3.9-safe.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: robin/11 K4: pooled-logistic switch point; LORO 0.0029-0.0035; the
#: ALL-rms plateau spans 0.002-0.008 (not finely tuned). Stack constant —
#: re-estimate if the wall model or the numerical-layer thickness changes.
KH_STAR_DEFAULT = 0.0034

_MASK_CACHE = {}     # (fingerprint) -> (mask, kh); survives shell writers


def _mesh_arrays(verts: np.ndarray, faces: np.ndarray):
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    cen = tri.mean(axis=1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(n, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.maximum(a2, 1e-300)[:, None]
    # orient outward (closed-body heuristic, same as the offline tool)
    cb = cen.mean(axis=0)
    flip = ((cen - cb) * n).sum(axis=1) < 0.0
    n[flip] *= -1.0
    edges = np.concatenate([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1],
                            tri[:, 0] - tri[:, 2]])
    med_edge = float(np.median(np.linalg.norm(edges, axis=1)))
    return v, f, cen, n, 0.5 * a2, med_edge


def quadric_curvature(verts: np.ndarray, faces: np.ndarray,
                      points: Optional[np.ndarray] = None,
                      radius: Optional[float] = None,
                      min_pts: int = 12) -> Dict[str, np.ndarray]:
    """Principal curvatures/directions at `points` (default: face centres)
    from a local quadric fit z = f + dx + ey + ax^2 + bxy + cy^2 in the
    frame of the area-weighted outward normal. Convex positive. Returns
    dict(k1, k2, dir1, dir2, normal, n_pts); NaN where the fit degenerates.
    Units follow the mesh (curvature = 1/mesh-unit)."""
    from scipy.spatial import cKDTree
    v, f, cen, fn, fa, med_edge = _mesh_arrays(verts, faces)
    if radius is None:
        radius = 2.2 * med_edge
    pts = cen if points is None else np.asarray(points, dtype=np.float64)
    vt = cKDTree(v)
    ft = cKDTree(cen)
    m = len(pts)
    out = {"k1": np.full(m, np.nan), "k2": np.full(m, np.nan),
           "dir1": np.full((m, 3), np.nan), "dir2": np.full((m, 3), np.nan),
           "normal": np.full((m, 3), np.nan), "n_pts": np.zeros(m, dtype=int)}
    vidx = vt.query_ball_point(pts, radius)
    fidx = ft.query_ball_point(pts, radius)
    for i in range(m):
        vi, fi = vidx[i], fidx[i]
        r = radius
        while len(vi) < min_pts and r < 4.0 * radius:
            r *= 1.5
            vi = vt.query_ball_point(pts[i], r)
            fi = ft.query_ball_point(pts[i], r)
        if len(vi) < 6 or not fi:
            continue
        n = (fn[fi] * fa[fi, None]).sum(axis=0)
        nn = np.linalg.norm(n)
        if nn == 0.0:
            continue
        n /= nn
        t1 = np.cross(n, [0.0, 0.0, 1.0])
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(n, [0.0, 1.0, 0.0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        Q = v[vi] - pts[i]
        x, y, z = Q @ t1, Q @ t2, Q @ n
        A = np.column_stack([np.ones_like(x), x, y, x ** 2, x * y, y ** 2])
        try:
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        except np.linalg.LinAlgError:
            continue
        _, d, e, a, b, c = coef
        g = np.array([[1 + d * d, d * e], [d * e, 1 + e * e]])
        II = (np.array([[2 * a, b], [b, 2 * c]])
              / np.sqrt(1 + d * d + e * e))
        w, V = np.linalg.eig(np.linalg.solve(g, II))
        k = -w.real                    # outward z: convex -> negative Hessian
        dirs = np.stack([V[0, j].real * t1 + V[1, j].real * t2
                         for j in range(2)])
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-300)
        s = np.argsort(k)[::-1]
        out["k1"][i], out["k2"][i] = k[s]
        out["dir1"][i], out["dir2"][i] = dirs[s[0]], dirs[s[1]]
        out["normal"][i] = n
        out["n_pts"][i] = len(vi)
    return out


def kappa_n_direction(curv: Dict[str, np.ndarray],
                      u_dir: Sequence[float]) -> np.ndarray:
    """Normal curvature along u_dir projected on the tangent plane
    (Euler: kn = k1 cos^2 + k2 sin^2). Signed, convex positive; NaN where
    degenerate (u_dir ~ normal, or failed fit)."""
    n = curv["normal"]
    u = np.broadcast_to(np.asarray(u_dir, dtype=np.float64), n.shape)
    t = u - (u * n).sum(axis=1, keepdims=True) * n
    tl = np.linalg.norm(t, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = t / tl[:, None]
        c1 = np.abs((t * curv["dir1"]).sum(axis=1))
    kn = curv["k1"] * c1 ** 2 + curv["k2"] * (1.0 - c1 ** 2)
    return np.where(tl > 1e-6, kn, np.nan)


def sknh_mask(verts: np.ndarray, faces: np.ndarray, h_lu: float,
              kh_star: float = KH_STAR_DEFAULT,
              flow_dir: Sequence[float] = (1.0, 0.0, 0.0)
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-triangle boolean mask: True -> take the outer (p_sample_h)
    channel. kh = kappa_n(flow)*h_lu; NaN kappa (failed fit, stagnation-
    degenerate projection, creases) -> False = wall-attached channel
    (conservative). Cached module-wide (cheap fingerprint) so shell
    writers recreated per step do not refit ~1e5 triangles each write."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    key = (v.shape[0], f.shape[0], float(v[0, 0]), float(v[-1, -1]),
           round(float(v.sum()), 6), float(h_lu), float(kh_star),
           tuple(float(c) for c in flow_dir))
    hit = _MASK_CACHE.get(key)
    if hit is not None:
        return hit
    curv = quadric_curvature(v, f)
    kn = kappa_n_direction(curv, flow_dir)
    kh = kn * float(h_lu)
    mask = np.where(np.isfinite(kh), kh >= float(kh_star), False)
    _MASK_CACHE[key] = (mask, kh)
    return mask, kh
