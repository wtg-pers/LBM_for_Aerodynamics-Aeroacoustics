"""Surface-field output on the SURFEL geometry — surfel track S8 (output).

Writes the body-resolved surface load as legacy-VTK POLYDATA whose cells are
the actual clipped surfel polygons (or, aggregated, the original STL
triangles) -- not the staircase voxel faces.

Why this is an extraction and not a reconstruction: Eq. (9) of [C1] is the
momentum flux THROUGH a facet, so the per-facet load is what the algorithm
computes; the total force is the sum of it. A link/staircase scheme has the
opposite structure -- link exchanges are primitive and any surface
distribution is a re-binning onto the geometry, which carries the lattice
directions with it.

PRESSURE (measured, patch 07/08). Eq. (23) reads F = (p + dp) n, and that
holds here to 2.4e-16, so there are two surface pressures and they differ by
the spurious mass-fix offset:
    `p`        from the normal traction  -> carries dp
    `p_state`  = rho^a theta from the facet state (Eq. 12) -> does not
    `p_use`    the right one per mode, and what Cp is built from
    mode="noslip"    dp == 0 identically (Eq. 10 needs no mass fix), so the
                     two agree; p_use = p.
    free-slip / wall-model
                     dp is NOT constant -- on a sphere its variation is ~79%
                     of the pressure signal (std/q 1.53 vs 1.94) and
                     correlates -0.91 with the near-wall normal velocity --
                     so p_use = p_state (std/q 0.49). `dp` is always written
                     as its own field so the affected quantity stays visible.
Wall shear is unaffected in every mode: the mass fix moves the facet force
only along n (gate s0 [M], dF.t = 2.4e-16).

Author: LBM Development Team
Date: 2026-08 (surfel track, surface output)
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_HEADER = "# vtk DataFile Version 3.0\n{title}\nASCII\nDATASET POLYDATA\n"


def _scalar_block(name, arr):
    out = [f"SCALARS {name} float 1", "LOOKUP_TABLE default"]
    out += ["%.7g" % v for v in np.asarray(arr, dtype=np.float64).ravel()]
    return "\n".join(out) + "\n"


def _vector_block(name, arr):
    a = np.asarray(arr, dtype=np.float64)
    out = [f"VECTORS {name} float"]
    out += ["%.7g %.7g %.7g" % tuple(r) for r in a]
    return "\n".join(out) + "\n"


def facet_surface_fields(
    facets,
    *,
    q_inf: Optional[float] = None,
    p_ref: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Per-facet surface fields from the last facet pass.

    Adds Cp = (p - p_ref)/q_inf and Cf = |tau|/q_inf when both are given.
    """
    t = facets.facet_traction()
    out = {'p': t['p'], 'p_state': t['p_state'], 'p_use': t['p_use'],
           'dp': t['dp'], 'tau_mag': t['tau_mag'], 'area': facets.area,
           'tau': t['tau'], 'traction': t['traction'],
           'normal': facets.normal}
    if q_inf is not None:
        # Cp is built from p_use: the normal traction carries the spurious
        # Eq. (24) dp in the slip modes (see the module doc)
        pu = t['p_use']
        pr = float(np.average(pu, weights=facets.area)) \
            if p_ref is None else float(p_ref)
        out['Cp'] = (pu - pr) / float(q_inf)
        out['Cf'] = t['tau_mag'] / float(q_inf)
    return out


def write_surfel_surface(
    path: str,
    facets,
    surfels: Dict[str, np.ndarray],
    fields: Dict[str, np.ndarray],
    *,
    origin_lu: Sequence[float] = (0.0, 0.0, 0.0),
    dx: float = 1.0,
    area_min: float = 0.0,
    title: str = "surfel surface",
) -> int:
    """Write the clipped surfel polygons with per-facet cell data.

    `surfels` must come from `build_surfels(..., return_polygons=True)`.
    Facets with area <= area_min are dropped (slivers carry no load but do
    make per-unit-area fields noisy). Returns the number of cells written.
    """
    if 'poly' not in surfels:
        raise ValueError("write_surfel_surface needs the polygons; build "
                         "with build_surfels(..., return_polygons=True)")
    keep = np.nonzero(surfels['area'] > float(area_min))[0]
    poly, nv = surfels['poly'][keep], surfels['nvert'][keep]
    org = np.asarray(origin_lu, dtype=np.float64)

    pts, conn, base = [], [], 0
    for r in range(keep.size):
        n = int(nv[r])
        pts.append(poly[r, :n, :] * float(dx) + org)
        conn.append([n] + list(range(base, base + n)))
        base += n
    P = np.concatenate(pts) if pts else np.zeros((0, 3))

    body = [_HEADER.format(title=title)]
    body.append("POINTS %d float\n" % P.shape[0])
    body.append("\n".join("%.7g %.7g %.7g" % tuple(p) for p in P) + "\n")
    body.append("POLYGONS %d %d\n" % (keep.size, sum(len(c) for c in conn)))
    body.append("\n".join(" ".join(str(v) for v in c) for c in conn) + "\n")
    body.append("CELL_DATA %d\n" % keep.size)
    for k, v in fields.items():
        a = np.asarray(v)
        if a.ndim == 1:
            body.append(_scalar_block(k, a[keep]))
        elif a.ndim == 2 and a.shape[1] == 3:
            body.append(_vector_block(k, a[keep]))
    with open(path, "w") as fh:
        fh.write("".join(body))
    return int(keep.size)


def aggregate_to_triangles(
    surfels: Dict[str, np.ndarray],
    fields: Dict[str, np.ndarray],
    n_faces: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Area-weighted aggregation of per-facet fields onto the STL triangles.

    The natural output unit: a triangle's surfels are its pieces in different
    voxels, so the area-weighted mean is the load on the ORIGINAL geometry,
    and it is immune to the sliver noise of per-unit-area fields.
    Returns (triangle area, aggregated fields); triangles with no surfel get
    zero area and zero fields.
    """
    tid = surfels['tri_id']
    w = surfels['area']
    a_tri = np.bincount(tid, w, n_faces)
    inv = 1.0 / np.maximum(a_tri, 1e-300)
    out = {}
    for k, v in fields.items():
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 1:
            out[k] = np.bincount(tid, w * a, n_faces) * inv
        elif a.ndim == 2 and a.shape[1] == 3:
            out[k] = np.stack(
                [np.bincount(tid, w * a[:, c], n_faces) * inv
                 for c in range(3)], axis=1)
    return a_tri, out


def finalize_surface_channels(
    fields: Dict[str, np.ndarray],
    verts_lu: np.ndarray,
    faces: np.ndarray,
    h_lu: Optional[float],
    kh_star: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """robin/13b surface output convention (applied per STL triangle, after
    aggregation/merge, at every write path):

        p_facet / Cp_facet   wall-attached channel (ex p_use / Cp; sampled
                             at sample_h inside the modeled layer)
        p_state_ph / Cp_ph   RAW outer channel (p_sample_h) — kept so the
                             kh* selection stays re-derivable offline
                             (kh* is a stack constant; robin/10b lesson:
                             never destroy a raw channel)
        p_sknh / Cp_sknh     the DEFAULT readout: kappa_n(flow)*h channel
                             selection (robin/11 K4), outer where
                             kh >= kh_star, wall-attached elsewhere;
                             NaN-kappa / NaN-ph triangles fall back to the
                             wall-attached channel.

    h_lu = p_sample_h in L0 lu (None or missing ph channel -> p_sknh is a
    copy of p_facet: legacy configs stay two-name-richer but value-equal).
    """
    out = dict(fields)
    if 'p_use' in out:
        out['p_facet'] = out.pop('p_use')
    if 'Cp' in out:
        out['Cp_facet'] = out.pop('Cp')
    pf = out.get('p_facet')
    if pf is None:
        return out
    pph = out.get('p_state_ph')
    sel = None
    if h_lu is not None and pph is not None and np.isfinite(pph).any():
        from src.boundary.surfel_kappa import KH_STAR_DEFAULT, sknh_mask
        ks = KH_STAR_DEFAULT if kh_star is None else float(kh_star)
        mask, kh = sknh_mask(verts_lu, faces, float(h_lu), ks)
        sel = mask & np.isfinite(np.asarray(pph))
    if sel is None:
        out['p_sknh'] = np.asarray(pf).copy()
        if 'Cp_facet' in out:
            out['Cp_sknh'] = np.asarray(out['Cp_facet']).copy()
        out['sknh_sel'] = np.zeros(len(np.asarray(pf)))
    else:
        out['p_sknh'] = np.where(sel, pph, pf)
        if 'Cp_facet' in out and 'Cp_ph' in out:
            out['Cp_sknh'] = np.where(sel, out['Cp_ph'], out['Cp_facet'])
        # audit channels (13b): the mask actually applied + kappa_n*h per
        # triangle — the selection is exactly reproducible (and kh* is
        # re-tunable) from the FILE alone; offline refits from the %.7g
        # vtk coordinates flip threshold-adjacent / sliver triangles.
        out['sknh_sel'] = sel.astype(np.float64)
        out['kh_flow'] = kh
    return out


def write_triangle_surface(
    path: str,
    triangles_lu: Tuple[np.ndarray, np.ndarray],
    a_tri: np.ndarray,
    fields: Dict[str, np.ndarray],
    *,
    origin_lu: Sequence[float] = (0.0, 0.0, 0.0),
    dx: float = 1.0,
    title: str = "surfel surface (per STL triangle)",
) -> int:
    """Write the ORIGINAL STL triangles with the aggregated cell data."""
    v, f = np.asarray(triangles_lu[0]), np.asarray(triangles_lu[1])
    keep = np.nonzero(a_tri > 0.0)[0]
    P = v * float(dx) + np.asarray(origin_lu, dtype=np.float64)
    body = [_HEADER.format(title=title)]
    body.append("POINTS %d float\n" % P.shape[0])
    body.append("\n".join("%.7g %.7g %.7g" % tuple(p) for p in P) + "\n")
    body.append("POLYGONS %d %d\n" % (keep.size, 4 * keep.size))
    body.append("\n".join("3 %d %d %d" % tuple(t) for t in f[keep]) + "\n")
    body.append("CELL_DATA %d\n" % keep.size)
    body.append(_scalar_block("area", a_tri[keep]))
    for k, val in fields.items():
        a = np.asarray(val)
        if a.ndim == 1:
            body.append(_scalar_block(k, a[keep]))
        elif a.ndim == 2 and a.shape[1] == 3:
            body.append(_vector_block(k, a[keep]))
    with open(path, "w") as fh:
        fh.write("".join(body))
    return int(keep.size)
