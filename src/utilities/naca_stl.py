"""Watertight extruded NACA 4-digit (symmetric) wing STL generator.

Produces the canonical geometry input for STL-track airfoil configs:
chord along +x in [0, chord], symmetric about y=0, span along z centered
on 0. Closed trailing edge (a4 = -0.1036 -> y(TE) = 0 exactly), single
welded TE/LE vertices, side surface extruded (triangles parallel to z --
the column-parity voxelizer takes its z-parity from the end caps alone,
the proven box-STL configuration), end caps triangulated as an
upper/lower chordwise strip (no TE-fan slivers).

The mesh is built as one vertex table + index triples, so the exact-bit
weld in stl_mesh reconstructs it losslessly, and check_watertight is run
before anything is written. Files are written atomically (tmp+rename) so
concurrent MPI ranks generating the same missing file cannot interleave.
"""

import os
import tempfile

import numpy as np

from src.boundary.stl_mesh import (
    StlMesh,
    _weld_soup,
    check_watertight,
    signed_volume,
    write_stl_binary,
)


def naca4_symmetric_thickness(x: np.ndarray, thickness: float) -> np.ndarray:
    """Half-thickness y_t(x/c) with the CLOSED-TE coefficient set.

    a4 = -0.1036 makes the coefficient sum exactly 0 at x=1 (sharp TE),
    which the watertight builder relies on (TE welds to one vertex).
    """
    x = np.asarray(x, dtype=np.float64)
    return (thickness / 0.2) * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
        + 0.2843 * x ** 3 - 0.1036 * x ** 4
    )


def _profile(n_profile: int, thickness: float):
    """Closed CCW profile ring + upper/lower index maps for the caps.

    Returns (ring_xy (M,2) f64, iu, il) with M = 2*n_profile; iu(k)/il(k)
    give the ring index of the upper/lower surface point at chord station
    k (k=0 LE, k=n TE; both map to the single shared LE/TE vertex).
    """
    n = int(n_profile)
    k = np.arange(n + 1, dtype=np.float64)
    x = 0.5 * (1.0 - np.cos(np.pi * k / n))          # cosine clustering
    y = naca4_symmetric_thickness(x, thickness)
    y[0] = 0.0                                        # LE exactly on axis
    y[n] = 0.0                                        # closed TE exactly
    # ring: TE -> upper (x decreasing) -> LE -> lower (x increasing), CCW
    ring = np.empty((2 * n, 2), dtype=np.float64)
    ring[0] = (x[n], 0.0)
    ring[1:n] = np.column_stack((x[n - 1:0:-1], y[n - 1:0:-1]))
    ring[n] = (0.0, 0.0)
    ring[n + 1:] = np.column_stack((x[1:n], -y[1:n]))

    def iu(kk):
        return 0 if kk == n else (n if kk == 0 else n - kk)

    def il(kk):
        return 0 if kk == n else (n if kk == 0 else n + kk)

    return ring, iu, il


def build_naca4_wing(chord: float = 1.0,
                     span: float = 0.1,
                     thickness: float = 0.12,
                     n_profile: int = 256) -> StlMesh:
    """Welded, watertight extruded symmetric NACA wing mesh (f64)."""
    n = int(n_profile)
    ring, iu, il = _profile(n, float(thickness))
    ring = ring * float(chord)
    m = ring.shape[0]                                 # 2n ring points

    verts = np.empty((2 * m, 3), dtype=np.float64)
    verts[:m, :2] = ring
    verts[:m, 2] = -0.5 * float(span)                 # bottom ring (z0)
    verts[m:, :2] = ring
    verts[m:, 2] = +0.5 * float(span)                 # top ring (z1)

    faces = []
    for i in range(m):                                # side surface
        j = (i + 1) % m
        faces.append((i, j, m + j))
        faces.append((i, m + j, m + i))
    for k in range(n):                                # caps, chordwise strip
        a, b = iu(k), iu(k + 1)                       # upper k, k+1
        c, d = il(k + 1), il(k)                       # lower k+1, k
        for tri in ((a, d, c), (a, c, b)):            # top cap: +z normal
            if len(set(tri)) == 3:
                faces.append(tuple(m + t for t in tri))
        for tri in ((a, c, d), (a, b, c)):            # bottom cap: -z normal
            if len(set(tri)) == 3:
                faces.append(tri)

    soup = verts[np.asarray(faces, dtype=np.int64)].reshape(-1, 3)
    mesh = _weld_soup(
        soup,
        f"<naca00{round(thickness * 100):02d} wing c={chord} s={span} "
        f"n={n}>",
    )
    check_watertight(mesh)
    vol = signed_volume(mesh)
    if vol <= 0.0:
        raise ValueError(f"NACA wing generator: signed volume {vol} <= 0 "
                         "(winding bug)")
    return mesh


def ensure_naca4_wing_stl(path: str,
                          chord: float = 1.0,
                          span: float = 0.1,
                          thickness: float = 0.12,
                          n_profile: int = 256) -> str:
    """Generate `path` if missing (atomic write; MPI-rank safe). Returns path."""
    if os.path.exists(path):
        return path
    mesh = build_naca4_wing(chord, span, thickness, n_profile)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".stl",
                               dir=os.path.dirname(path))
    os.close(fd)
    try:
        write_stl_binary(mesh, tmp)
        os.replace(tmp, path)                         # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path
