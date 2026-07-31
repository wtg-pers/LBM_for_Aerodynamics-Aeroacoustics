"""STL mesh loading, validation, and transformation (STL body track, S1).

Provides:
    StlMesh                       welded triangle mesh (vertices f64, faces i64)
    load_stl()                    binary/ASCII STL reader (file normals discarded)
    check_watertight()            edge-sharing diagnosis -> hard ValueError
    load_stl_checked()            load + watertight check, (mtime_ns, size) cache
    transform_vertices_to_l0lu()  raw -> L0 lattice-unit single-path transform
    generate_icosphere()          inscribed icosphere (validation reference)
    write_stl_binary()            binary STL writer (tests / roundtrip)
    signed_volume()               divergence-theorem volume (winding check)

Design decisions (patch_notes/stl_body/PLAN.md, user-approved):
    - numpy only, no new dependencies. A dirty STL is rejected with a clear
      diagnostic error; this module never repairs geometry.
    - Binary detection trusts only ``filesize == 84 + 50 * n_triangles``.
      The "solid" prefix is unreliable (many binary exporters write it).
    - File normals are discarded: column-parity voxelization and
      Moller-Trumbore ray-triangle q are winding-independent. Winding
      consistency is still enforced by check_watertight() (directed-edge
      duplicates), because signed_volume() and outward-orientation
      diagnostics rely on it.
    - Vertex weld is an exact bitwise ``np.unique(axis=0)``: binary STL
      repeats shared vertices with identical f32 bit patterns. No tolerance
      re-weld fallback -- a mesh that does not weld exactly fails the
      watertight check and is rejected.
    - All consumers (config validate, L0 build, fine-level rebuilds) must go
      through load_stl_checked(); cached meshes are returned with read-only
      arrays so shared state cannot be mutated.

Author: LBM Development Team
Date: 2026-07
"""

import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Binary STL record: normal (3 f32), 3 vertices (3 f32 each), attribute (u2).
_BINARY_RECORD = np.dtype([
    ("normal", "<f4", (3,)),
    ("verts", "<f4", (3, 3)),
    ("attr", "<u2"),
])
assert _BINARY_RECORD.itemsize == 50

_HEADER_BYTES = 80


@dataclass
class StlMesh:
    """Welded triangle mesh.

    vertices : (n_v, 3) float64, bitwise-unique rows (lexicographically sorted)
    faces    : (n_f, 3) int64 indices into vertices, file face order preserved
    n_dropped_degenerate : faces removed because >=2 welded indices coincided
    source   : file path or "<generated>" tag, for diagnostics only
    """

    vertices: np.ndarray
    faces: np.ndarray
    n_dropped_degenerate: int
    source: str

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def _weld_soup(soup: np.ndarray, source: str) -> StlMesh:
    """Canonicalize a (3*n_f, 3) f64 triangle soup into a welded StlMesh."""
    if soup.size == 0:
        raise ValueError(f"STL '{source}': contains no triangles")
    if not np.isfinite(soup).all():
        raise ValueError(f"STL '{source}': non-finite vertex coordinates")

    vertices, inverse = np.unique(soup, axis=0, return_inverse=True)
    faces = np.asarray(inverse).ravel().reshape(-1, 3).astype(np.int64)

    degenerate = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    n_dropped = int(degenerate.sum())
    if n_dropped:
        faces = faces[~degenerate]
    if faces.shape[0] == 0:
        raise ValueError(f"STL '{source}': all {n_dropped} faces degenerate")

    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces)
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return StlMesh(vertices, faces, n_dropped, source)


def _parse_ascii(data: bytes, source: str) -> np.ndarray:
    """Collect 'vertex x y z' tokens from an ASCII STL -> (3*n_f, 3) f64."""
    text = data.decode("utf-8", errors="replace")
    coords: List[Tuple[float, float, float]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts or parts[0].lower() != "vertex":
            continue
        if len(parts) != 4:
            raise ValueError(
                f"STL '{source}' line {lineno}: expected 'vertex x y z', "
                f"got {line.strip()!r}"
            )
        try:
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError as exc:
            raise ValueError(
                f"STL '{source}' line {lineno}: bad vertex float: {exc}"
            ) from None
    if not coords:
        raise ValueError(
            f"STL '{source}': no vertex data found -- neither a valid binary "
            f"STL (size != 84 + 50*n) nor a parsable ASCII STL"
        )
    if len(coords) % 3 != 0:
        raise ValueError(
            f"STL '{source}': {len(coords)} vertices is not a multiple of 3"
        )
    return np.asarray(coords, dtype=np.float64)


def load_stl(path: str) -> StlMesh:
    """Load a binary or ASCII STL file into a welded StlMesh.

    Binary detection: ``filesize == 84 + 50 * count`` with count read from
    bytes 80:84. Anything else is parsed as ASCII. File normals are discarded.
    """
    with open(path, "rb") as fh:
        data = fh.read()

    if len(data) >= _HEADER_BYTES + 4:
        (count,) = struct.unpack_from("<I", data, _HEADER_BYTES)
        if count > 0 and len(data) == _HEADER_BYTES + 4 + 50 * count:
            tris = np.frombuffer(
                data, dtype=_BINARY_RECORD, count=count, offset=_HEADER_BYTES + 4
            )
            soup = tris["verts"].reshape(-1, 3).astype(np.float64)
            return _weld_soup(soup, path)

    return _weld_soup(_parse_ascii(data, path), path)


def check_watertight(mesh: StlMesh) -> None:
    """Raise ValueError with diagnostic counts unless the mesh is watertight.

    Watertight == every undirected edge is shared by exactly 2 faces AND no
    directed edge appears twice (consistent winding). Diagnoses:
        undirected count == 1  -> hole (boundary edge)
        undirected count  > 2  -> non-manifold edge
        directed  count   > 1  -> inconsistent winding (flipped/duplicated face)
    """
    faces = mesh.faces
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    _, dir_counts = np.unique(edges, axis=0, return_counts=True)
    n_winding = int((dir_counts > 1).sum())

    undirected = np.sort(edges, axis=1)
    _, und_counts = np.unique(undirected, axis=0, return_counts=True)
    n_hole = int((und_counts == 1).sum())
    n_nonmanifold = int((und_counts > 2).sum())

    if n_hole or n_nonmanifold or n_winding:
        raise ValueError(
            f"STL mesh '{mesh.source}' is not watertight "
            f"({mesh.n_vertices} vertices, {mesh.n_faces} faces, "
            f"{mesh.n_dropped_degenerate} degenerate dropped): "
            f"{n_hole} hole edge(s) (shared by 1 face), "
            f"{n_nonmanifold} non-manifold edge(s) (shared by >2 faces), "
            f"{n_winding} directed edge(s) with inconsistent winding. "
            f"Repair the mesh externally; this loader never repairs."
        )


# Per-path single-slot cache: abspath -> ((mtime_ns, size), mesh).
_MESH_CACHE: Dict[str, Tuple[Tuple[int, int], StlMesh]] = {}


def load_stl_checked(path: str) -> StlMesh:
    """load_stl + check_watertight with an (abspath, mtime_ns, size) cache.

    This is the single entry point for all consumers (validate, L0 build,
    fine-level rebuilds) -- do not bypass the cache. Returned arrays are
    read-only; treat the mesh as immutable shared state.
    """
    abspath = os.path.abspath(path)
    st = os.stat(abspath)
    sig = (st.st_mtime_ns, st.st_size)
    hit = _MESH_CACHE.get(abspath)
    if hit is not None and hit[0] == sig:
        return hit[1]
    mesh = load_stl(abspath)
    check_watertight(mesh)
    _MESH_CACHE[abspath] = (sig, mesh)
    return mesh


def clear_stl_mesh_cache() -> None:
    """Drop all cached meshes (tests / long-lived driver hygiene)."""
    _MESH_CACHE.clear()


def _rotation_matrix(rotation_deg: Sequence[float]) -> np.ndarray:
    """Fixed-axis X -> Y -> Z rotation, i.e. R = Rz @ Ry @ Rx (column vectors)."""
    rx, ry, rz = (np.deg2rad(float(a)) for a in rotation_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    r_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return r_z @ r_y @ r_x


def transform_vertices_to_l0lu(
    mesh: StlMesh,
    scale_to_lu: float,
    center_lu: Sequence[float],
    rotation_deg: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Raw mesh vertices -> L0 lattice units. The single canonical path.

    scale -> rotate (Rz@Ry@Rx about the scaled-bbox center) -> translate the
    rotated bbox center onto center_lu. Every consumer (mask, q, outline,
    fine levels) must re-derive from this function; never chain lu -> lu
    transforms. Returns a new (n_v, 3) float64 array.
    """
    scale = float(scale_to_lu)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale_to_lu must be finite and > 0, got {scale_to_lu}")
    center = np.asarray(center_lu, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"center_lu must have 3 components, got {center_lu!r}")

    v = mesh.vertices * scale
    if rotation_deg is not None:
        rot = np.asarray(rotation_deg, dtype=np.float64)
        if rot.shape != (3,):
            raise ValueError(
                f"rotation_deg must have 3 components, got {rotation_deg!r}"
            )
        if np.any(rot != 0.0):
            pivot = 0.5 * (v.min(axis=0) + v.max(axis=0))
            v = (v - pivot) @ _rotation_matrix(rot).T + pivot

    bbox_center = 0.5 * (v.min(axis=0) + v.max(axis=0))
    return v + (center - bbox_center)


# Canonical icosahedron (unit-sphere projected), CCW-outward winding.
_ICO_T = (1.0 + np.sqrt(5.0)) / 2.0
_ICO_VERTS = np.array([
    (-1.0, _ICO_T, 0.0), (1.0, _ICO_T, 0.0),
    (-1.0, -_ICO_T, 0.0), (1.0, -_ICO_T, 0.0),
    (0.0, -1.0, _ICO_T), (0.0, 1.0, _ICO_T),
    (0.0, -1.0, -_ICO_T), (0.0, 1.0, -_ICO_T),
    (_ICO_T, 0.0, -1.0), (_ICO_T, 0.0, 1.0),
    (-_ICO_T, 0.0, -1.0), (-_ICO_T, 0.0, 1.0),
], dtype=np.float64) / np.sqrt(1.0 + _ICO_T * _ICO_T)
_ICO_FACES = [
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
]


def generate_icosphere(
    radius: float,
    center: Sequence[float],
    subdivisions: int,
) -> StlMesh:
    """Inscribed icosphere: every vertex projected onto the sphere.

    Chords lie strictly inside, so mask(icosphere STL) is a subset of
    mask(analytic sphere) -- the one-sided containment used by the S2/S3
    parity validation (chord error e_s ~ 0.153 R / 4**s).
    """
    if float(radius) <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if int(subdivisions) < 0:
        raise ValueError(f"subdivisions must be >= 0, got {subdivisions}")

    verts: List[np.ndarray] = [v.copy() for v in _ICO_VERTS]
    faces: List[Tuple[int, int, int]] = list(_ICO_FACES)

    for _ in range(int(subdivisions)):
        midpoint_cache: Dict[Tuple[int, int], int] = {}

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            k = midpoint_cache.get(key)
            if k is None:
                m = verts[i] + verts[j]
                m /= np.linalg.norm(m)
                verts.append(m)
                k = len(verts) - 1
                midpoint_cache[key] = k
            return k

        new_faces: List[Tuple[int, int, int]] = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces

    vertices = np.asarray(verts, dtype=np.float64) * float(radius)
    vertices += np.asarray(center, dtype=np.float64)
    soup = vertices[np.asarray(faces, dtype=np.int64)].reshape(-1, 3)
    return _weld_soup(
        soup, f"<icosphere r={radius} s={int(subdivisions)}>"
    )


def write_stl_binary(mesh: StlMesh, path: str) -> None:
    """Write a binary STL (f32). Normals recomputed from winding; attr = 0."""
    soup = np.ascontiguousarray(
        mesh.vertices[mesh.faces], dtype=np.float32
    )  # (n_f, 3, 3)
    e1 = soup[:, 1, :].astype(np.float64) - soup[:, 0, :].astype(np.float64)
    e2 = soup[:, 2, :].astype(np.float64) - soup[:, 0, :].astype(np.float64)
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    np.divide(normals, norms, out=normals, where=norms > 0.0)

    records = np.zeros(mesh.n_faces, dtype=_BINARY_RECORD)
    records["normal"] = normals.astype(np.float32)
    records["verts"] = soup
    header = b"binary STL written by src/boundary/stl_mesh.py"
    header = header[:_HEADER_BYTES].ljust(_HEADER_BYTES, b"\x00")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", mesh.n_faces))
        fh.write(records.tobytes())


def signed_volume(mesh: StlMesh) -> float:
    """Divergence-theorem volume: positive for outward (CCW) winding."""
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
