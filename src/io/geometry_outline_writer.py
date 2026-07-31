"""Geometry outline writer — emit obstacle silhouette as a VTK PolyData
file so ParaView can overlay the true geometry on the (staircased)
solid_mask field.

Output is a legacy VTK ASCII PolyData file at L0 lattice coordinates.
MLG VTK writer registers each level with origin/spacing in L0 lu
(`mlg_vtk_writer_2d.py:85-128`), so the same outline file overlays
all levels correctly without rescaling.

Supported geometry types:
    circle / cylinder (2D)  → closed polyline sampled around the circle
    airfoil                 → polygon_lu polyline closed
    box (2D)                → 4-corner rectangle
    stl (3D)                → triangle surface as POLYGONS (true geometry
                              overlay on the staircased mask)

Other 3D obstacles are skipped with a warning (silhouette of curved 3D
surfaces is non-trivial; ParaView is fine with raw mask there).
"""

from __future__ import annotations

import os
import math
from typing import Dict, Any, List, Tuple


def _format_polydata_ascii(
    points_xyz: List[Tuple[float, float, float]],
    polyline_indices: List[int],
    title: str = "Geometry outline",
) -> str:
    """Build a legacy VTK PolyData ASCII string.

    polyline_indices: ordered vertex indices into points_xyz, with the
    first index repeated at the end if the contour is closed.
    """
    n_pts = len(points_xyz)
    n_idx = len(polyline_indices)

    lines: List[str] = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {n_pts} float",
    ]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points_xyz)
    lines.append("")
    # LINES <n_polylines> <total_size>; total_size = (count + count_n)
    # for one polyline:  total_size = 1 + n_idx
    lines.append(f"LINES 1 {1 + n_idx}")
    lines.append(f"{n_idx} " + " ".join(str(i) for i in polyline_indices))
    return "\n".join(lines) + "\n"


def _circle_polyline(
    center: Tuple[float, float],
    radius: float,
    num_samples: int = 128,
) -> Tuple[List[Tuple[float, float, float]], List[int]]:
    """Closed polyline around a circle in the z=0 plane (L0 lu)."""
    cx, cy = float(center[0]), float(center[1])
    R = float(radius)
    pts: List[Tuple[float, float, float]] = []
    for k in range(num_samples):
        theta = 2.0 * math.pi * k / num_samples
        pts.append((cx + R * math.cos(theta), cy + R * math.sin(theta), 0.0))
    indices = list(range(num_samples)) + [0]   # close the loop
    return pts, indices


def _airfoil_polyline(
    x_poly, y_poly,
) -> Tuple[List[Tuple[float, float, float]], List[int]]:
    """Closed polyline from polygon_lu vertices (z=0)."""
    xs = [float(x) for x in (x_poly.tolist() if hasattr(x_poly, "tolist") else x_poly)]
    ys = [float(y) for y in (y_poly.tolist() if hasattr(y_poly, "tolist") else y_poly)]
    if len(xs) < 3:
        raise ValueError("airfoil polygon must have ≥ 3 vertices")
    # Drop duplicate trailing point if already closed
    if math.isclose(xs[0], xs[-1]) and math.isclose(ys[0], ys[-1]):
        xs = xs[:-1]
        ys = ys[:-1]
    pts = [(x, y, 0.0) for x, y in zip(xs, ys)]
    indices = list(range(len(xs))) + [0]
    return pts, indices


def _format_polydata_triangles(
    vertices,
    faces,
    title: str = "Geometry surface",
) -> str:
    """Legacy VTK PolyData ASCII with POLYGONS (triangle surface).

    vertices: (n_v, 3) array-like, L0 lu. faces: (n_f, 3) int indices.
    """
    n_pts = len(vertices)
    n_tri = len(faces)
    lines: List[str] = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {n_pts} float",
    ]
    lines.extend(
        f"{float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}"
        for v in vertices
    )
    lines.append("")
    lines.append(f"POLYGONS {n_tri} {4 * n_tri}")
    lines.extend(
        f"3 {int(f[0])} {int(f[1])} {int(f[2])}" for f in faces
    )
    return "\n".join(lines) + "\n"


def _box_polyline(
    corner_min: Tuple[float, float],
    corner_max: Tuple[float, float],
) -> Tuple[List[Tuple[float, float, float]], List[int]]:
    x0, y0 = float(corner_min[0]), float(corner_min[1])
    x1, y1 = float(corner_max[0]), float(corner_max[1])
    pts = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
    indices = [0, 1, 2, 3, 0]
    return pts, indices


def write_geometry_outline(
    geom_info: Dict[str, Any],
    output_path: str,
    num_circle_samples: int = 128,
    verbose: bool = True,
) -> bool:
    """Write the obstacle silhouette to `output_path` as a VTK PolyData.

    Args:
        geom_info: dict returned by `create_geometry_mask`.
        output_path: target file (will be created/overwritten).
        num_circle_samples: vertex count for circle/cylinder sampling.
        verbose: print one-line status.

    Returns:
        True on success, False if the geometry type is unsupported / 3D.
    """
    gtype = geom_info.get("type", "none")

    if gtype in ("circle", "cylinder") and "radius" in geom_info:
        # 2D circle (cylinder in 2D maps to circle).
        center = geom_info["center"]
        if len(center) != 2:
            if verbose:
                print(f"  [outline] {gtype} is 3D — outline skipped")
            return False
        pts, idx = _circle_polyline(center, geom_info["radius"],
                                    num_circle_samples)
        title = f"Geometry outline ({gtype})"

    elif gtype == "airfoil" and "polygon_lu" in geom_info:
        x_poly, y_poly = geom_info["polygon_lu"]
        pts, idx = _airfoil_polyline(x_poly, y_poly)
        title = "Geometry outline (airfoil)"

    elif gtype == "box":
        cmin = geom_info["corner_min"]
        cmax = geom_info["corner_max"]
        if len(cmin) != 2:
            if verbose:
                print(f"  [outline] box is 3D — outline skipped")
            return False
        pts, idx = _box_polyline(cmin, cmax)
        title = "Geometry outline (box)"

    elif gtype == "stl" and "triangles_lu" in geom_info:
        vertices_lu, faces = geom_info["triangles_lu"]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        body = _format_polydata_triangles(
            vertices_lu, faces, title="Geometry surface (stl)"
        )
        with open(output_path, "w") as f:
            f.write(body)
        if verbose:
            print(f"  [outline] wrote {len(faces)}-triangle stl surface "
                  f"→ {output_path}")
        return True

    else:
        if verbose:
            print(f"  [outline] geometry type '{gtype}' has no outline support — skipped")
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    body = _format_polydata_ascii(pts, idx, title=title)
    with open(output_path, "w") as f:
        f.write(body)

    if verbose:
        print(f"  [outline] wrote {len(pts)}-vertex {gtype} polyline → {output_path}")
    return True
