"""Surface-distribution post-processor (2D): Cp(x/c), Cf(x/c), VTK polyline.

Consumes the per-link NPZ produced by `ForceManager` when
`force_calculation.save_link_forces = True`, and the simulation config
(for geometry + reference quantities). Emits:

    <run_dir>/csv/surface_cp.csv       airfoil: x/c, side, Cp_mean, Cp_std
                                       circle : theta_deg, Cp_mean, Cp_std
    <run_dir>/csv/surface_cf.csv       same layout, skin friction
    <run_dir>/csv/surface_cp.vtp       VTK PolyData polyline — ParaView
                                       "Plot Over Line" / "Plot Data" friendly

CLI:
    python -m src.utilities.surface_distribution <run_dir> <config_path>
        [--window-start <step>] [--window-end <step>] [--bins <N>]

Formula (per link):
    F_link_norm = F_link · n̂_link             pressure contribution
    F_link_tang = F_link · t̂_link             skin-friction contribution
    p_link      = F_link_norm / Δl_link         surface pressure
    τ_link      = F_link_tang / Δl_link         shear stress
    Cp_link     = 2·(p_link − p_∞) / (ρ_∞·U²)
    Cf_link     = 2· τ_link        / (ρ_∞·U²)

where Δl_link is the per-link surface-arc weight from NPZ (`link_dl`).
After time-averaging per link, we bin links by (θ, side) or (x/c, side)
and average within each bin — the −c_i/|c_i| normal approximation's
scatter washes out in the bin average.

Author: LBM Development Team
Date: 2026-04
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────────
# Config loading
# ────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> Dict:
    spec = importlib.util.spec_from_file_location("cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "config"):
        raise AttributeError(f"'config' dict missing in {path}")
    return mod.config


def _detect_geometry(internal_geom: Dict) -> Tuple[str, Dict]:
    """Return (geom_type, geom_cfg) for the single enabled obstacle."""
    for key in ("airfoil", "circle", "cylinder"):
        cfg = internal_geom.get(key)
        if isinstance(cfg, dict) and cfg.get("enabled", False):
            # 2D cylinder is treated as circle
            return ("circle" if key in ("cylinder", "circle") else "airfoil"), cfg
    raise ValueError("No enabled 2D obstacle found in internal_geometry")


# ────────────────────────────────────────────────────────────────────
# Per-link time-average
# ────────────────────────────────────────────────────────────────────

def _time_average_links(
    npz_path: str,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load NPZ, slice to [window_start, window_end] (steps), return per-link stats.

    Both bounds are in actual simulation-step units (NOT sample indices).
    window_start=None → first sample. window_end=None → last sample.
    Inclusive on both sides.
    """
    d = np.load(npz_path)
    steps = d["steps"]
    F_link_ts = d["F_link"].astype(np.float64)  # (S, N, 2)

    if window_start is None:
        start = 0
    else:
        above = steps >= window_start
        start = int(np.argmax(above)) if above.any() else len(steps)

    if window_end is None:
        end = len(steps)
    else:
        below = steps <= window_end
        end = int(below.sum())  # # of samples ≤ window_end

    if start >= end:
        raise ValueError(
            f"Empty window: start={window_start}, end={window_end}, "
            f"available steps={int(steps.min())}..{int(steps.max())} "
            f"({len(steps)} samples)."
        )

    F_used = F_link_ts[start:end]

    # Density per link — new in option-B NPZ. Older NPZs lack it; caller
    # will see all-NaN rho_mean and fall back to the MEM projection path.
    if "rho_link" in d:
        rho_used = d["rho_link"][start:end].astype(np.float64)     # (S, N)
        rho_mean = rho_used.mean(axis=0)                           # (N,)
        rho_std  = rho_used.std(axis=0)
    else:
        N = len(d["link_dir"])
        rho_mean = np.full(N, np.nan, dtype=np.float64)
        rho_std  = np.full(N, np.nan, dtype=np.float64)

    return {
        "F_mean":  F_used.mean(axis=0),              # (N, 2)
        "F_std":   F_used.std(axis=0),               # (N, 2)
        "rho_mean": rho_mean,                        # (N,)
        "rho_std":  rho_std,
        "n_used":  len(F_used),
        "steps_used": steps[start:end],
        "link_dir": d["link_dir"],
        "link_xf":  d["link_xf"],
        "link_yf":  d["link_yf"],
        "link_cx":  d["link_cx"].astype(np.float64),
        "link_cy":  d["link_cy"].astype(np.float64),
        "link_xw":  d["link_xw"].astype(np.float64),
        "link_yw":  d["link_yw"].astype(np.float64),
        "link_nx":  d["link_nx"].astype(np.float64),
        "link_ny":  d["link_ny"].astype(np.float64),
        "link_dl":  d["link_dl"].astype(np.float64),
        "link_w":   d["link_w"].astype(np.float64) if "link_w" in d
                    else np.full(d["link_dir"].shape, np.nan, dtype=np.float64),
    }


# ────────────────────────────────────────────────────────────────────
# Physics — Cp, Cf per link
# ────────────────────────────────────────────────────────────────────

def _compute_dF_link_vec(
    data: Dict[str, np.ndarray],
    rho_ref: float,
) -> np.ndarray:
    """Per-link **vector** F_link − F_link_baseline (shape (N, 2)).

    Baseline in quiescent fluid: F_link_baseline = 2·c_i·w_i·ρ_∞ (vector).
    Subtracting it cancels the ambient-pressure offset so the bin-summed
    projection on the true wall normal directly gives (p − p_∞)·ds_bin.

    We keep the full vector (not just the normal projection) so the binning
    step can project on the *actual* wall normal at the bin center, which
    avoids the per-link n̂ ≈ c_i/|c| approximation error.
    """
    F = data["F_mean"]                    # (N, 2)
    w = data["link_w"]
    cx, cy = data["link_cx"], data["link_cy"]
    F_base_x = 2.0 * cx * w * rho_ref
    F_base_y = 2.0 * cy * w * rho_ref
    dF = np.empty_like(F)
    dF[:, 0] = F[:, 0] - F_base_x
    dF[:, 1] = F[:, 1] - F_base_y
    return dF


# ────────────────────────────────────────────────────────────────────
# Circle / Cylinder — Cp(θ)
# ────────────────────────────────────────────────────────────────────

def _bin_circle(
    data: Dict[str, np.ndarray],
    dF_link: np.ndarray,            # (N, 2) baseline-subtracted F_link
    center: Tuple[float, float],
    radius: float,
    rho_ref: float,
    u_ref: float,
    bins: int = 60,
) -> Dict[str, np.ndarray]:
    """Bin by polar angle θ.

    Cp is computed from **density at the fluid cell adjacent to the wall**
    (the Option-B path):
        Cp_bin = mean(ρ_link in bin) · cs² / q  −  ρ_∞·cs² / q
               = (ρ_bin − ρ_∞)·cs² / q
    which is insensitive to bin width and to the lattice-axis link clustering
    that affects the ΔF_link / ds_bin formulation.

    Cf is still derived from the MEM tangential force (no density analogue
    for shear stress), reported as a secondary diagnostic.
    """
    cx, cy = center
    xw = data["link_xw"] - cx
    yw = data["link_yw"] - cy
    theta = np.mod(np.arctan2(yw, xw), 2.0 * np.pi)

    # Edge-shift: place bin centers at 0°, Δθ°, 2Δθ°, … so 0°, 90°, 180°,
    # 270° fall on centers (provided 360/bins divides 90 — default 60 works).
    dtheta = 2.0 * np.pi / bins
    theta_shifted = np.mod(theta + 0.5 * dtheta, 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
    idx = np.clip(np.digitize(theta_shifted, edges) - 1, 0, bins - 1)
    theta_c = np.arange(bins) * dtheta
    ds_bin = radius * dtheta
    q = 0.5 * rho_ref * u_ref * u_ref
    cs2 = 1.0 / 3.0

    rho_link = data["rho_mean"]            # (N,) — NaN if old NPZ
    has_rho  = np.all(np.isfinite(rho_link))

    Cp = np.full(bins, np.nan)
    Cf = np.full(bins, np.nan)
    count = np.zeros(bins, dtype=int)

    for b in range(bins):
        m = idx == b
        if not np.any(m):
            continue

        # ── Density-based Cp (primary) ──
        if has_rho:
            rho_bin = rho_link[m].mean()
            Cp[b] = (rho_bin - rho_ref) * cs2 / q

        # ── MEM tangential Cf (secondary, bin-sum / arc length) ──
        Fx_sum = dF_link[m, 0].sum()
        Fy_sum = dF_link[m, 1].sum()
        tx = -(-np.sin(theta_c[b]))   # = sin(θ_c)
        ty =  (-np.cos(theta_c[b]))   # = -cos(θ_c)
        # (tangent = 90° CCW of inward normal n̂=(−cosθ,−sinθ))
        tx, ty = np.sin(theta_c[b]), -np.cos(theta_c[b])
        Ft = Fx_sum * tx + Fy_sum * ty
        Cf[b] = (Ft / ds_bin) / q

        # ── Fallback Cp if no ρ in NPZ: old MEM-projection formula ──
        if not has_rho:
            nx = -np.cos(theta_c[b]); ny = -np.sin(theta_c[b])
            Fn = Fx_sum * nx + Fy_sum * ny
            Cp[b] = (Fn / ds_bin) / q

        count[b] = int(m.sum())

    return {
        "theta_deg": np.rad2deg(theta_c),
        "Cp":        Cp,
        "Cf":        Cf,
        "n_links":   count,
        "mode":      "density" if has_rho else "MEM-projection (no ρ in NPZ)",
    }


# ────────────────────────────────────────────────────────────────────
# Airfoil — Cp(x/c) with upper/lower split
# ────────────────────────────────────────────────────────────────────

def _project_airfoil_coords(
    data: Dict[str, np.ndarray],
    chord: float,
    center: Tuple[float, float],
    aoa_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate wall points into airfoil body frame; return (x/c, y_body/c).

    Inverse of the forward transform used in
    `geometry.transform_airfoil_coords_to_lu`:
        lattice = rot(α) · (body - mid_chord) + center
    So body = rot(-α) · (lattice - center) + mid_chord.
    """
    cx, cy = center
    xw = data["link_xw"] - cx
    yw = data["link_yw"] - cy

    a = np.deg2rad(aoa_deg)
    ca, sa = np.cos(a), np.sin(a)
    x_body = xw * ca + yw * sa + 0.5 * chord
    y_body = -xw * sa + yw * ca
    return x_body / chord, y_body / chord


def _bin_airfoil(
    data: Dict[str, np.ndarray],
    dF_link: np.ndarray,            # (N, 2) baseline-subtracted F_link
    chord: float,
    center: Tuple[float, float],
    aoa_deg: float,
    rho_ref: float,
    u_ref: float,
    bins: int = 100,
) -> Dict[str, np.ndarray]:
    """Equal-link-count binning per side along the polyline.

    Each surface side (upper / lower) is sorted by x/c and split into
    `bins` chunks of (nearly) equal link count.  The bin x/c is the
    link-mean within the chunk; Cp is the density-mean within the chunk;
    Cf uses the actual per-link arc-weight sum for ds_bin.

    Why not x/c-uniform binning:  link density along x/c is far from
    uniform — the suction-side LE has many links per Δx/c (high curvature
    plus all 8 lattice directions hit the cell), while the pressure-side
    mid-chord has few (near-flat surface, only ±x lattice directions hit).
    Uniform x/c bins therefore mix bins of 30+ links (smooth) with bins
    of 5 (noisy), and reducing bin count to recover smoothness sacrifices
    the LE peak Cp by averaging over too wide an arc.  Equal-count
    chunking gives the same statistical scatter at every x/c, so the
    plot is bin-count-robust:  more bins → finer LE peak resolution
    without the pressure-side blowing up into noise.

    Cp formula:  Option-B density form
        Cp_bin = (mean ρ_link − ρ_∞)·cs² / q
    Cf formula:  per-link tangential force projected on body-frame ±x,
        normalised by the bin's actual arc length ds_bin = Σ link_dl
        (NPZ already stores per-link arc weights, so this includes the
        local slope sec(θ) factor that the old chord·Δ(x/c) approximation
        ignored).

    Returns a dict whose `x_c_upper` / `x_c_lower` arrays are independent
    (different bins occupy different x/c on the two sides), so plots and
    Tecplot zones must use the side-specific x_c arrays.
    """
    xc, yc = _project_airfoil_coords(data, chord, center, aoa_deg)

    # Upper/lower split via the inward wall-normal direction in body frame.
    # This is robust for cambered airfoils (camber line is above/below the
    # chord, so splitting by y_body=0 misclassifies points — using the
    # local surface normal avoids that ambiguity).
    a = np.deg2rad(aoa_deg)
    ca, sa = np.cos(a), np.sin(a)
    nx_lat = data["link_nx"]
    ny_lat = data["link_ny"]
    # Rotate inward normal: n_body = R_{-α} · n_lattice
    ny_body = -nx_lat * sa + ny_lat * ca
    # Upper surface: inward normal points DOWNWARD (into body) in body frame
    #   → ny_body < 0
    # Lower surface: inward normal points UPWARD (into body)
    #   → ny_body > 0
    side = np.where(ny_body < 0.0, +1, -1)

    # Inward body-frame normals rotated to lattice frame (for Cf tangent
    # and the MEM-projection Cp fallback).
    n_up  = np.array([+sa, -ca])
    n_low = np.array([-sa, +ca])
    t_up  = np.array([+ca, +sa])
    t_low = np.array([-ca, -sa])

    q   = 0.5 * rho_ref * u_ref * u_ref
    cs2 = 1.0 / 3.0

    rho_link = data["rho_mean"]
    has_rho  = np.all(np.isfinite(rho_link))
    link_dl  = data["link_dl"]

    def _bin_side(s: int, n_hat: np.ndarray, t_hat: np.ndarray):
        m = (side == s)
        N = int(m.sum())
        if N == 0:
            empty = np.array([], dtype=np.float64)
            return empty, empty, empty, np.array([], dtype=int)

        xc_s  = xc[m]
        rho_s = rho_link[m] if has_rho else None
        dF_s  = dF_link[m]
        dl_s  = link_dl[m]

        # Sort links by x/c, then carve into nearly-equal chunks.
        order = np.argsort(xc_s, kind="stable")
        n_bins_s = min(bins, N)
        cuts = np.linspace(0, N, n_bins_s + 1).astype(int)

        x_c_b = np.full(n_bins_s, np.nan, dtype=np.float64)
        Cp_b  = np.full(n_bins_s, np.nan, dtype=np.float64)
        Cf_b  = np.full(n_bins_s, np.nan, dtype=np.float64)
        n_b   = np.zeros(n_bins_s, dtype=int)

        for b in range(n_bins_s):
            chunk = order[cuts[b]:cuts[b + 1]]
            if len(chunk) == 0:
                continue
            x_c_b[b] = xc_s[chunk].mean()
            n_b[b]   = len(chunk)
            ds_bin   = float(dl_s[chunk].sum())
            if ds_bin <= 0.0:
                ds_bin = 1e-30  # numerical safety; should never trigger

            if has_rho:
                Cp_b[b] = (rho_s[chunk].mean() - rho_ref) * cs2 / q

            Fx_sum = dF_s[chunk, 0].sum()
            Fy_sum = dF_s[chunk, 1].sum()
            Ft = Fx_sum * t_hat[0] + Fy_sum * t_hat[1]
            Cf_b[b] = (Ft / ds_bin) / q

            if not has_rho:
                Fn = Fx_sum * n_hat[0] + Fy_sum * n_hat[1]
                Cp_b[b] = (Fn / ds_bin) / q

        return x_c_b, Cp_b, Cf_b, n_b

    x_c_u, Cp_u, Cf_u, n_u = _bin_side(+1, n_up,  t_up)
    x_c_l, Cp_l, Cf_l, n_l = _bin_side(-1, n_low, t_low)

    return {
        "x_c_upper": x_c_u, "x_c_lower": x_c_l,
        "Cp_upper":  Cp_u,  "Cp_lower":  Cp_l,
        "Cf_upper":  Cf_u,  "Cf_lower":  Cf_l,
        "n_upper":   n_u,   "n_lower":   n_l,
        "mode":      "density" if has_rho else "MEM-projection (no ρ in NPZ)",
    }


# ────────────────────────────────────────────────────────────────────
# Writers
# ────────────────────────────────────────────────────────────────────

def _write_csv(path: str, header: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _write_tecplot_xy(
    path: str,
    title: str,
    var_names: list,
    zones: list,
) -> None:
    """Tecplot ASCII point-format for XY plots.

    Each zone = (zone_title: str, columns: list[1D array]) where the column
    order matches `var_names`. NaN rows are dropped per zone (so each zone's
    I count may differ).  One file, multiple zones → Tecplot's XY plot picks
    any zone by name, or overlays them.

    Args:
        path:      Output .dat path.
        title:     Tecplot file title.
        var_names: Variable names, length = # columns per zone.
        zones:     List of (zone_title, [col0, col1, ...]) tuples.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f'TITLE = "{title}"',
        'VARIABLES = ' + ' '.join(f'"{v}"' for v in var_names),
    ]
    for zt, cols in zones:
        cols = [np.asarray(c, dtype=np.float64) for c in cols]
        # drop rows where ANY column is NaN (per-zone basis)
        keep = np.all(np.isfinite(np.column_stack(cols)), axis=1)
        if not np.any(keep):
            continue
        cols = [c[keep] for c in cols]
        I = len(cols[0])
        lines.append(
            f'ZONE T = "{zt}", I = {I}, F = POINT'
        )
        for row_i in range(I):
            lines.append(' '.join(f'{c[row_i]: .6e}' for c in cols))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _write_vtp_points(
    path: str,
    points_xyz: np.ndarray,            # (N, 3)
    scalars: Dict[str, np.ndarray],    # name → (N,)
) -> None:
    """ASCII VTP PolyData with one vertex cell per point + point scalars.

    Used to dump per-link wall intersection points (link_xw, link_yw) so that
    ParaView can render the actual q-interpolated wall position used by IBB.
    For HWBB the q is fixed (effectively 0.5) and these points collapse onto
    the staircase mid-pixel positions; for IBB they spread along the real
    airfoil surface — the visual diagnostic the user is after.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = len(points_xyz)

    pts_str = "\n".join(
        f"{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}" for p in points_xyz
    )
    conn_str = " ".join(str(i) for i in range(N))
    offset_str = " ".join(str(i + 1) for i in range(N))

    scalar_blocks = []
    for name, arr in scalars.items():
        vals = " ".join(
            f"{v:.6g}" if np.isfinite(v) else "nan" for v in arr
        )
        scalar_blocks.append(
            f'        <DataArray type="Float32" Name="{name}" '
            f'format="ascii">\n          {vals}\n        </DataArray>'
        )
    scalar_xml = "\n".join(scalar_blocks) if scalar_blocks else ""
    scalars_attr = (f' Scalars="{next(iter(scalars.keys()))}"'
                    if scalars else "")

    content = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        f'  <PolyData>\n'
        f'    <Piece NumberOfPoints="{N}" NumberOfVerts="{N}" '
        f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n'
        f'      <Points>\n'
        f'        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'
        f'          {pts_str}\n'
        f'        </DataArray>\n'
        f'      </Points>\n'
        f'      <Verts>\n'
        f'        <DataArray type="Int32" Name="connectivity" format="ascii">\n'
        f'          {conn_str}\n'
        f'        </DataArray>\n'
        f'        <DataArray type="Int32" Name="offsets" format="ascii">\n'
        f'          {offset_str}\n'
        f'        </DataArray>\n'
        f'      </Verts>\n'
        f'      <PointData{scalars_attr}>\n'
        f'{scalar_xml}\n'
        f'      </PointData>\n'
        f'    </Piece>\n'
        f'  </PolyData>\n'
        f'</VTKFile>\n'
    )

    with open(path, 'w') as f:
        f.write(content)


def _write_raw_links_vtp(
    path: str,
    data: Dict[str, np.ndarray],
    rho_ref: float,
    u_ref: float,
) -> None:
    """Dump per-link wall intersection points with Cp, q, and direction scalars.

    Scalars saved for ParaView inspection:
        Cp_mem    — pressure coefficient via momentum-exchange projection
                    on the link normal: -dF·n̂ / (q_dyn · dl)
        Cp_rho    — density-based Cp = (ρ_link − ρ_ref)·cs² / q_dyn (NaN if
                    rho_mean was missing from the NPZ)
        q         — sub-pixel wall fraction reconstructed from
                    (link_xw − link_xf) / link_cx (or _y / _cy when cx=0).
                    HWBB: ≈0.5 everywhere. IBB: spreads through (0, 1).
        link_dir  — lattice direction index (D2Q9 → 1..8)
    """
    cs2 = 1.0 / 3.0
    q_dyn = 0.5 * rho_ref * u_ref * u_ref

    xw = data["link_xw"]
    yw = data["link_yw"]
    xf = data["link_xf"].astype(np.float64)
    yf = data["link_yf"].astype(np.float64)
    cx = data["link_cx"]
    cy = data["link_cy"]
    nx = data["link_nx"]
    ny = data["link_ny"]
    dl = np.maximum(data["link_dl"], 1e-30)

    # MEM-projected Cp (baseline-subtracted force projected on inward normal,
    # divided by q_dyn · dl). Sign: -n̂ is outward, p_link = -dF·n̂/dl.
    dF = _compute_dF_link_vec(data, rho_ref)
    p_link = -(dF[:, 0] * nx + dF[:, 1] * ny) / dl
    Cp_mem = p_link / q_dyn

    # Density-based Cp if rho_mean was saved
    rho_mean = data.get("rho_mean")
    if rho_mean is not None and np.isfinite(rho_mean).any():
        Cp_rho = (rho_mean - rho_ref) * cs2 / q_dyn
    else:
        Cp_rho = np.full_like(Cp_mem, np.nan)

    # q reconstruction: prefer cx-based, fall back to cy-based, NaN if both 0
    q = np.full_like(xw, np.nan, dtype=np.float64)
    use_cx = np.abs(cx) > 0
    use_cy = (~use_cx) & (np.abs(cy) > 0)
    q[use_cx] = (xw[use_cx] - xf[use_cx]) / cx[use_cx]
    q[use_cy] = (yw[use_cy] - yf[use_cy]) / cy[use_cy]

    points = np.column_stack([xw, yw, np.zeros_like(xw)])
    _write_vtp_points(
        path,
        points.astype(np.float32),
        scalars={
            "Cp_mem":   Cp_mem.astype(np.float32),
            "Cp_rho":   Cp_rho.astype(np.float32),
            "q":        q.astype(np.float32),
            "link_dir": data["link_dir"].astype(np.float32),
        },
    )


def _write_vtp_polyline(
    path: str,
    points_xyz: np.ndarray,            # (N, 3)
    scalars: Dict[str, np.ndarray],    # name → (N,)
) -> None:
    """Minimal ASCII VTP PolyData with a single polyline and point scalars."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = len(points_xyz)

    pts_str = "\n".join(
        f"{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}" for p in points_xyz
    )
    conn_str = " ".join(str(i) for i in range(N))
    offset_str = str(N)

    scalar_blocks = []
    for name, arr in scalars.items():
        vals = " ".join(
            f"{v:.6g}" if np.isfinite(v) else "nan" for v in arr
        )
        scalar_blocks.append(
            f'        <DataArray type="Float32" Name="{name}" '
            f'format="ascii">\n          {vals}\n        </DataArray>'
        )
    scalar_xml = "\n".join(scalar_blocks) if scalar_blocks else ""

    content = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        f'  <PolyData>\n'
        f'    <Piece NumberOfPoints="{N}" NumberOfLines="1">\n'
        f'      <Points>\n'
        f'        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'
        f'          {pts_str}\n'
        f'        </DataArray>\n'
        f'      </Points>\n'
        f'      <Lines>\n'
        f'        <DataArray type="Int32" Name="connectivity" format="ascii">\n'
        f'          {conn_str}\n'
        f'        </DataArray>\n'
        f'        <DataArray type="Int32" Name="offsets" format="ascii">\n'
        f'          {offset_str}\n'
        f'        </DataArray>\n'
        f'      </Lines>\n'
        f'      <PointData Scalars="{next(iter(scalars.keys()))}">\n'
        f'{scalar_xml}\n'
        f'      </PointData>\n'
        f'    </Piece>\n'
        f'  </PolyData>\n'
        f'</VTKFile>\n'
    )
    with open(path, "w") as f:
        f.write(content)


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────

def write_surface_distribution(
    csv_dir: str,
    data: Dict[str, np.ndarray],
    geom_type: str,
    geom_cfg: Dict,
    rho_ref: float,
    u_ref: float,
    bins: Optional[int] = None,
) -> Dict[str, str]:
    """Write CSV + Tecplot .dat + VTP from an already-averaged data dict.

    Used both by `extract_surface_distribution` (loading from NPZ) and by
    `ForceManager.close()` (in-memory buffers). The `data` dict must contain
    `F_mean` and all `link_*` metadata (same keys as `_time_average_links`).
    """
    dF_link = _compute_dF_link_vec(data, rho_ref)
    written: Dict[str, str] = {}

    # Always write raw per-link wall points for ParaView inspection
    # (independent of bin-aggregated polyline). HWBB shows points
    # collapsed onto half-pixel positions; IBB spreads them through
    # the actual airfoil surface — direct visual diagnostic of which
    # wall-BC was active.
    raw_vtp_path = os.path.join(csv_dir, "surface_links_raw.vtp")
    _write_raw_links_vtp(raw_vtp_path, data, rho_ref, u_ref)
    written["vtp_raw_links"] = raw_vtp_path

    if geom_type == "circle":
        center = geom_cfg["center"]
        radius = float(geom_cfg["radius"])
        nbins = bins if bins is not None else 60
        res = _bin_circle(
            data, dF_link, center=center, radius=radius,
            rho_ref=rho_ref, u_ref=u_ref, bins=nbins,
        )

        # CSV
        csv_path = os.path.join(csv_dir, "surface_cp_cf.csv")
        rows = [
            (res["theta_deg"][i], res["Cp"][i], res["Cf"][i], res["n_links"][i])
            for i in range(nbins)
        ]
        _write_csv(csv_path, ["theta_deg", "Cp", "Cf", "n_links"], rows)
        written["csv"] = csv_path

        # Tecplot ASCII (one zone, Cp vs θ)
        dat_path = os.path.join(csv_dir, "surface_cp.dat")
        _write_tecplot_xy(
            dat_path,
            title="Cylinder surface Cp vs θ",
            var_names=["theta_deg", "Cp"],
            zones=[("cylinder", [res["theta_deg"], res["Cp"]])],
        )
        written["tecplot"] = dat_path

        # VTP polyline at wall (lift into 3D with z=0)
        r = float(geom_cfg["radius"])
        cx, cy = center
        theta = np.deg2rad(res["theta_deg"])
        pts = np.column_stack([
            cx + r * np.cos(theta),
            cy + r * np.sin(theta),
            np.zeros_like(theta),
        ])
        vtp_path = os.path.join(csv_dir, "surface_cp.vtp")
        _write_vtp_polyline(
            vtp_path, pts,
            scalars={"Cp": res["Cp"], "Cf": res["Cf"],
                     "n_links": res["n_links"].astype(np.float32)},
        )
        written["vtp"] = vtp_path

    elif geom_type == "airfoil":
        chord = float(geom_cfg["chord"])
        center = tuple(geom_cfg["center"])
        aoa = float(geom_cfg.get("angle_of_attack", 0.0))
        nbins = bins if bins is not None else 100
        res = _bin_airfoil(
            data, dF_link, chord=chord, center=center, aoa_deg=aoa,
            rho_ref=rho_ref, u_ref=u_ref, bins=nbins,
        )

        # Equal-count binning: upper and lower bins occupy independent
        # x/c grids. CSV writes them side-by-side, padding the shorter
        # column with NaN so each row is one (upper bin, lower bin) pair.
        csv_path = os.path.join(csv_dir, "surface_cp_cf.csv")
        header = [
            "x_c_upper", "Cp_upper", "Cf_upper", "n_upper",
            "x_c_lower", "Cp_lower", "Cf_lower", "n_lower",
        ]
        n_u = len(res["x_c_upper"])
        n_l = len(res["x_c_lower"])
        n_rows = max(n_u, n_l)

        def _pad_f(a: np.ndarray, n: int) -> np.ndarray:
            if len(a) == n:
                return a
            out = np.full(n, np.nan, dtype=np.float64)
            out[:len(a)] = a
            return out

        def _pad_i(a: np.ndarray, n: int) -> np.ndarray:
            if len(a) == n:
                return a
            out = np.zeros(n, dtype=int)
            out[:len(a)] = a
            return out

        xu = _pad_f(res["x_c_upper"], n_rows)
        cu = _pad_f(res["Cp_upper"],  n_rows)
        fu = _pad_f(res["Cf_upper"],  n_rows)
        nu = _pad_i(res["n_upper"],   n_rows)
        xl = _pad_f(res["x_c_lower"], n_rows)
        cl = _pad_f(res["Cp_lower"],  n_rows)
        fl = _pad_f(res["Cf_lower"],  n_rows)
        nl = _pad_i(res["n_lower"],   n_rows)
        rows = [
            (xu[i], cu[i], fu[i], int(nu[i]),
             xl[i], cl[i], fl[i], int(nl[i]))
            for i in range(n_rows)
        ]
        _write_csv(csv_path, header, rows)
        written["csv"] = csv_path

        # Tecplot ASCII — separate zones for upper / lower so Tecplot's
        # XY-plot tab shows "upper" and "lower" as two toggleable curves.
        # Each zone has its own I count (different bin counts allowed).
        dat_path = os.path.join(csv_dir, "surface_cp.dat")
        _write_tecplot_xy(
            dat_path,
            title=f"Airfoil surface Cp vs x/c (AoA={aoa:+.1f}°)",
            var_names=["x_c", "Cp"],
            zones=[
                ("upper", [res["x_c_upper"], res["Cp_upper"]]),
                ("lower", [res["x_c_lower"], res["Cp_lower"]]),
            ],
        )
        written["tecplot"] = dat_path

        # VTP: upper+lower concatenated (upper LE→TE, lower TE→LE) in lattice frame
        a = np.deg2rad(aoa)
        ca, sa = np.cos(a), np.sin(a)
        cx, cy = center

        def to_lattice(xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
            x_body = (xc - 0.5) * chord
            y_body = yc * chord
            xl = x_body * ca - y_body * sa + cx
            yl = x_body * sa + y_body * ca + cy
            return np.column_stack([xl, yl, np.zeros_like(xl)])

        # Polyline: upper LE→TE then lower TE→LE.  y=0 placeholder
        # (per-bin y_body unknown) is good enough for ParaView "Plot Data
        # by x coordinate" — points show Cp vs x/c at their lattice x.
        pts_u = to_lattice(res["x_c_upper"], np.zeros_like(res["x_c_upper"]))
        pts_l = to_lattice(res["x_c_lower"][::-1], np.zeros_like(res["x_c_lower"]))
        pts = np.concatenate([pts_u, pts_l], axis=0)

        Cp_concat = np.concatenate([res["Cp_upper"], res["Cp_lower"][::-1]])
        Cf_concat = np.concatenate([res["Cf_upper"], res["Cf_lower"][::-1]])
        side_concat = np.concatenate([
            np.full(len(res["x_c_upper"]), +1.0, dtype=np.float32),
            np.full(len(res["x_c_lower"]), -1.0, dtype=np.float32),
        ])

        vtp_path = os.path.join(csv_dir, "surface_cp.vtp")
        _write_vtp_polyline(
            vtp_path, pts,
            scalars={"Cp": Cp_concat, "Cf": Cf_concat, "side": side_concat},
        )
        written["vtp"] = vtp_path

    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    # Sectional integrated loads (sanity check)
    # Sum F_link over all links, normalize → should match run's Cd/Cl
    F_tot = data["F_mean"].sum(axis=0)
    ref_area = (geom_cfg.get("chord", None)
                or (2.0 * geom_cfg.get("radius", 1.0)))
    q = 0.5 * rho_ref * u_ref * u_ref
    Cd_check = F_tot[0] / (q * ref_area)
    Cl_check = F_tot[1] / (q * ref_area)

    print(f"  [{geom_type}] wrote:")
    for k, v in written.items():
        print(f"    {k}: {v}")
    mode = res.get("mode", "unknown")
    print(f"  Cp mode: {mode}")
    print(f"  Sanity Cd (from per-link sum, time-averaged): {Cd_check:+.4f}")
    print(f"  Sanity Cl (from per-link sum, time-averaged): {Cl_check:+.4f}")

    return written


def _override_geom_from_npz(
    geom_cfg: Dict, npz_path: str,
) -> Tuple[Dict, bool]:
    """Replace chord/center/AoA in `geom_cfg` with NPZ-stored values when
    available. ForceManager dumps these in *finest-lattice units* on MLG
    runs — matching `link_xw`/`link_yw` — so they must override the L0
    values pulled from the config module.

    Returns (new_cfg, overridden). `overridden` is True iff at least one
    `obstacle_*` key was present, signalling that the config-module
    chord/center is being ignored.
    """
    out = dict(geom_cfg)
    overridden = False
    with np.load(npz_path, allow_pickle=False) as d:
        if 'obstacle_center' in d.files:
            out['center'] = tuple(
                float(v) for v in np.asarray(d['obstacle_center'])
            )
            overridden = True
        if 'obstacle_chord' in d.files:
            out['chord'] = float(d['obstacle_chord'])
            overridden = True
        if 'obstacle_radius' in d.files:
            out['radius'] = float(d['obstacle_radius'])
            overridden = True
        if 'obstacle_aoa_deg' in d.files:
            out['angle_of_attack'] = float(d['obstacle_aoa_deg'])
            overridden = True
    return out, overridden


def extract_surface_distribution(
    run_dir: str,
    config_path: str,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
    bins: Optional[int] = None,
) -> Dict[str, str]:
    """CLI entry: NPZ + config → surface_cp.{dat,vtp,csv}.

    `window_start` and `window_end` are in actual simulation-step units
    (inclusive). Inspect `force_history.csv` to pick an appropriate
    averaging window after the transient has decayed.
    """
    cfg = _load_config(config_path)
    geom_type, geom_cfg = _detect_geometry(cfg.get("internal_geometry", {}))

    ref = cfg.get("force_calculation", {}).get("reference", {})
    rho_ref = ref.get("rho", 1.0)
    u_ref = ref.get("velocity", 0.1)

    csv_dir = os.path.join(run_dir, "csv")
    npz_path = os.path.join(csv_dir, "surface_link_forces.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found. Re-run with an obstacle to produce "
            "surface_link_forces.npz."
        )

    geom_cfg, used_npz_meta = _override_geom_from_npz(geom_cfg, npz_path)
    if used_npz_meta:
        print(f"  Using finest-lattice chord/center from NPZ "
              f"(MLG-aware): "
              f"chord={geom_cfg.get('chord', geom_cfg.get('radius'))}, "
              f"center={geom_cfg.get('center')}")
    else:
        print(f"  WARNING: NPZ has no obstacle metadata — falling back "
              f"to config-module values (correct only for single-level "
              f"runs).")

    data = _time_average_links(
        npz_path, window_start=window_start, window_end=window_end,
    )
    print(f"  Averaged {data['n_used']} samples "
          f"from step {int(data['steps_used'][0])} "
          f"to {int(data['steps_used'][-1])}")
    return write_surface_distribution(
        csv_dir=csv_dir,
        data=data,
        geom_type=geom_type,
        geom_cfg=geom_cfg,
        rho_ref=rho_ref,
        u_ref=u_ref,
        bins=bins,
    )


def write_surface_distribution_from_buffer(
    csv_dir: str,
    F_link_ts: np.ndarray,            # (S, N, 2)
    steps: np.ndarray,                # (S,)
    link_meta: Dict[str, np.ndarray],
    internal_geometry: Dict,
    rho_ref: float,
    u_ref: float,
    rho_link_ts: Optional[np.ndarray] = None,   # (S, N)
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
    bins: Optional[int] = None,
) -> Dict[str, str]:
    """In-memory counterpart of `extract_surface_distribution`.

    Not currently used by the solver (the solver only writes the NPZ at
    close); kept for programmatic use that has per-link arrays in memory
    and wants to skip the NPZ round-trip.
    """
    try:
        geom_type, geom_cfg = _detect_geometry(internal_geometry)
    except ValueError:
        return {}

    if window_start is None:
        start = 0
    else:
        above = steps >= window_start
        start = int(np.argmax(above)) if above.any() else len(steps)
    if window_end is None:
        end = len(steps)
    else:
        end = int((steps <= window_end).sum())
    if start >= end:
        return {}

    F_used = F_link_ts[start:end].astype(np.float64)
    if rho_link_ts is not None:
        rho_used = rho_link_ts[start:end].astype(np.float64)
        rho_mean = rho_used.mean(axis=0)
        rho_std  = rho_used.std(axis=0)
    else:
        N = F_used.shape[1]
        rho_mean = np.full(N, np.nan, dtype=np.float64)
        rho_std  = np.full(N, np.nan, dtype=np.float64)

    data = {
        "F_mean":  F_used.mean(axis=0),
        "F_std":   F_used.std(axis=0),
        "rho_mean": rho_mean,
        "rho_std":  rho_std,
        "n_used":  len(F_used),
        "steps_used": steps[start:end],
        **{k: (v.astype(np.float64) if v.dtype.kind == "f" else v)
           for k, v in link_meta.items()},
    }
    return write_surface_distribution(
        csv_dir=csv_dir,
        data=data,
        geom_type=geom_type,
        geom_cfg=geom_cfg,
        rho_ref=rho_ref,
        u_ref=u_ref,
        bins=bins,
    )


def _main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="Result directory (parent of csv/)")
    ap.add_argument("config_path", help="Config file used for the run")
    ap.add_argument("--window-start", type=int, default=None,
                    dest="window_start",
                    help="Include samples with step ≥ this value "
                         "(default: first available).")
    ap.add_argument("--window-end", type=int, default=None,
                    dest="window_end",
                    help="Include samples with step ≤ this value "
                         "(default: last available).")
    ap.add_argument("--bins", type=int, default=None,
                    help="Number of bins (default: 180 for circle, 100 for airfoil)")
    args = ap.parse_args()
    extract_surface_distribution(
        args.run_dir, args.config_path,
        window_start=args.window_start,
        window_end=args.window_end,
        bins=args.bins,
    )


if __name__ == "__main__":
    _main()
