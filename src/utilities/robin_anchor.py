"""ROBIN (TM-80051) rotor-off Cp anchor — loader, station tables, sim overlay.

Anchor data (data/robin/, digitised 2026-08-26 from NASA TM-80051,
Freeman & Mineck 1979 — appendix listings + Table III orifice
coordinates; provenance in patch_notes/robin/02):

    data/robin/tm80051_orifices.csv       176 orifices: station, side, x/R y/R z/R
    data/robin/tm80051_cp_rotor_off.csv   Run 12 pts 88-91 (alpha -10/-5/0/+5,
                                          beta 0, 81.5-81.7 kt), Cp per orifice
    data/robin/tm80051_raw/pt<NN>_visual.txt   raw transcriptions with flags

Conventions
    * Coordinates are in ROTOR RADII (R); the body is 2R long, nose at
      x/R = 0. Same frame as input_files/geom/robin_mod.stl (verified: every
      orifice lies within 0.0035 R of the STL surface).
    * Orifices 89-176 are the Y-mirrors (port side) of 1-88 (starboard).
      For beta = 0 the mirror pair is a repeatability/asymmetry measure
      (|dCp| mean 0.010-0.017, max ~0.06 near the nose).
    * phi = polar angle about the body axis, atan2(y, z): 0 = top (dorsal),
      +90 = starboard, 180 = bottom (TM eq. 8 uses Y = r sin phi, Z = r cos
      phi + Z0). Station numbering 1-14 follows TM strips 1-14 (x/R 0.0517
      ... 1.5303).

Simulation overlay: surface_<step>.vtk written by the surfel stack (per
STL triangle: p_use, tau, traction, area, Cp). Cp is recomputed here as
(p_use - p_inf)/q_inf with p_inf = rho_inf/3 (lattice, rho_inf = 1) and
q_inf = 0.5 U_lu^2, because the file's own Cp uses the AREA-MEAN surface
pressure as reference (fine for a wing, biased for a closed body). The
orifice positions are mapped into the run's L0-lu frame with the SAME
transform the solver applies to the STL (scale -> rotate about the scaled
bbox centre -> shift the rotated bbox centre onto center_lu), then each
orifice samples the area-weighted mean of the triangle Cp within a
radius r_s (default 1.5 fine cells) — nearest triangle as fallback.

CLI (main dir):
    python -m src.utilities.robin_anchor --point 90                 # tables
    python -m src.utilities.robin_anchor --point 90 --plot cp_pt90.png
    python -m src.utilities.robin_anchor --point 90 \
        --config configs/robin/robin_r0_musker.py \
        --surface results_robin_r0/vtk/surface_00007000.vtk [more files...] \
        --plot overlay.png --csv overlay.csv
Cluster is Python 3.9: no PEP 604 annotations here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO, "data", "robin")
ORIFICE_CSV = os.path.join(DATA_DIR, "tm80051_orifices.csv")
CP_CSV = os.path.join(DATA_DIR, "tm80051_cp_rotor_off.csv")

#: TM-80051 Table IV, Run 12 (rotor off, closed test section)
POINT_CONDITIONS = {
    88: {"run": 12, "V_kt": 81.5, "alpha_deg": -10.0, "beta_deg": 0.0},
    89: {"run": 12, "V_kt": 81.7, "alpha_deg": -5.0, "beta_deg": 0.0},
    90: {"run": 12, "V_kt": 81.7, "alpha_deg": 0.0, "beta_deg": 0.0},
    91: {"run": 12, "V_kt": 81.6, "alpha_deg": 5.0, "beta_deg": 0.0},
}
#: station groups used by the pre-registered discriminators (patch robin/02)
FORE_STATIONS = (1, 2)           # x/R 0.05, 0.09 — carried-over question
NOSE_STATIONS = (1, 2, 3, 4)     # x/R <= 0.20
PYLON_STATIONS = (8, 9, 10, 11)  # x/R 0.47-1.00 (pylon span 0.40-1.018)
AFT_STATIONS = (12, 13, 14)      # x/R 1.16-1.53 (sting-side caveat)


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────
def load_orifices(path: str = ORIFICE_CSV) -> Dict[str, np.ndarray]:
    """Orifice table -> dict of arrays (orifice, station, side, xyz, phi_deg)."""
    rows = list(csv.DictReader(open(path)))
    o = np.array([int(r["orifice"]) for r in rows])
    st = np.array([int(r["station"]) for r in rows])
    side = np.array([r["side"].split("(")[0] for r in rows])
    xyz = np.array([[float(r["x_R"]), float(r["y_R"]), float(r["z_R"])]
                    for r in rows])
    phi = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 2]))
    order = np.argsort(o)
    return {"orifice": o[order], "station": st[order], "side": side[order],
            "xyz": xyz[order], "phi_deg": phi[order]}


def load_cp(path: str = CP_CSV) -> Dict[int, Dict[str, np.ndarray]]:
    """Cp table -> {point: {'orifice', 'cp', 'flag', ...conditions}}."""
    out: Dict[int, Dict[str, list]] = {}
    for r in csv.DictReader(open(path)):
        pt = int(r["point"])
        d = out.setdefault(pt, {"orifice": [], "cp": [], "flag": []})
        d["orifice"].append(int(r["orifice"]))
        d["cp"].append(float(r["cp"]) if r["cp"] != "" else float("nan"))
        d["flag"].append(r["flag"])
        d["run"] = int(r["run"]); d["V_kt"] = float(r["V_kt"])
        d["alpha_deg"] = float(r["alpha_deg"]); d["beta_deg"] = float(r["beta_deg"])
    res: Dict[int, Dict[str, np.ndarray]] = {}
    for pt, d in out.items():
        idx = np.argsort(d["orifice"])
        cp = np.array(d["cp"])[idx]
        assert len(cp) == 176, (pt, len(cp))
        res[pt] = {"orifice": np.array(d["orifice"])[idx], "cp": cp,
                   "flag": np.array(d["flag"])[idx],
                   "run": d["run"], "V_kt": d["V_kt"],
                   "alpha_deg": d["alpha_deg"], "beta_deg": d["beta_deg"]}
    return res


def mirror_pairs(cp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cp_starboard[88], cp_port[88], mirror-mean[88]) for orifices 1-88."""
    s, p = cp[:88], cp[88:]
    mean = np.where(np.isnan(s), p, np.where(np.isnan(p), s, 0.5 * (s + p)))
    return s, p, mean


def station_table(orf: Dict[str, np.ndarray], cp: np.ndarray, station: int
                  ) -> List[Tuple[int, float, float, str]]:
    """[(orifice, phi_deg, cp, side)] sorted by phi (-180..180) for one station."""
    sel = np.where(orf["station"] == station)[0]
    rows = [(int(orf["orifice"][i]), float(orf["phi_deg"][i]), float(cp[i]),
             str(orf["side"][i])) for i in sel]
    return sorted(rows, key=lambda t: t[1])


# ──────────────────────────────────────────────────────────────────────
# Simulation surface files (surfel stack)
# ──────────────────────────────────────────────────────────────────────
def read_surface_vtk(path: str):
    """surface_<step>.vtk (legacy ASCII polydata, per-triangle cell data)."""
    txt = open(path).read()
    m = re.search(r"POINTS (\d+) float\n", txt)
    n_pts = int(m.group(1)); p0 = m.end()
    m2 = re.search(r"\nPOLYGONS (\d+) (\d+)\n", txt)
    pts = np.array(txt[p0:m2.start()].split(), dtype=np.float64).reshape(n_pts, 3)
    n_poly = int(m2.group(1)); q0 = m2.end()
    m3 = re.search(r"\nCELL_DATA (\d+)\n", txt)
    poly = np.array(txt[q0:m3.start()].split(), dtype=np.int64).reshape(n_poly, 4)[:, 1:]
    rest = txt[m3.end():]
    fields = {}
    for mm in re.finditer(r"(SCALARS|VECTORS) (\w+) float( 1)?\n(LOOKUP_TABLE default\n)?", rest):
        s = mm.end(); nxt = re.search(r"\n(SCALARS|VECTORS) ", rest[s:])
        blk = rest[s: s + nxt.start()] if nxt else rest[s:]
        arr = np.array(blk.split(), dtype=np.float64)
        fields[mm.group(2)] = arr.reshape(-1, 3) if mm.group(1) == "VECTORS" else arr
    return pts, poly, fields


def load_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("robin_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def orifices_to_l0lu(xyz_R: np.ndarray, stl_cfg: dict) -> np.ndarray:
    """Map orifice coordinates (STL frame) into the run's global L0-lu frame
    with the solver's own STL transform (src.boundary.stl_mesh)."""
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from src.boundary.stl_mesh import load_stl_checked, _rotation_matrix
    mesh = load_stl_checked(stl_cfg["file"])
    scale = float(stl_cfg["scale_to_lu"])
    v = mesh.vertices * scale
    p = np.asarray(xyz_R, dtype=np.float64) * scale
    rot = stl_cfg.get("rotation_deg")
    if rot is not None and np.any(np.asarray(rot, dtype=np.float64) != 0.0):
        pivot = 0.5 * (v.min(axis=0) + v.max(axis=0))
        Rm = _rotation_matrix(np.asarray(rot, dtype=np.float64)).T
        v = (v - pivot) @ Rm + pivot
        p = (p - pivot) @ Rm + pivot
    bbox_center = 0.5 * (v.min(axis=0) + v.max(axis=0))
    return p + (np.asarray(stl_cfg["center_lu"], dtype=np.float64) - bbox_center)


def surface_cp_mean(paths: Sequence[str], p_inf: float, q_inf: float,
                    channel: str = "p_use"):
    """Time-mean per-triangle Cp over several surface files.

    Returns (centroids, cp_mean, cp_std_over_files, area, file_cp_offset)
    where file_cp_offset = mean(Cp_file - Cp_here) documents the writer's
    area-mean p_ref choice."""
    cps, areas, cen, off = [], None, None, []
    for path in paths:
        pts, poly, F = read_surface_vtk(path)
        tri = pts[poly]
        c = tri.mean(axis=1)
        a = F["area"]
        if channel == "auto":                    # 13b default readout
            channel = "p_sknh" if "p_sknh" in F else "p_use"
        if channel not in F:
            alias = {"p_state": "p_use", "p_use": "p_state"}
            channel = alias.get(channel, channel)
        if channel not in F:
            raise SystemExit(f"surface file has no '{channel}' array "
                             f"(p_sknh/p_state = robin/13b writer; "
                             f"p_state_ph = robin/10b two-channel writer; "
                             f"older files: p_use)")
        nan_frac = float(np.mean(~np.isfinite(F[channel])))
        if nan_frac > 0.0:
            # step-0 files predate the first facet pass (all NaN); a partial
            # NaN share means a diverged run -> never average it in silently
            print(f"   [surface] {os.path.basename(path)}: p_use NaN share "
                  f"{nan_frac:.3f} -> SKIPPED")
            continue
        cp = (F[channel] - p_inf) / q_inf
        _cpw = F.get("Cp_state", F.get("Cp"))
        if _cpw is not None:
            off.append(float(np.average(_cpw - cp, weights=np.maximum(a, 1e-300))))
        cps.append(cp); areas = a if areas is None else np.maximum(areas, a); cen = c
    if not cps:
        raise SystemExit("no usable surface file (all skipped: NaN p_use)")
    cps = np.array(cps)
    return cen, cps.mean(axis=0), cps.std(axis=0), areas, (float(np.mean(off)) if off else float("nan"))


def orifice_curvature_quadric(stl_path: str, xyz_R: np.ndarray,
                              radius: float = 0.012, min_pts: int = 12
                              ) -> Dict[str, np.ndarray]:
    """Principal curvatures/directions at surface points from a local quadric
    fit on the STL (robin/11 K0; radius 0.012 R validated vs the analytic
    genROBIN surface: |k|max rel err median 1.0 %, p90 7.6 %). Since
    robin/13b this delegates to src.boundary.surfel_kappa (the runtime
    home of the same fit — intercept term REQUIRED, see there)."""
    import trimesh
    from src.boundary.surfel_kappa import quadric_curvature
    m = trimesh.load(stl_path)
    return quadric_curvature(np.asarray(m.vertices), np.asarray(m.faces),
                             points=np.asarray(xyz_R, dtype=np.float64),
                             radius=radius, min_pts=min_pts)


from src.boundary.surfel_kappa import kappa_n_direction  # noqa: E402  (13b: single source)


def flow_dir_body(alpha_deg: float) -> np.ndarray:
    """Freestream direction in the body/STL frame for pitch alpha (deg).

    Sign fixed EMPIRICALLY from pt 88 (alpha -10): exp st1 Cp is +0.72 on the
    upper side (phi 47.6) and -0.48 below (phi 150) -> windward = UPPER at
    nose-down, i.e. u = (cos a, 0, +sin a) (robin/11; note the swapped
    windward/acceleration labels in robin/10 s2-s3)."""
    a = math.radians(alpha_deg)
    return np.array([math.cos(a), 0.0, math.sin(a)])


def read_sim_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """(cp_exp[176], cp_sim[176]) back from a robin_anchor --csv output."""
    cp_exp = np.full(176, np.nan)
    cp_sim = np.full(176, np.nan)
    for r in csv.DictReader(open(path)):
        i = int(r["orifice"]) - 1
        if r["cp_exp"] != "":
            cp_exp[i] = float(r["cp_exp"])
        if r["cp_sim"] != "":
            cp_sim[i] = float(r["cp_sim"])
    return cp_exp, cp_sim


def select_channel_cp(cp05: np.ndarray, cp15: np.ndarray, kh: np.ndarray,
                      kh_star: float) -> np.ndarray:
    """Per-orifice wall-Cp channel choice (robin/11 K4): the outer channel
    (h 1.5, outside the wall-model numerical layer) where the flow-direction
    normal curvature satisfies kappa_n*h >= kh_star, the wall-attached
    p_state (h 0.5) elsewhere. Hard switch: the logistic blend measured
    slightly worse on all five rotor-off runs (robin/11 K4)."""
    return np.where(kh >= kh_star, cp15, cp05)


def orifice_normals(stl_path: str, xyz_R: np.ndarray) -> np.ndarray:
    """Outward unit normals at the orifices = normal of the nearest STL facet."""
    import trimesh
    from scipy.spatial import cKDTree
    m = trimesh.load(stl_path)
    _, fi = cKDTree(m.triangles_center).query(xyz_R)
    n = np.array(m.face_normals[fi], dtype=np.float64)
    cb = m.triangles_center.mean(axis=0)
    flip = ((m.triangles_center[fi] - cb) * n).sum(axis=1) < 0.0
    n[flip] *= -1.0
    return n


def volume_cp_at_height(level_dir: str, steps: Sequence[int], stl_cfg: dict, xyz_R: np.ndarray,
                        h_cells: float, n_levels: int, rho_phys: float, u_phys: float,
                        scale_R: float) -> np.ndarray:
    """Cp read from the FINEST-level volume output (lbm_<step>_level<k>.vti,
    p_gauge_pa) at h_cells fine cells along the outward normal of every
    orifice, time-averaged over `steps` (robin/08: the surfel p_state is
    sampled 0.5 cell from the wall, inside the slip/cut-cell numerical
    layer whose pressure carries the centrifugal gradient down to the
    wall — on the ROBIN nose that reads 0.05-0.09 low; the pressure at the
    edge of that layer, ~1.5-2 cells, matches the measured wall Cp)."""
    import glob as _glob
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from scipy.ndimage import map_coordinates
    q = 0.5 * rho_phys * u_phys ** 2
    n = orifice_normals(stl_cfg["file"], xyz_R)
    cell_R = 1.0 / scale_R / 2 ** (n_levels - 1)               # fine cell in R
    pts_lu = orifices_to_l0lu(xyz_R + n * h_cells * cell_R, stl_cfg)
    acc = None
    for s in steps:
        f = sorted(_glob.glob(os.path.join(level_dir, f"*{int(s):08d}*level{n_levels - 1}.vti")))
        if not f:
            raise SystemExit(f"no level-{n_levels - 1} vti for step {s} in {level_dir}")
        r = vtk.vtkXMLImageDataReader(); r.SetFileName(f[0]); r.Update(); im = r.GetOutput()
        dims = im.GetDimensions(); org = np.array(im.GetOrigin()); sp = np.array(im.GetSpacing())
        src = im.GetPointData() if im.GetPointData().GetNumberOfArrays() else im.GetCellData()
        p = vtk_to_numpy(src.GetArray("p_gauge_pa")).reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0).astype(np.float64)
        sm = vtk_to_numpy(src.GetArray("solid_mask")).reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0) > 0
        p = np.where(sm, np.nan, p)
        g = (pts_lu - org) / sp
        v = map_coordinates(p, g.T, order=1, mode="nearest") / q
        acc = v if acc is None else acc + v
    return acc / len(steps)


def sample_at_points(pts_lu: np.ndarray, cen: np.ndarray, val: np.ndarray,
                     area: np.ndarray, r_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Area-weighted mean of `val` over triangles (with surfel coverage
    area > 0) within r_s of each point; nearest covered triangle otherwise.
    Returns (sampled, n_triangles_used)."""
    from scipy.spatial import cKDTree
    live = area > 0.0
    tree = cKDTree(cen[live])
    vl, al = val[live], area[live]
    out = np.full(len(pts_lu), np.nan); n_used = np.zeros(len(pts_lu), dtype=int)
    for i, p in enumerate(pts_lu):
        idx = tree.query_ball_point(p, r_s)
        if not idx:
            _, j = tree.query(p); idx = [j]
        w = al[idx]
        out[i] = float(np.sum(vl[idx] * w) / np.sum(w)); n_used[i] = len(idx)
    return out, n_used


# ──────────────────────────────────────────────────────────────────────
# Metrics (pre-registered readouts, patch robin/02)
# ──────────────────────────────────────────────────────────────────────
def compare(orf: Dict[str, np.ndarray], cp_exp: np.ndarray, cp_sim: np.ndarray
            ) -> Dict[str, object]:
    """Station-wise and grouped RMS / mean offsets, sim - exp, mirror-mean
    exp basis (both sides of the sim are sampled and averaged the same way)."""
    _, _, exp_m = mirror_pairs(cp_exp)
    _, _, sim_m = mirror_pairs(cp_sim)
    st = orf["station"][:88]
    d = sim_m - exp_m
    per = {}
    for s in range(1, 15):
        sel = (st == s) & np.isfinite(d)
        per[s] = {"x_R": float(orf["xyz"][:88][st == s, 0].mean()),
                  "n": int(sel.sum()),
                  "rms": float(np.sqrt(np.mean(d[sel] ** 2))) if sel.any() else float("nan"),
                  "mean": float(np.mean(d[sel])) if sel.any() else float("nan"),
                  "exp_min": float(np.nanmin(exp_m[st == s])),
                  "sim_min": float(np.nanmin(sim_m[st == s]))}
    def grp(stations):
        sel = np.isin(st, stations) & np.isfinite(d)
        return {"rms": float(np.sqrt(np.mean(d[sel] ** 2))),
                "mean": float(np.mean(d[sel]))}
    ok = np.isfinite(d)
    return {"per_station": per,
            "all": {"rms": float(np.sqrt(np.mean(d[ok] ** 2))),
                    "mean": float(np.mean(d[ok]))},
            "fore": grp(FORE_STATIONS), "nose": grp(NOSE_STATIONS),
            "pylon": grp(PYLON_STATIONS), "aft": grp(AFT_STATIONS),
            "sim_LR_asym_rms": float(np.sqrt(np.nanmean((cp_sim[:88] - cp_sim[88:]) ** 2))),
            "exp_LR_asym_rms": float(np.sqrt(np.nanmean((cp_exp[:88] - cp_exp[88:]) ** 2)))}


def print_report(pt: int, cpd: Dict[str, np.ndarray], orf: Dict[str, np.ndarray],
                 cp_sim: Optional[np.ndarray] = None) -> None:
    cp = cpd["cp"]
    print(f"== TM-80051 Run {cpd['run']} point {pt}: alpha {cpd['alpha_deg']:+.0f} deg, "
          f"beta {cpd['beta_deg']:.0f}, V {cpd['V_kt']} kt (rotor off)")
    s, p, m = mirror_pairs(cp)
    dd = np.abs(s - p)
    print(f"   mirror asymmetry |dCp|: mean {np.nanmean(dd):.4f}, max {np.nanmax(dd):.3f}; "
          f"flags: {int(np.sum(cpd['flag'] != ''))} cells, unreadable: "
          f"{int(np.sum(np.isnan(cp)))}")
    hdr = "   st  x/R    n   Cp_min(exp)  Cp_max(exp)  phi(Cp_min)"
    if cp_sim is not None:
        hdr += "   | Cp_min(sim)  rms(sim-exp)  mean(sim-exp)"
    print(hdr)
    rep = compare(orf, cp, cp_sim) if cp_sim is not None else None
    for st in range(1, 15):
        rows = station_table(orf, cp, st)
        vals = np.array([r[2] for r in rows]); phis = np.array([r[1] for r in rows])
        k = int(np.nanargmin(vals))
        line = (f"   {st:2d}  {rows[0][0] and orf['xyz'][orf['station'] == st, 0].mean():.3f}  "
                f"{len(rows):2d}   {np.nanmin(vals):+.3f}       {np.nanmax(vals):+.3f}      "
                f"{phis[k]:+6.1f}")
        if rep is not None:
            ps = rep["per_station"][st]
            line += f"   | {ps['sim_min']:+.3f}       {ps['rms']:.4f}        {ps['mean']:+.4f}"
        print(line)
    if rep is not None:
        print(f"   ALL  rms {rep['all']['rms']:.4f} mean {rep['all']['mean']:+.4f} | "
              f"fore(1-2) rms {rep['fore']['rms']:.4f} mean {rep['fore']['mean']:+.4f} | "
              f"nose(1-4) rms {rep['nose']['rms']:.4f} | pylon(8-11) rms {rep['pylon']['rms']:.4f} | "
              f"aft(12-14) rms {rep['aft']['rms']:.4f}")
        print(f"   L/R asymmetry rms: sim {rep['sim_LR_asym_rms']:.4f}  exp {rep['exp_LR_asym_rms']:.4f}")


def plot_stations(orf, cp_exp, cp_sim, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 7, figsize=(22, 7), sharey=True)
    for st, ax in zip(range(1, 15), axes.ravel()):
        rows = station_table(orf, cp_exp, st)
        phi = np.array([r[1] for r in rows]); v = np.array([r[2] for r in rows])
        ax.plot(phi, v, "ko", ms=4, label="TM-80051")
        if cp_sim is not None:
            rs = station_table(orf, cp_sim, st)
            ax.plot([r[1] for r in rs], [r[2] for r in rs], "r-s", ms=3, lw=1, label="LBM")
        ax.set_title(f"st {st}  x/R={orf['xyz'][orf['station'] == st, 0].mean():.3f}", fontsize=9)
        ax.set_xlim(-185, 185); ax.grid(alpha=0.3); ax.invert_yaxis()
        ax.set_xlabel("phi [deg] (0=top, +90=stbd)", fontsize=7)
    axes[0, 0].set_ylabel("Cp"); axes[0, 0].legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=130); print(f"   plot -> {path}")


def _write_csv(path: str, orf, cpd, cp_sim) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orifice", "station", "side", "x_R", "y_R", "z_R", "phi_deg",
                    "cp_exp", "flag", "cp_sim"])
        for i in range(176):
            w.writerow([int(orf["orifice"][i]), int(orf["station"][i]), orf["side"][i],
                        *[f"{v:.4f}" for v in orf["xyz"][i]], f"{orf['phi_deg'][i]:.2f}",
                        "" if np.isnan(cpd["cp"][i]) else f"{cpd['cp'][i]:.3f}",
                        cpd["flag"][i],
                        "" if cp_sim is None else f"{cp_sim[i]:.4f}"])
    print(f"   csv -> {path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--point", type=int, default=90, choices=sorted(POINT_CONDITIONS))
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--config", default=None, help="run config (.py) for the sim overlay")
    ap.add_argument("--surface", nargs="*", default=None,
                    help="surface_*.vtk files (time-averaged); globs allowed")
    ap.add_argument("--r-sample", type=float, default=1.5,
                    help="sampling radius in FINE (surfel-level) cells")
    ap.add_argument("--p-inf", type=float, default=1.0 / 3.0, help="lattice p_inf (rho_inf/3)")
    ap.add_argument("--p-channel", default="auto",
                    choices=["auto", "p_sknh", "p_state", "p_state_ph", "p_use"],
                    help="surface pressure array: auto = p_sknh (robin/13b "
                         "kappa_n*h-selected default) with p_use fallback on "
                         "older files; p_state(=old p_use) = wall-attached "
                         "sample_h channel; p_state_ph = raw p_sample_h "
                         "channel (robin/10b)")
    ap.add_argument("--volume-dir", default=None,
                    help="finest-level vti directory (…/vtk/level3): read the wall Cp from the VOLUME "
                         "at --p-h fine cells along the orifice normal instead of the surface file (robin/08)")
    ap.add_argument("--select-from", nargs=2, default=None, metavar=("CSV_H05", "CSV_H15"),
                    help="kappa_n*h channel selection (robin/11 K4) from two existing "
                         "--csv outputs: the h0.5 surface p_state readout and the h1.5 "
                         "readout (volume or p_state_ph). Needs --config (STL + levels) "
                         "and --point (alpha). Overrides --surface/--volume-dir.")
    ap.add_argument("--kh-star", type=float, default=0.0034,
                    help="kappa_n(flow)*h switch threshold for --select-from "
                         "(robin/11: pooled logistic 0.0034, LORO 0.0029-0.0035, "
                         "ALL-rms plateau 0.002-0.008)")
    ap.add_argument("--p-h", type=float, default=1.5, help="normal sampling height [fine cells] for --volume-dir")
    ap.add_argument("--steps", nargs="*", type=int, default=None, help="steps for --volume-dir (default: from --surface file names)")
    ap.add_argument("--plot", default=None); ap.add_argument("--csv", default=None)
    ap.add_argument("--stations", action="store_true", help="print orifice-level tables")
    a = ap.parse_args(argv)

    orf = load_orifices(os.path.join(a.data, "tm80051_orifices.csv"))
    cpd = load_cp(os.path.join(a.data, "tm80051_cp_rotor_off.csv"))[a.point]
    cp_sim = None
    if a.select_from:
        if a.config is None:
            sys.exit("--select-from needs --config (STL placement + levels)")
        cfg = load_config(a.config)
        stl = cfg["internal_geometry"]["stl"]
        nlev = int(cfg.get("mlg", {}).get("num_levels", 1))
        cell_R = 1.0 / float(stl["scale_to_lu"]) / 2 ** (nlev - 1)
        _, cp05 = read_sim_csv(a.select_from[0])
        _, cp15 = read_sim_csv(a.select_from[1])
        curv = orifice_curvature_quadric(stl["file"], orf["xyz"])
        alpha = POINT_CONDITIONS[a.point]["alpha_deg"]
        kn = kappa_n_direction(curv, flow_dir_body(alpha))
        kh = kn * a.p_h * cell_R
        cp_sim = select_channel_cp(cp05, cp15, kh, a.kh_star)
        n15 = int(np.sum(kh >= a.kh_star))
        print(f"== sim (kappa_n*h channel selection, robin/11): kh* {a.kh_star:g}, "
              f"h {a.p_h:g} fine cells (cell {cell_R:.5f} R), alpha {alpha:+.0f}; "
              f"h1.5 channel at {n15}/176 orifices")
        print_report(a.point, cpd, orf, cp_sim)
        if a.csv:
            _write_csv(a.csv, orf, cpd, cp_sim)
        if a.plot:
            plot_stations(orf, cpd["cp"], cp_sim, a.plot,
                          f"ROBIN rotor-off Cp — TM-80051 pt {a.point} "
                          f"(kappa_n*h selected, kh*={a.kh_star:g})")
        return
    if a.surface:
        if a.config is None:
            sys.exit("--surface needs --config (STL placement + U_lu)")
        cfg = load_config(a.config)
        files = sorted(sum((glob.glob(s) for s in a.surface), []))
        if not files:
            sys.exit("no surface files matched")
        u_lu = float(cfg["physics"]["initial_flow_velocity"][0])
        q_inf = 0.5 * u_lu ** 2
        stl = cfg["internal_geometry"]["stl"]
        nlev = int(cfg.get("mlg", {}).get("num_levels", 1))
        r_s = a.r_sample / 2 ** (nlev - 1)          # fine cells -> L0 lu
        pts_lu = orifices_to_l0lu(orf["xyz"], stl)
        if a.volume_dir:
            steps = a.steps or [int(re.search(r"surface_(\d+)", f).group(1)) for f in files]
            cp_sim = volume_cp_at_height(a.volume_dir, steps, stl, orf["xyz"], a.p_h, nlev,
                                         float(cfg["physics"]["rho"]), float(cfg["physics"]["U_inf"]),
                                         float(stl["scale_to_lu"]))
            print(f"== sim (VOLUME, h = {a.p_h:g} fine cells along the orifice normal): steps {steps}; "
                  f"nan {int(np.isnan(cp_sim).sum())}")
            print_report(a.point, cpd, orf, cp_sim)
            if a.csv:
                _write_csv(a.csv, orf, cpd, cp_sim)
            if a.plot:
                plot_stations(orf, cpd["cp"], cp_sim, a.plot,
                              f"ROBIN rotor-off Cp — TM-80051 pt {a.point} (volume p at h={a.p_h:g} cells)")
            return
        cen, cpm, cps, area, off = surface_cp_mean(files, a.p_inf, q_inf,
                                                   channel=a.p_channel)
        cp_sim, n_used = sample_at_points(pts_lu, cen, cpm, area, r_s)
        print(f"== sim: {len(files)} surface file(s) {os.path.basename(files[0])} .. "
              f"{os.path.basename(files[-1])}; U_lu {u_lu:.5f} q_inf {q_inf:.4e} "
              f"p_inf {a.p_inf:.6f}; r_s {r_s:.3f} L0 lu; triangles/orifice "
              f"min {n_used.min()} median {int(np.median(n_used))}; "
              f"file-Cp offset (area-mean p_ref) {off:+.4f}; "
              f"per-triangle time-std median {np.median(cps):.4f}")
    print_report(a.point, cpd, orf, cp_sim)
    if a.stations:
        for st in range(1, 15):
            print(f"   -- station {st}")
            for o, phi, v, side in station_table(orf, cpd["cp"], st):
                extra = f"  sim {cp_sim[o - 1]:+.3f}" if cp_sim is not None else ""
                print(f"      orifice {o:3d} phi {phi:+7.1f} Cp {v:+.3f} {side}{extra}")
    if a.csv:
        _write_csv(a.csv, orf, cpd, cp_sim)
    if a.plot:
        plot_stations(orf, cpd["cp"], cp_sim, a.plot,
                      f"ROBIN rotor-off Cp — TM-80051 pt {a.point} "
                      f"(alpha {cpd['alpha_deg']:+.0f}, {cpd['V_kt']} kt)")


if __name__ == "__main__":
    main()
