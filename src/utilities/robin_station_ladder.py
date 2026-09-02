"""ROBIN grid-ladder station-Cp figure (robin/16 s9/s11): one panel per TM-80051
station, the experiment + one line per run, every run read the SAME way:

    kappa_n(flow)*h selective composite at the TRUE sampling height
      p_state  (kernel wall-attached channel, h = 0.5 cell)      where kh <  kh*
      volume   (finest-level vti, orifice PROJECTED onto the STL,
                h fine cells along the outward normal)            where kh >= kh*
    kh = kappa_n * h * cell_R,  kh* = surfel_kappa.kh_star_for(h)

so runs made before robin/16 (p_sample_h 1.5 in their surface files) and
after (1.1) land on one footing (robin/16 s3-correction: the TM orifice
points sit 0.00145 R inside the STL; the raw-orifice probe of 08-15 read
0.2-0.5 cells low). Window = the last N surface files (default 5) and the
matching finest-level volumes.

Usage (main dir):
    python -m src.utilities.robin_station_ladder \\
        --tags robin_g3_r20 robin_g4_r20 robin_g5_r20 robin_g6_r20 \\
        --out temp_results/robin/robin_ladder_g3_g6_station_cp
Missing run folders are reported and skipped (the g6 line appears as soon
as results_robin_g6_r20 exists under --base / the grid_test drop / cwd).
Cluster is Python 3.9: no PEP 604 annotations.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.boundary.surfel_kappa import kh_star_for
from src.utilities.robin_anchor import (_REPO, _write_csv, compare,
                                        flow_dir_body, kappa_n_direction,
                                        load_config, load_cp, load_orifices,
                                        orifice_curvature_quadric,
                                        orifices_to_l0lu, sample_at_points,
                                        select_channel_cp, station_table,
                                        surface_cp_mean, surface_projection)
from src.utilities.robin_g2_readout import find_dir, RES


def _read_volume(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(p_gauge_pa with solid -> NaN, origin, spacing, dims); only the two
    arrays needed are read (a 6-level finest volume is ~4 GB on disk)."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    rd = vtk.vtkXMLImageDataReader()
    rd.SetFileName(path)
    try:
        rd.UpdateInformation()
        for i in range(rd.GetNumberOfPointArrays()):
            nm = rd.GetPointArrayName(i)
            rd.SetPointArrayStatus(nm, 1 if nm in ("p_gauge_pa", "solid_mask") else 0)
    except Exception:
        pass
    rd.Update()
    im = rd.GetOutput()
    dims = im.GetDimensions()
    org = np.array(im.GetOrigin())
    sp = np.array(im.GetSpacing())
    src = im.GetPointData()
    p = vtk_to_numpy(src.GetArray("p_gauge_pa")).reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0).astype(np.float64)
    sm = vtk_to_numpy(src.GetArray("solid_mask")).reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0) > 0
    return np.where(sm, np.nan, p), org, sp, np.array(dims)


def composite_for_run(tag: str, d: str, h: float, n_files: int,
                      orf: dict, proj: Tuple[np.ndarray, np.ndarray],
                      kn: np.ndarray) -> Dict[str, object]:
    """Selective composite Cp[176] for one run + metadata."""
    from scipy.ndimage import map_coordinates
    cfg = load_config(os.path.join(_REPO, "configs", "robin", f"{tag}.py"))
    stl = cfg["internal_geometry"]["stl"]
    nlev = int(cfg["mlg"]["num_levels"])
    r = float(stl["scale_to_lu"])
    cell_R = 1.0 / r / 2 ** (nlev - 1)
    lv = f"level{nlev - 1}"
    q = 0.5 * float(cfg["physics"]["rho"]) * float(cfg["physics"]["U_inf"]) ** 2
    u_lu = float(cfg["physics"]["initial_flow_velocity"][0])
    q_lu = 0.5 * u_lu ** 2
    r_s = 1.5 / 2 ** (nlev - 1)
    files = sorted(glob.glob(os.path.join(d, "vtk", "surface_*.vtk")))[-n_files:]
    if not files:
        raise SystemExit(f"{tag}: no surface files under {d}/vtk")
    steps = [int(re.search(r"surface_(\d+)", f).group(1)) for f in files]
    cen, cpm, _, area, _ = surface_cp_mean(files, 1.0 / 3.0, q_lu, channel="p_state")
    h05, _ = sample_at_points(orifices_to_l0lu(orf["xyz"], stl), cen, cpm, area, r_s)
    closest, n = proj
    acc = np.zeros(176)
    used = []
    for s in steps:
        vf = sorted(glob.glob(os.path.join(d, "vtk", lv, f"*{s:08d}*{lv}.vti")))
        if not vf:
            print(f"   [{tag}] no {lv} volume for step {s} — skipped")
            continue
        p, org, sp, _ = _read_volume(vf[0])
        g = (orifices_to_l0lu(closest + n * h * cell_R, stl) - org) / sp
        acc += map_coordinates(p, g.T, order=1, mode="nearest") / q
        used.append(s)
    if not used:
        raise SystemExit(f"{tag}: no finest-level volumes under {d}/vtk/{lv}")
    vol = acc / len(used)
    kh = kn * h * cell_R
    ks = kh_star_for(h)
    sel = select_channel_cp(h05, vol, kh, ks)
    fine = int(round(r * 2 ** (nlev - 1)))
    return {"tag": tag, "cp": sel, "h05": h05, "vol": vol, "nlev": nlev, "r": int(r),
            "fine": fine, "label": f"R/{fine} {tag.replace('robin_', '')}",
            "n_outer": int((kh >= ks).sum()), "steps": used}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="+", required=True, help="run tags = configs/robin/<tag>.py")
    ap.add_argument("--base", default=RES, help="results root (results_<tag> looked up here, the grid_test drop, then cwd)")
    ap.add_argument("--h", type=float, default=1.1, help="TRUE outer sample height [fine cells] (robin/16: layer edge 1.0-1.1)")
    ap.add_argument("--n-files", type=int, default=5, help="window = last N surface files")
    ap.add_argument("--point", type=int, default=90)
    ap.add_argument("--out", default=None, help="output stem (.png + per-run _<tag>_cp_sel.csv)")
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = os.path.join(RES, "robin_ladder_station_cp_h%g" % a.h)
    orf = load_orifices()
    cpd = load_cp()[a.point]
    stl0 = load_config(os.path.join(_REPO, "configs", "robin", f"{a.tags[0]}.py"))["internal_geometry"]["stl"]
    closest, n, signed = surface_projection(stl0["file"], orf["xyz"])
    kn = kappa_n_direction(orifice_curvature_quadric(stl0["file"], orf["xyz"]), flow_dir_body(0.0))
    print(f"== selective composite, TRUE h = {a.h:g} cells, kh* = {kh_star_for(a.h):.5f}; "
          f"orifice->STL offset mean {signed.mean():+.5f} R (projected out)")
    runs: List[Dict[str, object]] = []
    for tag in a.tags:
        d = find_dir(a.base, tag)
        if d is None:
            print(f"== {tag}: NOT FOUND — skipped (drop results_{tag} under {a.base}, the grid_test folder or cwd)")
            continue
        res = composite_for_run(tag, d, a.h, a.n_files, orf, (closest, n), kn)
        c = compare(orf, cpd["cp"], res["cp"])
        res["cmp"] = c
        runs.append(res)
        _write_csv(f"{a.out}_{tag}_cp_sel.csv", orf, cpd, res["cp"])
        print(f"   {res['label']:16s} {res['nlev']} levels, outer {res['n_outer']}/176, steps {res['steps']} | "
              f"ALL {c['all']['rms']:.4f} ({c['all']['mean']:+.4f}) fore {c['fore']['rms']:.4f} "
              f"nose {c['nose']['rms']:.4f} pylon {c['pylon']['rms']:.4f} aft {c['aft']['rms']:.4f}")
    if not runs:
        print("nothing to plot")
        return
    runs.sort(key=lambda t: t["fine"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    marks = ["v", "d", "s", "o", "^", "P", "X"]
    fig, axes = plt.subplots(2, 7, figsize=(22, 7), sharey=True)
    for st, ax in zip(range(1, 15), axes.ravel()):
        rows = station_table(orf, cpd["cp"], st)
        ax.plot([t[1] for t in rows], [t[2] for t in rows], "ko", ms=4.5, label="TM-80051 (exp)", zorder=6)
        for i, res in enumerate(runs):
            rs = station_table(orf, res["cp"], st)
            col = cmap(0.15 + 0.7 * i / max(1, len(runs) - 1))
            ax.plot([t[1] for t in rs], [t[2] for t in rs], "-", marker=marks[i % len(marks)],
                    color=col, ms=3, lw=1.0 + 0.3 * i, label=res["label"])
        ax.set_title(f"st {st}  x/R={orf['xyz'][orf['station'] == st, 0].mean():.3f}", fontsize=9)
        ax.set_xlim(-185, 185)
        ax.grid(alpha=0.3)
        ax.invert_yaxis()
        ax.set_xlabel("phi [deg] (0=top, +90=stbd)", fontsize=7)
    axes[0, 0].set_ylabel("Cp")
    axes[0, 0].legend(fontsize=6.5, loc="lower left")
    txt = "ALL rms: " + "  |  ".join(f"{r['label'].split()[0]} {r['cmp']['all']['rms']:.4f}" for r in runs)
    fig.suptitle(f"ROBIN rotor-off Cp — TM-80051 pt {a.point}, kappa_n*h selective "
                 f"(p_state h0.5 vs TRUE h={a.h:g} projected probe, kh*={kh_star_for(a.h):.5f}) — "
                 f"{', '.join(r['tag'].replace('robin_', '') for r in runs)}\n{txt}", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=130)
    print(f"   plot -> {a.out}.png")


if __name__ == "__main__":
    main()
