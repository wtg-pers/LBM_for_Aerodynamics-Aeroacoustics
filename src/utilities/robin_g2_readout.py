"""robin/13 Phase-1 grid-ladder readout (2-level r10/20/30/40).

One command once the run directories exist. Per run, in the pre-registered
order (patch robin/13 s3):
  A  representability gate: setup-log overlap-cap/crease stats, surface
     coverage per x-band (incl. the boom tip 1.9-2.0R), NaN share, boom
     solid thickness [fine cells] from the last level-1 volume
  B  forces: window (last 2.5 FT = last 25 % of steps) mean/std of Cd, Cz
     from force_history.csv + half-window drift = the noise denominator
  C  pressure/friction split: scale-invariant fractions from the window
     surface files (F = sum traction*a, F_f = sum tau*a) applied to the
     window Cd -> Cd_p (PRIMARY QoI with Cz), Cd_f (operating-point row)
  D  wall-model operating point: area-weighted Cf -> y+ at h_law = 3 cells
  E  Cp rms (secondary): h05 (p_use) / ph (p_state_ph) / selective
     kappa_n*h (offline robin/11 selection; h = the config's p_sample_h
     [fine cells], kh* derived from it -- robin/16 sec. 3) -- old-code runs
     carry no p_sknh channel, so the selection is applied HERE
     (post-processing), per the user's call.
Outputs: table to stdout, temp_results/robin/robin_g2_ladder.csv + .png.

Usage (main dir):
    python -m src.utilities.robin_g2_readout            # all four runs
    python -m src.utilities.robin_g2_readout --runs 10 20
    python -m src.utilities.robin_g2_readout --base temp_results/robin
Cluster is Python 3.9: no PEP 604 annotations.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.utilities.robin_anchor import (POINT_CONDITIONS, _REPO, compare,
                                        flow_dir_body, kappa_n_direction,
                                        load_config, load_cp, load_orifices,
                                        orifice_curvature_quadric,
                                        orifices_to_l0lu, read_surface_vtk,
                                        sample_at_points, surface_cp_mean)

RES = os.path.join(_REPO, "temp_results", "robin")
#: user-designated drop point for the Phase-1 ladder results (0901)
RES_GRID = os.path.join(RES, "grid_test", "01_baseline_L1_difference_r_values")
#: outer-channel height / kh* come from the run config (p_sample_h) via
#: surfel_kappa.kh_star_for (robin/16); this is only the pre-16 fallback
#: for configs that set no p_sample_h.
H_CELLS_FALLBACK = 1.5
U_PHYS, NU_PHYS = 42.03, 1.4610e-5
R_PHYS = 1.574


def find_dir(base: str, tag: str) -> Optional[str]:
    for root in (base, RES_GRID, RES, "."):
        for name in (f"results_{tag}", tag):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                return p
    return None


def gate_representability(d: str, cfg: dict, r: int, nlev: int = 2) -> Dict[str, object]:
    out: Dict[str, object] = {}
    log = os.path.join(d, "csv", "setup_log.txt")
    if os.path.exists(log):
        txt = open(log).read()
        mm = re.findall(r"overlap cap.*?: (\d+) \(cell,dir\) sums capped, "
                        r"max g/dV ([\d.]+)", txt)
        if mm:                       # per-level build lines; last = finest
            out["cap_sums"] = int(mm[-1][0]); out["cap_max"] = float(mm[-1][1])
        m = re.search(r"crease_mode=noslip: (\d+) of (\d+) facets", txt)
        if m:
            out["crease_frac"] = int(m.group(1)) / int(m.group(2))
    surfs = sorted(glob.glob(os.path.join(d, "vtk", "surface_*.vtk")))
    if surfs:
        pts, poly, F = read_surface_vtk(surfs[-1])
        cen = pts[poly].mean(axis=1)
        xR = (cen[:, 0] - 4.0 * r) / r
        area = F["area"]
        pch = F.get("p_use", F.get("p_state"))
        out["nan_share"] = float(np.mean(~np.isfinite(pch[area > 0])))
        cov = {}
        for a, b in ((0.0, 1.0), (1.0, 1.9), (1.9, 2.0)):
            m2 = (xR >= a) & (xR < b)
            cov[f"{a:g}-{b:g}"] = float((area[m2] > 0).mean()) if m2.any() else np.nan
        out["coverage"] = cov
    # boom solid thickness from the last FINEST-level volume (fine cells)
    lv = f"level{nlev - 1}"
    vols = sorted(glob.glob(os.path.join(d, "vtk", lv, f"*{lv}.vti")))
    if vols:
        import vtk as _vtk
        from vtk.util.numpy_support import vtk_to_numpy
        rd = _vtk.vtkXMLImageDataReader(); rd.SetFileName(vols[-1]); rd.Update()
        im = rd.GetOutput(); dims = im.GetDimensions()
        org = np.array(im.GetOrigin()); sp = np.array(im.GetSpacing())
        src = im.GetPointData() if im.GetPointData().GetNumberOfArrays() else im.GetCellData()
        sm = vtk_to_numpy(src.GetArray("solid_mask")).reshape(
            dims[2], dims[1], dims[0]).transpose(2, 1, 0) > 0
        def _box_solid(x0, x1, half):
            """solid cells + max z-run inside a body-frame box around the
            axis (y 0 +- half, z Z0 +- half)."""
            lo = ((np.array([4.0 * r + x0 * r, 3.0 * r - half * r,
                             3.0 * r - half * r]) - org) / sp)
            hi = ((np.array([4.0 * r + x1 * r, 3.0 * r + half * r,
                             3.0 * r + (half + 0.05) * r]) - org) / sp)
            lo = np.clip(np.floor(lo).astype(int), 0, np.array(dims) - 1)
            hi = np.clip(np.ceil(hi).astype(int) + 1, 1, np.array(dims))
            box = sm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            runs = 0
            if box.any():
                zproj = box.any(axis=(0, 1))
                runs = max(len(list(g)) for k, g in __import__("itertools")
                           .groupby(zproj) if k)
            return int(box.sum()), int(runs)
        out["boom19_solid"], out["boom19_zrun"] = _box_solid(1.85, 1.95, 0.10)
        out["st14_solid"], out["st14_zrun"] = _box_solid(1.48, 1.58, 0.12)
    return out


def force_window(d: str, max_steps: int) -> Dict[str, float]:
    rows = list(csv.DictReader(open(os.path.join(d, "csv", "force_history.csv"))))
    st = np.array([int(x["step"]) for x in rows])
    cd = np.array([float(x["Cd"]) for x in rows])
    cz = np.array([float(x["Cz"]) for x in rows])
    w = st >= 0.75 * max_steps
    if not w.any():                          # partial run: use what exists
        print(f"   [B force] WARNING: run shorter than the registered "
              f"window (max step {st.max()} < 0.75*{max_steps}) — using "
              f"the last 25% of available steps")
        w = st >= 0.75 * st.max()
        max_steps = int(st.max())
    h2 = st >= 0.875 * max_steps            # second half of the window
    h1 = w & ~h2
    return {"Cd": float(cd[w].mean()), "Cd_std": float(cd[w].std()),
            "Cz": float(cz[w].mean()), "Cz_std": float(cz[w].std()),
            "Cd_halfdiff": float(abs(cd[h2].mean() - cd[h1].mean())),
            "Cz_halfdiff": float(abs(cz[h2].mean() - cz[h1].mean())),
            "n": int(w.sum())}


def split_and_cf(files: Sequence[str]) -> Dict[str, float]:
    """Window-mean pressure fraction of Fx + area-weighted Cf (surface)."""
    fr, cf = [], []
    for f in files:
        pts, poly, F = read_surface_vtk(f)
        a = F["area"]
        if not np.isfinite(F.get("p_use", F.get("p_state"))[a > 0]).all():
            continue
        Ft = (F["traction"] * a[:, None]).sum(axis=0)
        Ff = (F["tau"] * a[:, None]).sum(axis=0)
        fr.append((Ft[0] - Ff[0]) / Ft[0])
        cf.append(float(np.average(F["Cf"][a > 0], weights=a[a > 0])))
    return {"frac_p": float(np.mean(fr)), "Cf_mean": float(np.mean(cf)),
            "n_files": len(fr)}


def cp_three_ways(d: str, cfg: dict, files: Sequence[str], r: int
                  ) -> Dict[str, float]:
    orf = load_orifices()
    cpd = load_cp()[90]
    stl = cfg["internal_geometry"]["stl"]
    u_lu = float(cfg["physics"]["initial_flow_velocity"][0])
    q_inf = 0.5 * u_lu ** 2
    nlev = int(cfg["mlg"]["num_levels"])
    cell_R = 1.0 / float(stl["scale_to_lu"]) / 2 ** (nlev - 1)
    r_s = 1.5 * cell_R * float(stl["scale_to_lu"])      # fine cells -> L0 lu
    pts_lu = orifices_to_l0lu(orf["xyz"], stl)
    have = set(read_surface_vtk(files[-1])[2].keys())
    sims = {}
    # channels: p_state (= old p_use) always; p_state_ph only on two-channel
    # files (robin/10b); p_sknh = the writer's own selection (robin/13b)
    for ch in [c for c in ("p_use", "p_state_ph", "p_sknh")
               if c in have or (c == "p_use" and "p_state" in have)]:
        cen, cpm, _, area, _ = surface_cp_mean(files, 1.0 / 3.0, q_inf, channel=ch)
        sims[ch], _ = sample_at_points(pts_lu, cen, cpm, area, r_s)
    nan = np.full(176, np.nan)
    from src.boundary.surfel_kappa import kh_star_for
    h_cells = float(stl.get("surfel", {}).get("p_sample_h") or H_CELLS_FALLBACK)
    kh_star = kh_star_for(h_cells)
    curv = orifice_curvature_quadric(stl["file"], orf["xyz"])
    kn = kappa_n_direction(curv, flow_dir_body(0.0))
    kh = kn * h_cells * cell_R
    if "p_sknh" in sims:                 # 13b+ file: the writer already selected
        sel = sims["p_sknh"]
    elif "p_state_ph" in sims:           # 10b two-channel file: select offline
        sel = np.where(kh >= kh_star, sims["p_state_ph"], sims["p_use"])
    else:                                # legacy single-channel file
        sel = sims["p_use"]
    out = {}
    for name, sim in (("h05", sims["p_use"]), ("ph", sims.get("p_state_ph", nan)),
                      ("sknh", sel)):
        out[name] = (compare(orf, cpd["cp"], sim)["all"]["rms"]
                     if np.isfinite(sim).any() else float("nan"))
    out["n_ph_sel"] = int((kh >= kh_star).sum())
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="*", type=int, default=[10, 20, 30, 40])
    ap.add_argument("--tags", nargs="*", default=[],
                    help="extra config tags (e.g. robin_g3_r20, robin/13 s7 H2)")
    ap.add_argument("--base", default=RES_GRID)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = os.path.join(a.base if os.path.isdir(a.base) else RES,
                             "robin_g2_ladder")

    table: List[Dict[str, object]] = []
    tags = [f"robin_g2_r{r}" for r in a.runs] + list(a.tags)
    for tag in tags:
        r = int(re.search(r"_r(\d+)$", tag).group(1))
        d = find_dir(a.base, tag)
        if d is None:
            print(f"== {tag}: NOT FOUND — skipped")
            continue
        cfg = load_config(os.path.join(_REPO, "configs", "robin", f"{tag}.py"))
        max_steps = int(cfg["time"]["max_steps"])
        nlev = int(cfg["mlg"]["num_levels"])
        kf = 2 ** (nlev - 1)
        dx_fine_R = 1.0 / (kf * r)
        print(f"== {tag} ({d}); {nlev} levels, fine = R/{kf * r}, "
              f"boom-min {0.05 * kf * r:.1f} cells")
        g = gate_representability(d, cfg, r, nlev)
        print(f"   [A gate] {g}")
        fw = force_window(d, max_steps)
        print(f"   [B force] Cd {fw['Cd']:.5f}±{fw['Cd_std']:.5f} "
              f"Cz {fw['Cz']:+.5f}±{fw['Cz_std']:.5f}  "
              f"half-window drift Cd {fw['Cd_halfdiff']:.2e} Cz {fw['Cz_halfdiff']:.2e}")
        surfs = sorted(glob.glob(os.path.join(d, "vtk", "surface_*.vtk")))[-5:]
        sc = split_and_cf(surfs)
        cd_p = fw["Cd"] * sc["frac_p"]
        cd_f = fw["Cd"] * (1.0 - sc["frac_p"])
        u_tau = np.sqrt(sc["Cf_mean"] / 2.0) * U_PHYS
        yplus = u_tau * 3.0 * dx_fine_R * R_PHYS / NU_PHYS
        print(f"   [C split] Cd_p {cd_p:.5f}  Cd_f {cd_f:.5f} "
              f"(frac_p {sc['frac_p']:.3f}, {sc['n_files']} files)")
        print(f"   [D wm] Cf {sc['Cf_mean']:.5f} -> y+(h=3) ~ {yplus:.0f}")
        cp = cp_three_ways(d, cfg, surfs, r)
        print(f"   [E Cp rms] h05 {cp['h05']:.4f}  ph {cp['ph']:.4f}  "
              f"sknh {cp['sknh']:.4f} (ph at {cp['n_ph_sel']}/176)")
        table.append({"tag": tag, "r": r, "nlev": nlev,
                      "dx_fine_R": dx_fine_R, **fw,
                      "Cd_p": cd_p, "Cd_f": cd_f, "yplus_h3": yplus,
                      "cp_h05": cp["h05"], "cp_ph": cp["ph"],
                      "cp_sknh": cp["sknh"],
                      "boom19_solid": g.get("boom19_solid"), "boom19_zrun": g.get("boom19_zrun"),
                      "nan_share": g.get("nan_share")})
    if not table:
        return
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        for row in table:
            w.writerow(row)
    print(f"   csv -> {a.out}.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    dx = [t["dx_fine_R"] for t in table]
    axs[0].errorbar(dx, [t["Cd_p"] for t in table],
                    yerr=[t["Cd_halfdiff"] for t in table], marker="o",
                    label="Cd_p (primary)")
    axs[0].plot(dx, [t["Cd_f"] for t in table], "s--", label="Cd_f (wm)")
    axs[0].plot(dx, [t["Cd"] for t in table], "^:", label="Cd total")
    axs[0].set_ylabel("Cd (A_ref = 0.5 R^2)")
    axs[1].errorbar(dx, [t["Cz"] for t in table],
                    yerr=[t["Cz_halfdiff"] for t in table], marker="o")
    axs[1].set_ylabel("Cz")
    axs[2].plot(dx, [t["cp_h05"] for t in table], "o-", label="h05")
    axs[2].plot(dx, [t["cp_ph"] for t in table], "s-", label="ph")
    axs[2].plot(dx, [t["cp_sknh"] for t in table], "^-", lw=2, label="sknh")
    axs[2].set_ylabel("Cp ALL rms (pt 90)")
    for ax in axs:
        ax.set_xlabel("fine cell [R]")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle("robin/13 Phase-1 2-level ladder (window = last 2.5 FT)")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=130)
    print(f"   plot -> {a.out}.png")


if __name__ == "__main__":
    main()
