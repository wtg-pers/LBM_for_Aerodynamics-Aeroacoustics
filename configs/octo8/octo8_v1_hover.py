"""Octo-8 v1 — 4-level MLG, 10 m outwash field. **5-rev SMOKE.**

What this config is for
-----------------------
The first structure sized against the actual production question (outwash in
ground effect) rather than against what fitted a single GPU. Decided with the
user, 2026-08-09; the sizing arithmetic is in patch_notes/octo8_v1/.

    far-field   10.0 x 10.0 m centred on the vehicle, ground to inflow
    기체        D/80  (dx 5.715 mm) — same as the validated mlg2
    로터        D/160 (dx 2.857 mm) — 2x better than mlg2 (D/80), half of the
                single-prop validation (configs/apc18x8e runs it at D/320)

Compare rotor resolution by dx, not by the summary's c_tip/dx: that metric
uses the chord at the OUTERMOST MARKER, so it moves with n_radial as well as
with the grid (this file prints 2.47 at n=48; the same grid at n=32 would
print ~3.2). The apc18x8e header's "9.8 cells" is a third thing again — the
chord at 0.95R over dx. Three definitions, not comparable to each other.

Why these numbers
-----------------
* 10 m: the user measured ~5 m of influence in another solver at a hover
  height of 1911 mm. This case hovers at 955.5 mm — h/D 2.09 vs 4.18, i.e.
  properly in ground effect where the wall jet is stronger and reaches
  further. 5 m from the vehicle centre is therefore a floor, not a ceiling.
  It gives 1.61 CD of clearance from the body surface (CD = 1912.8 mm, the
  vehicle's circumscribed radius), against the solver's 0.5*L_body rule which
  is numerically the same thing as 1.0 CD.
* L0 = D/20: a 10 m field at D/40 is 155 M cells and does not build on a
  24 GB card. The far field does not need to resolve the rotor.
* 4 levels, not 5: level count was the agreed thing to spend, but each level
  adds a coupling interface and the interface is the one error source we have
  actually measured (~5% on body force when it cuts a wall,
  patch_notes/wall_coupling/01). A 5-level variant reaches the same rotor
  resolution for 20% fewer cells but with a 4th interface — rejected.
* D/320 (matching the single-prop validation) needs 303 M cells and a 53 GB
  build peak. It does not fit 4x RTX 4090 and would take days. Deferred.

★ The rotor level cuts the airframe — deliberately
--------------------------------------------------
Measured: all 8 rotor blocks intersect the airframe on 3-4 faces each
(41,490 STL vertices inside every block). The arm sits at z ~ 121 mm, the
disk at 133.7 mm, and the ALM Gaussian support needs 0.25 D below the disk,
so the block cannot be lifted clear of the arm. The arms run to the booms and
the booms to the fuselage, so no finite block avoids the cut.

This is why `wall_coupling` exists. `mode='allow'` and not `'exclude'`: this
case uses wall_bc='ibb', and the C2F wall exclusion was measured to HELP with
hwbb (+1.90) but HURT with ibb (-2.89) — IBB's deposit-rewrite pass already
corrects what C2F writes, so excluding those cells only starves them
(patch_notes/wall_coupling/01 §3a).

Consequence to keep in mind when reading results: rotor thrust is unaffected
by the cut (measured: 7e-6 relative), but **body download carries ~5%
uncertainty**. If the airframe load is the quantity of interest, judge it
against a whole-vehicle-fine run, not against this one.

Smoke scope (this file)
-----------------------
5 revolutions, full-field VTK every 30 deg from step 0. The goal is that it
builds, runs, conserves, and produces sane rotor thrust — not a physics
answer. 5 rev with a 1-rev ALM ramp is nowhere near a settled wake.

Run (cluster 1, 4x RTX 4090):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \\
        --config configs/octo8/octo8_v1_hover.py --gpu 0,1,2,3 --cuda-aware 1

Single GPU (slower, for a first look):
    LBM_ESOTERIC=1 python main.py \\
        --config configs/octo8/octo8_v1_hover.py --gpu 0
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import (  # noqa: E402
    _BBOX_MAX_MM, _BBOX_MIN_MM, _LAYOUT, D_PHYS, ROTOR_Z_MM,
    build_config, grid_map_centered)

# ── L0: rotor at D/20, domain +-5 m about the vehicle centre ──────────
D_LU_0 = 20
HALF_XY_MM = 5000.0

# markers/blade: delta_r = span/n must stay <= eps, and eps has a 2*dx floor.
# At the rotor level (D/160, dx = 2.857 mm) that floor is 5.71 mm, so the
# active span 178 mm needs n >= 31. 48 gives 3.7 mm.  (The single-prop
# validation runs 64 at D/320 by the same rule.)
N_RADIAL = 48

config = build_config(rpm=5000.0, n_rev=5, n_radial=N_RADIAL,
                      vtk_deg=30.0, vtk_fields_last_rev=5, wall_bc="ibb",
                      d_lu=D_LU_0, half_xy_mm=HALF_XY_MM)

_dx, _mm2lu, _O, _N = grid_map_centered(D_LU_0, HALF_XY_MM)


def _lu(p_mm):
    return _O + np.asarray(p_mm, dtype=np.float64) * _mm2lu


# ── boxes, all in L0 lattice units (setup.py:1518 — regions are L0 coords) ──
_b_lo, _b_hi = _lu(_BBOX_MIN_MM), _lu(_BBOX_MAX_MM)        # airframe bbox

# L3 rotor blocks: the proportions the single-prop validation used
#   (EXTENTS[-1] = up 0.125 D, down 0.25 D, lat 0.6875 D)
_LAT = 0.6875 * D_LU_0                                     # 13.75 lu
_UP, _DN = 0.125 * D_LU_0, 0.25 * D_LU_0
_rz = float(_lu([0.0, 0.0, ROTOR_Z_MM])[2])
_rotor_boxes = []
for _name, _x_mm, _y_mm, _hand in _LAYOUT:
    _h = _lu([_x_mm, _y_mm, ROTOR_Z_MM])
    _rotor_boxes.append({
        "name": _name,
        "x_min": float(_h[0] - _LAT), "x_max": float(_h[0] + _LAT),
        "y_min": float(_h[1] - _LAT), "y_max": float(_h[1] + _LAT),
        "z_min": float(_rz - _DN),    "z_max": float(_rz + _UP),
    })

# L2 (D/80) must contain the airframe AND every L3 block, plus L3's band
# (2 parent cells = 2 * dx_L2 = 0.5 L0 lu). PAD2 is on top of that.
PAD2 = 2.0
_r_lo = np.min([[b["x_min"], b["y_min"], b["z_min"]] for b in _rotor_boxes], 0)
_r_hi = np.max([[b["x_max"], b["y_max"], b["z_max"]] for b in _rotor_boxes], 0)
_l2_lo = np.floor(np.minimum(_b_lo, _r_lo) - PAD2)
_l2_hi = np.ceil(np.maximum(_b_hi, _r_hi) + PAD2)

# L1 (D/40) contains L2 + L2's band (2 L0 cells) + PAD1.
PAD1 = 5.0
_l1_lo = np.floor(_l2_lo - PAD1)
_l1_hi = np.ceil(_l2_hi + PAD1)


def _box(name, lo, hi):
    return {"name": name,
            "x_min": float(lo[0]), "x_max": float(hi[0]),
            "y_min": float(lo[1]), "y_max": float(hi[1]),
            "z_min": float(lo[2]), "z_max": float(hi[2])}


config["mlg"] = {
    "enabled": True, "num_levels": 4, "overlap_width": 2,
    "interpolation": "cubic", "filter_level": 1,
    "levels": [
        {},                                                    # L0  D/20  far
        {"regions": [_box("near", _l1_lo, _l1_hi)]},            # L1  D/40
        {"regions": [_box("airframe", _l2_lo, _l2_hi)]},        # L2  D/80
        {"regions": _rotor_boxes},                              # L3  D/160 x8
    ],
    # The rotor blocks cut the airframe (see header). 'allow' because this
    # case is IBB — 'exclude' was measured to make IBB worse, not better.
    "wall_coupling": {"mode": "allow"},
}

_tag = "result_octo8_v1_smoke"
config["output"]["output_dir"] = "./%s/vtk" % _tag
config["output"]["checkpoint_dir"] = "./%s/checkpoints" % _tag
config["output"]["csv_dir"] = "./%s/csv" % _tag
config["output"]["vtk"]["fields_start_step"] = 0       # smoke: fields from t=0


def _fine_n(lo, hi, k, ow=2):
    """Fine nodes per axis for a region given in L0 lu, INCLUDING the band.

    OverlapRegion grows the user region by `ow` PARENT cells on each face
    before refining, so the fine extent is region*2^k + 4*ow + 1 — not
    region*2^k + 1. Verified against the solver's own summary.
    """
    return [(hi[i] - lo[i]) * 2 ** k + 4 * ow + 1 for i in range(3)]


if __name__ == "__main__":
    CD = 1912.8
    print(f"  L0  D/{D_LU_0:<4d} dx={_dx*1000:6.3f} mm  "
          f"{_N[0]}x{_N[1]}x{_N[2]} = {np.prod(_N)/1e6:6.2f} M   far-field")
    tot = float(np.prod(_N))
    for k, (nm, lo, hi) in enumerate(
            (("near", _l1_lo, _l1_hi), ("airframe", _l2_lo, _l2_hi)), start=1):
        n = _fine_n(lo, hi, k)
        tot += float(np.prod(n))
        print(f"  L{k}  D/{D_LU_0*2**k:<4d} dx={_dx*1000/2**k:6.3f} mm  "
              f"{int(n[0])}x{int(n[1])}x{int(n[2])} = {np.prod(n)/1e6:6.2f} M"
              f"   {nm}  x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}]"
              f" z[{lo[2]:.0f},{hi[2]:.0f}]")
    b = _rotor_boxes[0]
    n3 = _fine_n([b["x_min"], b["y_min"], b["z_min"]],
                 [b["x_max"], b["y_max"], b["z_max"]], 3)
    tot += 8 * float(np.prod(n3))
    print(f"  L3  D/{D_LU_0*8:<4d} dx={_dx*1000/8:6.3f} mm  "
          f"{int(n3[0])}x{int(n3[1])}x{int(n3[2])} x8 = "
          f"{8*np.prod(n3)/1e6:6.2f} M   rotors")
    print(f"  ---- total {tot/1e6:.2f} M cells  (measured build peak 18.6 GB)")
    dom = (_N[0] - 1) / _mm2lu
    print(f"  far-field {dom:.0f} x {dom:.0f} mm = {dom/2/CD:.2f} CD 반경 "
          f"| steps {config['time']['max_steps']:,} "
          f"(VTK every {config['time']['output_interval']} = 30 deg)")
