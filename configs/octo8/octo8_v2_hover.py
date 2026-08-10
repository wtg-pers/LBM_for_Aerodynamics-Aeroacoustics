"""Octo-8 v2 — 4-level MLG, 14 m field, ground-resolved outwash. **100 rev.**

The production configuration. v1 (`octo8_v1_hover.py`) was the 5-rev smoke
that got the structure running; everything below is what the smoke and the
rig campaign changed about it.

    far-field   14.0 x 14.0 m  (vehicle centre +- 7 m), ground to inflow
    ground+wake D/40  (dx 11.430 mm) out to +- 5 m — the whole outwash field
    기체        D/80  (dx  5.715 mm)
    로터        D/160 (dx  2.857 mm), one block per rotor

    L0 D/20  616x616x100    =  37.95 M
    L1 D/40  883x883x139    = 108.60 M
    L2 D/80  401x701x141    =  39.64 M
    L3 D/160 229x229x69 x8  =  28.95 M
    ------------------------------------
    총 215.1 M cells | 645.3 M updates / coarse step

What changed from v1, and why
-----------------------------
1. **L1 reaches the GROUND and spans +-5 m.** v1's L1 floated 503 mm above
   the ground, so the wall jet — the thing the wide field exists to measure —
   was resolved only at L0 (22.86 mm). It now starts at z=0 and covers the
   full +-5 m influence radius the user measured in another solver. The
   vehicle's downwash and the radial ground flow are therefore both at D/40
   or finer, continuously.

2. **L0 grown to +-7 m** so L1 (+-5 m) does not touch the domain face. That
   is not cosmetic: fine levels are built with `boundaries_config={}` — they
   have NO domain BC of their own. A fine region flush against the domain
   face gets no C2F inflow (flush faces skip coupling by design) and no BC
   either, leaving the kernel's %N wrap exposed — outwash leaving +x would
   re-enter at -x. 2 m of L0 margin keeps L1 interior.

3. **Sides are `neumann`, not `sponge`.** A sponge force-damps inside its
   20-cell layer, which destroys the quantity being measured — how far the
   wall jet actually reaches. Zero-gradient lets it leave. The cost is
   reflectivity: when the target becomes acoustics (toroidal-propeller
   noise), this must go back to sponge or a non-reflecting BC.

4. **`overlap_width` stays 2.** Daeian et al. use 4 for a curved interface,
   so it was tested: `patch_notes/wall_coupling/gates/ow4_sweep.py` measured
   ow=4 as +0.06 pt (ibb) / +0.07 pt (hwbb) — i.e. no effect, within one SEM,
   for +23% cells. The residual is read-side, and a wider band does not
   remove the body from the stencil.

Known, quantified limitation
----------------------------
The 8 rotor blocks cut the airframe on 3-4 faces each (measured: 41,490 STL
vertices inside every block). The arm sits just under the disk and the ALM
Gaussian support needs 0.25 D below it, so no block can clear the arm, and
the arms run to the booms and the booms to the fuselage — no finite block
avoids the cut. `wall_coupling.mode='allow'` (NOT 'exclude': this case is
IBB, where the C2F wall exclusion was measured to make things worse, not
better — patch_notes/wall_coupling/01 §3a).

Consequence: rotor thrust is unaffected (measured 7e-6 relative), but
**airframe download carries ~5% uncertainty**. Judge body load against a
whole-vehicle-fine run, not against this one.

Memory / runtime
----------------
Running state is 7.0 GB per rank at 4 ranks — comfortable. The REPLICATED
build would be ~42 GB and does not fit a 24 GB card, so this runs with
`--dist-init`. That is no longer a restriction: the restore path became
slab-scoped, so --dist-init and restart now coexist (gate:
`mpi_blocks_gate.py` case `restart/dist-init`). The only thing given up is
the conservation diagnostic.

Run (cluster 1, 4x RTX 4090):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \\
        --config configs/octo8/octo8_v2_hover.py --gpu 0,1,2,3 \\
        --cuda-aware 1 --dist-init

Restart (checkpoints every 5 rev):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \\
        --config configs/octo8/octo8_v2_hover.py --gpu 0,1,2,3 \\
        --cuda-aware 1 --dist-init --restart-latest
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import (  # noqa: E402
    _BBOX_MAX_MM, _BBOX_MIN_MM, _LAYOUT, D_PHYS, ROTOR_Z_MM,
    build_config, grid_map_centered)

D_LU_0 = 20                       # L0: rotor diameter in cells
HALF_XY_MM = 7000.0               # L0 half-extent about the vehicle centre
L1_HALF_MM = 5000.0               # ground-resolved outwash radius
N_RADIAL = 48                     # delta_r 3.7 mm <= eps floor 5.71 mm @D/160
N_REV = 100

# full-field VTK 를 전 구간 30° 로 뽑는다(사용자 확정). 용량:
#   1,207 스냅샷 x 212.7 M 셀 x ~32 B/셀 = **약 7~8 TB**
# (32 B/셀 = density+pressure+velocity(3)+velocity_magnitude 4B 씩 + nu_t
#  + solid_mask; farfield40 실측 1.7 GB / 52.3 M 셀 = 32.5 B/셀 과 일치)
# KISTI /scratch 100 TB 에 들어간다. **/home01 은 64 GB 라 절대 불가** —
# 리포와 출력을 /scratch 에 두어야 한다.
# 줄이려면 vtk_deg 를 키우거나 아래를 마지막 N 바퀴로 낮춘다.
VTK_FIELDS_REV = N_REV       # 전 구간
config = build_config(rpm=5000.0, n_rev=N_REV, n_radial=N_RADIAL,
                      vtk_deg=30.0, vtk_fields_last_rev=VTK_FIELDS_REV,
                      wall_bc="ibb",
                      d_lu=D_LU_0, half_xy_mm=HALF_XY_MM, side_bc="neumann")

_dx, _mm2lu, _O, _N = grid_map_centered(D_LU_0, HALF_XY_MM)


def _lu(p_mm):
    return _O + np.asarray(p_mm, dtype=np.float64) * _mm2lu


# ── boxes in L0 lattice units (setup.py:1518 — regions are L0 coords) ──
_b_lo, _b_hi = _lu(_BBOX_MIN_MM), _lu(_BBOX_MAX_MM)          # airframe bbox
_CX, _CY = float(_lu([0, 0, 0])[0] + _BBOX_MAX_MM[0] * 0), None  # placeholder
_ctr = _lu([0.5 * (_BBOX_MIN_MM[0] + _BBOX_MAX_MM[0]), 0.0, 0.0])

# L3: rotor blocks, at the proportions the single-prop validation used
#     (apc18x8e EXTENTS[-1] = up 0.125 D, down 0.25 D, lat 0.6875 D)
_LAT = 0.6875 * D_LU_0
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

# L2 (D/80) contains the airframe AND every L3 block plus L3's band
# (2 parent cells = 2 * dx_L2 = 0.5 L0 lu); PAD2 is on top of that.
PAD2 = 2.0
_r_lo = np.min([[b["x_min"], b["y_min"], b["z_min"]] for b in _rotor_boxes], 0)
_r_hi = np.max([[b["x_max"], b["y_max"], b["z_max"]] for b in _rotor_boxes], 0)
_l2_lo = np.floor(np.minimum(_b_lo, _r_lo) - PAD2)
_l2_hi = np.ceil(np.maximum(_b_hi, _r_hi) + PAD2)

# L1 (D/40): +-L1_HALF_MM about the vehicle centre, floor ON THE GROUND
# (z=0). Must also contain L2 + L2's band (2 L0 cells).
#
# ★ z_min is 2, NOT 0. A fine region placed ON the domain floor is FLUSH:
# the C2F band collapses to zero width there (by design — OverlapRegion
# treats flush as a first-class state), and since fine levels are built with
# `boundaries_config={}` they have no wall of their own either. L1's floor
# would then have neither coarse inflow nor the hwbb ground — the wall jet
# would run over nothing. Verified by shape: z_min=0 gave 887x887x135 where
# a two-banded region gives 139, i.e. one band missing.
#
# With z_min = overlap_width the bottom two L0 cells become the band: L1
# still has fine cells down to the ground node, fed by C2F from L0, which
# does carry the wall. The consequence to keep in mind is that the very
# near-wall layer (0..46 mm) is L0-resolved and only the jet above it is
# D/40. Putting the wall on the fine level needs domain BCs on fine levels
# — a real gap, tracked separately.
_L1_ZMIN = 2.0
_half1 = L1_HALF_MM * _mm2lu
_l1_lo = np.array([np.floor(_ctr[0] - _half1), np.floor(_ctr[1] - _half1),
                   _L1_ZMIN])
_l1_hi = np.array([np.ceil(_ctr[0] + _half1), np.ceil(_ctr[1] + _half1),
                   float(np.ceil(_l2_hi[2] + 5.0))])
assert (_l1_lo <= _l2_lo - 2).all() and (_l1_hi >= _l2_hi + 2).all(), \
    f"L1 {_l1_lo}..{_l1_hi} must contain L2 {_l2_lo}..{_l2_hi} + band"
assert _l1_lo[0] >= 2 and _l1_hi[0] <= _N[0] - 3, \
    "L1 must stay clear of the domain face (fine levels carry no domain BC)"


def _box(name, lo, hi):
    return {"name": name,
            "x_min": float(lo[0]), "x_max": float(hi[0]),
            "y_min": float(lo[1]), "y_max": float(hi[1]),
            "z_min": float(lo[2]), "z_max": float(hi[2])}


config["mlg"] = {
    "enabled": True, "num_levels": 4, "overlap_width": 2,
    "interpolation": "cubic", "filter_level": 1,
    "levels": [
        {},                                                 # L0  D/20  far
        {"regions": [_box("outwash", _l1_lo, _l1_hi)]},      # L1  D/40  ground
        {"regions": [_box("airframe", _l2_lo, _l2_hi)]},     # L2  D/80
        {"regions": _rotor_boxes},                           # L3  D/160 x8
    ],
    # Rotor blocks cut the airframe (see header). 'allow', not 'exclude':
    # this case is IBB, where the exclusion was measured to hurt.
    "wall_coupling": {"mode": "allow"},
}

_tag = "result_octo8_v2"
config["output"]["output_dir"] = "./%s/vtk" % _tag
config["output"]["checkpoint_dir"] = "./%s/checkpoints" % _tag
config["output"]["csv_dir"] = "./%s/csv" % _tag


def _fine_n(lo, hi, k, ow=2):
    """Fine nodes per axis INCLUDING the coupling band (region*2^k + 4*ow + 1)."""
    return [int(round((hi[i] - lo[i]) * 2 ** k)) + 4 * ow + 1
            for i in range(3)]


if __name__ == "__main__":
    CD = 1912.8
    tot = float(np.prod(_N))
    upd = float(np.prod(_N))
    print(f"  L0  D/{D_LU_0:<4d} dx={_dx*1000:6.3f} mm  "
          f"{_N[0]}x{_N[1]}x{_N[2]} = {np.prod(_N)/1e6:7.2f} M   far-field")
    for k, (nm, lo, hi) in enumerate(
            (("outwash", _l1_lo, _l1_hi), ("airframe", _l2_lo, _l2_hi)), 1):
        n = _fine_n(lo, hi, k)
        tot += np.prod(n); upd += np.prod(n) * 2 ** k
        print(f"  L{k}  D/{D_LU_0*2**k:<4d} dx={_dx*1000/2**k:6.3f} mm  "
              f"{n[0]}x{n[1]}x{n[2]} = {np.prod(n)/1e6:7.2f} M   {nm}"
              f"  x[{lo[0]:.0f},{hi[0]:.0f}] z[{lo[2]:.0f},{hi[2]:.0f}]")
    b = _rotor_boxes[0]
    n3 = _fine_n([b["x_min"], b["y_min"], b["z_min"]],
                 [b["x_max"], b["y_max"], b["z_max"]], 3)
    tot += 8 * np.prod(n3); upd += 8 * np.prod(n3) * 8
    print(f"  L3  D/{D_LU_0*8:<4d} dx={_dx*1000/8:6.3f} mm  "
          f"{n3[0]}x{n3[1]}x{n3[2]} x8 = {8*np.prod(n3)/1e6:7.2f} M   rotors")
    dom = (_N[0] - 1) / _mm2lu
    print(f"  ---- {tot/1e6:.2f} M cells | {upd/1e6:.1f} M upd/coarse step")
    print(f"  far-field {dom/1000:.2f} m  | L1 ground radius "
          f"{L1_HALF_MM/1000:.2f} m = {L1_HALF_MM/CD:.2f} CD")
    print(f"  steps {config['time']['max_steps']:,} ({N_REV} rev) | "
          f"VTK every {config['time']['output_interval']} | "
          f"ckpt every {config['time']['checkpoint_interval']:,}")
    print(f"  running mem @4 ranks ~ {tot*130/4/1e9:.1f} GB/rank")
    n_out = config['time']['max_steps'] // config['time']['output_interval']
    n_fld = int(VTK_FIELDS_REV * 628 / config['time']['output_interval'])
    print(f"  VTK: {n_out} 출력 중 full-field {min(n_fld,n_out)}개 "
          f"~ {min(n_fld,n_out)*tot*32/1e12:.1f} TB  (/scratch 100 TB 내)")
