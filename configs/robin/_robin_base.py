"""ROBIN fuselage, rotor-off — shared builder for the TM-80051 anchor runs.

Case identity (patch_notes/robin/01-02): NASA TM-80051 Run 12 point 90 —
alpha_F = 0, beta_F = 0, V = 81.7 kt (42.03 m/s), atmospheric closed test
section (blockage ~0.5 %, ignored), rotor OFF. Body = ROBIN (AIAA J 2021
corrected coefficients, body + pylon boolean union, unit = rotor radius R,
length 2R; R = 1.574 m -> L = 3.148 m). Flow +x along the body axis, z up
(TM frame: X from the nose, Z up, Y starboard; identical to the STL frame,
verified against the 176 orifice coordinates, 02 sec. 2).

Re-targeting convention (memory: acoustic scaling / nu-only): L_char = L
is FIXED, RE is the anchor and NU_PHYS = U*L/RE is the derived knob.
At SLS air the anchor Re_L = 42.03*3.148/1.461e-5 = 9.06e6 (the tunnel is
atmospheric; the TM gives no temperature -> SLS is the assumption, a few
% on Re, immaterial for Cp). Ma = U/c_s = 0.1235 physical -> U_lu = Ma/sqrt(3).

Grid (r_lu0 = R in L0 cells; production 32 -> dx0 = 49.2 mm, dx3 = 6.15 mm,
L = 512 fine cells; smoke twin 12):
    L0  x [-4R, 10R]  y [-3R, 3R]  z [-3R, 3R]     far field, body axis z = 3R
    L1  x [-1, 3.5]   y +-1.125     z [-1.125, 1.25]
    L2  x [-0.5, 2.75] y +-0.625    z [-0.625, 0.75]
    L3  x [-0.125, 2.25] y +-0.25   z [-0.25, 0.34]   BODY LEVEL (surfel)
Region bounds are in R and rounded to L0 nodes (inclusive); the L3 pad
from the body bbox (x[0,2], y+-0.125, z[-0.125,0.1975]) is >= 0.125R =
0.5 D_body (memory: MLG region padding >= 0.5D or 3 delta; delta_tail ~
0.03R). Node counts (production): L0 16.5M, L1 6.4M, L2 11.9M, L3 12.0M
= 46.8M — inside the replicated-build ceiling (~50-60M) and the 2x24 GiB
z-slab working set (52.1M ran for span16, patch 64/75). Body-level y+ of
dx3: u_tau ~ 0.039 U -> y+ ~ 690 (WMLES, Musker h_law = 3 cells).

Timing: dt0 = dx0*U_lu/U = 8.35e-5 s; body flow-through L/U = 0.0749 s =
897 L0 steps; domain (14R) = 6.3k steps. Production 9000 steps; the
surface files (every output_interval) from step >= 6500 form the Cp
averaging window (>= 2.2 body FT after >= 7 body FT of development).

Geometry asset = robin_mod_v1c.stl (user decision 0826, 02 sec. 7.10-7.11):
the genROBIN/makerobin boolean union (v1: fuselage 400x128 + pylon 120x64,
manifold3d) with the seam slivers collapsed by robin_src/clean_seam.py
(relative sliver criterion, pylon tips frozen, cap flips) — the genROBIN
triangulation everywhere else, 0 folds, surface within 0.0017 R of v1.
The ~50 hair triangles at the pylon tips (tangential-contact singularity
of any boolean) are sub-cell and harmless for the surfel build. v3
(analytic union loft, robin_mod_v3.stl) is kept as the analytic
reference / quad-export source. Surfel needs overlap_cap=True here (02
sec. 7: the concave junction is the first non-convex surfel body); the
crease pathology (spurious pocket, 02 sec. 7.7) is cured by
crease_mode="noslip" (02 sec. 7.12).

Surfel stack = NACA campaign parity (patch 45/54 defaults): law musker,
h_law 3, tau_model ON, mode wallmodel, orient as_is, march_axis z
(default; every z-line crosses the closed body <= twice, pylon included),
SGS smagorinsky Cs 0.1. intermittency (gamma, patch 80) is airfoil-only
(chord-frame suction/pressure) -> OFF; trip forcing OFF; pressure_gradient
(patch 81) is the R1 knob.

MPI: surfel z-slab (patch 62-64; surfel+MPI is z-cut ONLY). 2 ranks split
at z = 3R = body axis; the body straddles the interface. A CLOSED body
has facets with z-normals next to the interface, so the wall-law sample
envelope needs **--ghost 8** (ghost 4 is refused by the slab build:
"sample envelope escapes the window" — robin/02 sec. 7.5/7.6; the span16
wing had no z-normals). Nz = 6R = 192 L0 also admits 4 ranks (own 48 >=
2*ghost = 16).

Run (cluster, main dir; --mca pml ucx REQUIRED, 64 sec. 19c):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    LBM_ESOTERIC=1 mpirun --mca pml ucx -n 2 python main.py --mpi \\
        --config configs/robin/robin_r0_musker.py \\
        --axis z --ghost 8 --cuda-aware 1 --gpu 0,1
Readout (main dir):
    python -m src.utilities.robin_anchor --point 90 \\
        --config configs/robin/robin_r0_musker.py \\
        --surface "results_robin_r0_musker/vtk/surface_0000[6-9]*.vtk" \\
        --plot robin_r0_cp.png --csv robin_r0_cp.csv
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(os.path.dirname(_here))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# ── physical case (TM-80051 Run 12 pt 90) ─────────────────────────────
R_PHYS   = 1.574                       # [m] rotor radius = STL unit
L_PHYS   = 2.0 * R_PHYS                # [m] body length 3.148 m = L_char
V_KT     = 81.7                        # [kt] Table IV pt 90
U_INF    = V_KT * 0.514444             # 42.03 m/s
C_S_PHYS = 340.3                       # [m/s] SLS
NU_SLS   = 1.461e-5                    # [m^2/s] SLS (atmospheric tunnel)
RE       = U_INF * L_PHYS / NU_SLS     # 9.06e6 = the anchor Re_L
NU_PHYS  = U_INF * L_PHYS / RE         # derived knob (== NU_SLS here)
MA       = U_INF / C_S_PHYS            # 0.1235
U_LU     = MA / math.sqrt(3.0)         # 0.07130 (acoustic scaling)

STL_PATH = os.path.join(_repo, "input_files", "geom", "robin_mod_v1c.stl")  # v1 boolean union, seam-cleaned (02 sec. 7.11)
BODY_BBOX_R = ((0.0, -0.125, -0.125), (2.0, 0.125, 0.1975))   # measured (01)

# region boxes in R, relative to the nose (x) and the body axis (y, z)
_BOXES_R = {
    1: ((-1.0, 3.5), (-1.125, 1.125), (-1.125, 1.25)),
    2: ((-0.5, 2.75), (-0.625, 0.625), (-0.625, 0.75)),
    3: ((-0.125, 2.25), (-0.25, 0.25), (-0.25, 0.34)),
}
DOMAIN_R = ((-4.0, 10.0), (-3.0, 3.0), (-3.0, 3.0))


def _stl_bbox_center_R():
    """Exact bbox centre of the STL (unit R) — the point the solver puts on
    center_lu (stl_mesh.transform_vertices_to_l0lu: rotated-bbox centre)."""
    try:
        from src.boundary.stl_mesh import load_stl_checked
        v = load_stl_checked(STL_PATH).vertices
        lo, hi = v.min(axis=0), v.max(axis=0)
    except Exception:                                  # config-only contexts
        lo, hi = BODY_BBOX_R
    return tuple(0.5 * (float(lo[d]) + float(hi[d])) for d in range(3))


#: L3 pads used by the alpha = 0 design, re-applied to the ROTATED bbox for
#: the alpha clones (fore, aft, y, z_lo, z_hi) [R]; L2/L1 = L3 + these
#: margins (the alpha = 0 boxes: L2 = L3 + 0.375R, L1 = L2 + 0.5/0.75R).
_PAD_L3 = (0.125, 0.25, 0.125, 0.125, 0.15)
_MARGIN_L2 = 0.375
_MARGIN_L1 = (0.5, 0.75, 0.5, 0.5, 0.5)

#: 2-level grid-ladder L1 box (patch robin/13, user-set round convention):
#: full body + 1R wake, >= 0.5R pad to the surface everywhere. The finest
#: level is L1 = R/(2 r_lu0); the 10-cells-across-the-thinnest-boom rule
#: (finest-level criterion, boom D = 0.05R at x/R 1.9 -> cell <= R/200)
#: is NOT met by this family (r40: 4 cells) BY DESIGN - the ladder
#: measures how the integral QoIs drift while under-resolved, and the
#: rule then sets the LEVEL DEPTH of the final grid (r20 -> 5, r30/r40
#: -> 4 levels; production r32 x 4 = R/256 = 12.8 cells passes).
GRID2_L1_BOX_R = ((-0.5, 3.0), (-1.0, 1.0), (-1.0, 1.0))
#: 3-level variant (robin/13 s7, user hypothesis H2): same L1 box + one
#: nested body box -> finest = R/(4 r). robin_g3_r20 (fine R/80) vs the
#: 2-level r40 (fine R/80) isolates the LEVEL-STRUCTURE axis at equal
#: finest dx (if equal within noise: deep-levels + coarse L0 wins ~6x).
GRID3_L2_BOX_R = ((-0.25, 2.5), (-0.5, 0.5), (-0.5, 0.5))


def _rotated_bbox_R(rotation_deg):
    """bbox of the STL after Rz@Ry@Rx about its bbox centre, in R, relative
    to the (unrotated) bbox centre (= the point the solver pins to center_lu)."""
    from src.boundary.stl_mesh import load_stl_checked, _rotation_matrix
    v = load_stl_checked(STL_PATH).vertices
    pivot = 0.5 * (v.min(axis=0) + v.max(axis=0))
    w = (v - pivot) @ _rotation_matrix(rotation_deg).T
    return w.min(axis=0), w.max(axis=0), pivot


def build(r_lu0: int = 32, tag: str = "robin_r0_musker", max_steps: int = 9000,
          output_interval: int = 500, surfel_extra: dict = None,
          rotation_deg=(0.0, 0.0, 0.0), num_levels: int = 4) -> dict:
    """Config for R = r_lu0 L0 cells. rotation_deg = (0, alpha, 0) pitches
    the body about its bbox centre: Ry gives z' = -sin(a) x + cos(a) z, so
    the nose (x = -1R from the pivot) rises for positive alpha = nose-up
    = positive angle of attack (report() prints nose/tail z as the check).
    alpha = 0 uses the registered boxes (_BOXES_R); for alpha != 0 the
    L3/L2/L1 boxes are rebuilt from the ROTATED bbox with the same pads
    (patch robin/07), the flow stays along +x."""
    r = float(r_lu0)
    (x0, x1), (y0, y1), (z0, z1) = DOMAIN_R
    Nx, Ny, Nz = (int(round((x1 - x0) * r)), int(round((y1 - y0) * r)),
                  int(round((z1 - z0) * r)))
    nose_x, axis_y, axis_z = -x0 * r, -y0 * r, -z0 * r      # L0 lu
    cx, cy, cz = _stl_bbox_center_R()
    center_lu = (nose_x + cx * r, axis_y + cy * r, axis_z + cz * r)

    def _r(box, outward=False):
        """Region in inclusive L0 nodes. outward=True (rotated-bbox boxes)
        rounds min down / max up so the pad rule is never under-cut by
        rounding; the alpha = 0 design boxes keep plain rounding (bit-
        identical to the registered R0 grid)."""
        (bx0, bx1), (by0, by1), (bz0, bz1) = box
        lo = (lambda v: int(math.floor(v))) if outward else (lambda v: int(round(v)))
        hi = (lambda v: int(math.ceil(v))) if outward else (lambda v: int(round(v)))
        return {"region": {
            "x_min": lo(nose_x + bx0 * r), "x_max": hi(nose_x + bx1 * r),
            "y_min": lo(axis_y + by0 * r), "y_max": hi(axis_y + by1 * r),
            "z_min": lo(axis_z + bz0 * r), "z_max": hi(axis_z + bz1 * r)}}

    rot = tuple(float(a) for a in rotation_deg)
    if num_levels in (2, 3):
        # grid-ladder family (robin/13): L0 everywhere + body box(es).
        # alpha = 0 only (the ladder is the alpha = 0 anchor case).
        if any(a != 0.0 for a in rot):
            raise ValueError("grid-ladder levels (2/3) are alpha=0 only")
        levels = [{}, _r(GRID2_L1_BOX_R)]
        if num_levels == 3:
            levels.append(_r(GRID3_L2_BOX_R))
    elif num_levels != 4:
        raise ValueError(f"num_levels must be 2, 3 or 4, got {num_levels}")
    elif any(a != 0.0 for a in rot):
        lo, hi, _ = _rotated_bbox_R(rot)
        # rotated bbox relative to the nose-x / axis frame used by _r():
        # the pivot (bbox centre) sits at (cx, cy, cz) in that frame
        bx = (cx + lo[0], cx + hi[0]); by = (cy + lo[1], cy + hi[1]); bz = (cz + lo[2], cz + hi[2])
        f, a_, py, zl, zh = _PAD_L3
        b3 = ((bx[0] - f, bx[1] + a_), (by[0] - py, by[1] + py), (bz[0] - zl, bz[1] + zh))
        m = _MARGIN_L2
        b2 = ((b3[0][0] - m, b3[0][1] + m), (b3[1][0] - m, b3[1][1] + m), (b3[2][0] - m, b3[2][1] + m))
        f1, a1, y1_, zl1, zh1 = _MARGIN_L1
        b1 = ((b2[0][0] - f1, b2[0][1] + a1), (b2[1][0] - y1_, b2[1][1] + y1_), (b2[2][0] - zl1, b2[2][1] + zh1))
        levels = [{}, _r(b1, True), _r(b2, True), _r(b3, True)]
    else:
        levels = [{}, _r(_BOXES_R[1]), _r(_BOXES_R[2]), _r(_BOXES_R[3])]

    surfel = {"tau_model": True, "law": "musker", "h_law": 3.0,
        # p_state sample height (robin/10, survey 09): surface pressure read
        # OUTSIDE the modeled layer (readout definition of patch 08). Output-
        # only; physics stays at sample_h/h_law.
        "p_sample_h": 1.5,
              "march_axis": 2, "orient": "as_is",
              # robin/02 sec. 7: concave body/pylon crease -> prism overlap
              # g_i > dV (up to 17x at the pylon TE) -> negative density in
              # 4 substeps. Per-(cell,dir) renormalisation (Chen-Teixeira-
              # Molvig sum P <= 1); no-op on a convex body (bit-identical).
              "overlap_cap": True,
              # robin/02 sec. 7.12: facets touching a capped (overlapping)
              # cell use the mass-conserving Eq. (10) reconstruction — the
              # free-slip-based wall-model reconstruction pumps a spurious
              # pressure pocket into the fully-captured crease cells
              # (measured Cp +23 -> +0.16; crease orifices +3.5 -> -0.2).
              "crease_mode": "noslip"}
    if surfel_extra:
        surfel.update(surfel_extra)

    boundaries = {
        "xmin": {"location": "xmin", "method": "eq", "velocity": [U_INF, 0.0, 0.0]},
        "xmax": {"location": "xmax", "method": "sponge",
                 "velocity": [U_INF, 0.0, 0.0], "density": 1.0,
                 "thickness": 20, "strength": 0.1},
        "ymin": {"location": "ymin", "method": "neumann"},
        "ymax": {"location": "ymax", "method": "neumann"},
        "zmin": {"location": "zmin", "method": "neumann"},
        "zmax": {"location": "zmax", "method": "neumann"},
    }
    folder = f"results_{tag}"
    return {
        "simulation": {"device_mode": "gpu", "precision": "float32",
                       "dimension": 3, "lattice_model": "D3Q27",
                       "collision_model": "cumulant"},
        "physics": {"rho": 1.225, "U_inf": U_INF, "nu": NU_PHYS,
                    "L_char": L_PHYS, "flow_direction": [1.0, 0.0, 0.0],
                    "initial_flow_velocity": [U_LU, 0.0, 0.0]},   # [lu]
        "grid": {"Nx": Nx, "Ny": Ny, "Nz": Nz, "resolution": int(2 * r_lu0)},
        "numerics": {"acoustic_scaling": True, "c_s_phys": C_S_PHYS,
                     "collision": "cumulant"},
        "boundaries": boundaries,
        "internal_geometry": {
            "stl": {"enabled": True, "file": STL_PATH,
                    "scale_to_lu": r,                    # STL unit = R
                    "center_lu": tuple(float(c) for c in center_lu),
                    "rotation_deg": tuple(float(a) for a in rotation_deg),
                    "wall_bc": "surfel",
                    "surfel": surfel},
        },
        "mlg": {"enabled": True, "num_levels": len(levels), "overlap_width": 2,
                "interpolation": "cubic", "filter_level": 1, "levels": levels},
        "sgs": {"enabled": True, "model": "smagorinsky", "Cs": 0.1},
        "conservation": {"enabled": True, "verbose": 0, "log_to_csv": True},
        "convergence": {"enabled": False},
        "force_calculation": {
            "enabled": True, "interval": 20,
            # monitor only: A_ref = L x W_max (planform-like), L0 lu
            "reference": {"rho": 1.0, "velocity": U_LU,
                          "char_length": 2.0 * r, "span_length": 0.25 * r},
        },
        "output": {"output_dir": f"./{folder}/vtk",
                   "checkpoint_dir": f"./{folder}/checkpoints",
                   "csv_dir": f"./{folder}/csv", "clear_previous": True,
                   "vtk": {"enabled": True, "precision": "float32",
                           "variables": ["density", "pressure", "velocity",
                                         "velocity_magnitude"]},
                   "checkpoint": {"enabled": True, "keep_last_n": 2}},
        "time": {"max_steps": int(max_steps), "output_interval": int(output_interval),
                 "logging_interval": 100, "checkpoint_interval": 2500,
                 "conservation_interval": 500},
    }


def report(cfg: dict, label: str = "") -> dict:
    """Node accounting (solver convention: fine_shape = extent*2^k + 1 per
    axis, inclusive L0 region nodes) + memory model (patch 60: 736 B/node,
    VRAM ~ x1.15 + 0.9 GiB) + timing."""
    g = cfg["grid"]; r = float(cfg["internal_geometry"]["stl"]["scale_to_lu"])
    dx0 = R_PHYS / r
    kf = 2 ** (len(cfg["mlg"]["levels"]) - 1)          # finest-level factor
    nodes = {0: g["Nx"] * g["Ny"] * g["Nz"]}
    print(f"  [{label}] R = {r:g} L0 cells, dx0 = {dx0 * 1e3:.2f} mm, "
          f"dx_fine = {dx0 * 1e3 / kf:.3f} mm (L{len(cfg['mlg']['levels']) - 1}), "
          f"L = {2 * r:g} L0 = {2 * r * kf:g} fine cells, "
          f"boom-min 0.05R = {0.05 * r * kf:.1f} fine cells")
    print(f"  L0  {g['Nx']}x{g['Ny']}x{g['Nz']} = {nodes[0] / 1e6:6.2f} M  "
          f"domain x[{-4:g},{10:g}]R y+-3R z+-3R")
    for k, lv in enumerate(cfg["mlg"]["levels"][1:], 1):
        rg = lv["region"]; ext = [rg["x_max"] - rg["x_min"], rg["y_max"] - rg["y_min"],
                                  rg["z_max"] - rg["z_min"]]
        sh = [e * 2 ** k + 1 for e in ext]
        nodes[k] = sh[0] * sh[1] * sh[2]
        print(f"  L{k}  {sh[0]}x{sh[1]}x{sh[2]} = {nodes[k] / 1e6:6.2f} M  "
              f"region x[{rg['x_min']},{rg['x_max']}] y[{rg['y_min']},{rg['y_max']}] "
              f"z[{rg['z_min']},{rg['z_max']}] (L0 nodes)")
    tot = sum(nodes.values())
    ws = tot * 736 / 2 ** 30
    print(f"  ---- {tot / 1e6:.2f} M nodes | working set {ws:.1f} GiB | "
          f"VRAM/rank ~ {ws * 1.15 / 2 + 0.9:.1f} GiB @2 ranks, "
          f"{ws * 1.15 / 4 + 0.9:.1f} GiB @4 ranks")
    dt0 = dx0 * U_LU / U_INF
    ft = L_PHYS / U_INF / dt0
    print(f"  dt0 = {dt0:.3e} s | body flow-through = {ft:.0f} L0 steps | "
          f"max_steps {cfg['time']['max_steps']} = {cfg['time']['max_steps'] / ft:.1f} body FT")
    # body placement + pad check
    c = cfg["internal_geometry"]["stl"]["center_lu"]; cR = _stl_bbox_center_R()
    rot = tuple(float(a) for a in cfg["internal_geometry"]["stl"].get("rotation_deg", (0, 0, 0)))
    if any(a != 0.0 for a in rot):
        rlo, rhi, _ = _rotated_bbox_R(rot)
        lo = [c[d] + rlo[d] * r for d in range(3)]; hi = [c[d] + rhi[d] * r for d in range(3)]
        from src.boundary.stl_mesh import _rotation_matrix
        Rm = _rotation_matrix(rot)
        nose = Rm @ np.array([0.0 - cR[0], 0.0, 0.0 - cR[2]]); tail = Rm @ np.array([2.0 - cR[0], 0.0, 0.04 - cR[2]])
        print(f"  rotation {rot}: nose tip z {nose[2]:+.3f} R vs tail z {tail[2]:+.3f} R -> "
              f"{'NOSE UP (positive alpha)' if nose[2] > tail[2] + 0.5 * (0.04) else 'nose down'} ; rotated bbox z [{rlo[2]:+.3f},{rhi[2]:+.3f}] R")
    else:
        lo = [c[d] + (BODY_BBOX_R[0][d] - cR[d]) * r for d in range(3)]
        hi = [c[d] + (BODY_BBOX_R[1][d] - cR[d]) * r for d in range(3)]
    rg = cfg["mlg"]["levels"][-1]["region"]
    pads = [(lo[0] - rg["x_min"]) / r, (rg["x_max"] - hi[0]) / r,
            (lo[1] - rg["y_min"]) / r, (rg["y_max"] - hi[1]) / r,
            (lo[2] - rg["z_min"]) / r, (rg["z_max"] - hi[2]) / r]
    print(f"  body bbox [L0 lu] x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] "
          f"z[{lo[2]:.2f},{hi[2]:.2f}] | L3 pad [R] fore {pads[0]:.3f} aft {pads[1]:.3f} "
          f"y {pads[2]:.3f}/{pads[3]:.3f} z {pads[4]:.3f}/{pads[5]:.3f} (rule >= 0.125)")
    print(f"  physics: U {U_INF:.2f} m/s, Ma {MA:.4f}, U_lu {U_LU:.5f}, "
          f"Re_L {RE:.3e}, nu {NU_PHYS:.4e} m^2/s")
    return {"nodes": nodes, "total": tot}


config = build()

if __name__ == "__main__":
    report(config, "production R0")
    print()
    report(build(r_lu0=12, tag="robin_smoke", max_steps=200, output_interval=100), "smoke r12")
