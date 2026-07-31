"""NACA0012 wing section — a=5 deg, M_inf=0.15 (physical), Re=6e6.

First production application of the STL body track: 3D extruded NACA0012
(closed TE, generated on demand by src/utilities/naca_stl.py -> a config
pull is enough on the cluster, no geometry file transfer).

Setup (chord c = 100 COARSE cells, MLG 4-level -> 800 fine cells/chord):
  - 5-level was ruled out arithmetically: an L4 region wrapping the wing
    alone exceeds 400M cells at this chord — over the 100M budget.
  - quasi-2D slab: span 0.1c (10 lu), wing tips 23 lu inboard of the
    neumann z-walls (STL bodies must sit strictly inside the domain; no
    periodic BC in the domain-BC policy). Tip leakage is accepted and
    documented — this is a shakeout run, not a lift-curve anchor.
  - AoA via geometry: rotation_deg = (0, 0, -5) (flow stays +x aligned).
  - far-field per request: xmin eq, xmax sponge (L=20, sigma=0.1),
    ymin/ymax/zmin/zmax neumann.
  - acoustic scaling ON: u_lu = 0.15/sqrt(3) = 0.0866, dt from c_s_phys.
    Re enters via physics.nu + L_char = Re*nu/U_inf = 1.7173 m.
  - Levels/cells (c=100): L0 960x600x56 = 32.3M, L1 7.9M, L2 14.1M,
    L3 (wing, ibb) 40.0M -> ~94M total, ~19.5 GB esoteric = 2x24GB OK.
    Padding to region faces >= ~5 lu everywhere (3xBL criterion for a
    streamlined body; expect 0.5*L_body WARNINGS since L_body ~ c).
  - 10k coarse steps ~ 8.7 chords of travel (~1 flow-through).

Run (esoteric 2-rank is the intended path; plain single-GPU float32
needs ~39 GB and does NOT fit a 24 GB card):
    LBM_ESOTERIC=1 mpirun -n 2 python main.py \
        --config configs/stl/naca0012_a5_m015_re6m.py --gpu 2,3
"""
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(os.path.dirname(_here))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from src.utilities.naca_stl import ensure_naca4_wing_stl

# ── physical case ───────────────────────────────────────────────────────
MA        = 0.15                      # inlet Mach (physical)
C_S_PHYS  = 340.3                     # [m/s] SLS
NU_PHYS   = 1.461e-5                  # [m^2/s] SLS
RE        = 6.0e6
AOA_DEG   = 5.0
U_INF     = MA * C_S_PHYS             # 51.045 m/s
L_CHAR    = RE * NU_PHYS / U_INF      # chord = 1.7173 m
U_LU      = MA / math.sqrt(3.0)       # = what acoustic scaling derives

SPAN_C    = 0.1                       # span / chord (quasi-2D slab)
STL_PATH  = ensure_naca4_wing_stl(
    os.path.join(_repo, "input_files", "geom", "naca0012_c1_s010_n256.stl"),
    chord=1.0, span=SPAN_C, thickness=0.12, n_profile=256)


def _build(c=100):
    """Config dict for chord = c L0 cells. All region bounds are linear
    in c and integer for c in {100, 50}; c=50 is the local-smoke twin
    (~12M cells), c=100 the production case (~94M)."""
    if c % 50 != 0:
        raise ValueError("chord must keep region bounds integral (50|c)")
    Nx, Ny, Nz = round(9.6 * c), 6 * c, round(0.56 * c)
    xc, yc, zc = 3.5 * c, 3 * c, round(0.28 * c)     # rotated-bbox center

    def _r(x0, x1, y0, y1, z0, z1):
        return {"region": {
            "x_min": round(x0 * c), "x_max": round(x1 * c),
            "y_min": round(y0 * c), "y_max": round(y1 * c),
            "z_min": round(z0 * c), "z_max": round(z1 * c)}}

    levels = [
        {},                                            # L0
        _r(2.50, 4.70, 2.44, 3.56, 0.08, 0.48),        # L1
        _r(2.84, 4.24, 2.72, 3.28, 0.14, 0.42),        # L2
        _r(2.90, 4.12, 2.84, 3.16, 0.18, 0.38),        # L3 (wing, pads>=5s)
    ]

    boundaries = {
        "xmin": {"location": "xmin", "method": "eq",
                 "velocity": [U_INF, 0.0, 0.0]},
        "xmax": {"location": "xmax", "method": "sponge",
                 "velocity": [U_INF, 0.0, 0.0], "density": 1.0,
                 "thickness": 20, "strength": 0.1},
        "ymin": {"location": "ymin", "method": "neumann"},
        "ymax": {"location": "ymax", "method": "neumann"},
        "zmin": {"location": "zmin", "method": "neumann"},
        "zmax": {"location": "zmax", "method": "neumann"},
    }

    folder = f"results_naca0012_a5_m015_re6m_c{c}"
    return {
        "simulation": {"device_mode": "gpu", "precision": "float32",
                       "dimension": 3, "lattice_model": "D3Q27",
                       "collision_model": "cumulant"},
        "physics": {"rho": 1.225, "U_inf": U_INF, "nu": NU_PHYS,
                    "L_char": L_CHAR, "flow_direction": [1.0, 0.0, 0.0],
                    "initial_flow_velocity": [U_LU, 0.0, 0.0]},   # [lu]
        "grid": {"Nx": Nx, "Ny": Ny, "Nz": Nz, "resolution": c},
        "numerics": {"acoustic_scaling": True, "c_s_phys": C_S_PHYS,
                     "collision": "cumulant"},
        "boundaries": boundaries,
        "internal_geometry": {
            "stl": {
                "enabled": True,
                "file": STL_PATH,
                "scale_to_lu": float(c),               # unit chord -> c lu
                "center_lu": (float(xc), float(yc), float(zc)),
                "rotation_deg": (0.0, 0.0, -AOA_DEG),  # -5 about z = +AoA
                "wall_bc": "ibb",
            },
        },
        "mlg": {"enabled": True, "num_levels": 4, "overlap_width": 2,
                "interpolation": "cubic", "filter_level": 1,
                "levels": levels},
        "sgs": {"enabled": True, "model": "dyn_smag"},
        "conservation": {"enabled": True, "verbose": 0, "log_to_csv": True},
        "convergence": {"enabled": False},
        "force_calculation": {
            "enabled": True,
            "interval": 20,
            # reference in L0 lu (setup rescales x2^k to the force level):
            # A_ref = chord x span (planform), U = u_lu -> standard Cl/Cd.
            "reference": {"rho": 1.0, "velocity": U_LU,
                          "char_length": float(c),
                          "span_length": SPAN_C * c},
        },
        "output": {"output_dir": f"./{folder}/vtk",
                   "checkpoint_dir": f"./{folder}/checkpoints",
                   "csv_dir": f"./{folder}/csv", "clear_previous": True,
                   "vtk": {"enabled": True, "precision": "float32",
                           "variables": ["density", "pressure", "velocity",
                                         "velocity_magnitude"]},
                   "checkpoint": {"enabled": True, "keep_last_n": 2}},
        "time": {"max_steps": 10000, "output_interval": 2000,
                 "logging_interval": 100, "checkpoint_interval": 2500,
                 "conservation_interval": 500},
    }


config = _build(100)
