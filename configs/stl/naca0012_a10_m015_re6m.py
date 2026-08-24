"""NACA0012 infinite wing (wall-piercing prism) — a=10 deg, M=0.15, Re=6e6.

Quasi-3D setup for 2D-section analysis: the user-supplied wing STL
(input_files/geom/naca0012_wing.stl — chord 1000 units along x, span 900
units along +y, 10-deg pitch about the span axis BAKED IN, watertight,
sharp TE) pierces BOTH z faces of a thin slab domain. Correctness
contract (STL track P2): the esoteric kernels address with periodic
wrap, so a body through both z faces with a z-INVARIANT mask makes every
wrapped read/write bit-equal to the true continued wing — periodic-z ==
infinite extrusion. validate + per-level mask checks enforce exactly
that ("span_through_axis": "z").

Geometry mapping (measured from the file, NOT the nominal c=1/l=1):
  scale_to_lu = c/1000 (chord -> c lu), rotation_deg = (-90, 0, 0)
  (Rx(-90): (x,y,z) -> (x, z, -y) — span y -> z, TE stays DOWN in y
  = +10 deg AoA with flow +x, lift +y). AoA comes from the file; the
  config rotation is a pure axis swap.

Grid: chord = 100 COARSE cells, MLG 4-level, wing on L3 (800 fine
cells/chord). Thin slab Nz = 0.16c; ALL fine regions span full z (their
z faces are domain faces — no z coupling interfaces). Cells (c=100):
L0 960x600x16 = 9.2M, L1 3.2M, L2 8.0M, L3 32.0M -> ~52.4M total.
BCs: xmin eq, xmax sponge, ymin/ymax neumann, z PERIODIC (omitted —
the canonical spanwise closure for an infinite wing).
0.5*L_body padding advisories vs the chord are expected (streamlined
body -> the 3xBL criterion applies; documented, not gated).

Run (esoteric; --dist-init avoids the replicated full-domain init that
OOMed the 94M finite-slab case — recommended even at this size):
    LBM_ESOTERIC=1 mpirun -n 2 python main.py \
        --config configs/stl/naca0012_a10_m015_re6m.py --gpu 2,3 --dist-init
"""
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(os.path.dirname(_here))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# ── physical case ───────────────────────────────────────────────────────
MA        = 0.15                      # inlet Mach (physical)
C_S_PHYS  = 340.3                     # [m/s] SLS
NU_PHYS   = 1.461e-5                  # [m^2/s] SLS
RE        = 6.0e6
AOA_DEG   = 10.0                      # baked into the STL (span-axis pitch)
U_INF     = MA * C_S_PHYS             # 51.045 m/s
L_CHAR    = RE * NU_PHYS / U_INF      # chord = 1.7173 m
U_LU      = MA / math.sqrt(3.0)       # = what acoustic scaling derives

STL_PATH   = os.path.join(_repo, "input_files", "geom", "naca0012_wing.stl")
STL_CHORD  = 1000.0                   # file units (measured; x-extent/cos10)


def _build(c=100, wall_bc="ibb", nz_frac=0.16):
    """Config dict for chord = c L0 cells. Region bounds are linear in c
    and integer for c in {100, 50}; c=50 is the local-smoke twin (~6.6M
    cells), c=100 the production case (~52.4M).

    nz_frac: slab thickness / chord. Under the PROVEN z-slice invariance
    (<= 7e-7) every Nz gives the same physics; a thin slab (e.g. 0.04 ->
    Nz=4, ~13M cells) fits the STANDARD path on one GPU. (The 2026-08-01
    esoteric force bias was root-caused to SGS wall pathologies and fixed
    — patch_notes/stl_body/12; the std nz4 run now serves as the
    cross-check twin for the repaired esoteric path.)"""
    if c % 50 != 0:
        # Resolution-ladder chords (patch 79): region bounds are rounded
        # by _r anyway, so a non-multiple only shifts box edges by <1 lu
        # (physically negligible; exact twin-comparability is what the
        # 50|c rule protected). Loudly noted, not refused.
        print(f"  [config] chord c={c} is not a multiple of 50 — region "
              f"bounds rounded (<1 lu shifts; resolution-ladder use)")
    Nx, Ny, Nz = round(9.6 * c), 6 * c, round(nz_frac * c)
    xc, yc, zc = 3.5 * c, 3 * c, 0.5 * Nz     # rotated-bbox center [L0 lu]

    def _r(x0, x1, y0, y1):
        return {"region": {
            "x_min": round(x0 * c), "x_max": round(x1 * c),
            "y_min": round(y0 * c), "y_max": round(y1 * c),
            "z_min": 0, "z_max": Nz - 1}}      # full z (inclusive nodes):
                                               # flush domain faces

    levels = [
        {},                                    # L0
        _r(2.50, 4.70, 2.44, 3.56),            # L1
        _r(2.84, 4.24, 2.72, 3.28),            # L2
        _r(2.90, 4.12, 2.84, 3.16),            # L3 (wing level)
    ]

    boundaries = {
        "xmin": {"location": "xmin", "method": "eq",
                 "velocity": [U_INF, 0.0, 0.0]},
        "xmax": {"location": "xmax", "method": "sponge",
                 "velocity": [U_INF, 0.0, 0.0], "density": 1.0,
                 "thickness": 20, "strength": 0.1},
        "ymin": {"location": "ymin", "method": "neumann"},
        "ymax": {"location": "ymax", "method": "neumann"},
        # zmin/zmax OMITTED = PERIODIC (faces absent from bc_manager stay
        # FLUID; wrap streaming is the periodic BC — Phase 1a convention).
        # Canonical for the infinite wing: periodic-z == continued
        # extrusion, exactly matching the span_through prism contract.
        # Under z-invariance neumann was equivalent to rounding (measured
        # 6e-7 vs periodic); periodic states the intent and stays correct
        # if Nz is ever enlarged to resolve real spanwise turbulence.
    }

    folder = f"results_naca0012_a10_m015_re6m_c{c}"
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
                "scale_to_lu": float(c) / STL_CHORD,
                "center_lu": (float(xc), float(yc), float(zc)),
                "rotation_deg": (-90.0, 0.0, 0.0),   # axis swap only
                "span_through_axis": "z",            # infinite-wing prism
                "wall_bc": wall_bc,
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
            # A_ref = chord x in-domain span (planform) -> standard Cl/Cd.
            "reference": {"rho": 1.0, "velocity": U_LU,
                          "char_length": float(c),
                          "span_length": float(Nz)},
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
