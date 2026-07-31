"""NACA0012 2D (D2Q9) — a=10 deg, M=0.15, Re=6e6. Cross-check twin.

Purpose: independent 2D anchor for the 3D infinite-wing case
(naca0012_a10_m015_re6m.py) — same chord resolution (100 coarse cells,
4-level MLG -> 800 cells/chord on L3), same domain cross-section,
same BC family (eq / sponge / neumann far-field), same physics
(acoustic scaling, u_lu = 0.0866), but the proven 2D polygon-PIP + IBB
path with none of the span-through machinery. Reference: Ladson (80-grit
trip) / NASA TMR 2D: Cl ~= 1.09, Cd ~= 0.0123 at a=10 deg — absolute Cd
comparison stays conservative here (skin friction unresolved at Re 6e6
with 800 cells/chord; judge sign/stability/pressure component).

Airfoil coordinates are generated on demand (closed-TE NACA0012, unit
chord, Selig numeric-only format) by src/utilities/naca_stl.py -> a git
pull is enough anywhere.

STATUS (measured 2026-07-31): ** Re 6e6 is OUTSIDE the D2Q9 stable
regime at this resolution ** (tau_L0 - 0.5 = 4e-6): dyn_smag, WALE and
no-SGS all NaN at the lower-surface LE by ~step 40; the constant-Cs
Smagorinsky below stays finite but the forces drift unphysically
(Cl -> 8 monotone). D3Q27 is robust at the same tau (the 3D twin runs
clean) — this is a D2Q9 tau->0.5 robustness gap, its own work item.
A stability-based ceiling tau_L0 >= ~0.5005 puts D2Q9 at Re <~ 5e4
for c = 100 L0 cells. VALIDATED reduced-Re anchor: override
physics.nu for Re = 2e4 (sgs off) -> Cd = +0.086 +- 0.011,
Cl = +/-0.789 exact mirror under AoA sign flip, converging windows.

AoA SIGN (latent convention defect, measured): the 2D transform
rotates CCW = TE UP for positive angle_of_attack (= nose-DOWN), while
its docstring claims "Nose-up" (src/boundary/geometry.py:592). This
config passes -AOA_DEG to realize true nose-up. The CLF5605 sweep's
alpha sign interpretation should be re-checked against this.

    python main.py --config configs/stl/naca0012_2d_a10_m015_re6m.py --gpu 0
"""
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(os.path.dirname(_here))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from src.utilities.naca_stl import ensure_naca4_selig_dat

# ── physical case (identical to the 3D twin) ────────────────────────────
MA        = 0.15
C_S_PHYS  = 340.3                     # [m/s] SLS
NU_PHYS   = 1.461e-5                  # [m^2/s] SLS
RE        = 6.0e6
AOA_DEG   = 10.0
U_INF     = MA * C_S_PHYS             # 51.045 m/s
L_CHAR    = RE * NU_PHYS / U_INF      # chord = 1.7173 m
U_LU      = MA / math.sqrt(3.0)

SELIG_PATH = ensure_naca4_selig_dat(
    os.path.join(_repo, "input_files", "geom", "naca0012_n256.dat"))

C = 100                               # chord in L0 cells (twin of 3D)
Nx, Ny = round(9.6 * C), 6 * C        # 960 x 600
CX, CY = 3.5 * C, 3 * C               # mid-chord placement (LE at 3c)

_folder = "results_naca0012_2d_a10_m015_re6m"
config = {
    "simulation": {"device_mode": "gpu", "precision": "float32",
                   "dimension": 2, "lattice_model": "D2Q9",
                   "collision_model": "cumulant",
                   "omega_3": 0.6, "omega_4": 1.4},
    "physics": {"rho": 1.225, "U_inf": U_INF, "nu": NU_PHYS,
                "L_char": L_CHAR, "flow_direction": [1.0, 0.0],
                "initial_flow_velocity": [U_LU, 0.0]},          # [lu]
    "grid": {"Nx": Nx, "Ny": Ny, "resolution": C},
    "numerics": {"acoustic_scaling": True, "c_s_phys": C_S_PHYS,
                 "collision": "cumulant"},
    "boundaries": {
        "xmin": {"location": "xmin", "method": "eq",
                 "velocity": [U_INF, 0.0]},
        "xmax": {"location": "xmax", "method": "sponge",
                 "velocity": [U_INF, 0.0], "density": 1.0,
                 "thickness": 20, "strength": 0.1},
        "ymin": {"location": "ymin", "method": "neumann"},
        "ymax": {"location": "ymax", "method": "neumann"},
    },
    "internal_geometry": {
        "airfoil": {
            "enabled": True,
            "selig_file": SELIG_PATH,
            "chord": float(C),
            "center": (float(CX), float(CY)),
            # SIGN (measured, 2026-07-31): the 2D transform rotates CCW,
            # which RAISES the TE for positive values (= nose-DOWN,
            # negative aerodynamic AoA; +10 gave Cl=-0.79 at Re 20k).
            # The docstring's "Nose-up" claim contradicts the math —
            # latent convention defect, also relevant to the CLF sweep.
            # Pass the NEGATED value to realize a true nose-up AoA.
            "angle_of_attack": -AOA_DEG,
            "wall_bc": "ibb",
        },
    },
    "mlg": {"enabled": True, "num_levels": 4, "overlap_width": 2,
            "interpolation": "cubic", "filter_level": 1,
            "levels": [
                {},                                            # L0
                {"region": {"x_min": 250, "x_max": 470,
                            "y_min": 244, "y_max": 356}},      # L1
                {"region": {"x_min": 284, "x_max": 424,
                            "y_min": 272, "y_max": 328}},      # L2
                {"region": {"x_min": 290, "x_max": 412,
                            "y_min": 284, "y_max": 316}},      # L3
            ]},
    # D2Q9 stability at tau ~= 0.500004: dyn_smag AND wale AND no-SGS all
    # blow up at the LE by step ~40 (both wall BCs — measured). They
    # vanish on the smooth impulsive-start field; the constant-Cs
    # Smagorinsky floor (nu_t ~ (Cs*dx)^2 |S|) damps the LE strain spike
    # and is stable. Model differs from the 3D twin (dyn_smag) — the 2D
    # case is a diagnostic anchor, not a model-matched comparison.
    "sgs": {"enabled": True, "model": "smagorinsky", "Cs": 0.17},
    "conservation": {"enabled": True, "verbose": 0, "log_to_csv": True},
    "convergence": {"enabled": False},
    "force_calculation": {
        "enabled": True,
        "interval": 20,
        # L0 lu; setup rescales x2^k to the force level (L3: 800).
        # 2D: A_ref = chord per unit span.
        "reference": {"rho": 1.0, "velocity": U_LU,
                      "char_length": float(C), "span_length": 1},
    },
    "output": {"output_dir": f"./{_folder}/vtk",
               "checkpoint_dir": f"./{_folder}/checkpoints",
               "csv_dir": f"./{_folder}/csv", "clear_previous": True,
               "vtk": {"enabled": True, "precision": "float32",
                       "variables": ["density", "pressure", "velocity",
                                     "velocity_magnitude", "solid_mask"]},
               "checkpoint": {"enabled": True, "keep_last_n": 2}},
    "time": {"max_steps": 10000, "output_interval": 2000,
             "logging_interval": 100, "checkpoint_interval": 2500,
             "conservation_interval": 500},
}
