"""NACA0012 infinite wing a=10 — surfel + MLG 2-level LOCAL smoke (S8a-2).

Mechanism smoke for the S8a-2 MLG levelization (patch_notes/surfel/50):
c = 50, num_levels = 2 -> wing on L1 at 100 fine cells/chord, ~770k
cells total. What this run checks (NOT aerodynamics):

  * per-level SurfelBoundary build: level-local triangles_lu, level nu
    (L1 nu = 2x L0 nu under acoustic scaling), band guards pass on a
    realistic region,
  * MLG advance with _advance_surfel on BOTH levels: stability, mass
    drift, C2F/F2C strips vs the periodic advect wrap,
  * per-level z closure: |Fz|/|Fy| ~ 0 on the L1 facet ledger (the
    patch-48 Fz metric, now on a fine level with Nz_f = 2*Nz-1 = 7),
  * force level binding: force_history must come from L1 (header
    "Force measurement: Level 1"), surface_*.vtk written in global
    L0-lu coordinates next to the .vth output.

Run (local, single GPU, ~300 steps):
    python main.py --config configs/stl/naca0012_a10_surfel_mlg2_smoke.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_surfel_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_C = 50
config = _m._build(_C, wall_bc="surfel", nz_frac=0.08)      # Nz = 4

# 2 levels: keep the base L1 region (wing + margin), drop L2/L3
config["mlg"] = dict(config["mlg"], num_levels=2,
                     levels=config["mlg"]["levels"][:2])

# surfel scope: no SGS (S8b), dV march along x (inlet is always fluid)
config["sgs"] = {"enabled": False}
config["internal_geometry"]["stl"]["surfel"] = {"march_axis": 0}

_folder = "results_naca0012_a10_surfel_mlg2_smoke"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
config["time"] = dict(config["time"], max_steps=300, output_interval=100,
                      logging_interval=50, checkpoint_interval=100000,
                      conservation_interval=100)
