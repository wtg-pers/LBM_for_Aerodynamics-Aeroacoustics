"""NACA0012 infinite wing a=10, Nz=4 — wall_bc="surfel" twin (S8c smoke).

S8c PRECURSOR, not a physics-quality run (patch_notes/surfel/47-48):
S8a is L0-single-grid only, so this twin runs the wing at chord = 100
cells on ONE level (the ibb/hwbb baseline resolves it on MLG L3 = 800
cells/chord). What this run decides is MECHANISM, not aerodynamics:

  * span-through facets: the wing STL pierces both z faces — do the
    surfel build / prism tables / periodic advect close momentum at the
    z boundary? (checks: mass drift, |Fz|/|Fx| ~ 0, facet count sanity)
  * production wiring at scale: 2.3M cells, force readout via the facet
    ledger instead of MEM links, stability over O(10k) steps.

S8a deviations from the ibb twin (naca0012_a10_std_nz4.py):
  * mlg DISABLED — kept single-grid as the L0 mechanism twin (surfel +
    MLG is supported since S8a-2, patch 50 — see
    naca0012_a10_surfel_mlg4.py for the leveled run).
  * sgs DISABLED — _advance_surfel passes scalar tau; SGS/tau-model on
    the production surfel path is S8b scope. (SGS-off cumulant is the
    stl_body convention; smoke does not read quantitative Cd.)
  * surfel.march_axis = 0: dV fluid-fraction march must enter from
    FLUID. The default axis=2 (testbed channel) would start INSIDE the
    wing cross-section for columns under the airfoil — march along x
    (inlet is always fluid).

Run (cluster, single GPU — std path, NO LBM_ESOTERIC):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    python main.py --config configs/stl/naca0012_a10_surfel_nz4.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_surfel_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m._build(100, wall_bc="surfel", nz_frac=0.04)     # Nz = 4

# S8a scope: single grid, no SGS (see module docstring)
config["mlg"] = {"enabled": False}
config["sgs"] = {"enabled": False}
config["internal_geometry"]["stl"]["surfel"] = {"march_axis": 0}

_folder = "results_naca0012_a10_surfel_nz4"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
