"""NACA0012 infinite wing a=10, Nz=4 — surfel + MLG 4-level (S8a-2 verdict).

The S8c precursor at L0 c=100 confirmed the surfel wiring but sat OUTSIDE
the wall model's design region: delta ~ 1.6 cells < h = 3 cells, so the
log-layer sample read the outer flow and the wing stalled numerically
(patch_notes/surfel/49). This twin puts the wing on MLG L3 = 800 fine
cells/chord — delta ~ 13 fine cells, h = 3 ~ 0.23 delta: the wall law
samples inside the boundary layer for the first time on the wing.

Setup mirrors the ibb baseline (naca0012_a10_std_nz4) except:
  * wall_bc = "surfel" on every level carrying the body (S8a-2:
    level-local triangles/nu, coupling-band + z-flush guards at setup),
  * sgs DISABLED — surfel + SGS is S8b scope (setup raises otherwise);
    NB the ibb baseline itself is outside quantitative range at this Re
    (tau->0.5 pumping, Cd < 0 — patch_notes/surfel/49 sec. 2), so the
    reading here is surfel-internal: surface Cp/Cf shape (attached vs
    stalled), Fz closure per level, force level = L3.
  * surfel.march_axis = 0 (dV march enters from fluid at the inlet).

Memory note: Nz = 4 slab -> ~11M cells across the 4 levels (L0 2.3M /
L1 0.7M / L2 1.6M / L3 6.2M). The surfel advect keeps dense (27, N)
float64 Q/g_field per level (~4.7 GB) on top of ~2.3 GB f-buffers —
total well under 20 GB, comfortable on one cluster GPU (sparse g_field
stays a listed follow-up for thicker slabs).

Run (cluster, single GPU — std path, NO LBM_ESOTERIC):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    python main.py --config configs/stl/naca0012_a10_surfel_mlg4.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_surfel_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m._build(100, wall_bc="surfel", nz_frac=0.04)     # Nz = 4

# surfel scope: no SGS (S8b), dV march along x (inlet is always fluid)
config["sgs"] = {"enabled": False}
config["internal_geometry"]["stl"]["surfel"] = {"march_axis": 0}

_folder = "results_naca0012_a10_surfel_mlg4"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
