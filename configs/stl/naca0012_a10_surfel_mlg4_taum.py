"""NACA0012 infinite wing a=10 — surfel MLG4 + tau-model band (S8b verdict).

The S8a-2 verdict run (patch_notes/surfel/51) ended stalled-signature-free
but left a SEPARATION BUBBLE on the suction side (x/c 0.20~0.35: Cf
0.0079 -> 0.00002 -> re-attach, Cp plateau -1.2) — the registered
signature of the missing near-wall turbulent stress (no SGS, no
tau-model band). This twin turns the band ON (patch 52: the channel
campaign's measured-closure supply, FS_CLASSES v3, zero free
parameters, generalized per-facet) and asks the REGISTERED question:

  does the band kill/shrink the bubble (Cf zero-crossing interval,
  Cp plateau) and move Cl (+0.754 -> ~1.05) / Cd (0.0314 -> ~0.011)
  toward the attached-flow values?

Read with s8a2_mlg4_read.py (same 4 items; compare item (3) against the
patch-51 table). Fidelity caveat (registered, patch 52): the supply
tables were inverted on a Cs = 0.1 SGS channel; the production surfel
path still runs SGS-off (S8b-2 = SGS integration) — the band supplies
the dominant near-wall stress share either way.

Run (cluster, single GPU — std path, NO LBM_ESOTERIC):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    python main.py --config configs/stl/naca0012_a10_surfel_mlg4_taum.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_surfel_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m._build(100, wall_bc="surfel", nz_frac=0.04)     # Nz = 4

config["sgs"] = {"enabled": False}
config["internal_geometry"]["stl"]["surfel"] = {
    "march_axis": 0,
    "tau_model": True,          # S8b band (defaults: FS_CLASSES v3, wall
}                               # src, inject->apply, no cut-cell inject)

_folder = "results_naca0012_a10_surfel_mlg4_taum"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
