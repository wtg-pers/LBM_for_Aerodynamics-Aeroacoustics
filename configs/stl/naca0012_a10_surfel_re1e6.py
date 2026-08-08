"""NACA0012 infinite wing a=10 — S8c ladder rung: Re 1e6 (attached anchor).

Re-anchoring after the 2e5 verdict (patch_notes/surfel/56): alpha=10 at
Re 2e5 sits ON the physical stall boundary and transition dominates the
loads there — the rung measured a (physically plausible) stalled-side
shedding regime, not the attached discriminator. Re 1e6 keeps the twin
chain (same wing, alpha, grid as 51/53) with an UNAMBIGUOUS attached
reference (stall alpha ~13-14 deg; XFOIL-class Cl ~ 1.05, Cd ~ 0.012)
and every numerical premise in range:
  tau_L3 - 0.5 = 2.1e-4 (6x the 6e6 margin)
  delta ~ 21 wing cells, Delta+ ~ 60-100 (campaign band 63~448)
  cell Re (wing) = 1250
Full stack: surfel + tau-model band + SGS smagorinsky Cs 0.1; fused
collide (patch 55).

Run (cluster, single GPU — std path, NO LBM_ESOTERIC):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    python main.py --config configs/stl/naca0012_a10_surfel_re1e6.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_surfel_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

RE_RUNG = 1.0e6

config = _m._build(100, wall_bc="surfel", nz_frac=0.04)     # Nz = 4

# ── Re knob (patch 54 convention): L_char is the case identity, nu is
#    the derived free variable. Everything else (Ma, grid, BCs) fixed.
config["physics"]["nu"] = _m.U_INF * _m.L_CHAR / RE_RUNG

# ── full stack: band + campaign-parity SGS
config["sgs"] = {"enabled": True, "model": "smagorinsky", "Cs": 0.1}
config["internal_geometry"]["stl"]["surfel"] = {
    "march_axis": 0,
    "tau_model": True,
}

_folder = "results_naca0012_a10_surfel_re1e6"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
