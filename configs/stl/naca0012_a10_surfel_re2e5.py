"""NACA0012 infinite wing a=10 — S8c ladder ANCHOR rung: Re 2e5, full stack.

First rung of the S8c Reynolds ladder (patch_notes/surfel/53 sec. 3
branch (a)+(b)): same wing, same grid (c=100, MLG4, wing L3 = 800
cells/chord), same Mach — ONLY Re moves, via the registered convention
(L_char FIXED as the case identity, nu derived = U*L/Re; patch 54).

Why 2e5 is the anchor: every premise of the validated wall-model
campaign returns to its design neighborhood at once —
  tau_L3 - 0.5 = 1.0e-3  (30x the 6e6 margin; the testbed campaign ran
                          at the same order, tau ~ 0.502)
  delta ~ 30 wing cells  (vs 13 at 6e6 — WMLES-resolvable outer BL)
  Delta+ ~ 30-60         (just under the campaign band 63~448)
  cell Re (wing) = 250   (far below the tau->0.5 pumping regime)

Full stack = surfel + tau-model band (52) + SGS smagorinsky Cs = 0.1
(S8b-2, patch 54 — the moment-based Stiebler route and the EXACT Cs the
campaign tables were measured with; s13 [S] pins parity 3.9e-7).

Registered caveats: span 0.04c (Nz = 4 slab) constrains the largest 3D
structures — the rung reads Cp/Cf/bubble first, span-widening is the
follow-up arm; physical reference at Re 2e5 alpha=10 is transition-
sensitive (XFOIL-class Cl ~ 1.0, Cd ~ 0.015-0.02, laminar bubbles are
PHYSICAL here) so the read is regime-level, not point-matching.

Run (cluster, single GPU — std path, NO LBM_ESOTERIC):
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    python main.py --config configs/stl/naca0012_a10_surfel_re2e5.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.join(_here, "naca0012_a10_m015_re6m.py")
_spec = importlib.util.spec_from_file_location("naca0012_surfel_base", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

RE_RUNG = 2.0e5

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

_folder = "results_naca0012_a10_surfel_re2e5"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
