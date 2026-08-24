"""LE-resolution ladder point c=100 (patch 79) — thin-span (nz4 lineage).

R1 discriminator without the partial-L4 machinery (patch 78: the
band-through-wall refinement box is a research problem) and without the
replicated-build ceiling (a global span16 upscale cannot even BUILD on
one 24 GiB card): scale the WHOLE grid at fixed physics (Re-targeting
rule: L_char fixed, nu derived) on the thin slab where each point is a
single-GPU job. r_LE = 12.7 / 17.9 / 21.6 L3 cells at c = 100/141/170.

Regime note (pre-registered): thin span = the 57-era span-locked
quasi-steady state — the ladder reads the RESOLUTION response of the
attached LE suction peak (quasi-2D), not the turbulent-BL absolute.
Readout: p_state Cp_min (x/c 0.005-0.01, span-avg) from the finalize
surface file. If c=100 already gives ~-5.3 the grid resolves the peak
(deficit = BL/wall-model physics); a monotone deepening across the
ladder confirms the resolution mechanism instead.

Run (ONE GPU each; two points can run in parallel on two cards):
    LBM_ESOTERIC=1 python main.py \
        --config configs/stl/naca0012_a10_surfel_re1e6_nz4_c100.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "re6m_base", os.path.join(_here, "naca0012_a10_m015_re6m.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

RE_RUNG = 1.0e6

config = _m._build(100, wall_bc="surfel", nz_frac=0.04)
config["physics"]["nu"] = _m.U_INF * _m.L_CHAR / RE_RUNG
config["sgs"] = {"enabled": True, "model": "smagorinsky", "Cs": 0.1}
config["internal_geometry"]["stl"]["surfel"] = {
    "march_axis": 0,
    "tau_model": True,
}
config["time"] = dict(config["time"], max_steps=10000)

_folder = "results_naca0012_a10_surfel_re1e6_nz4_c100"
config["output"] = dict(
    config["output"],
    output_dir=f"./{_folder}/vtk",
    checkpoint_dir=f"./{_folder}/checkpoints",
    csv_dir=f"./{_folder}/csv",
)
