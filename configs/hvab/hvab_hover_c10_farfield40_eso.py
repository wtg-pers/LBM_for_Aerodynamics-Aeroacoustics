"""HVAB hover c10 M0.65 — far-field D40 (cells/R 320), ESOTERIC single-24GB.

4-case matrix CASE 1 (handoff §5): pure ALM — isotropic Gaussian, N=64
markers/blade, corrections OFF (no eps_correction, no Prandtl), NASA
OVERFLOW C81 deck, n_rev=25 (converged; bench5 lesson: 3 rev is not).

Grid: preset 'farfield40' — the literature-anchored far-field domain
(radial 6R / upstream 3R / downstream 10R, blockage 2.7%) with the
validated slab5 4-fine telescope at D0=40 -> cells/R 320, tip chord ~17
cells. ~102M cells.

MEMORY — REQUIRES the Esoteric Pull path (patch_notes/hpc_upgrade/15):
    LBM_ESOTERIC=1 python main.py --config configs/hvab/hvab_hover_c10_farfield40_eso.py
Standard path needs ~42GB (f + f_post); esoteric fits ~19GB on a single
24GB 4090 (f_post removed; f_prev sub-volume; region-scoped MLG bridge).
Do NOT enable MLG_CUDA_GRAPH (esoteric is parity-dependent; auto-rejected).

Other 3 cases of the matrix: archB (radial truncation + FLLC-straight) and
the KSAS/psu deck variants — deck wiring pending (handoff §4).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    eps_correction=None,          # corrections OFF (pure-ALM baseline)
    prandtl_loss=False,           # tip loss OFF (case 1)
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,                  # handoff §4: N=64/blade at cR320
    n_rev=25,
    polar_source="nasa_overflow",
    run_tag="farfield40_eso_case1",
)
