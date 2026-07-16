"""HVAB hover c10 M0.65 — far-field D40, CASE 1': pure ALM, WENDLAND kernel.

beta-kernel D40 matrix member (patch_notes/alm_beta_kernel/01_design #14.5):
physics-identical to CASE 1 (pure ALM, corrections OFF, NASA deck) except
actuator_line.kernel = Wendland C2 — eps-equivalent support
R_s = sqrt(7.5) eps replaces the gaussian 3-eps cutoff automatically
(spreading + sampling; no deficit kernel in play). Judged against the
case-1 gaussian archive: peak-region (0.85-0.97R) M2cn and CT
(success criteria: 00_handoff #5.4). Rank count is free (decomposition
is bit-identical); the beta driver runs 2-rank on GPUs 0,1:
    ./run_beta_d40.sh    (or the mpirun line inside it)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
    eps_correction=None,
    prandtl_loss=False,
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="nasa_overflow",
    run_tag="farfield40_eso_case1w_wendland",
)
config["actuator_line"]["kernel"] = {"type": "wendland"}
