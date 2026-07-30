"""HVAB hover c10 M0.65 — NEW FORMULATION baseline, no tip loss, 30 rev.

2026-07-22 formulation revision on the confirmed baseline (mlg4 + K17L +
SGS off = ILES): spreading stays iso variable-eps (0.25c chord-based,
2dx floor); sampling switches to coupled LineAverage (Melani 2024,
constant rs=0.25c, N_s=80, EMA time filter 6 deg of azimuth). Expected
vs gaussian-sampling twin (mlg4_k17l): midspan unchanged, very-tip
(>0.97R) alpha lower ~1.2 deg (offline DELA diagnostic), CT -0.5~-1%.

30 rev ON PURPOSE: establishes the convergence rev count of the new
formulation (k17l showed dCT/rev +8.5e-5 drift at rev 20) before the
Prandtl(g=1)/Shen g-sweep re-anchors on it. Judge: CT(rev) plateau
location, spanwise M2cn, wake metrics from last-2-rev fields.

4-GPU run (65.1M cells, 30 rev = 37710 steps):
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \\
        --config configs/hvab/hvab_hover_c10_farfield40_eso_mlg4_k17l_la.py \\
        --steps 37710 --log-every 64 --cuda-aware 1 --dist-init \\
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 2 \\
        --ckpt-every 37710 --csv mlg4_k17l_la.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_mlg4",
    use_sgs=False,
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "lineavg", "profile": "constant",
              "n_points": 80, "rs_chord_factor": 0.25,
              "time_filter_deg": 6.0},
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=30,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_mlg4_k17l_la",
)
config["simulation"]["omega_3"] = 0.0
config["simulation"]["omega_4"] = 0.0
config["simulation"]["omega_5"] = 0.0
config["simulation"]["cumulant_limiter"] = 0.01
# esoteric streaming via config (env LBM_ESOTERIC no longer required on the
# command line; env still overrides for A/B). entry_unification note 11.
config["numerics"]["esoteric"] = True
