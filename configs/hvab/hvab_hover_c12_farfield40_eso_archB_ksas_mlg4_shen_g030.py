"""HVAB hover c12.0 M0.65 — archB + Shen g=0.3 mlg4 (workshop collective sweep).

Identical formulation to the validated c10 point (CT +9.7% EXP, dFM +0.030,
best spanwise/wake of campaign): all-one cumulant + dyn_smag, gaussian
sampling/spreading + Merabet radial truncation + Kleine straight + Shen
g=0.3, KSAS psu, n64, mlg4, 25 rev. ONLY collective changes.

Purpose (2026-07-24): HVAB workshop figures — pitch vs CT/sigma,
CT/sigma vs FM, wake-age trajectories (shadowgraphy TH08/TH10/TH12
available; c06 has no SG reference). Sweep set: c06/c08/c12 (+c10 done).

4-GPU run: driver run_hvab_archb_collective_sweep.sh, or
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \
        --config configs/hvab/hvab_hover_c12_farfield40_eso_archB_ksas_mlg4_shen_g030.py \
        --steps 31425 --log-every 64 --cuda-aware 1 --dist-init \
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 5 \
        --ckpt-every 31425 --csv archB_mlg4_shen_g030_c12.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=12.0, mtip=0.65, preset="farfield40_mlg4",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "straight", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss={"enabled": True, "model": "shen", "g": 0.3,
                  "tip": True, "root": False, "eps_offset": False},
    radial_truncation=True,
    sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_archB_ksas_mlg4_shen_g030",
)
