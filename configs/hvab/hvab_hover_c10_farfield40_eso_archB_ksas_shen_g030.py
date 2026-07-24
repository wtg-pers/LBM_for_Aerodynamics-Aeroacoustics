"""HVAB hover c10 M0.65 — archB (era-B best FM) + Shen g=0.3 tip loss.

CASE 4 formulation exactly (mlg5 farfield40, all-one cumulant + dyn_smag
SGS, iso gaussian sampling/spreading + Merabet radial truncation + Kleine
straight correction, KSAS psu deck, n64, 25 rev) with Shen tip loss added
on top (force-multiplier stage, downstream of the Kleine sampling
correction — no code interaction).

Purpose (2026-07-23, workshop-pragmatic track): archB gave the best
CT/sigma-FM placement (dFM +0.014); Shen g<=0.3 moves the point left
toward the EXP c10 anchor (clean-baseline g030 gave CT -11%). Judge on
the CT/sigma-FM plane; wake only sanity-checked (not the metric).
NOTE: current code has post-era-B fixes -> expect small drift vs the
era-B archB numbers (0.00955/0.7412); attribute vs a fresh archB anchor
if exact attribution is needed.

4-GPU run:
    mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \
        --config configs/hvab/hvab_hover_c10_farfield40_eso_archB_ksas_shen_g030.py \
        --steps 31425 --log-every 64 --cuda-aware 1 --dist-init \
        --devices 0,1,2,3 --vtk-every 1257 --vtk-fields-last 5 \
        --ckpt-every 31425 --csv archB_shen_g030.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40",
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
    run_tag="farfield40_eso_archB_ksas_shen_g030",
)
