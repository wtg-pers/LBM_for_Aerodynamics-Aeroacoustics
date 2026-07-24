"""HVAB hover c10 M0.65 — FINAL FORMULATION + Shen tip-loss g=0.5.

Formulation (2026-07-23 anchor run mlg4_k17l_la, 30 rev): ALM on L3
(interface-free), K17L ILES (omega_345=0, limiter 0.01, SGS off), iso
variable-eps spreading (0.25c), coupled LineAverage sampling (constant
rs=0.25c, N_s=80, EMA 6 deg). Anchor without tip loss: CT 0.00964
(+23.0% EXP / +15.9% GT), peak M2cn +8.1% vs KSAS (best yet), converged
from rev ~16-20 -> 20 rev + last-4-rev averaging is sufficient.

Shen 2005 load factor F=(2/pi)arccos(exp(-g f)), f=(B/2)(R-r)/(r sin phi),
R_eff=R (eps_offset False, standard Prandtl), tip only. g=1.0 == Prandtl
(sweep reference), g<1 broadens the roll-off inboard. Judge: peak M2cn /
CT vs EXP 0.00784 / roll-off shape / double-counting with the resolved +
LineAverage-read tip relief.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="farfield40_mlg4",
    use_sgs=False,
    eps_correction=None,
    prandtl_loss={"enabled": True, "model": "shen", "g": 0.5,
                  "tip": True, "root": False, "eps_offset": False},
    sampling={"mode": "lineavg", "profile": "constant",
              "n_points": 80, "rs_chord_factor": 0.25,
              "time_filter_deg": 6.0},
    anisotropic=None,
    marker_distribution="uniform",
    n_radial=64,
    n_rev=25,
    polar_source="ksas_psu",
    run_tag="farfield40_eso_mlg4_k17l_shen_g050",
)
config["simulation"]["omega_3"] = 0.0
config["simulation"]["omega_4"] = 0.0
config["simulation"]["omega_5"] = 0.0
config["simulation"]["cumulant_limiter"] = 0.01
