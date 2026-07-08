"""TEST B smoke — light_slab5 first-ever 5-level stability check (CLUSTER ONLY).

Same build as testB_slab5_dag_straight.py (grid + physics identical) but with
the NaN trap ON (check every 10 coarse steps): light_slab5 is the first
5-level MLG+ALM config to run at all — the old fine/tip5 5-level presets
NaN-crashed at step 1 — so trap early and cheap. Run a few hundred steps:

    python main.py --config configs/0703_dag_edge_fix/testB_slab5_smoke.py --max-steps 300

Pass criteria: (1) nvidia-smi ~18.6 GB (abort if >22), (2) no NaN trap fire,
(3) setup log shows 5 levels / 45.3M cells (L4 17.4M), (4) thrust finite and
ramping. Then run the real 15rev case (testB_slab5_dag_straight, nan_trap off).

Do NOT run locally — 45.3M cells; the local PC hits its CPU thermal limit.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light_slab5",
    eps_correction={"enabled": True, "method": "dag", "target": "inviscid",
                    "relax": 0.5, "smooth": 2, "wake": "straight"},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testB_slab5_smoke",
)
config["numerics"]["nan_trap"] = True
config["numerics"]["nan_check_every"] = 10
