"""Pure-LBM GPU-util DIAGNOSTIC — identical HVAB grid / MLG / domain / BC to the
Test A cases, but the ACTUATOR LINE IS DISABLED (enabled=False → al_model=None →
the pure-LBM `_advance_fused` path, no per-step CPU ALM sample→BEM→project).

Purpose — separate the two possible causes of the <100% GPU utilisation:
  • If THIS run saturates the GPU (~100% util) → the Test A <100% util is the ALM
    CPU serialisation (sample → CPU BEM/correction → project each step).
    → fix = overlap the ALM with the GPU (#2, 1-step lag) or port it to GPU (#3).
  • If THIS run is ALSO <100% util → the ceiling is MLG OCCUPANCY (coarse levels
    too small to fill the GPU) and/or kernel-launch overhead, INDEPENDENT of the
    ALM → a separate structural issue (#6 / CUDA graphs), NOT fixable by ALM overlap.

Everything except the ALM is byte-identical to testA_dag_* (same preset="light"
MLG levels, D3Q27 cumulant, domain, BCs), so the LBM kernel sizes/occupancy match —
the util you read here is the LBM-only ceiling for this exact grid.

Short run (n_rev=2 ≈ 2010 steps). Watch `nvidia-smi` during the steady stepping
phase and Ctrl+C once the util pattern is clear. The flow is quiescent (no rotor
forcing) — irrelevant to the diagnostic, since kernel sizes are unchanged.

    python main.py --config configs/0703_dag_edge_fix/diag_pure_lbm.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    n_rev=2, polar_source="nasa_overflow", run_tag="diag_pure_lbm",
)
config["actuator_line"]["enabled"] = False   # ← DISABLE ALM → pure-LBM path
