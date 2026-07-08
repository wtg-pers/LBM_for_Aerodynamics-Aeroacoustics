"""HPC-upgrade regression/perf baseline — bench5 (5-level mini slab), 2 rev.

Purpose: a minutes-scale run that exercises EVERY production code path
(D3Q27 cumulant fused kernels, 5-level MLG with the rotor-disk L4 slab,
dyn_smag SGS, ALM + NASA C81 decks, outlet sponge) so speed/memory/multi-GPU
work can be validated against it. ALM correction = **Kleine free wake**
(2026-07-05 decision): the physically-preferred production model AND the
heaviest ALM code path (per-step solve, exact-Phi, wake convect D2H batch,
influence rebuild) — exactly what the Phase-1c ALM-overlap work must not
break. The Phase-0 pure-LBM toggle run covers the LBM-only regression.

Gates against the archived reference:
  1. bit-identical: checkpoints/checkpoint_*.npz must match the reference
     for algorithm-preserving changes (same GPU + env).
  2. physics: CT over rev 2 (rotor_performance.csv tail) within thrust-CV
     of the reference for physics-preserving changes (e.g. domain split).
  3. speed: MLUPS / wall-per-step from performance.csv, same hardware.

NOT a physics case — D=16 (rotor diameter 16 lu) is far below any
resolution requirement; numbers are anchors, not predictions.

~9.05M cells / ~3.5 GB, 503 steps/rev, 1006 steps total (ramp = rev 1).

    python main.py --config configs/hpc_bench/bench5_baseline.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_baseline",
)
