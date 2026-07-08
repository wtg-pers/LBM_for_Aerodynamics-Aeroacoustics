"""HPC Phase 0 diagnostic — bench5 with the ALM disabled (pure LBM).

Identical to bench5_baseline.py in EVERY respect (same 5-level topology incl.
the rotor-disk L4 slab, same D3Q27 cumulant fused kernels, dyn_smag, sponge,
same cells/level -> same updates_per_step) EXCEPT actuator_line.enabled=False.
Disabling ALM makes al_model=None (setup.py:985), so advance() takes the
_advance_fused (no-ALM) path.

Purpose (PLAN.md §0): subtract this run's wall/MLUPS from bench5_baseline's to
split the coarse step into
    LBM-only cost   (this run)
    ALM serial cost (baseline - this run)
and cross-check the MLG_PROFILE / ALM_PROFILE breakdown. Because the topology
(and thus updates_per_step) is unchanged, MLUPS is directly comparable.

NOT a physics case (D=16); numbers are timing anchors only.

    MLG_PROFILE=1 python main.py --config configs/hpc_bench/bench5_pure_lbm.py
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
    n_rev=2, polar_source="nasa_overflow", run_tag="bench5_pure_lbm",
)
# ── Phase 0: pure-LBM toggle — the ONLY difference from bench5_baseline. ──
# al_enabled=False -> al_model=None -> _advance_fused path (no rotor pipeline).
config["actuator_line"]["enabled"] = False
