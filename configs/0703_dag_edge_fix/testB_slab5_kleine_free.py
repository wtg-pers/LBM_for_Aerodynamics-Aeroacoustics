"""TEST B run #2 — slab-L4 grid (light_slab5), Kleine free wake, 15 rev.

SECOND slab5 run, after testB_slab5_pure_alm establishes the pure grid effect.
Question: does the free-wake correction behave differently once the LBM field
actually resolves tip-vortex roll-up/contraction? (Kleine's own caveat — and
our testA finding that on the light grid the free-wake tip correction turns
upwash-signed — both hinge on the resolved contraction, so the model response
itself is grid-dependent. This run measures that response; the pure-ALM run
must come first to separate grid from model.)

Physics = testA_kleine_free (kleine, wake=free, rebuild_every=1, all edges,
smooth=2, NASA deck, gaussian, no Prandtl); grid = light_slab5.
NOTE: kleine free has only ever run on light 4-level — run a short smoke
(~300 steps, nan_trap on) before committing 15 rev on this 5-level grid.

    python main.py --config configs/0703_dag_edge_fix/testB_slab5_kleine_free.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light_slab5",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testB_slab5_kleine_free",
)
