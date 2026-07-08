"""TEST A — Kleine non-iterative + FREE wake (15 rev), with the downwash-sign fix.

Runs alongside testA_dag_{helix,straight} for the sign check. The 2026-07-04 fix
made the free-wake PROJECTION use the downwash direction (−thrust_axis) instead of
−sign(ω)·rotation_axis, which for HVAB (thrust=−axis) was the THRUST side and FLIPPED
the correction sign. (Only the straight limit had ever been validated, not this
projection on the real rotor — so past Kleine-free runs had the wrong sign.)

How to read (15 rev, ~converged):
  - vtk/markers/markers_*.vtp, colour by `w_corr`: the tip should now be > 0 (added
    downwash → de-loads the over-predicted tip), same sign as testA_dag_straight.
  - vtk/markers/wake/wake_*.vtp: the CONVECTED free wake (should descend on the
    downwash side +x and roll up at the tip) — contrast with the helix (prescribed)
    and straight (radial semi-infinite lines).

wake=free, rebuild_every=1 (EXACT baseline — free-wake influence matrix rebuilt every
step). testA_kleine_free_fast.py is the rebuild_every=5 approximation to compare.
n_w auto = 2 rev @ 2° (Dağ), wake_markers=all. Kleine solve needs no relaxation.

    python main.py --config configs/0703_dag_edge_fix/testA_kleine_free.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "rebuild_every": 1,   # EXACT; n_w auto = 2 rev @ 2°
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testA_kleine_free",
)
