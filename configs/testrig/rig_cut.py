"""Sphere STRADDLING the L1 interface — wall-aware coupling rig. See _rig_base.

The sphere spans z 21..35; L1's floor is raised to z=22, so its lower cap
sits on L0 while the rest is on L1 and the C2F band on the z_low face is
full of solid cells. This is the octo8 airframe problem at rig scale:
`_check_body_vs_coupling_band` rejects the build outright unless the
coupling is told to skip the wall neighbourhood.

    mlg.wall_coupling = {"mode": "exclude", "wall_margin": 1}

Judge against `rig_single.py` — the identical case with the body fully
enclosed by L1 — not against `rig_off.py`: the question is what the
interface does to a body it cuts, and `single` isolates exactly that.
Rotor thrust symmetry (counter-rotating, ~1e-4) stays a free check that
the ALM side is undisturbed.

Read the caveat in src/grid/wall_coupling.py first: only the coupling
WRITE side is excluded. The C2F read still interpolates across the body,
so near-wall quantities on the cut face are outside what this rig
validates.

Run:
    LBM_ESOTERIC=1 python main.py --config configs/testrig/rig_cut.py --gpu 0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _rig_base import build  # noqa: E402

config = build(mlg="cut", forces=True,
               wall_coupling={"mode": "exclude", "wall_margin": 1})
