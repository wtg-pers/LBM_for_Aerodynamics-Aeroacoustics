"""ROBIN rotor-off grid campaign Phase 2 — 4-level ladder boxes, r = 20.

User decision (0901, robin/14 s6): the rotor-only recipe D40 + MLG-4
mapped onto the fuselage (rotor D = 2R -> r20 = D40). finest = R/160,
boom-min 8.0 cells = a DELIBERATE test of the 10-cell finest rule
(boom passes -> rule was conservative; aft degrades vs the r32-4 control
-> rule confirmed, move to r25-4 / r20-5). Boxes: ladder L1/L2 frozen +
GRID4_L3_BOX_R (pads >= 0.125R). ~9.3M nodes, ~0.13x the production
compute. Control = robin_r0_musker (r32-4, production).

Run (main dir):
    LBM_ESOTERIC=1 python main.py --config configs/robin/robin_g4_r20.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build(r_lu0=20, tag="robin_g4_r20", max_steps=5600,
                  output_interval=280, num_levels=4, ladder_boxes=True)

if __name__ == "__main__":
    _m.report(config, "robin_g4_r20")
