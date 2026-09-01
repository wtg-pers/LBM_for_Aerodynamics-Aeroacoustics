"""ROBIN rotor-off grid ladder — 3-level, r = 20 (patch robin/13 s7).

User hypothesis H2 (registered BEFORE the ladder results, 0901): at equal
finest dx the level structure / far field is neutral — this run (L0 +
L1(-0.5..3,+-1,+-1)R + L2(-0.25..2.5,+-0.5,+-0.5)R, finest = R/80) should
match the 2-level r40 (finest = R/80) within the noise floor, at ~6x fewer
nodes. Boom-min is 4 fine cells in BOTH (identical representability).
Stack/steps identical to robin_g2_r20 (10 FT = 5600 steps, output 280,
window = last 5 outputs).

Run (main dir):
    LBM_ESOTERIC=1 python main.py --config configs/robin/robin_g3_r20.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build(r_lu0=20, tag="robin_g3_r20", max_steps=5600,
                  output_interval=280, num_levels=3)

if __name__ == "__main__":
    _m.report(config, "robin_g3_r20")
