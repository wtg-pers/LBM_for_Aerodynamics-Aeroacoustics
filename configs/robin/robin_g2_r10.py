"""ROBIN rotor-off grid ladder — 2-level, r = 10 (patch robin/13, 1-knob = r).

Family: L0 = registered domain (-4..+10, +-3, +-3)R frozen; single L1 box
(-0.5..+3, +-1, +-1)R (round convention, user-set); finest = L1 = R/(2 r);
see _robin_base.GRID2_L1_BOX_R note. Stack identical to R0 (v1c, Musker h3,
tau_model, overlap_cap, crease noslip, Cs 0.1, p_sample_h 1.5). Steps =
~10 FT (281.25 * r), output = FT/2 -> readout window = last 5 outputs
(2.5 FT). QoI rules, representability gates, noise floor: patch robin/13.

Run (main dir):
    LBM_ESOTERIC=1 python main.py --config configs/robin/robin_g2_r10.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build(r_lu0=10, tag="robin_g2_r10", max_steps=2800,
                  output_interval=140, num_levels=2)

if __name__ == "__main__":
    _m.report(config, "robin_g2_r10")
