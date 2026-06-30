"""CT smoke + eps_correction ON (inviscid) — plumbing/regression gate.

Same deck as ct_hover_smoke.py (prandtl_loss=True kept identical) with only the
viscous-core smearing correction added, so any ΔT_lu vs the 0.080959 baseline is
purely the correction engaging end-to-end.

    python main.py --config configs/caradonna_tung/ct_hover_smoke_epscorr.py --max-steps 6 --no-vtk --clear
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.439, smoke=True)
config["actuator_line"]["eps_correction"] = {"enabled": True, "target": "inviscid"}
