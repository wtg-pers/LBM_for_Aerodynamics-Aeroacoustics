import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(collective_deg=8.0, mtip=0.65, smoke=True,
                      eps_correction={"enabled": True, "target": "inviscid"},
                      prandtl_loss=False)
