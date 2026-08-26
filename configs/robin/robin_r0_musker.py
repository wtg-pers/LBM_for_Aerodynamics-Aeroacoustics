"""ROBIN rotor-off R0 — Musker baseline (patch_notes/robin/02).

TM-80051 Run 12 pt 90 (alpha 0, beta 0, 81.7 kt, Re_L 9.06e6). Stack =
NACA campaign parity: surfel + tau-model ON + Musker h=3 + Smagorinsky
Cs 0.1; gamma (intermittency) OFF — airfoil-only logic; trip OFF; PG OFF.
Everything (grid, physics, discriminators, runbook) is in _robin_base.py
and patch_notes/robin/02. R1 (robin_r1_pg.py) is the ONE-knob clone.
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build(r_lu0=32, tag="robin_r0_musker", max_steps=9000,
                  output_interval=500)

if __name__ == "__main__":
    _m.report(config, "robin_r0_musker")
