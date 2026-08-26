"""ROBIN rotor-off R1 — R0 + pressure-gradient wall function (patch 81).

ONE knob over robin_r0_musker: internal_geometry.stl.surfel.pressure_gradient
= {"ds": 2.0} (per-facet tau_turb = tau_Musker * R(beta), R == 1 at beta = 0).
Discriminators pre-registered in patch_notes/robin/02 sec. 5 (R1 vs R0:
nose/fore station Cp, pylon-junction adverse-gradient stations, global
RMS). Same grid, same steps, same window.
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "robin_base", os.path.join(_here, "_robin_base.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = _m.build(r_lu0=32, tag="robin_r1_pg", max_steps=9000,
                  output_interval=500,
                  surfel_extra={"pressure_gradient": {"ds": 2.0}})

if __name__ == "__main__":
    _m.report(config, "robin_r1_pg")
