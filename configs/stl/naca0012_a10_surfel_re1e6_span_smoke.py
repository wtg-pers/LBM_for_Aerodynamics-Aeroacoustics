"""Local preflight for the wide-span arm — same geometry, c=50, 300 steps.

Purpose is wiring/stability only, NOT physics: the production span run
(naca0012_a10_surfel_re1e6_span.py) is the first time the 0.16c slab meets
the surfel path, and the two failure modes that have actually bitten this
track are both geometry-specific and z-dependent:

  - sliver cut cells (patch 50 sec. 4c/4e): dV is built from the raw
    facet prism decomposition and is NOT z-symmetrized, so a new slab
    thickness re-slices the tessellation and can expose dV values the
    Nz=4 run never saw. The 1e-2 floor was calibrated on that run; this
    checks it on the new slicing before 5.5 h of cluster time.
  - the S8a-2 coupling-band guards (span-through z flush, band invasion
    by prism/partial cells and wall-law sample points).

Everything except chord and step count comes from build_span(), so the
preflight cannot drift from the production config.

PASS = 300 steps with no NaN in f/rho, conservation drift ~1e-3 %,
|Fz|/|Fy| ~ 1e-3 (z closure), and the setup header showing "Force
measurement: Level 3" plus a per-level "surfel wall BC ... dv_min 0.001"
line for each of the 4 levels. Cost ~7 M cells, a couple of minutes.

Run (local, single GPU):
    python main.py --config configs/stl/naca0012_a10_surfel_re1e6_span_smoke.py --gpu 0
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "re1e6_span_base", os.path.join(_here, "naca0012_a10_surfel_re1e6_span.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

# frac 0.18 at c=50 == Nz 9 == the production slab thickness in L0 lu.
# (Passing the production 0.09 would give Nz=4 — the old 57/59 slab, which
#  previews nothing; see build_span's docstring.)
config = _m.build_span(
    c=50, steps=300, frac=2.0 * _m.SPAN_FRAC,
    folder="results_naca0012_a10_surfel_re1e6_span_smoke")

config["time"] = dict(config["time"], output_interval=300,
                      checkpoint_interval=10**9,
                      conservation_interval=100, logging_interval=50)
config["output"] = dict(config["output"],
                        checkpoint=dict(config["output"]["checkpoint"],
                                        enabled=False))
