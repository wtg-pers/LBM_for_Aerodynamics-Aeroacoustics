"""TEST B production — slab-L4 grid (light_slab5), PURE ALM, 15 rev.

Model of record for the GRID-axis experiment (2026-07-05 decision): NO smearing
correction at all. testA closed the correction axis (Dag/Kleine cannot fix the
tip overload and Dag adds numerical artifacts — mid-span w'' oscillation, root
spike, helix pitch floor), so the cleanest resolution test is raw ALM: any tip
de-load seen here comes from the RESOLVED induction of the finer L4 slab alone,
with zero model confounding.

Grid: preset light_slab5 (see _hvab_hover_base) — 45.3M cells / ~18.6 GB,
tip chord 6.8 -> 13.6 cells, tip eps/dx 2.0 -> 3.4. Smoke passed 2026-07-05.

A/B companions (same physics, light grid):
  - pureALM_nasa 25rev  (260630_results_nasa_c81/pureALM_csv)  <- primary
  - diag_pure_alm 3rev  (260703_dag_edge_fix)                  <- same-folder ref
Judge on outer-band (r/R 0.85-1.0) spanwise loading / tip alpha / tip u_n,
NOT absolute CT/FM (domain shrink shifts the hover baseline).

    python main.py --config configs/0703_dag_edge_fix/testB_slab5_pure_alm.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light_slab5",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testB_slab5_pure_alm",
)
# eps_correction intentionally omitted -> pure ALM (no Dag/Kleine, no smoothing).
