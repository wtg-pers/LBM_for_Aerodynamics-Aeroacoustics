"""TEST B — slab-L4 grid (light_slab5), Dağ straight, 15 rev.

GRID axis (testA = correction axis): domain shrunk to the old L1 box
(3.25D×2.5D², hub 0.75D from inlet), L1 nearly domain-filling, and a NEW
rotor-disk slab L4 (dx=L0/16): lat ±1.0625R, 48 fine cells thick (spans the
full ±3ε ALM force/sampling support; a 10-cell slab would truncate the
Gaussian kernel). ε/Δx at the tip 2.0 → 3.4 (0.25c takes over from the 2Δx
floor), tip chord 6.8 → 13.6 cells — light_tip5-class ALM resolution at
45.3M cells / ~18.6 GB (24 GB GPU OK; light_tip5 needed 137M/56GB).

Physics identical to testA_dag_straight (Dağ straight, inviscid target,
relax 0.5, smooth 2, NASA deck, gaussian sampling, no Prandtl) so the grid
is the only ALM-model-side variable. ⚠ BUT the domain shrink (upstream
0.75D, lateral 1.25D) shifts the hover baseline (blockage/recirculation) —
judge against testA via SPANWISE loading / w_corr / tip alpha, not absolute
CT/FM; for absolute numbers a same-domain 4-level control would be needed.

Wall clock ~3.3× light (435M vs 131M updates/coarse step).

    python main.py --config configs/0703_dag_edge_fix/testB_slab5_dag_straight.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light_slab5",
    eps_correction={"enabled": True, "method": "dag", "target": "inviscid",
                    "relax": 0.5, "smooth": 2, "wake": "straight"},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testB_slab5_dag_straight",
)
