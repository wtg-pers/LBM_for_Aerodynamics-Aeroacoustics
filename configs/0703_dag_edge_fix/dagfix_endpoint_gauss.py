"""Dağ edge-fix re-run: endpoint markers + gauss + FIXED viscous-core correction.

_viscous_core_correction was reimplemented edge-based (Dağ&Sørensen 2020 Eq.17-18):
trailed vortices on the N+1 panel EDGES incl. the TIP vortex (Γw=−Γ_tip) and root,
+1/(4π).  The old marker dΓ/dr code dropped the end vortices → tip correction was
~10-30× too weak (Dağ did nothing).  See patch_notes/alm_dag_edge_fix/.

Correction = the STANDARD relaxed scheme (Martínez-Tossas 2019 / Dağ iterative,
one update per step): w = relax·w_new + (1−relax)·w_prev.  The fixed point is the
FULL self-consistent Dağ correction REGARDLESS of relax — relax only sets the
convergence rate/stability (Kleine: relax→1 may go unstable).  This is the faithful
baseline; the earlier `relax·w_new` alone converged to relax·full (under-applied).
relax=0.5 is a starting point; lower it if the tip loading oscillates.

    python main.py --config configs/0703_dag_edge_fix/dagfix_endpoint_gauss.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction={"enabled": True, "method": "dag",
                    "target": "inviscid", "relax": 0.5},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="endpoint",
    n_rev=25, polar_source="nasa_overflow", run_tag="dagfix_endpoint_gauss",
)
