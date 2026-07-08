"""Pure-ALM diagnostic — ALM + LBM with NO smearing correction (eps_correction off).

Runs alongside diag_pure_lbm.py (ALM disabled). Two purposes:

  1. GPU util — isolates the ALM BASE cost (velocity sample → BEM/polar → force
     project each step) WITHOUT the heavy straight/free-wake correction. Ladder:
        diag_pure_lbm  (no ALM)          → LBM kernel ceiling
        diag_pure_alm  (ALM, no corr)    → + ALM base CPU serialisation
        testA_dag_*    (ALM + correction)→ + correction CPU (pruned freewake)
     Comparing the three util levels locates exactly where the GPU stall comes from.

  2. Sawtooth check — pure ALM has NO correction feedback (the sawtooth is the
     edge-kernel w ∝ −Γ″ instability, which lives in the CORRECTION). So the loading
     MUST be a smooth spanwise line (marker roughness ~0.05), NOT the high-low-high
     sawtooth (~1.6) seen in the uncorrected-but-corrected straight run. This
     confirms the sawtooth originates in the correction, not the ALM itself, and is
     the smooth CT baseline (~0.0104, +32% — the tip over-loading with no correction).

Same grid/MLG/domain/BC as testA_dag_* (preset="light", D3Q27 cumulant, dyn_smag),
only eps_correction is absent. n_rev=3 (diagnostic; extend for a converged CT/FM
baseline). Watch nvidia-smi; confirm spanwise alpha/F_n is a smooth line.

    python main.py --config configs/0703_dag_edge_fix/diag_pure_alm.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=3, polar_source="nasa_overflow", run_tag="diag_pure_alm",
)
# NOTE: eps_correction intentionally omitted → pure ALM (no Dağ/Kleine, no
# smoothing) → no correction feedback → no sawtooth. ALM itself stays enabled.
