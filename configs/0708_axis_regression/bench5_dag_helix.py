"""bench5 회귀 — canonical-axis Step 1: prescribed-helix 경로 bit-identical 확인.

Step 1이 helix 하강/투영축(`downwash=descent=proj`)을 `−thrust_axis / +sign(ω)·rot` fallback →
`rotor.axial_inflow_dir` 단일소스로 치환. HVAB에선 bit-identical이어야 함. 이 config가 그 경로
(_dag_prescribed_helix_wcorr)를 exercise.

grid=bench5(D16 5-level 슬랩). physics=dag prescribed_helix(relax 0.5, smooth 2, inviscid,
NASA덱, gaussian, prandtl OFF, uniform).

    python main.py --config configs/0708_axis_regression/bench5_dag_helix.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction={"enabled": True, "method": "dag", "target": "inviscid",
                    "relax": 0.5, "smooth": 2, "wake": "prescribed_helix"},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=3, polar_source="nasa_overflow", run_tag="axisreg_dag_helix",
)
