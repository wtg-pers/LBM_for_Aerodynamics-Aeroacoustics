"""bench5 회귀 — canonical-axis Step 1: free-wake 경로 bit-identical 확인.

Step 1이 free-wake 투영축을 `−thrust_axis / −sign(ω)·rot` fallback → `rotor.axial_inflow_dir`
단일소스로 치환. HVAB(thrust=[−1,0,0])에선 axial_inflow_dir=[1,0,0]=−thrust_axis라 **bit-identical**
이어야 함. 이 config가 그 경로(_kleine_w_corr free-wake)를 exercise.

grid=bench5(D16 5-level 슬랩, ~9M셀, 2rev≈수분/1×4090). physics=testB_slab5_kleine_free와
동일(kleine free, rebuild_every=1, all edges, smooth=2, NASA덱, gaussian, prandtl OFF, uniform).

    python main.py --config configs/0708_axis_regression/bench5_kleine_free.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=3, polar_source="nasa_overflow", run_tag="axisreg_kleine_free",
)
