"""Case 1 baseline: uniform(셀중심) + gauss. == 기존 hvab_hover_c10_pureALM_nasa.
결과 보유 중이면 재실행 불요(이 파일은 매트릭스 완성/재현용).
    python main.py --config configs/0630_markers_test/md_uniform_gauss.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=25, polar_source="nasa_overflow", run_tag="md_uniform_gauss",
)
