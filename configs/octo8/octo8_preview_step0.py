"""Octo-8 프리뷰 — ParaView 지오메트리/회전방향 확인용 (step 0 + 105 두 프레임).

full-field를 step 0부터 출력(vtk_fields_last_rev=n_rev → fields_start_step=0).
step 105 = 30.07° 회전 후 → 마커 VTP 두 프레임으로 각 로터 회전방향 검증.

Run:
    LBM_ESOTERIC=1 python main.py \\
        --config configs/octo8/octo8_preview_step0.py --gpu 0 --max-steps 106
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import build_config

config = build_config(rpm=5000.0, n_rev=40, vtk_deg=30.0,
                      vtk_fields_last_rev=40, wall_bc="ibb")
config["output"]["output_dir"] = "./result_octo8_preview/vtk"
config["output"]["checkpoint_dir"] = "./result_octo8_preview/checkpoints"
config["output"]["csv_dir"] = "./result_octo8_preview/csv"
config["output"]["checkpoint"]["enabled"] = False
