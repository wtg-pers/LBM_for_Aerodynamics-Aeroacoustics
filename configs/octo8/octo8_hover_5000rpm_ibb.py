"""Octo-8 vehicle hover @ 5000 RPM — 8x APC 18x8E ALM + STL body(IBB), 40 rev.

uniform D40 + 지면 IGE z/D=2.09(32.1M cells, eso ~6.7GB), 단일 GPU 전용(멀티로터=MPI 미지원).
VTK: 마커 30.07°마다 전 구간, full-field는 마지막 5바퀴만.

Run (메인 디렉토리 기준, 단일 GPU — --mpi 금지):
    LBM_ESOTERIC=1 python main.py \\
        --config configs/octo8/octo8_hover_5000rpm_ibb.py --gpu 2
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import build_config

config = build_config(rpm=5000.0, n_rev=40, vtk_deg=30.0,
                      vtk_fields_last_rev=5, wall_bc="ibb")
