"""APC 18x8E hover @ 2446 RPM — ALM mlg4, tip-loss OFF.

HVAB 파이널 런 포뮬레이션 이식(빌더 참조). UIUC static 대조점(프로펠러 conv.,
apce_18x8_static_2184od.txt) 있음. GPU 2 = notl 시리즈.

Run (메인 디렉토리 기준):
    LBM_ESOTERIC=1 python main.py --mpi \\
        --config configs/apc18x8e/apc18x8e_hover_2446rpm_notl.py \\
        --gpu 2 --steps 31425 --log-every 64 --vtk-every 35 \\
        --vtk-fields-last 180 --ckpt-every 31425 --csv apc18x8e_2446_notl.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _apc18x8e_hover_base import build_config

config = build_config(rpm=2446, tip_loss="off")
