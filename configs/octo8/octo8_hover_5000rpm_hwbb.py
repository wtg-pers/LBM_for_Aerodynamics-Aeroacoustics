"""[A/B 예비] Octo-8 hover 5000 RPM — 바디 벽처리 hwbb 변형.

★본 런은 `octo8_hover_5000rpm_ibb.py`(IBB, 0808 사용자 확정). 이 파일은 계단벽
효과를 대조하고 싶을 때만 쓰는 hwbb 변형으로 남겨둠 — 결과 폴더가 wall_bc로
구분되므로(result_..._hwbb vs _ibb) 충돌하지 않음.

Run:
    LBM_ESOTERIC=1 python main.py \\
        --config configs/octo8/octo8_hover_5000rpm_hwbb.py --gpu 3
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import build_config

config = build_config(rpm=5000.0, n_rev=40, vtk_deg=30.0,
                      vtk_fields_last_rev=5, wall_bc="hwbb")
