"""octo8 v4 축소 스모크 — 구조 보존, 해상도 절반 (v3_smoke 패턴).

v4 의 조합 그대로 — ★8기 방향반전(flip_handedness=True, 이름까지 반전)
★output.units="phys" ★planes 6장 매 L0 스텝(전 AMR 계층) — 에 지면
implicit wall(eso_wall §4 완결)까지 얹어 로컬 2랭크로 관통 확인하는
용도다. v3_smoke 와 같은 축소(d_lu 20→10, hover 2배로 지면 여유 lu
보존, 총 셀 ~1/8). **물리 판독 금지**: c_tip/dx~1.2.

용법 (로컬 2랭크):
    LBM_ESOTERIC=1 mpirun -n 2 python main.py \\
        --config configs/octo8/octo8_v4_smoke.py \\
        --gpu 0 --dist-init --max-steps 4 --no-vtk \\
        --results-dir /tmp/octo8_v4_smoke
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import (  # noqa: E402
    build_config, build_mlg_4level, mlg_report)

RPM = 4000.0
HOVER_H_MM = 900.0            # v4=450. dx 2배 -> 지면 여유를 lu 로 보존
D_LU_0 = 10                   # v4 본 런 = 20. 여기만 다르다.
HALF_XY_MM = 7000.0
L1_HALF_MM = 5000.0
N_RADIAL = 48
N_REV = 1

config = build_config(rpm=RPM, n_rev=N_REV, n_radial=N_RADIAL,
                      vtk_deg=30.0, vtk_fields_last_rev=N_REV,
                      wall_bc="ibb",
                      d_lu=D_LU_0, half_xy_mm=HALF_XY_MM, side_bc="neumann",
                      hover_h_mm=HOVER_H_MM,
                      theta0=np.pi / 2,
                      flip_handedness=True)       # ★ v4 그대로

_info = build_mlg_4level(config, d_lu0=D_LU_0, half_xy_mm=HALF_XY_MM,
                         l1_half_mm=L1_HALF_MM, hover_h_mm=HOVER_H_MM,
                         overlap_width=2, pad2=2.0, l1_zmin=2.0,
                         wall_coupling_mode="allow",
                         flip_handedness=True)    # ★ 블록 이름 일치

config["output"]["units"] = "phys"                # ★ v4 그대로

# ★ v4 의 planes 6장 그대로 (CAD mm -> L0 lu 는 _info 매핑으로 유도라
# 해상도 축소에 자동 추종; 위치는 최코스 교차 레벨 격자에 스냅됨)
WING_CENTER_X_MM = -837.55
_PLANES_MM = [
    ("x_front_rotors", "x", -302.24),
    ("x_wing_center",  "x", WING_CENTER_X_MM),
    ("x_aft_rotors",   "x", -1325.24),
    ("y_outer_rotors", "y", 1288.5),
    ("y_inner_rotors", "y", 548.5),
    ("y_fuselage",     "y", 0.0),
]
_AX = {"x": 0, "y": 1}
config["output"]["planes"] = [
    {"name": name, "normal": ax,
     "position": float(_info["origin"][_AX[ax]] + mm * _info["mm2lu"]),
     "units": "lu", "level": "all",
     "fields": ["p", "u"], "interval": 1}
    for name, ax, mm in _PLANES_MM
]

_tag = "result_octo8_v4_smoke"
config["output"]["output_dir"] = "./%s/vtk" % _tag
config["output"]["checkpoint_dir"] = "./%s/checkpoints" % _tag
config["output"]["csv_dir"] = "./%s/csv" % _tag

if __name__ == "__main__":
    mlg_report(config, _info)
