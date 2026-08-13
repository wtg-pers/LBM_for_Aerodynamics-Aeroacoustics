"""octo8 v3 축소 스모크 — 구조 보존, 해상도 절반.

본 런(octo8_v3_hover)과 같은 조합 — 4레벨 MLG, 로터별 L3 블록 8기,
ground outwash L1(l1_zmin=2.0 = overlap_width → 밴드-온-월 배치),
hover 높이만 2배(dx 2배에 맞춰 지면 여유를 lu 로 보존),
STL 기체 IBB, neuralfoil 다중 익형(e63+naca4412), eso — 을 그대로 두고
d_lu 만 20→10 으로 줄인 빌드/관통 확인용이다. 총 셀 ~1/8 (~22 M).
**물리 판독 금지**: c_tip/dx~1.2 라 하중은 무의미하다.

용법 (2랭크, GPU 2,3 예):
    LBM_ESOTERIC=1 mpirun -n 2 python main.py \\
        --config configs/octo8/octo8_v3_smoke.py \\
        --gpu 2,3 --cuda-aware 1 --dist-init \\
        --max-steps 2 --no-vtk --results-dir result_legacy_smoke/octo8
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import (  # noqa: E402
    build_config, build_mlg_4level, mlg_report)

RPM = 4000.0
HOVER_H_MM = 900.0            # v3=450. dx 가 2배라 지면 여유를
                              # lu 로 보존(밴드-온-월 적층 유지)

D_LU_0 = 10                       # v3 본 런 = 20. 여기만 다르다.
HALF_XY_MM = 7000.0               # 물리 범위는 v3 그대로 (블록 구조 보존)
L1_HALF_MM = 5000.0
N_RADIAL = 48                     # delta_r 는 물리 고정 — eps 바닥이 2배로
                                  # 커지므로 제약은 오히려 여유
N_REV = 1

config = build_config(rpm=RPM, n_rev=N_REV, n_radial=N_RADIAL,
                      vtk_deg=30.0, vtk_fields_last_rev=N_REV,
                      wall_bc="ibb",
                      d_lu=D_LU_0, half_xy_mm=HALF_XY_MM, side_bc="neumann",
                      hover_h_mm=HOVER_H_MM,
                      theta0=np.pi / 2)

_info = build_mlg_4level(config, d_lu0=D_LU_0, half_xy_mm=HALF_XY_MM,
                         l1_half_mm=L1_HALF_MM, hover_h_mm=HOVER_H_MM,
                         overlap_width=2, pad2=2.0, l1_zmin=2.0,
                         wall_coupling_mode="allow")

_tag = "result_octo8_v3_smoke"
config["output"]["output_dir"] = "./%s/vtk" % _tag
config["output"]["checkpoint_dir"] = "./%s/checkpoints" % _tag
config["output"]["csv_dir"] = "./%s/csv" % _tag


if __name__ == "__main__":
    mlg_report(config, _info, D_LU_0, L1_HALF_MM, N_REV, tag="octo8 v3 smoke")
