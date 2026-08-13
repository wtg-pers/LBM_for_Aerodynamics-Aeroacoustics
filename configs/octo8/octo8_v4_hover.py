"""Octo-8 v4 — v3 격자·실험조건 그대로, 회전방향 반전 + 출력 3종 변경.

v3 대비 바뀐 것은 정확히 셋이다:

  1) **회전방향 전부 반전** (flip_handedness=True). 8기 각각 CW<->CCW.
     체커보드 패턴 자체는 유지되고 위상만 뒤집힌다 —
         전방열 왼(-y)→오(+y): CCW,CW,CCW,CW  ->  **CW,CCW,CW,CCW**
         후방열 왼(-y)→오(+y): CW,CCW,CW,CCW  ->  **CCW,CW,CCW,CW**
     로터 이름 접미사도 실제 방향을 따라간다(f1_ccw -> f1_cw). 위치·rpm
     크기·추력 방향(+z)은 불변 — rpm 부호만 8기 모두 뒤집힌다.

  2) **출력 단위 physical** (output.units="phys" 명시). VTK/plane 의
     p -> p' [Pa], u -> [m/s] (전 레벨 단일 상수; 좌표는 L0-lu 유지).
     v3 은 units 키가 없어 당시 코드(0811)로는 lu 로 출력됐다 — 같은 결과
     시리즈에 단위가 섞이지 않도록 v4 는 명시 고정한다(restart 단위 전환은
     하드에러, a2f6abe).

  3) **section plane 6장, 매 L0 스텝 출력** (output.planes, interval=1).
     full-field VTK 는 v3 그대로 30° 간격(52 steps) 전 구간을 유지하고,
     plane 채널이 조밀한 시간축을 담당한다. 전 AMR 계층이 잘리며(레벨별
     .vti + 스텝별 .vtm + plane 당 .pvd 1개), 위치는 최코스 교차 레벨
     격자에 스냅되어 전 레벨 공면이다.

     좌표(CAD mm; 노즈=+x, 왼쪽 날개=+y):
         x_front_rotors  x = -302.24   전방 로터열 허브 중심
         x_wing_center   x = -837.55   날개 중심 (STL 실측: |y|>1400 mm
                                       밴드는 순수 날개, x[-1081.2,-593.9]
                                       의 중점. 로터열 중점 -813.7 아님)
         x_aft_rotors    x = -1325.24  후방 로터열 허브 중심
         y_outer_rotors  y = +1288.5   +y 최외곽(왼쪽 날개 끝단) 로터 허브
         y_inner_rotors  y = +548.5    +y 내곽 로터 허브
         y_fuselage      y = 0.0       동체 중심 종단면
     fields=["p","u"] — 용량은 __main__ 리포트가 계층 실측으로 찍는다.
     줄이려면 fields 를 ["p"] 로(1/4), 또는 interval 을 키운다.

나머지(4000 rpm, 호버 450 mm, 4레벨 격자, 14 m 원방, neumann 측면,
100 rev, n_radial=48, theta0=pi/2, wall_bc=ibb, 30° full-field 전 구간)는
v3 과 같고, 격자 구성은 `_octo8_hover_base.build_mlg_4level` 을 공유한다.
flip_handedness 기본값(False) 경로는 v3 config dict 와 비트 동일함을
확인했다(sha256 비교, 0813).

★★ 실행 노트 (0813 갱신)
------------------------
output.planes 는 **MPI 소유랭크 스트립으로 지원된다**(0813 구현,
patch_notes/acoustic_probes/03): 각 랭크가 자기 슬랩 조각 .vti 를 직접
쓰고 rank0 이 .vtm/.pvd 를 인덱스한다(통신 0). 분해축과 평행한 plane
은 랭크별 s<start> 조각으로 나뉘고 조각 사이 1노드 표시 심이 남는다
(데이터 무손실 — 조립 비트 동일, 게이트 PASS). solid 셀은 rest state
(볼륨 규약 동일). probes 도 MPI 소유랭크 샘플링 지원(0813 후속) —
flush 때 rank0 이 csv 를 조립한다(단일GPU 와 비트 동일).
클러스터 4랭크 실측은 아직 안 했다 — 로컬 2랭크 패리티/restart 만 실증.

Run (cluster, 4x24GB):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \\
        --config configs/octo8/octo8_v4_hover.py --gpu 0,1,2,3 \\
        --cuda-aware 1 --dist-init

Restart:
    ... --dist-init --restart-latest
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import (  # noqa: E402
    build_config, build_mlg_4level, mlg_report)

RPM = 4000.0                      # = v3
HOVER_H_MM = 450.0                # = v3 (지면 -> 로터면)

D_LU_0 = 20                       # L0: rotor diameter in cells
HALF_XY_MM = 7000.0               # L0 half-extent about the vehicle centre
L1_HALF_MM = 5000.0               # ground-resolved outwash radius
N_RADIAL = 48                     # delta_r 3.7 mm <= eps floor 5.71 mm @D/160
N_REV = 100

# full-field VTK 는 v3 그대로 전 구간 30°(~7 TB). plane 채널이 여기에
# 얹히므로 KISTI /scratch(100 TB) 기준으로 잡을 것 — /home01(64 GB) 불가.
VTK_FIELDS_REV = N_REV       # 전 구간

config = build_config(rpm=RPM, n_rev=N_REV, n_radial=N_RADIAL,
                      vtk_deg=30.0, vtk_fields_last_rev=VTK_FIELDS_REV,
                      wall_bc="ibb",
                      d_lu=D_LU_0, half_xy_mm=HALF_XY_MM, side_bc="neumann",
                      hover_h_mm=HOVER_H_MM,
                      theta0=np.pi / 2,       # 8기 y축 평행 정렬 (= v3)
                      flip_handedness=True)   # ★ v4: 8기 전부 CW<->CCW

_info = build_mlg_4level(config, d_lu0=D_LU_0, half_xy_mm=HALF_XY_MM,
                         l1_half_mm=L1_HALF_MM, hover_h_mm=HOVER_H_MM,
                         overlap_width=2, pad2=2.0, l1_zmin=2.0,
                         wall_coupling_mode="allow",   # v3 과 동일 근거
                         flip_handedness=True)         # 블록 이름 일치

# ── ★ v4: 출력 단위 physical (p'[Pa], u[m/s]) ──
config["output"]["units"] = "phys"

# ── ★ v4: section plane 6장, 매 L0 스텝 ──
# 위치는 로터 허브와 같은 단일소스 매핑(CAD mm -> 전역 L0 lu)으로 유도:
# pos_lu = origin[ax] + mm * mm2lu (build_mlg_4level 이 돌려준 값).
WING_CENTER_X_MM = -837.55        # STL 실측 (헤더 참조)
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
     "units": "lu",              # 전역 L0-lu (위에서 CAD mm 로부터 유도)
     "level": "all",             # 전 AMR 계층
     "fields": ["p", "u"],
     "interval": 1}              # 매 L0 스텝
    for name, ax, mm in _PLANES_MM
]

_tag = "result_octo8_v4"
config["output"]["output_dir"] = "./%s/vtk" % _tag
config["output"]["checkpoint_dir"] = "./%s/checkpoints" % _tag
config["output"]["csv_dir"] = "./%s/csv" % _tag


def _plane_report():
    """plane 별 교차 계층 실측 -> 셀수/용량. (fields=p,u = 4 float32/셀)"""
    n = _info["shape"]
    lv_boxes = {0: [([0.0, 0.0, 0.0], [n[0] - 1.0, n[1] - 1.0, n[2] - 1.0])],
                1: [tuple(_info["l1"])], 2: [tuple(_info["l2"])],
                3: [([b["x_min"], b["y_min"], b["z_min"]],
                     [b["x_max"], b["y_max"], b["z_max"]])
                    for b in _info["rotor_boxes"]]}
    steps = config["time"]["max_steps"]
    total = 0.0
    for pc in config["output"]["planes"]:
        ax, pos = _AX[pc["normal"]], pc["position"]
        cells = 0
        for lv, boxes in lv_boxes.items():
            for lo, hi in boxes:
                if lo[ax] - 1e-9 <= pos <= hi[ax] + 1e-9:
                    dims = [int(round((hi[i] - lo[i]) * 2 ** lv)) + 1
                            for i in range(3) if i != ax]
                    cells += dims[0] * dims[1]
        gb = cells * 4 * 4 * steps / 1e9        # p+u = 4 x float32
        total += gb
        print(f"    {pc['name']:15s} {pc['normal']}={pos:7.2f} lu  "
              f"~{cells / 1e3:6.1f} k cells/step  ~{gb:6.1f} GB 총")
    print(f"    planes 합계 ~{total / 1e3:.2f} TB "
          f"({steps:,} steps, fields=p+u; p 만이면 1/4)")


if __name__ == "__main__":
    mlg_report(config, _info, D_LU_0, L1_HALF_MM, N_REV, tag="octo8 v4")
    _n_out = config['time']['max_steps'] // config['time']['output_interval']
    _n_fld = int(VTK_FIELDS_REV * 628 / config['time']['output_interval'])
    print(f"  steps {config['time']['max_steps']:,} ({N_REV} rev) | "
          f"VTK every {config['time']['output_interval']} | "
          f"ckpt every {config['time']['checkpoint_interval']:,}")
    print(f"  VTK: {_n_out} 출력 중 full-field {min(_n_fld, _n_out)}개 "
          f"~ {min(_n_fld, _n_out) * _info['total_cells'] * 32 / 1e12:.1f} TB")
    print(f"  units: {config['output']['units']} (p'[Pa], u[m/s])")
    print("  rotors (v3 에서 전부 반전):")
    for e in config["actuator_line"]["rotors"]:
        r = e["rotor"]
        print(f"    {e['name']:7s} rpm={r['rpm']:+7.0f}  "
              f"hub_lu=({r['hub_center'][0]:6.2f},{r['hub_center'][1]:6.2f},"
              f"{r['hub_center'][2]:6.2f})")
    print("  planes (매 L0 스텝, 전 계층):")
    _plane_report()
