"""Octo-8 v5 — v4 인프라 그대로, 회전방향만 v3 원 방향으로 복귀.

v4 대비 바뀐 것은 정확히 하나다: **flip_handedness=False** (v3 원 방향).

    전방열 왼(-y)→오(+y): CCW,CW,CCW,CW   (로터명 f1_ccw..f4_cw)
    후방열 왼(-y)→오(+y): CW,CCW,CW,CCW   (a1_cw..a4_ccw)

즉 v5 = "v3 의 개선판": v3 의 회전방향 위에 v4 에서 확립된 인프라 전부 —
  * 버티포트 도메인 (L0=SA 2.6CD 438², L1=FATO 2PD 지면 flush,
    L2 l2_zmin 적층; v3 의 616²/10m L1 아님 — 113.34M 셀)
  * 진짜 지면 벽 (eso_wall §4: 전 경로 halfway, L1 flush q=1 face
    -> 반사 평면 전역 -0.5dx0 정확, tau 불변. v3 은 EQ(u=0) 무음 강등)
  * output.units="phys" (p'[Pa], u[m/s])
  * section plane 6장 매 L0 스텝 (+.vth AMR 합성 뷰, <plane>.vth.series)
  * 지면 마이크 probe 22점 (TLOF/FATO/외곽 링, z=126mm, flush 200)
— 를 얹은 구성이다. base 의 flip_handedness=False 경로는 v3 config dict
와 비트 동일 확인(sha256, 0813) — 로터 배선은 문자 그대로 v3 이다.

★시리즈 가드: v5 는 새 결과 시리즈(result_octo8_v5)다. v3 체크포인트로
재시작 불가(wall_mail 키 부재 하드에러 — 벽 의미론이 다름), v4 와도
회전방향이 달라 물리적으로 섞을 수 없다.

실행·MPI·plane/probe 상세 노트는 v4 헤더(octo8_v4_hover.py) 참조 —
인프라가 동일하므로 그대로 적용된다.

Run (cluster, 4x24GB):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \\
        --config configs/octo8/octo8_v5_hover.py --gpu 0,1,2,3 \\
        --cuda-aware 1 --dist-init

Restart:
    ... --dist-init --restart-latest
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import (  # noqa: E402
    build_config, build_mlg_4level, mlg_report,
    ground_cad_mm, _BBOX_CTR_MM)

RPM = 4000.0                      # = v3
HOVER_H_MM = 450.0                # = v3 (지면 -> 로터면)

# ── 버티포트 구역 -> 격자 계층 (0814, 설계자 치수) ───────────────────
# CD(기준직경, 기체 최외각 원) = 3822 mm  (STL 실측 3826, 0.1% 일치)
# PD(추력직경)                = 2622.72 mm
#   L0 = SA   전폭 2.6 CD     L1 = FATO 전폭 2 PD (지면 flush 밴드)
#   L2 = 기체+로터 컨테인먼트를 지면까지 내림(l2_zmin, 밴드-온-월 적층)
#   TLOF 전폭 1 PD 는 격자가 아니라 지면 개념 구역(probe 링 기준):
#   L2 가 기체·로터 L3 를 담아야 하므로 1 PD 로는 물리적으로 축소 불가
#   (날개 팁 y +-1911, 로터 블록 y +-1616 > 1PD half 1311).
CD_MM = 3822.0
PD_MM = 2622.72
D_LU_0 = 20                       # L0: rotor diameter in cells
HALF_XY_MM = 2.6 * CD_MM / 2      # SA half = 4968.6 (구 7000)
L1_HALF_MM = 2.0 * PD_MM / 2      # FATO half = 2622.72 (구 5000)
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
                      flip_handedness=False)  # ★ v5: v3 원 방향 복귀

_info = build_mlg_4level(config, d_lu0=D_LU_0, half_xy_mm=HALF_XY_MM,
                         l1_half_mm=L1_HALF_MM, hover_h_mm=HOVER_H_MM,
                         overlap_width=2, pad2=2.0,
                         l1_zmin=0.0,   # ★ L1 지면 flush (벽 승계+q=1 face)
                         l2_zmin=1.0,   # ★ L2 밴드 바닥 = 지면 위 0.5 L0셀(11.4mm)
                         wall_coupling_mode="allow",   # v3 과 동일 근거
                         flip_handedness=False)        # 블록 이름 일치

# ── 출력 단위 physical (v4 승계) (p'[Pa], u[m/s]) ──
config["output"]["units"] = "phys"

# ── section plane 6장, 매 L0 스텝 (v4 승계) ──
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

# ── 지면 마이크 probe 22점 (v4 승계, 0814 사용자 사양, 단위 mm) ──
# 패드 중심(=도메인 중심=기체 bbox 중심) 기준 수평좌표, z=지상고 126 mm.
# 링 구조: TLOF 경계(|r|=1310=0.5PD, 6점) / FATO 경계(2620=1PD, 8점) /
# 외곽(4530, 8점), x 오프셋 {0, +-918}. 원 사양의 6번째 점
# (-918, +4530)은 2번째와 중복 — (-918, -4530) 오기로 정정,
# ★0814 사용자 확정. 패드 중심 = 기체 bbox 중심도 사용자 확정.
_PROBES_MM = [
    (-4530,     0), (-918,  4530), (0,  4530), (+918,  4530),
    ( 4530,     0), (-918, -4530), (0, -4530), (+918, -4530),
    ( -918,  2620), (0,  2620), (+918,  2620), ( 2620,     0),
    ( -918, -2620), (0, -2620), (+918, -2620), (-2620,     0),
    ( -918,  1310), (0,  1310), (+918,  1310),
    ( +918, -1310), (0, -1310), (-918, -1310),
]
_PROBE_Z_MM = 126.0

def _probe_lu(x_mm, y_mm):
    """패드 좌표(mm) -> 전역 L0 lu (planes 와 같은 단일소스 매핑).
    z 는 지면(GROUND_CAD) 기준 높이 — 벽면(노드0 -0.5)과 정합."""
    cad = (x_mm + _BBOX_CTR_MM[0], y_mm,
           ground_cad_mm(HOVER_H_MM) + _PROBE_Z_MM)
    return [float(_info["origin"][d] + cad[d] * _info["mm2lu"])
            for d in range(3)]

config["output"]["probes"] = {
    "points": [_probe_lu(x, y) for x, y in _PROBES_MM],
    "units": "lu",
    "interval": 1,            # 매 L0 스텝 (음향 시계열)
    # flush 전까지 csv 는 헤더만 보인다(정상). 0814 실측: flush 도달
    # 순간 일괄 기록. 근실시간 모니터링을 위해 200 (비용 무시 가능).
    "flush_every": 200,
}

_tag = "result_octo8_v5"
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
    mlg_report(config, _info, D_LU_0, L1_HALF_MM, N_REV, tag="octo8 v5")
    _n_out = config['time']['max_steps'] // config['time']['output_interval']
    _n_fld = int(VTK_FIELDS_REV * 628 / config['time']['output_interval'])
    print(f"  steps {config['time']['max_steps']:,} ({N_REV} rev) | "
          f"VTK every {config['time']['output_interval']} | "
          f"ckpt every {config['time']['checkpoint_interval']:,}")
    print(f"  VTK: {_n_out} 출력 중 full-field {min(_n_fld, _n_out)}개 "
          f"~ {min(_n_fld, _n_out) * _info['total_cells'] * 32 / 1e12:.1f} TB")
    print(f"  units: {config['output']['units']} (p'[Pa], u[m/s])")
    print("  rotors (v3 원 방향):")
    for e in config["actuator_line"]["rotors"]:
        r = e["rotor"]
        print(f"    {e['name']:7s} rpm={r['rpm']:+7.0f}  "
              f"hub_lu=({r['hub_center'][0]:6.2f},{r['hub_center'][1]:6.2f},"
              f"{r['hub_center'][2]:6.2f})")
    print("  planes (매 L0 스텝, 전 계층):")
    _plane_report()
