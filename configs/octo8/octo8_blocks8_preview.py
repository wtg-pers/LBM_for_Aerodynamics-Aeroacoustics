"""Octo-8 — L0 + 로터별 L1 8블록. **토폴로지 확인용 프리뷰.**

무엇을 보려는 config인가
------------------------
지금까지의 octo8 MLG config(`octo8_hover_5000rpm_mlg2.py`)는 fine 박스가 하나뿐인
**체인**이라, 다중 블록 MPI 코드를 한 줄도 타지 않는다. 이 파일은 그 반대다 —
L0 위에 **로터마다 하나씩 L1 블록 8개**를 놓아, 리그(`configs/testrig/rig_blocks.py`)
에서 검증한 형태를 실제 기체 STL로 재현한다. ParaView에서 `.vth`를 열면 L1 블록
8개가 로터를 하나씩 감싼 모습이 보인다.

도메인은 **처음 테스트한 그대로**(uniform D/40, 324x496x200 = 32.1M). 확장하지
않았다 — 이건 배치 확인용이지 물리 결과를 얻는 런이 아니다.

    L0  D/40  = 11.43 mm   324x496x200 = 32.1 M
    L1  D/80  =  5.71 mm   113x113x81 x 8블록 = 8.3 M
    -> 40.4 M cells

블록 크기 근거 (L0 lattice units, R_rotor = 20)
-----------------------------------------------
    hub +- 26  ->  팁 밖 6 L0셀 = 12 fine셀 여유 (가이드 10 fine셀 이상)
    밴드 포함 반폭 28  ->  형제 간극  y +8.7 / x +33.5 L0셀  (전부 disjoint)
    z 는 hub +- 18 (411 mm) — 디스크는 평면이지만 근접 후류를 조금 담는다

★★ 이 config 는 빌드되지 않는다 — 기록된 음성 결과다 (0809 로컬 실측)
--------------------------------------------------------------------
    ValueError: Level 1: obstacle intersects the C2F/F2C coupling band on
    face(s): x_low (1787 solid cells), y_low (64), y_high (64).

`_check_body_vs_coupling_band`가 막는다. 밴드 안의 solid 셀은 매 coarse step마다
보간값으로 두 번 덮이므로 벽 처리가 성립하지 않는다 — 가드가 옳다.

**왜 박스를 옮겨도 안 되는지** (L0 solid mask 실측, f1_ccw 창 hub±26):

    로터 디스크 평면          z = 83.1
    이 창 안의 solid z 범위   z = 78 ~ 82   ← 암/붐, 디스크 바로 1셀 아래
    z 83 위쪽 / 77 아래쪽     solid 0

    x 분포: x=213 에 60셀(스팬 방향 붐), x=217~241 에 7셀씩(로터로 뻗는 암)

즉 **암이 디스크 바로 밑 1.1 L0셀(=12.6mm)에 붙어 있다.** 디스크를 excised에
넣으면서(ALM 배치 규칙) 밴드(2 coarse셀)를 암 위로 빼낼 z 여유가 없다:
  z_min=83 -> 밴드 81~83 이 암(z=82)을 문다
  z_min=84 -> 디스크(83.1)가 excised 밖 -> ALM 배치 가드가 거부
수평으로도 같다. 붐은 y로, 암은 x로 뻗어 어느 면을 잡아도 기체를 자른다.
기체가 x·y·z 세 방향 모두로 연결돼 있어 **이 기체를 자르지 않는 유한 블록 분할이
존재하지 않는다.** `octo8_hover_5000rpm_mlg2.py`의 "L1이 로터만이 아니라 차량
전체를 감싼다 — 선호가 아니라 강제"가 측정으로 확인된 셈이다.

남은 실낱 하나 (미확인)
-----------------------
위 z 여유 1.1 L0셀은 **L0 격자(11.43mm)로 잰 값**이다. L1(5.71mm) 마스크에서
암의 실제 상단이 z=82.0 L0 에 딱 붙어 있다면 디스크까지 2.2 L1셀이 남고, L2 블록의
밴드는 2 L1셀이므로 **아슬아슬하게 통과할 수도 있다**(차량 전체 L1 + 로터별 L2 8블록).
확정하려면 L1 fine mask 에서 암 상단 z 를 직접 재야 한다 — L0 해상도로는 판정 불가.

그래서 이 파일은 그대로 둔다: 로터별 블록이 왜 이 기체에 안 맞는지를 숫자와 함께
남긴 기록이고, 다음에 누군가 같은 배치를 다시 시도하기 전에 읽을 문서다.
빌드 단계에서 즉시 실패하므로 GPU 시간은 들지 않는다.

Run (단일 GPU — 배치만 보면 되므로 몇 스텝이면 충분):
    LBM_ESOTERIC=1 python main.py \\
        --config configs/octo8/octo8_blocks8_preview.py --gpu 0 --max-steps 4

Run (다중 블록 MPI 경로까지 함께 확인):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \\
        --config configs/octo8/octo8_blocks8_preview.py \\
        --gpu 0,1,2,3 --cuda-aware 1 --max-steps 4 --log-every 2
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import (  # noqa: E402
    _LAYOUT, ROTOR_Z_MM, build_config, grid_map)

config = build_config(rpm=5000.0, n_rev=40, vtk_deg=30.0,
                      vtk_fields_last_rev=40, wall_bc="ibb")

# 로터 허브를 L0 lattice 좌표로 (config 의 region 박스는 전 레벨 L0 좌표계다 —
# src/solver/setup.py:1518 "Config values are in L0 physical coordinates")
_dx, _mm2lu, _O, _N = grid_map()             # 처음 테스트한 도메인 그대로
_RXY = 26                                    # 수평 반폭 [L0 cells]
_RZ = 18                                     # 수직 반폭 [L0 cells]

_regions = []
for _name, _x_mm, _y_mm, _hand in _LAYOUT:
    _h = _O + np.array([_x_mm, _y_mm, ROTOR_Z_MM]) * _mm2lu
    _regions.append({
        "name": _name,
        "x_min": float(round(_h[0] - _RXY)), "x_max": float(round(_h[0] + _RXY)),
        "y_min": float(round(_h[1] - _RXY)), "y_max": float(round(_h[1] + _RXY)),
        "z_min": float(round(_h[2] - _RZ)),  "z_max": float(round(_h[2] + _RZ)),
    })

config["mlg"] = {
    "enabled": True, "num_levels": 2, "overlap_width": 2,
    "interpolation": "cubic", "filter_level": 1,
    "levels": [{}, {"regions": _regions}],
}

# 프리뷰: step 0 부터 full-field VTK, 체크포인트 없음
config["output"]["output_dir"] = "./result_octo8_blocks8/vtk"
config["output"]["checkpoint_dir"] = "./result_octo8_blocks8/checkpoints"
config["output"]["csv_dir"] = "./result_octo8_blocks8/csv"
config["output"]["checkpoint"]["enabled"] = False
config["output"]["vtk"]["fields_start_step"] = 0
config["time"]["output_interval"] = 1


if __name__ == "__main__":
    n0 = _N[0] * _N[1] * _N[2]
    print(f"  L0 {_N[0]}x{_N[1]}x{_N[2]} = {n0/1e6:.2f} M   dx={_dx*1000:.2f} mm")
    nf = (2 * (2 * (_RXY + 2)) + 1) ** 2 * (2 * (2 * (_RZ + 2)) + 1)
    print(f"  L1 8 blocks, {nf/1e6:.3f} M each = {8*nf/1e6:.2f} M")
    print(f"  total {(n0 + 8*nf)/1e6:.2f} M cells")
    for r in _regions:
        print(f"    {r['name']:8s} x[{r['x_min']:.0f},{r['x_max']:.0f}] "
              f"y[{r['y_min']:.0f},{r['y_max']:.0f}] "
              f"z[{r['z_min']:.0f},{r['z_max']:.0f}]")
