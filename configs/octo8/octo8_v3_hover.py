"""Octo-8 v3 — v2 의 격자 구성 그대로, 실험 조건 두 개만 바뀐 본 런.

v2 대비 바뀐 것은 정확히 둘이다:

    회전수     5000 rpm -> **4000 rpm**
    호버 높이  지면->로터면 955.5 mm (2.09 D) -> **450 mm (0.98 D)**

나머지(4레벨 구조, 14 m 원방, 지면까지 D/40, neumann 측면, 100 rev,
n_radial=48, theta0=pi/2, wall_bc=ibb)는 v2 와 같은 값이고, 격자 구성 자체는
`_octo8_hover_base.build_mlg_4level` 을 v2 와 **공유**한다. v2 config 는 이
공유 전후로 완전 동일함을 확인했다(config dict 전체 비교).

    L0 D/20  far field
    L1 D/40  ground outwash (+-5 m, 바닥은 지면)
    L2 D/80  airframe
    L3 D/160 로터 8기, 블록 하나씩

두 조건이 무엇을 바꾸는가
-------------------------
**rpm 4000.** 격자·스텝 수는 안 바뀐다. `u_max` 가 0.1 lu 로 고정이고
`STEPS_REV = 2*pi*R_lu / u_max` 라 팁이 항상 0.1 lu/step 로 돈다 — 즉 회전당
스텝 수는 rpm 과 무관하고, rpm 은 **물리 dt** 를 바꾼다(dt = u_max*dx/(Omega R)).
따라서 nu_lu = nu_phys*dt/dx^2 가 함께 커져 Re 가 자동으로 따라간다.
0.75R 폴라 Re 가 약 151k -> **121k** 로 내려가므로 neuralfoil 테이블도 그
Re 로 다시 뽑힌다(Re_min/max 10k~300k 안에 있다).

**호버 높이 450 mm.** 지면(hwbb, zmin)이 CAD z=-821.8 mm 에서 **-316.3 mm**
로 올라온다. 기체·로터의 LU z 가 그만큼 내려앉고, L2/L3 박스도 따라 내려간다
(`build_mlg_4level` 이 전부 유도한다). 기어 최하단은 지면 위 **184.2 mm** 로
남는다(v2 는 689.7 mm).

★ 물리적으로 v2 와 같은 런이 아니다. 0.98 D 는 강한 IGE 영역이다 —
지면 와류가 디스크에 훨씬 가깝고, 벽 제트가 더 세고 더 멀리 간다. L1 의
+-5 m 아웃워시 반경은 v2 의 2.09 D 에서 정한 값이므로, 이 높이에서 도달거리가
그 밖으로 나가는지는 **결과를 보고 판단해야 한다**(나가면 L1 반경을 키운다).

★ L2 바닥이 지면에 매우 가까워진다. v2 는 L2 바닥이 z=27 L0 셀이었는데 v3 는
한 자릿수로 내려온다. L1 바닥(z=2)과의 간격이 밴드(2 L0 셀)를 겨우 넘기므로
`build_mlg_4level` 의 assert 가 이걸 지킨다 — 걸리면 l1_zmin/pad2 가 아니라
**높이 자체가 이 격자 구성의 하한**이라는 뜻이다.

v2 에서 그대로 물려받는 한계
----------------------------
로터 블록 8개가 기체를 3~4 면에서 자른다(어떤 유한 블록도 암을 피할 수 없다 —
`octo8_blocks8_preview.py` 측정 확정). 로터 추력은 영향 없고(7e-6), **기체
다운로드는 ~5% 불확실**하다. 그리고 그 절단면 때문에 L3 링크의 ~17% 가 박스
이음매를 넘는 유령 링크가 된다(patch_notes/ibb_sparse/02) — 현재 구성에서는
양성이지만(힘 측정 off + C2F 밴드 안), 이 config 로 기체 힘을 재려 하면 먼저
그 트랙을 끝내야 한다.

Run (cluster, 4x24GB):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \\
        --config configs/octo8/octo8_v3_hover.py --gpu 0,1,2,3 \\
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

RPM = 4000.0                      # v2: 5000
HOVER_H_MM = 450.0                # v2: 955.5 (지면 -> 로터면)

D_LU_0 = 20                       # L0: rotor diameter in cells
HALF_XY_MM = 7000.0               # L0 half-extent about the vehicle centre
L1_HALF_MM = 5000.0               # ground-resolved outwash radius
N_RADIAL = 48                     # delta_r 3.7 mm <= eps floor 5.71 mm @D/160
N_REV = 100

# full-field VTK 를 전 구간 30° 로. 용량은 v2 와 같은 셀 수라 ~8 TB —
# KISTI /scratch(100 TB) 안. **/home01(64 GB) 은 불가**: 리포와 출력을
# /scratch 에 둘 것. 줄이려면 vtk_deg 를 키우거나 아래를 낮춘다.
VTK_FIELDS_REV = N_REV       # 전 구간

config = build_config(rpm=RPM, n_rev=N_REV, n_radial=N_RADIAL,
                      vtk_deg=30.0, vtk_fields_last_rev=VTK_FIELDS_REV,
                      wall_bc="ibb",
                      d_lu=D_LU_0, half_xy_mm=HALF_XY_MM, side_bc="neumann",
                      hover_h_mm=HOVER_H_MM,
                      # 8기 전부 y축(날개 방향)에 평행하게 정렬해 시작.
                      # e_perp=-y_hat 이라 theta=pi/2 가 -y, +pi 더한 2번째
                      # 블레이드가 +y. 스태거를 포기하는 대신 초기 위상이 통제된다.
                      theta0=np.pi / 2)

_info = build_mlg_4level(config, d_lu0=D_LU_0, half_xy_mm=HALF_XY_MM,
                         l1_half_mm=L1_HALF_MM, hover_h_mm=HOVER_H_MM,
                         overlap_width=2, pad2=2.0, l1_zmin=2.0,
                         # 로터 블록이 기체를 자른다(헤더 참조). 'exclude' 가
                         # 아니라 'allow': 이 케이스는 IBB 이고, C2F 벽 제외는
                         # 오히려 나쁘다고 실측됐다(wall_coupling/01 3a).
                         wall_coupling_mode="allow")

_tag = "result_octo8_v3"
config["output"]["output_dir"] = "./%s/vtk" % _tag
config["output"]["checkpoint_dir"] = "./%s/checkpoints" % _tag
config["output"]["csv_dir"] = "./%s/csv" % _tag


if __name__ == "__main__":
    mlg_report(config, _info, D_LU_0, L1_HALF_MM, N_REV, tag="octo8 v3")
    _n_out = config['time']['max_steps'] // config['time']['output_interval']
    _n_fld = int(VTK_FIELDS_REV * 628 / config['time']['output_interval'])
    print(f"  steps {config['time']['max_steps']:,} ({N_REV} rev) | "
          f"VTK every {config['time']['output_interval']} | "
          f"ckpt every {config['time']['checkpoint_interval']:,}")
    print(f"  VTK: {_n_out} 출력 중 full-field {min(_n_fld, _n_out)}개 "
          f"~ {min(_n_fld, _n_out) * _info['total_cells'] * 32 / 1e12:.1f} TB")
    _l2 = _info["l2"]
    print(f"  L2 z [{_l2[0][2]:.0f}, {_l2[1][2]:.0f}] L0 lu  |  "
          f"L1 z [{_info['l1'][0][2]:.0f}, {_info['l1'][1][2]:.0f}]")
