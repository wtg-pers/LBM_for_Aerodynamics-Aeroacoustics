"""
Octo-8 vehicle — 8x APC 18x8E + STL body, HOVER, uniform single-level grid.

지오메트리: input_files/geom/assembly_full.stl (CATIA ASCII, mm 단위,
352,960 facets). CAD bbox: x[-2125.79, -55.00], y[±1911.00],
z[-132.04, +505.00] mm — y=0 대칭. 로터 8기 (APC 18x8E, R=228.6mm,
z=+133.711mm 평면, CAD mm):
    inner |y|=548.5  (x=-302.24, -1325.24, y=±548.5)  → CCW (위에서 볼 때)
    outer |y|=1288.5 (x=-302.24, -1325.24, y=±1288.5) → CW
미러측(y<0)은 사용자 확인대로 동일 패턴(내측 CCW·외측 CW).

핸디드니스 인코딩 (프로브로 실증, 2026-08-08):
    rotation_axis=[0,0,-1], thrust_direction=[0,0,+1] 전 로터 공통(추력 +z,
    후류 -z). rpm 부호가 회전방향: +rpm=CW(위에서), -rpm=CCW(위에서).
    Rotor.project_blade_forces가 rotation_sign=sign(ω)를 적용, 양쪽 모두
    +z 추력·동일 |C_T| 확인.

솔버 제약 (2026-08-08 코드 확정):
    - 멀티로터 ALM은 MPI 미지원 → 단일 GPU 실행 (python main.py --gpu N)
    - 단일 레벨(uniform, MLG off) — dx = D/40 = 11.43mm, 로터는 L0 해상
    - ε=0.25c는 전 스팬에서 2Δx floor에 걸림(ε≈22.9mm≈0.1R) — 절대 CT는
      단일로터 캠페인(dx=D/320, configs/apc18x8e)이 앵커, 여기는 기체 통합
    - VTK: config output_interval=105(30.07°) + vtk.fields_start_step로
      field는 마지막 N바퀴만(마커 VTP는 전 구간) — 단일 GPU 경로 옵션(0808 추가)

포뮬레이션: 단일로터 notl 확정 구성 이식 — kleine straight eps-correction +
Merabet radial truncation + iso gaussian, tip loss OFF, dyn_smag, cumulant
D3Q27, convective u_max=0.1. hwbb 바디(블로케이지 취급; τ→0.5 정류 펌핑
병리는 stl 트랙 문서 참조 — 바디 힘 정밀도가 목표가 아닐 때만).

CAD → LU 매핑: dx=11.43mm, CAD 원점(노즈 원점)을 O_LU=(266, 248, 172)에
배치. STL은 bbox 중심이 center_lu로 가는 변환이라 center_lu를 CAD bbox
중심에서 역산(하드코딩된 bbox는 0808 스캔값 — STL 교체 시 갱신 필요).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "apc18x8e"))
from _apc18x8e_hover_base import (  # noqa: E402  (blade/폴라 단일소스)
    C_S_PHYS, D_PHYS, N_BLADES, NU_PHYS, R_PHYS, RHO_PHYS,
    _blade_sections, _load_geometry)

# =============================================================================
# S1. GRID / CAD MAPPING
# =============================================================================
D_LU     = 40                                   # rotor diameter [cells]
DX_PHYS  = D_PHYS / D_LU                        # 0.011430 m
MM2LU    = 0.001 / DX_PHYS                      # 0.0874891 lu/mm

# CAD bbox (mm) — awk 스캔(0808): 재스캔 없이 하드코딩. STL 교체 시 갱신!
_BBOX_MIN_MM = np.array([-2125.790, -1911.000, -132.044])
_BBOX_MAX_MM = np.array([-55.000, 1911.000, 505.000])
_BBOX_CTR_MM = 0.5 * (_BBOX_MIN_MM + _BBOX_MAX_MM)   # (-1090.395, 0, 186.478)

ROTOR_Z_MM = 133.711                            # 로터 디스크 평면 (CAD z)

# 호버링 높이(0808 사용자 확정): 지면 → **로터면** 955.5mm = 2.09D (IGE).
#   → 지면 z_cad = -821.789mm. 기어 최하단(-132.044)은 지면 위 689.7mm.
HOVER_H_MM    = 955.5
GROUND_CAD_MM = ROTOR_Z_MM - HOVER_H_MM          # -821.789

# 도메인 (lu): CAD 하드웨어 x[-186.0,-4.8], y[±167.2] lu 기준
#   x: 후방(테일측 -x) 2D, 전방(노즈 너머 +x) ~1.5D
#   y: 윙팁 밖 ±2D
#   z: zmin = 지면(hwbb 벽, half-way라 벽면은 노드0보다 0.5셀 아래) ~ 바디 상부
#      위 ~2.1D(로터면 위 ~2.9D). 지면 아래 여백 없음(IGE).
NX, NY, NZ = 324, 496, 200
_O_Z = -GROUND_CAD_MM * MM2LU - 0.5             # 71.898 - 0.5 = 71.398
O_LU = np.array([266.0, 248.0, _O_Z])           # CAD 원점(노즈)의 LU 좌표


def grid_map(d_lu: int = D_LU, side_clearance_lb: float = None):
    """(dx_m, mm2lu, O_LU, (Nx, Ny, Nz)) for a base resolution of D/`d_lu`.

    d_lu=40 with side_clearance_lb=None reproduces the module constants above
    EXACTLY — the legacy domain, kept so the validated uniform/mlg2 configs do
    not move.

    side_clearance_lb: lateral (x, y) clearance from the body surface to the
    domain boundary, in units of L_body (the vehicle's own largest bbox span).
    geometry_manager warns below 0.5; the legacy domain sits at 0.18 (x) and
    0.24 (y), so the wall jet leaving the ground reaches the sponge while still
    developing — an outwash study cannot be read at face value there.

    z is NEVER expanded and is not part of this knob:
      * zmin IS the ground (IGE hover). Its "clearance" is the hover height —
        a physical quantity, not a far-field margin. The axis-2 warning is a
        FALSE POSITIVE for this case and stays no matter how big the box gets.
      * zmax is the inflow, 2.1 D above the body, which the wake never reaches.
    """
    dx = D_PHYS / d_lu
    mm2lu = 0.001 / dx
    o_z = -GROUND_CAD_MM * mm2lu - 0.5
    if side_clearance_lb is None:                       # legacy domain
        s = d_lu / 40.0
        return dx, mm2lu, np.array([266.0 * s, 248.0 * s, o_z]), \
            (int(round(324 * s)), int(round(496 * s)), int(round(200 * s)))
    span = (_BBOX_MAX_MM - _BBOX_MIN_MM) * mm2lu        # body bbox, this level
    lb = float(span.max())                              # L_body (y span)
    pad = side_clearance_lb * lb
    o = np.array([pad - _BBOX_MIN_MM[0] * mm2lu,
                  pad - _BBOX_MIN_MM[1] * mm2lu,
                  o_z])
    # even extents: MLG halves boxes level by level, and an odd outer extent
    # buys nothing
    n = [int(np.ceil((span[0] + 2 * pad) / 2) * 2),
         int(np.ceil((span[1] + 2 * pad) / 2) * 2),
         int(round(200 * d_lu / 40.0))]                 # z: legacy height
    return dx, mm2lu, o, tuple(n)

def grid_map_centered(d_lu: int, half_xy_mm: float):
    """(dx_m, mm2lu, O_LU, (Nx,Ny,Nz)) for a domain CENTRED on the vehicle.

    half_xy_mm: half-extent in x and y measured from the vehicle centre (the
    STL bbox centre; y=0 by symmetry). This is the sizing knob when the
    far-field requirement is stated as "N metres around the aircraft" rather
    than as a clearance from its surface — e.g. an outwash study whose
    influence radius was measured in another solver.

    z is NOT centred and is not part of this knob, for the same reason as in
    grid_map: zmin IS the ground (IGE hover, height = HOVER_H_MM) and zmax is
    the inflow. z keeps the legacy height scaled to `d_lu`.
    """
    dx = D_PHYS / d_lu
    mm2lu = 0.001 / dx
    n_xy = int(np.ceil(2.0 * half_xy_mm * mm2lu))
    n_xy = int(np.ceil(n_xy / 2) * 2) + 2          # even, and >= the request
    nz = int(round(200 * d_lu / 40.0))
    o = np.array([(n_xy - 1) * 0.5 - _BBOX_CTR_MM[0] * mm2lu,
                  (n_xy - 1) * 0.5 - _BBOX_CTR_MM[1] * mm2lu,
                  -GROUND_CAD_MM * mm2lu - 0.5])
    return dx, mm2lu, o, (n_xy, n_xy, nz)


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
# repaired(0808): 원본 assembly_full.stl은 비수밀(나사 32개 역와인딩 + 로드
# 4개 캡 없음) → 로더 거부. 수리: 나사 제거(2.9mm, dx=11.4mm 서브그리드),
# 로드 캡 실린더 재구성, manifold 불리언 union. bbox 원본 동일.
STL_PATH = os.path.join(_REPO, "input_files", "geom",
                        "assembly_full_repaired.stl")


def _cad_mm_to_lu(p_mm):
    """CAD mm 좌표 → L0 LU (단일소스: O_LU + p*MM2LU)."""
    return (O_LU + np.asarray(p_mm, dtype=np.float64) * MM2LU).tolist()


STL_CENTER_LU = _cad_mm_to_lu(_BBOX_CTR_MM)     # bbox 중심의 LU 위치

# =============================================================================
# S2. ROTOR LAYOUT  (CAD mm; +rpm=CW(위에서), -rpm=CCW)
# =============================================================================
# ROTOR_Z_MM 은 S1에 정의(지면 기준 계산이 먼저 필요).
# (name, x_mm, y_mm, handedness)  — 사용자 확정(0808 2차): 위에서 내려다보고
# 노즈(+x)가 화면 아래일 때(화면 왼쪽=−y), 전방열 왼→오 CCW,CW,CCW,CW /
# 후방열 왼→오 CW,CCW,CW,CCW = 체커보드. f=전방(x=−302.24), a=후방(x=−1325.24),
# 번호는 왼(−y)→오(+y).
_LAYOUT = [
    ("f1_ccw", -302.24,  -1288.5, "ccw"),
    ("f2_cw",  -302.24,   -548.5, "cw"),
    ("f3_ccw", -302.24,    548.5, "ccw"),
    ("f4_cw",  -302.24,   1288.5, "cw"),
    ("a1_cw",  -1325.24, -1288.5, "cw"),
    ("a2_ccw", -1325.24,  -548.5, "ccw"),
    ("a3_cw",  -1325.24,   548.5, "cw"),
    ("a4_ccw", -1325.24,  1288.5, "ccw"),
]


# =============================================================================
# S3. CONFIG BUILDER
# =============================================================================
def build_config(rpm=5000.0, n_rev=40, n_radial=32,
                 vtk_deg=30.0, vtk_fields_last_rev=5, wall_bc="ibb",
                 d_lu=None, half_xy_mm=None, side_bc="sponge", theta0=None):
    """Octo-8 hover config.

    rpm                : 공칭 회전수(부호 없는 크기; 방향은 _LAYOUT이 결정)
    n_rev              : 총 회전수 (기본 40)
    n_radial           : markers/blade. dx=D/40에서 δr=R/n ≤ ε(=2Δx=R/10)
                         필요 → n≥10; 32면 δr=0.625Δx (테이퍼·전이 해상 여유)
    vtk_deg            : VTK 출력 각도 간격 (30° → 105 steps)
    vtk_fields_last_rev: full-field VTK는 마지막 N바퀴만 (마커는 전 구간)
    d_lu               : L0 로터 직경 해상도 [cells]. None = 모듈 기본(40).
                         MLG를 쓸 때 L0 는 원방 담당이므로 40 보다 낮게 잡는다.
    half_xy_mm         : 기체 중심 기준 x,y 반폭 [mm]. 주면 도메인이 그 값으로
                         재계산된다(grid_map_centered). None 이면 legacy 도메인.
    theta0             : 전 로터 공통 초기 방위각 [rad]. None(기본)이면 로터마다
                         i*pi/8 스태거(아래 참조). 값을 주면 8기 모두 그 값.
                         rotation_axis=[0,0,-1] 이라 e_ref=x_hat, e_perp=-y_hat 이므로
                         블레이드 방향 = (cos t, -sin t, 0):
                           t=0    -> +x (노즈)      t=pi/2 -> -y (날개 평행)
                         2엽이라 t0 와 t0+pi 에 놓인다 -> theta0=pi/2 면 두 블레이드가
                         -y/+y 를 향해 **y축에 평행**하게 정렬된다.
    side_bc            : 측면 4면(+-x, +-y) BC. "sponge"(기본) | "neumann".
                         아웃워시 도달거리가 목적이면 neumann — sponge 는 흡수층
                         20셀 안에서 벽 제트를 **강제 감쇠**시켜 재려는 양을
                         훼손한다. 대가는 무반사성 상실이므로, 음향이 목적이 되면
                         다시 sponge(또는 무반사 BC)로 돌아와야 한다.

    d_lu=None, half_xy_mm=None 이면 모듈 상수를 그대로 써서 기존 config 들이
    비트 단위로 움직이지 않는다.
    """
    if d_lu is None:
        d_lu = D_LU
    if half_xy_mm is not None:
        dx_m, mm2lu, o_lu, (nx, ny, nz) = grid_map_centered(d_lu, half_xy_mm)
    elif d_lu != D_LU:
        dx_m, mm2lu, o_lu, (nx, ny, nz) = grid_map(d_lu)
    else:                                    # legacy — 모듈 상수 그대로
        dx_m, mm2lu, o_lu = DX_PHYS, MM2LU, O_LU
        nx, ny, nz = NX, NY, NZ

    def _to_lu(p_mm):
        return (o_lu + np.asarray(p_mm, dtype=np.float64) * mm2lu).tolist()

    U_MAX_LU  = 0.1
    R_LU      = 0.5 * d_lu
    STEPS_REV = int(round(2.0 * np.pi * R_LU / U_MAX_LU))          # 1257 @D/40
    MAX_STEPS = n_rev * STEPS_REV
    VTK_EVERY = int(round(STEPS_REV * vtk_deg / 360.0))            # 105
    FIELDS_START = MAX_STEPS - int(round(vtk_fields_last_rev * STEPS_REV))

    omega_mag = abs(rpm) * 2.0 * np.pi / 60.0                      # [rad/s]

    # 폴라 Re 범위 (0.75R 기준; 5000rpm → Re_75 ≈ 151k)
    rR, chord, _ = _load_geometry()
    chord_75 = float(np.interp(0.75, rR, chord))
    re_75 = int(round(0.75 * omega_mag * R_PHYS * chord_75 / NU_PHYS))

    simulation = {"device_mode": "gpu", "precision": "float32", "dimension": 3,
                  "lattice_model": "D3Q27", "collision_model": "cumulant"}
    physics = {"rho": RHO_PHYS, "U_inf": 0.0, "nu": NU_PHYS,
               "L_char": D_PHYS, "flow_direction": [0.0, 0.0, -1.0]}
    grid = {"Nx": nx, "Ny": ny, "Nz": nz, "resolution": d_lu}
    numerics = {"acoustic_scaling": False, "u_max": U_MAX_LU,
                "c_s_phys": C_S_PHYS, "collision": "cumulant"}

    # IGE 하버링: zmin = 지면(no-slip hwbb 벽). 다운워시가 지면에서 radial
    # wall jet로 꺾여 측면 4면으로 유출 → 측면은 sponge(흡수), 유입은 zmax eq.
    def _sponge(loc):
        return {"location": loc, "method": "sponge",
                "velocity": [0.0, 0.0, 0.0], "density": 1.0,
                "thickness": 20, "strength": 0.1}

    def _side(loc):
        if side_bc == "neumann":            # zero-gradient: f[bdy] = f[int]
            return {"location": loc, "method": "neumann"}
        if side_bc != "sponge":
            raise ValueError(f"side_bc must be 'sponge' or 'neumann', "
                             f"got {side_bc!r}")
        return _sponge(loc)
    boundaries = {
        "xmin": _side("xmin"),
        "xmax": _side("xmax"),
        "ymin": _side("ymin"),
        "ymax": _side("ymax"),
        "zmin": {"location": "zmin", "method": "hwbb"},
        "zmax": {"location": "zmax", "method": "eq", "velocity": [0.0, 0.0, 0.0]},
    }

    internal_geometry = {
        "stl": {
            "enabled": True,
            "file": STL_PATH,
            "scale_to_lu": mm2lu,
            "center_lu": tuple(float(c) for c in _to_lu(_BBOX_CTR_MM)),
            "rotation_deg": (0.0, 0.0, 0.0),
            "wall_bc": wall_bc,
        },
    }

    def _nf(name):
        return {"method": "neuralfoil", "airfoil_name": name,
                "Re_target": re_75, "Re_min": 10000, "Re_max": 300000,
                "Re_steps": 12, "mode": "asb", "ncrit": 9.0}
    airfoil_polar = {"method": "multi", "default": "e63",
                     "airfoils": {"e63": _nf("e63"), "naca4412": _nf("naca4412")}}

    secs = [{"r": r, "chord": c, "twist": tw, "airfoil": af, "active": act}
            for r, c, tw, af, act in _blade_sections()]

    def _rotor_entry(i, name, x_mm, y_mm, hand):
        sign = +1.0 if hand == "cw" else -1.0       # +rpm=CW(위에서 볼 때)
        return {
            "name": name,
            "rotor": {
                "rpm": sign * abs(rpm),
                "radius": R_PHYS,
                "omega": sign * omega_mag,
                "n_blades": N_BLADES,
                "hub_center": _to_lu([x_mm, y_mm, ROTOR_Z_MM]),  # L0 LU
                "rotation_axis": [0, 0, -1],
                "thrust_direction": [0, 0, 1],
                # 기본은 블레이드 통과 위상 스태거(동기화된 8기 blade-passage
                # 아티팩트 방지; 실기체도 위상 무작위). theta0 를 주면 8기를
                # 그 값으로 정렬한다 — 초기 위상을 통제하고 싶을 때. 다만 8기가
                # 동기화되면 blade-passage 성분이 인위적으로 결맞을 수 있다.
                "theta_0": (float(i * np.pi / 8.0) if theta0 is None
                            else float(theta0)),
                "blade": {"sections": secs},
                "grid": {"n_radial": int(n_radial),
                         "marker_distribution": "uniform",
                         "cosine_side": "both"},
                "epsilon_chord_factor": 0.25,
            },
        }

    actuator_line = {
        "enabled": True,
        "rotors": [_rotor_entry(i, *row) for i, row in enumerate(_LAYOUT)],
        "gaussian_cutoff": 3.0, "rho_ref": 1.0, "coeff_mode": "rotorcraft",
        "ramp_steps": STEPS_REV,
        "prandtl_loss": False,                       # 단일로터 캠페인 판정=notl
        "eps_correction": {"enabled": True, "method": "kleine",
                           "wake": "straight", "rebuild_every": 1,
                           "wake_markers": "all", "target": "inviscid",
                           "smooth": 2},
        "sampling": {"mode": "gaussian"},
        "spreading": {"radial_truncation": True},
    }

    time = {"max_steps": MAX_STEPS,
            "output_interval": VTK_EVERY,
            "logging_interval": max(1, STEPS_REV // 20),
            "checkpoint_interval": 5 * STEPS_REV,
            "conservation_interval": STEPS_REV}

    folder = "result_octo8_hover_%04drpm_D%d_%s" % (round(abs(rpm)), d_lu, wall_bc)
    output = {"output_dir": "./%s/vtk" % folder,
              "checkpoint_dir": "./%s/checkpoints" % folder,
              "csv_dir": "./%s/csv" % folder, "clear_previous": True,
              "vtk": {"enabled": True, "precision": "float32",
                      "variables": ["density", "pressure", "velocity",
                                    "velocity_magnitude"],
                      "fields_start_step": max(0, FIELDS_START)},
              "checkpoint": {"enabled": True, "keep_last_n": 2}}

    return {"simulation": simulation, "physics": physics, "grid": grid,
            "numerics": numerics, "boundaries": boundaries,
            "internal_geometry": internal_geometry,
            "mlg": {"enabled": False},
            "sgs": {"enabled": True, "model": "dyn_smag"},
            "airfoil_polar": airfoil_polar, "actuator_line": actuator_line,
            "conservation": {"enabled": True, "verbose": 0, "log_to_csv": True},
            "convergence": {"enabled": False},
            "force_calculation": {"enabled": False},
            "output": output, "time": time}


if __name__ == "__main__":
    cfg = build_config()
    n_cells = NX * NY * NZ
    print("  [OCTO-8 HOVER]  %dx%dx%d = %.1fM cells, dx=%.2fmm" %
          (NX, NY, NZ, n_cells / 1e6, DX_PHYS * 1000))
    print("  mem ~%.1f GB (eso 207B) / ~%.1f GB (std 410B)" %
          (n_cells * 207 / 1e9, n_cells * 410 / 1e9))
    print("  steps/rev=1257, 40rev=%d steps, vtk every %d (%.2f deg)" %
          (cfg["time"]["max_steps"], cfg["time"]["output_interval"],
           360 * cfg["time"]["output_interval"] / 1257))
    print("  fields_start_step=%d" %
          cfg["output"]["vtk"]["fields_start_step"])
    print("  STL center_lu=%s" % (cfg["internal_geometry"]["stl"]["center_lu"],))
    for e in cfg["actuator_line"]["rotors"]:
        r = e["rotor"]
        print("  %-7s rpm=%+6.0f hub_lu=(%7.2f,%7.2f,%7.2f)" %
              (e["name"], r["rpm"], *r["hub_center"]))
