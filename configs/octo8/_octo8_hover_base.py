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

# 호버링 높이: 지면 → **로터면** [mm]. 실험 조건이므로 rpm 과 같은 급의
# 노브다 — 모듈 상수는 기본값이고, build_config/grid_map* 의 hover_h_mm 로
# 케이스별로 준다(v3 = 450mm).
#   0808 사용자 확정 기본값 955.5mm = 2.09D (IGE)
#     → 지면 z_cad = -821.789mm. 기어 최하단(-132.044)은 지면 위 689.7mm.
HOVER_H_MM    = 955.5
GROUND_CAD_MM = ROTOR_Z_MM - HOVER_H_MM          # -821.789


def ground_cad_mm(hover_h_mm=None):
    """지면의 CAD z [mm]. hover_h_mm=None 이면 모듈 기본값."""
    h = HOVER_H_MM if hover_h_mm is None else float(hover_h_mm)
    if h <= 0.0:
        raise ValueError(f"hover_h_mm must be > 0, got {h}")
    g = ROTOR_Z_MM - h
    # 기체가 지면을 뚫으면 마스크가 지면 아래로 내려가 hwbb 벽과 겹친다.
    if _BBOX_MIN_MM[2] <= g:
        raise ValueError(
            f"hover_h_mm={h}: 지면 z_cad={g:.3f}mm 이 기체 최하단 "
            f"{_BBOX_MIN_MM[2]:.3f}mm 위에 있다 — 기체가 지면에 박힌다")
    return g

# 도메인 (lu): CAD 하드웨어 x[-186.0,-4.8], y[±167.2] lu 기준
#   x: 후방(테일측 -x) 2D, 전방(노즈 너머 +x) ~1.5D
#   y: 윙팁 밖 ±2D
#   z: zmin = 지면(hwbb 벽, half-way라 벽면은 노드0보다 0.5셀 아래) ~ 바디 상부
#      위 ~2.1D(로터면 위 ~2.9D). 지면 아래 여백 없음(IGE).
NX, NY, NZ = 324, 496, 200
_O_Z = -GROUND_CAD_MM * MM2LU - 0.5             # 71.898 - 0.5 = 71.398
O_LU = np.array([266.0, 248.0, _O_Z])           # CAD 원점(노즈)의 LU 좌표


def grid_map(d_lu: int = D_LU, side_clearance_lb: float = None,
             hover_h_mm: float = None):
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
    o_z = -ground_cad_mm(hover_h_mm) * mm2lu - 0.5
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

def grid_map_centered(d_lu: int, half_xy_mm: float, hover_h_mm: float = None):
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
                  -ground_cad_mm(hover_h_mm) * mm2lu - 0.5])
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


def _layout(flip_handedness=False):
    """로터 배치. flip_handedness=True 면 8기 전부 반대 방향(cw<->ccw).

    이름 접미사가 핸디드니스를 인코딩하므로 이름도 함께 뒤집는다
    (f1_ccw -> f1_cw). 위치·행 구조는 불변 — 방향 실험(v4)의 단일 노브.
    """
    if not flip_handedness:
        return _LAYOUT
    flip = {"cw": "ccw", "ccw": "cw"}
    return [(name.rsplit("_", 1)[0] + "_" + flip[hand], x, y, flip[hand])
            for name, x, y, hand in _LAYOUT]


# =============================================================================
# S3. CONFIG BUILDER
# =============================================================================
def build_config(rpm=5000.0, n_rev=40, n_radial=32,
                 vtk_deg=30.0, vtk_fields_last_rev=5, wall_bc="ibb",
                 d_lu=None, half_xy_mm=None, side_bc="sponge", theta0=None,
                 hover_h_mm=None, flip_handedness=False):
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
    flip_handedness    : True 면 8기 전부 회전방향 반전(cw<->ccw, 이름 포함).
                         위치는 불변. build_mlg_4level 에도 같은 값을 줄 것
                         (로터 블록 이름 일치).

    d_lu=None, half_xy_mm=None 이면 모듈 상수를 그대로 써서 기존 config 들이
    비트 단위로 움직이지 않는다.
    """
    if d_lu is None:
        d_lu = D_LU
    if half_xy_mm is not None:
        dx_m, mm2lu, o_lu, (nx, ny, nz) = grid_map_centered(
            d_lu, half_xy_mm, hover_h_mm)
    elif d_lu != D_LU:
        dx_m, mm2lu, o_lu, (nx, ny, nz) = grid_map(d_lu,
                                                   hover_h_mm=hover_h_mm)
    else:                                    # legacy — 모듈 상수 그대로
        dx_m, mm2lu, o_lu = DX_PHYS, MM2LU, O_LU
        nx, ny, nz = NX, NY, NZ
        if hover_h_mm is not None:           # 지면만 옮긴 legacy 도메인
            o_lu = np.array([O_LU[0], O_LU[1],
                             -ground_cad_mm(hover_h_mm) * MM2LU - 0.5])

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
        "rotors": [_rotor_entry(i, *row)
                   for i, row in enumerate(_layout(flip_handedness))],
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


# =============================================================================
# S4. 4-LEVEL MLG (far field / ground outwash / airframe / per-rotor blocks)
# =============================================================================
# 로터 블록 비율: 단일프롭 검증(apc18x8e EXTENTS[-1])이 쓴 값 그대로.
_ROTOR_LAT_D = 0.6875      # 디스크 중심 기준 좌우 반폭 [D]
_ROTOR_UP_D  = 0.125       # 디스크 위 [D]
_ROTOR_DN_D  = 0.25        # 디스크 아래 [D] (ALM Gaussian 지지)


def build_mlg_4level(config, d_lu0, half_xy_mm, l1_half_mm,
                     hover_h_mm=None, overlap_width=2, pad2=2.0,
                     l1_zmin=2.0, l2_zmin=None,
                     wall_coupling_mode="allow",
                     interpolation="cubic", filter_level=1,
                     flip_handedness=False):
    """v2/v3 가 공유하는 4레벨 구성. `config["mlg"]` 를 채우고 정보를 돌려준다.

        L0 D/d_lu0     far field          (도메인 전체)
        L1 D/2x        ground outwash     (+-l1_half_mm, 바닥은 지면)
        L2 D/4x        airframe           (기체 bbox + 전 로터 블록 + 밴드)
        L3 D/8x        로터 8기, 블록 하나씩

    ★ l1_zmin 은 0 이 아니라 overlap_width 다. 도메인 바닥에 붙인 fine region
    은 flush 라서 C2F 밴드가 폭 0 으로 붕괴하고(설계된 상태), fine 레벨은
    boundaries_config={} 라 자기 벽도 없다. 그러면 L1 바닥은 coarse 유입도
    벽도 없이 남는다. 바닥 두 L0 셀을 밴드로 내주면 L1 은 지면 노드까지 fine
    셀을 갖되 그 밴드를 L0(지면 hwbb 를 가진)에서 받는다.

    반환: dict(dx, mm2lu, origin, shape, l1, l2, rotor_boxes, n_cells, ...)
    """
    dx, mm2lu, o, n = grid_map_centered(d_lu0, half_xy_mm, hover_h_mm)

    def _lu(p_mm):
        return o + np.asarray(p_mm, dtype=np.float64) * mm2lu

    b_lo, b_hi = _lu(_BBOX_MIN_MM), _lu(_BBOX_MAX_MM)      # airframe bbox
    ctr = _lu([_BBOX_CTR_MM[0], 0.0, 0.0])

    lat = _ROTOR_LAT_D * d_lu0
    up, dn = _ROTOR_UP_D * d_lu0, _ROTOR_DN_D * d_lu0
    rz = float(_lu([0.0, 0.0, ROTOR_Z_MM])[2])
    rotor_boxes = []
    for name, x_mm, y_mm, _hand in _layout(flip_handedness):
        h = _lu([x_mm, y_mm, ROTOR_Z_MM])
        rotor_boxes.append({
            "name": name,
            "x_min": float(h[0] - lat), "x_max": float(h[0] + lat),
            "y_min": float(h[1] - lat), "y_max": float(h[1] + lat),
            "z_min": float(rz - dn),    "z_max": float(rz + up),
        })

    # L2: 기체 AND 전 로터 블록 + L3 밴드(2 부모셀 = 0.5 L0 lu) + pad2
    r_lo = np.min([[b["x_min"], b["y_min"], b["z_min"]] for b in rotor_boxes], 0)
    r_hi = np.max([[b["x_max"], b["y_max"], b["z_max"]] for b in rotor_boxes], 0)
    l2_lo = np.floor(np.minimum(b_lo, r_lo) - pad2)
    l2_hi = np.ceil(np.maximum(b_hi, r_hi) + pad2)
    # l2_zmin(v4): L2 를 지면 쪽으로 최대한 내리는 노브 [L0 lu].
    # 하한은 **부모 내부**다: 두 독립 가드가 지킨다 — ①부모-포함 계약
    # (setup._mlg_resolve_parent) ②중첩 검증기(validate_block_tree:
    # 자식 영역이 부모의 C2F 밴드를 침범 금지 — 밴드 행은 L0 가 매 스텝
    # 처방하므로 F2C 와 충돌). l1_zmin=2, ow=2 기준 하한 = 3(L0 lu) →
    # L2 영역 z=3 + 자기 밴드(ow x L1셀 = 1 L0 lu) = **도메인 바닥이
    # 지면 위 2 L0셀**. 완전 지면 flush 는 fine 벽 승계(open-face
    # mailbox 일반화, eso_wall 이월) 필요. None = 기존 auto(v2/v3 비트
    # 불변).
    if l2_zmin is not None:
        _lo_min = l1_zmin + int(overlap_width) / 2.0   # 부모 내부 하한
        if l2_zmin < _lo_min:
            raise ValueError(
                f"l2_zmin={l2_zmin} < {_lo_min}: 부모(L1) 내부 하한 위반"
                " — validate_block_tree 가 부모 밴드 침범을 막는다")
        l2_lo[2] = float(l2_zmin)

    # L1: 기체 중심 기준 +-l1_half_mm, 바닥은 지면 위 l1_zmin
    half1 = l1_half_mm * mm2lu
    l1_lo = np.array([np.floor(ctr[0] - half1), np.floor(ctr[1] - half1),
                      float(l1_zmin)])
    l1_hi = np.array([np.ceil(ctr[0] + half1), np.ceil(ctr[1] + half1),
                      float(np.ceil(l2_hi[2] + 5.0))])

    ow = int(overlap_width)
    chk_lo = l2_lo - ow
    if l2_zmin is not None:
        # 지면 적층: z 하한 컨테인먼트는 영역이 아니라 L1 도메인(밴드
        # 포함 z>=0)이 담보 — z 성분만 L1 영역 하한으로 면제.
        chk_lo[2] = max(chk_lo[2], l1_lo[2])
    if not ((l1_lo <= chk_lo).all() and (l1_hi >= l2_hi + ow).all()):
        raise ValueError(
            f"L1 {l1_lo}..{l1_hi} 이 L2 {l2_lo}..{l2_hi} + 밴드({ow})를 담지 "
            f"못한다 — l1_half_mm 을 키우거나 hover 높이를 확인할 것")
    if not (l1_lo[0] >= ow and l1_hi[0] <= n[0] - 1 - ow):
        raise ValueError(
            "L1 이 도메인 면에 닿는다 — fine 레벨은 도메인 BC 가 없다")
    if 0 < l1_lo[2] < ow:
        raise ValueError(
            f"l1_zmin={l1_zmin} in (0, overlap_width={ow}): 바닥 밴드가 안 "
            "잡힌다 — 0(지면 flush, 벽 승계 + q-face) 또는 >= ow(밴드-온-월)")

    def _box(name, lo, hi):
        return {"name": name,
                "x_min": float(lo[0]), "x_max": float(hi[0]),
                "y_min": float(lo[1]), "y_max": float(hi[1]),
                "z_min": float(lo[2]), "z_max": float(hi[2])}

    config["mlg"] = {
        "enabled": True, "num_levels": 4, "overlap_width": ow,
        "interpolation": interpolation, "filter_level": filter_level,
        "levels": [
            {},                                              # L0 far field
            {"regions": [_box("outwash", l1_lo, l1_hi)]},     # L1 ground
            {"regions": [_box("airframe", l2_lo, l2_hi)]},    # L2 airframe
            {"regions": rotor_boxes},                         # L3 rotors x8
        ],
        "wall_coupling": {"mode": wall_coupling_mode},
    }

    def _fine_n(lo, hi, k):
        """밴드 포함 fine 노드 수 (region*2^k + 4*ow + 1)."""
        return [int(round((hi[i] - lo[i]) * 2 ** k)) + 4 * ow + 1
                for i in range(3)]

    n3 = _fine_n([rotor_boxes[0]["x_min"], rotor_boxes[0]["y_min"],
                  rotor_boxes[0]["z_min"]],
                 [rotor_boxes[0]["x_max"], rotor_boxes[0]["y_max"],
                  rotor_boxes[0]["z_max"]], 3)
    shapes = {"L0": tuple(int(v) for v in n),
              "L1": _fine_n(l1_lo, l1_hi, 1),
              "L2": _fine_n(l2_lo, l2_hi, 2),
              "L3": n3}
    cells = {k: float(np.prod(v)) for k, v in shapes.items()}
    cells["L3"] *= len(rotor_boxes)
    updates = (cells["L0"] + cells["L1"] * 2 + cells["L2"] * 4
               + cells["L3"] * 8)
    return {"dx": dx, "mm2lu": mm2lu, "origin": o, "shape": tuple(n),
            "l1": (l1_lo, l1_hi), "l2": (l2_lo, l2_hi),
            "rotor_boxes": rotor_boxes, "shapes": shapes, "cells": cells,
            "total_cells": sum(cells.values()), "updates": updates,
            "ground_cad_mm": ground_cad_mm(hover_h_mm),
            "hover_h_mm": HOVER_H_MM if hover_h_mm is None else hover_h_mm}


def mlg_report(cfg, info, d_lu0, l1_half_mm, n_rev, tag=""):
    """build_mlg_4level 결과를 사람이 읽는 표로. config 의 __main__ 용."""
    dxmm = info["dx"] * 1000.0
    print(f"  [{tag}] hover {info['hover_h_mm']:.1f} mm = "
          f"{info['hover_h_mm'] / (D_PHYS * 1000):.2f} D  "
          f"(지면 z_cad {info['ground_cad_mm']:.1f} mm)")
    rpm = abs(cfg["actuator_line"]["rotors"][0]["rotor"]["rpm"])
    print(f"        rpm {rpm:.0f} | {n_rev} rev = "
          f"{cfg['time']['max_steps']:,} steps")
    for k, lbl in (("L0", "far-field"), ("L1", "outwash"),
                   ("L2", "airframe"), ("L3", "rotors x8")):
        s = info["shapes"][k]
        lev = int(k[1])
        print(f"  {k}  D/{d_lu0 * 2 ** lev:<4d} dx={dxmm / 2 ** lev:6.3f} mm  "
              f"{s[0]}x{s[1]}x{s[2]} = {info['cells'][k] / 1e6:7.2f} M   {lbl}")
    print(f"  ---- {info['total_cells'] / 1e6:.2f} M cells | "
          f"{info['updates'] / 1e6:.1f} M upd/coarse step")
    dom = (info["shape"][0] - 1) / info["mm2lu"]
    print(f"  far-field {dom / 1000:.2f} m | L1 ground radius "
          f"{l1_half_mm / 1000:.2f} m")
    print(f"  running mem @4 ranks ~ "
          f"{info['total_cells'] * 130 / 4 / 1e9:.1f} GB/rank")


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
