"""
APC 18x8E Thin Electric — single-rotor ALM HOVER, Cumulant + MLG 4-level.

HVAB 파이널 런(hvab_hover_c10_farfield40_eso_archB_ksas_mlg4_shen_g030) 포뮬레이션
이식: all-one cumulant + dyn_smag SGS, iso gaussian sampling/spreading + Merabet
radial truncation + Kleine straight eps-correction, uniform markers n64, 25 rev,
convective scaling(u_max=0.1), farfield40_mlg4 격자 토폴로지(D-상대 동일).
토글은 tip loss function 하나만:
    tip_loss="off"  → prandtl_loss=False               (GPU 2 시리즈)
    tip_loss="shen" → Shen g=0.3 (HVAB 테스트 조건 그대로)  (GPU 3 시리즈)

    import sys, os; sys.path.insert(0, os.path.dirname(__file__))
    from _apc18x8e_hover_base import build_config
    config = build_config(rpm=2446, tip_loss="off")

================================  GEOMETRY  ===================================
출처: input_files/apc18x8e/
  - apc18x8_chord_twist_distribution.csv — r/R, chord[m], twist[deg] (51 stations,
    APC 공식 지오메트리에서 추출; twist = LE-TE 기하 피치각, 하버링이라 추가
    collective 없음. x_qc(sweep)·t/c 열은 현재 미사용 — HVAB 파이널 런도 sweep off)
  - 18x8E-PERF.PE0 — APC 공식: R=9.0 in, hub 0.62 in, hub transition 2.00 in,
    2 blades. AIRFOIL SECTIONS: E63 (transition start r=2.00 in) → APC12(≡NACA4412,
    transition end r=5.76 in), 두께비로 스케일.
  - apce_18x8_static_2184od.txt — UIUC-style static 실험 CT/CP(프로펠러 convention,
    T/(ρn²D⁴)). RPM 2446.667/3460/4446.667/5446.667 포함 = 본 스윕 대조 데이터.

익형 배치(PE0 기반): r/R < 0.4311(=전이구간 (2.00+5.76)/2/9.0 중간점) → E63,
이후 NACA4412. 공력 활성 r/R ≥ 0.2222(hub transition 끝 = E63 시작).
폴라: neuralfoil(asb) Re-보간 덱 — APC 9x4.5MR 하버링 스윕과 동일 경로.
  *** t/c 스케일(내측 0.17~0.31)은 폴라에 미반영(명목 익형 폴라) — 한계 명시 ***
  *** Mach 보정 없음(Re-only 폴라): M_tip 0.17(2446)~0.38(5446) — 한계 명시 ***

==============================  CONVENTIONS  ==================================
coeff_mode="rotorcraft" (HVAB 파이널과 동일): CT=T/(ρA(ΩR)²), CP=P/(ρA(ΩR)³).
UIUC static 데이터(프로펠러 convention)와 비교 시:
    CT_prop = (π³/4)·CT_rc ≈ 7.7516·CT_rc,   CP_prop = (π⁴/4)·CP_rc ≈ 24.352·CP_rc
"""

import os

import numpy as np

# =============================================================================
# S1. PHYSICAL CONSTANTS (APC 18x8E; PE0 + SLS air)
# =============================================================================
IN2M         = 0.0254
R_PHYS       = 9.0 * IN2M                            # [m] = 0.2286
D_PHYS       = 2.0 * R_PHYS                          # [m] = 0.4572
N_BLADES     = 2
ROOT_CUT_RR  = 2.00 / 9.0                            # 0.2222 — PE0 hub transition 끝
AF_SWITCH_RR = 0.5 * (2.00 + 5.76) / 9.0             # 0.4311 — E63→NACA4412 전이 중간점

RHO_PHYS     = 1.225                                 # [kg/m^3] SLS
C_S_PHYS     = 340.3                                 # [m/s] SLS sound speed
NU_PHYS      = 1.461e-5                              # [m^2/s] SLS

N_RADIAL     = 64                                    # markers/blade (HVAB farfield n64 권장 동일)

_HERE    = os.path.dirname(os.path.abspath(__file__))
_REPO    = os.path.abspath(os.path.join(_HERE, "..", ".."))
GEOM_CSV = os.path.join(_REPO, "input_files", "apc18x8e",
                        "apc18x8_chord_twist_distribution.csv")

# =============================================================================
# S2. BLADE DISTRIBUTIONS (APC 공식 CSV 그대로, 51 stations)
# =============================================================================
def _load_geometry():
    """CSV → (rR, chord[m], twist[deg]) 배열. 헤더: r/R,chord(m),twist(deg),..."""
    rows = np.genfromtxt(GEOM_CSV, delimiter=",", skip_header=1)
    return rows[:, 0], rows[:, 1], rows[:, 2]


def _blade_sections():
    """(r[m], chord[m], twist[deg], airfoil, active) — CSV 전 station 사용.

    active: r/R ≥ 0.2222 (PE0 hub transition 끝). 내측 station들은 비활성
    지오메트리로 유지(마커 보간 연속성).  airfoil: PE0 전이 중간점에서 스위치.
    """
    rR, chord, twist = _load_geometry()
    secs = []
    for r, c, tw in zip(rR, chord, twist):
        af = "e63" if r < AF_SWITCH_RR else "naca4412"
        secs.append((float(r * R_PHYS), float(c), float(tw), af,
                     bool(r >= ROOT_CUT_RR)))
    return secs


# =============================================================================
# S3. GRID — HVAB farfield40_mlg4 프리셋 이식 (D-상대 기하 동일, 4-level)
# =============================================================================
# 도메인 반경 6R(±3D)·상류 1.5D·하류 5D → blockage 2.7%. ALM은 L3(dx=D/320)에
# 직접 앉음(L4 슬랩 없음). HVAB 실측 셀: L0 15.0 + L1 7.7 + L2 17.4 + L3 25.1
# = 65.1M (D-상대 동일이라 여기도 같음) → esoteric ~207B/cell ≈ 13.5GB, 24GB OK.
D_LU    = 40                                         # prop diameter [cells on L0]
EXTENTS = [(0.65625, 1.84375, 1.15625),              # L1
           (0.4,     1.0,     0.84375),              # L2
           (0.125,   0.25,    0.6875)]               # L3 (rotor disk)
_DOM    = {"nx_D": 6.5, "ny_D": 6.0, "nz_D": 6.0, "hub_x_D": 1.5}

# tip chord(0.95R) 13.9mm / dx_fine 1.429mm ≈ 9.8 cells; ε=0.25c(tip)=2.44Δx
# (2Δx floor 위), inboard ε≈6.7Δx. n64 δr(활성 스팬 0.178m/64)≈1.9 fine cells ≤ ε ✓

# =============================================================================
# S4. TIP LOSS VARIANTS
# =============================================================================
TIP_LOSS = {
    # GPU 2 시리즈: tip loss function 없음
    "off":  False,
    # GPU 3 시리즈: HVAB 파이널 런 테스트 조건 그대로 (Shen g=0.3, tip만)
    "shen": {"enabled": True, "model": "shen", "g": 0.3,
             "tip": True, "root": False, "eps_offset": False},
}
_TAG = {"off": "notl", "shen": "shen030"}


# =============================================================================
# S5. CONFIG BUILDER
# =============================================================================
def build_config(rpm, tip_loss, n_rev=25):
    """APC 18x8E hover config (주어진 RPM, tip loss variant).

    rpm      : float — 회전수. UIUC static 대조점: 2446/3460/4446/5446
               (실험 원값 2446.667/3460.000/4446.667/5446.667의 반올림).
    tip_loss : "off" | "shen" — S4 variants (HVAB 파이널 A/B 축).
    n_rev    : int — production 회전수 (기본 25 = HVAB 파이널 동일).
    """
    omega     = rpm * 2.0 * np.pi / 60.0             # [rad/s]
    tip_speed = omega * R_PHYS                       # [m/s]
    m_tip     = tip_speed / C_S_PHYS

    # 폴라 Re 범위: 0.75R 기준 target, 활성 span 전역(root@2446 ~29k,
    # near-tip@5446 ~170k)을 10k~300k 12-step 보간으로 커버.
    rR, chord, _ = _load_geometry()
    chord_75 = float(np.interp(0.75, rR, chord))
    re_75    = int(round(0.75 * tip_speed * chord_75 / NU_PHYS))

    U_MAX_LU  = 0.1                                  # convective scaling (HVAB 동일)
    STEPS_REV = int(round(np.pi * D_LU / U_MAX_LU))  # 1257
    RHO_LU    = 1.0

    Nx = int(round(_DOM["nx_D"] * D_LU))             # 260
    Ny = int(round(_DOM["ny_D"] * D_LU))             # 240
    Nz = int(round(_DOM["nz_D"] * D_LU))             # 240
    HUB_X = int(round(_DOM["hub_x_D"] * D_LU))       # 60
    HUB_Y, HUB_Z = Ny // 2, Nz // 2                  # 120, 120

    def _box(up, down, lat):
        return {"x_min": HUB_X - int(up * D_LU),  "x_max": HUB_X + int(down * D_LU),
                "y_min": HUB_Y - int(lat * D_LU), "y_max": HUB_Y + int(lat * D_LU),
                "z_min": HUB_Z - int(lat * D_LU), "z_max": HUB_Z + int(lat * D_LU)}
    levels = [{}] + [{"region": _box(*e)} for e in EXTENTS]

    simulation = {"device_mode": "gpu", "precision": "float32", "dimension": 3,
                  "lattice_model": "D3Q27", "collision_model": "cumulant"}
    physics = {"rho": RHO_PHYS, "U_inf": 0.0, "nu": NU_PHYS,
               "L_char": D_PHYS, "flow_direction": [1.0, 0.0, 0.0]}
    grid = {"Nx": Nx, "Ny": Ny, "Nz": Nz, "resolution": D_LU}
    numerics = {"acoustic_scaling": False, "u_max": U_MAX_LU,
                "c_s_phys": C_S_PHYS, "collision": "cumulant"}
    boundaries = {
        "xmin": {"location": "xmin", "method": "eq", "velocity": [0.0, 0.0, 0.0]},
        "xmax": {"location": "xmax", "method": "sponge", "velocity": [0.0, 0.0, 0.0],
                 "density": RHO_LU, "thickness": 20, "strength": 0.1},
        "ymin": {"location": "ymin", "method": "eq", "velocity": [0.0, 0.0, 0.0]},
        "ymax": {"location": "ymax", "method": "eq", "velocity": [0.0, 0.0, 0.0]},
        "zmin": {"location": "zmin", "method": "eq", "velocity": [0.0, 0.0, 0.0]},
        "zmax": {"location": "zmax", "method": "eq", "velocity": [0.0, 0.0, 0.0]},
    }

    # E63 + NACA4412 neuralfoil Re-보간 덱 (APC 9x4.5MR 스윕과 동일 경로).
    def _nf(name):
        return {"method": "neuralfoil", "airfoil_name": name,
                "Re_target": re_75, "Re_min": 10000, "Re_max": 300000,
                "Re_steps": 12, "mode": "asb", "ncrit": 9.0}
    airfoil_polar = {
        "method": "multi",
        "default": "e63",
        "airfoils": {"e63": _nf("e63"), "naca4412": _nf("naca4412")},
    }

    actuator_line = {
        "enabled": True,
        "rotor": {
            "rpm": float(rpm), "radius": R_PHYS, "omega": omega,
            "n_blades": N_BLADES,
            "hub_center": [HUB_X, HUB_Y, HUB_Z],     # L0 LU 좌표
            "rotation_axis": [1, 0, 0],
            "thrust_direction": [-1, 0, 0], "theta_0": 0.0,
            "blade": {"sections": [
                {"r": r, "chord": c, "twist": tw, "airfoil": af, "active": act}
                for r, c, tw, af, act in _blade_sections()]},
            "grid": {"n_radial": N_RADIAL,
                     "marker_distribution": "uniform",
                     "cosine_side": "both"},
            "epsilon_chord_factor": 0.25,
        },
        "gaussian_cutoff": 3.0, "rho_ref": 1.0, "coeff_mode": "rotorcraft",
        "ramp_steps": STEPS_REV,
        "prandtl_loss": TIP_LOSS[tip_loss],
        # HVAB 파이널 런 포뮬레이션 (archB): Kleine straight + inviscid target
        "eps_correction": {"enabled": True, "method": "kleine",
                           "wake": "straight", "rebuild_every": 1,
                           "wake_markers": "all", "target": "inviscid",
                           "smooth": 2},
        "sampling": {"mode": "gaussian"},
        "spreading": {"radial_truncation": True},    # Merabet tip/root truncation
    }

    mlg = {"enabled": True, "num_levels": len(EXTENTS) + 1, "overlap_width": 2,
           "interpolation": "cubic", "filter_level": 1, "levels": levels}
    sgs = {"enabled": True, "model": "dyn_smag"}
    time = {"max_steps": n_rev * STEPS_REV, "output_interval": STEPS_REV,
            "logging_interval": max(1, STEPS_REV // 20),
            "checkpoint_interval": 5 * STEPS_REV,
            "conservation_interval": max(1, STEPS_REV // 2)}

    folder = "result_apc18x8e_hover_%04drpm_mlg4_%s" % (round(rpm), _TAG[tip_loss])
    output = {"output_dir": "./%s/vtk" % folder,
              "checkpoint_dir": "./%s/checkpoints" % folder,
              "csv_dir": "./%s/csv" % folder, "clear_previous": True,
              "vtk": {"enabled": True, "precision": "float32",
                      "variables": ["density", "pressure", "velocity",
                                    "velocity_magnitude"]},
              "checkpoint": {"enabled": True, "keep_last_n": 2}}

    return {"simulation": simulation, "physics": physics, "grid": grid,
            "numerics": numerics, "boundaries": boundaries,
            "internal_geometry": {"type": "none"}, "mlg": mlg, "sgs": sgs,
            "airfoil_polar": airfoil_polar, "actuator_line": actuator_line,
            "conservation": {"enabled": True, "verbose": 0, "log_to_csv": True},
            "convergence": {"enabled": False},
            "force_calculation": {"enabled": False}, "output": output,
            "time": time}


if __name__ == "__main__":
    print("  [APC 18x8E HOVER — tip-loss A/B sweep]")
    rR, chord, twist = _load_geometry()
    c75 = float(np.interp(0.75, rR, chord))
    print("  R=%.4f m, c(0.75R)=%.4f m, %d blades, root cut r/R=%.4f, "
          "E63→NACA4412 @ r/R=%.4f" % (R_PHYS, c75, N_BLADES,
                                       ROOT_CUT_RR, AF_SWITCH_RR))
    print("  sections=%d, active=%d, n_radial=%d"
          % (len(rR), int(np.sum(rR >= ROOT_CUT_RR)), N_RADIAL))
    print("\n  RPM    M_tip   V_tip[m/s]  Re_75     steps/rev  steps(25rev)")
    for rpm in (2446, 3460, 4446, 5446):
        om = rpm * 2 * np.pi / 60.0
        vt = om * R_PHYS
        re75 = 0.75 * vt * c75 / NU_PHYS
        spr = int(round(np.pi * D_LU / 0.1))
        print("  %4d   %.3f   %8.2f   %8.0f   %6d     %7d"
              % (rpm, vt / C_S_PHYS, vt, re75, spr, 25 * spr))
    print("\n  UIUC static 대조(프로펠러 conv.): CT_prop=(π³/4)·CT_rc, "
          "CP_prop=(π⁴/4)·CP_rc")
    for k in ("off", "shen"):
        cfg = build_config(2446, k)
        print("  variant %-4s → prandtl_loss=%s" % (k, cfg["actuator_line"]["prandtl_loss"]))
