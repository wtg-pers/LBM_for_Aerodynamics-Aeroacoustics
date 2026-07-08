"""
HVAB rotor — HOVER, Cumulant + ALM + MLG.  AIAA Hover Prediction Workshop case.

검증 전환용: tip M=0.65(HART2 0.64 근접), **sectional airloads 공개**(NASA HVAB)라
ALM과 직접 비교 가능.  테이퍼+다종 RC라 ALM **Mach-pass + multi-airfoil** 사용.

    import sys, os; sys.path.insert(0, os.path.dirname(__file__))
    from _hvab_hover_base import build_config
    config = build_config(collective_deg=8.0)              # production
    # config = build_config(collective_deg=8.0, smoke=True) # CPU 배관 테스트

================================  GEOMETRY  ===================================
출처: NASA/TM-2020-5002153 (HVAB Blade Set Definition, NTRS 20205002153) Table 1/2.
상세/provenance: docs/hvab_geometry_kr.md.
  - 4 blades, R = 66.50 in = 1.6891 m, σ ≈ 0.1033
  - chord 5.45 in 등현 → r/R≈0.95023(sweep break)부터 tip 3.27 in 선형 테이퍼 (외측 ~5%)
  - twist 선형 −14.05 deg/R (0.75R 기준 ~0), root fairing(<0.252) 무양력
  - airfoil: RC(4)-12(~0.65) → RC(4)-10(0.70-0.80) → RC(6)-08(0.85-0.95) → RC(6)-08T(tip)
  - 공력 활성 스팬 r/R = 0.25188 ~ 1.0
조건: M_tip 0.60/0.65/0.675(주력 0.65), RPM≈1250, collective 6/8/10/12.76 deg, SLS.

  *** Tip sweep 30°@95%R: ALM 직선 마커선으로 근사(표현 불가) — 한계 명시. ***
  *** Precone: TM Table 1 미확인 → 미적용(HART2처럼; 효과 ~0.2%). ***

==============================  WHY THESE CHOICES  ============================
Mach: HART2와 동일 — LBM 격자는 저Mach 유도/wake만, 팁 천음속은 C81 lookup.
  단 **테이퍼라 등현 Re→M 트릭이 깨짐** → ALM Mach-pass(M=u_rel/a 직접계산)로
  각 단면을 올바른 Mach로 조회.  RC 익형 4종 = airfoil_polar method="multi".
  (이론: docs/alm_mach_pass_theory_kr.md)
Scaling: CONVECTIVE (acoustic_scaling=False, u_max=0.1).  SGS: dyn_smag (Re_tip~1-2M).

RC polar provenance (docs/hvab_geometry_kr.md): RC(4)-10·RC(6)-08 authentic
(asb rc410=NASA-TP-3009 / NASA-TM-4264), RC(4)-12 placeholder(12% 스케일),
RC(6)-08T≈RC(6)-08(tab 근사).  production 전 FileShare로 교체 권장.
"""

import os

import numpy as np

# =============================================================================
# S1. PHYSICAL CONSTANTS (HVAB; NASA/TM-2020-5002153)
# =============================================================================
IN2M        = 0.0254
R_PHYS      = 66.50 * IN2M                           # [m] = 1.68910
D_PHYS      = 2.0 * R_PHYS                           # [m] = 3.37820
CHORD_REF   = 5.45 * IN2M                            # [m] = 0.138430  (등현부)
CHORD_TIP   = 3.27 * IN2M                            # [m] = 0.083058  (팁)
TAPER_START = 0.95023                                # r/R, sweep break = 테이퍼 시작
N_BLADES    = 4
ROOT_CUT_RR = 0.25188                                # 공력 활성 시작 r/R (fairing 끝)
TWIST_RATE  = -14.05                                 # [deg/R] 선형 (Table 2 적합)
TWIST_REF   = 0.75                                   # collective @ 0.75R (built-in ~0)
PRECONE_DEG = 0.0                                    # 미적용 (TBD)

SIGMA       = N_BLADES * CHORD_REF / (np.pi * R_PHYS)  # ~0.1044 (TM 정의상 0.1033)

RHO_PHYS    = 1.225                                  # [kg/m^3] SLS
C_S_PHYS    = 340.3                                  # [m/s] SLS sound speed
NU_PHYS     = 1.461e-5                               # [m^2/s] SLS

# RC 익형 Mach C81 덱 — provenance: docs/hvab_geometry_kr.md. build_config의
# polar_source 인자로 per-config 선택(전역 아님 → 기존 결과 폴더와 충돌 방지):
#   "neuralfoil" (기본): 자체 생성 덱 (data/airfoils/<name>.C81, 9 Mach 0~0.8 × 61 α).
#       기존 8-case는 전부 이걸로 돌림. CL_max 과대·CD 과소 경향.
#   "nasa_overflow": NASA/TM 20250006213 부록 OVERFLOW 2.3d Spalart-Allmaras 덱
#       (data/airfoils/hvab_nasa_overflow/<name>_OVERFLOW_SA.c81, 11 Mach 0~1.0 ×
#       99 α ±180°, Viterna-Corrigan 후실속). 공식·권위 → 검증/비교용.
# [[reference_hvab_cfd_benchmarks]]
_HERE       = os.path.dirname(os.path.abspath(__file__))
_REPO       = os.path.abspath(os.path.join(_HERE, "..", ".."))
def _c81(name, source="neuralfoil"):
    if source == "nasa_overflow":
        return os.path.join(_REPO, "data", "airfoils", "hvab_nasa_overflow",
                            name + "_OVERFLOW_SA.c81")
    return os.path.join(_REPO, "data", "airfoils", name + ".C81")

N_RADIAL    = 48                                     # markers/blade (테이퍼·다익형 해상)


# =============================================================================
# S2. BLADE DISTRIBUTIONS (taper + linear twist + multi-airfoil)
# =============================================================================
def _twist_deg(rR, collective_deg):
    """기하 피치 = collective(@0.75R) + 선형 −14.05 deg/R."""
    return collective_deg + TWIST_RATE * (rR - TWIST_REF)


def _chord(rR):
    """등현 5.45in, r/R>0.95023 부터 tip 3.27in 선형 테이퍼  [m]."""
    if rR <= TAPER_START:
        return CHORD_REF
    f = (rR - TAPER_START) / (1.0 - TAPER_START)
    return CHORD_REF + f * (CHORD_TIP - CHORD_REF)


def _airfoil(rR):
    """반경별 RC 익형 (Table 2 station 전이 경계 중간값)."""
    if rR < 0.675:
        return "RC4-12"
    elif rR < 0.825:
        return "RC4-10"
    elif rR < 0.975:
        return "RC6-08"
    return "RC6-08T"


TIP_SWEEP_DEG = 30.0                                 # TM Table 2: 팁 후퇴각 (ΔY=-1.911in)


def _sweep_deg(rR):
    """LE 후퇴각 Λ [deg]: r/R≤0.95023(sweep break) 0, 외측 후퇴(TM 30°).
    simple-sweep 보정용(폴라 Mach·동압에 cosΛ). 직선 마커선이라 기하 위치는
    못 옮기지만 단면이 느끼는 LE-법선 속도를 cosΛ로 반영."""
    return 0.0 if rR <= TAPER_START else TIP_SWEEP_DEG


def _blade_sections(collective_deg):
    """(r[m], chord[m], twist[deg], airfoil, active).
    r=0 비활성 hub + 공력 활성 (r/R=0.25188~1.0).  테이퍼·익형전이·twist를
    선형보간으로 재현하도록 station 배치(전이 경계·팁 테이퍼 포함)."""
    rR_active = [ROOT_CUT_RR, 0.30, 0.40, 0.50, 0.60, 0.675, 0.70, 0.75,
                 0.80, 0.825, 0.85, 0.90, 0.95, 0.975, 1.00]
    secs = [(0.0, CHORD_REF, _twist_deg(0.0, collective_deg), "RC4-12", False, 0.0)]
    for rR in rR_active:
        secs.append((rR * R_PHYS, _chord(rR),
                     _twist_deg(rR, collective_deg), _airfoil(rR), True,
                     _sweep_deg(rR)))
    return secs


# =============================================================================
# S3. GRID PRESETS
# =============================================================================
# tip chord(3.27in)이 가장 작아 "tip chord ≥16 cell"(ε=0.25c & ε>4Δ) 목표가 빡빡.
# chord_fine_tip[cells] = CHORD_TIP / dx_fine,  dx_fine = (D_PHYS/D)/2^(nl-1).
#     (D_on_L0, [(up, down, lat) per fine level in D-units])
GRID_PRESETS = {
    # 4 levels(3 fine). tip chord cells / 총셀 / ~GB (BYTES_PER_CELL=410):
    "light":   (32, [(0.75, 2.5, 1.25), (0.4, 1.0, 0.75), (0.15, 0.25, 0.6)]),
    #          tip 6.3 cell / 30.5M / ~12.5 GB  → 24GB GPU 안전 (DEFAULT)
    # light_wide: 최정밀 레벨 여유 확대(팁 너머 0.1D→0.3D, wake 0.25D→0.4D).
    # 팁 다운워시 starvation(φ→0 artifact) 진단/수정용. nesting L1⊇L2⊇L3 유지.
    "light_wide": (32, [(0.75, 2.5, 1.25), (0.4, 1.0, 0.9), (0.2, 0.4, 0.8)]),
    #          tip 6.3 cell / ~48M / ~19.7 GB → 24GB GPU (4090) 적합
    "medium":  (40, [(0.75, 2.5, 1.25), (0.4, 1.0, 0.75), (0.15, 0.25, 0.6)]),
    #          tip 7.9 cell / 60.2M / ~24.7 GB → 24GB OOM 위험, ~32GB GPU 필요
    # 5 levels(4 fine) — tip chord ~16 cell. ★nesting 안정화(2026-06-26): 원래
    #   (...,(0.2,0.4,0.7),(0.1,0.2,0.55))는 **L2→L3 lat_sep=0.05D + tip margin 0.05D**로 너무
    #   빡빡 → overlap_width=2 coupling stencil 자리 부족 → 5-level coupling 폭주(step1 NaN,
    #   로터 근처 Level1). medium(sep 0.15D)은 정상. 고침: 모든 interface lat_sep≥0.15D, tip≥0.08D.
    "fine":    (40, [(0.6, 2.0, 1.0), (0.4, 0.85, 0.85),
                     (0.25, 0.5, 0.7), (0.14, 0.22, 0.58)]),
    #          tip 15.7 / eps/dx 3.93 / 208M / ~85 GB → DGX(128GB). (안정nesting으로 셀↑)
    # light_tip5: D32 5-level, **안정 nesting**(원래 light+빡빡슬랩은 NaN). ε/Δx 3.15, 팁 12.6.
    "light_tip5": (32, [(0.75, 2.5, 1.25), (0.4, 1.0, 0.9), (0.25, 0.5, 0.75),
                        (0.15, 0.3, 0.6)]),
    #          tip 12.6 / eps/dx 3.15 / 137M / ~56 GB → 80GB GPU (24/32/40GB 불가)
    # DEBUG: fine과 동일 nesting 비율을 D16로 축소(~13M/~5GB) → 24GB 로컬에서 5-level
    #   coupling 크래시(CUDA_ILLEGAL_ADDRESS @ coarse_to_fine, DGX fine 재현)용. production 아님.
    "fine_mini": (16, [(0.6, 2.0, 1.0), (0.4, 0.85, 0.85),
                       (0.25, 0.5, 0.7), (0.14, 0.22, 0.58)]),
    # DEBUG: 단일레벨(extents 없음) D76 → 8·5·5·D^3 = 87.78M 셀 > 79.5M int32 임계.
    #   레벨당 셀수 천장(q*N+idx overflow) before/after 검증용. SGS off로 24GB 적합.
    "ovf87": (76, []),
    # ★slab5 (2026-07-05): 도메인을 기존 L1 크기로 축소(3.25D×2.5D², hub 상류 0.75D)
    #   + L1을 도메인 거의 전체로 + 로터 디스크 슬랩 L4(dx=L0/16) 추가.
    #   L4 슬랩: lat ±17lu=1.0625R(≈사용자 1.05R 의도), x [hub−1, hub+2]lu = 48 fine
    #   cells 두께 — 하한 근거: ALM Gaussian force/sampling 지지폭 ±3ε(등현부
    #   ε=0.25c=5.7 L4셀 → ±17셀) + overlap. 10셀 두께는 커널이 잘려 불가.
    #   nesting: 전 interface lat_sep≥0.15D(5lu), L3→L4 axial 3lu=0.094D(tip 가이드
    #   0.08D 충족). L1은 outlet sponge(L=20lu, x∈[84,104]) 침범 않게 x_max=83.
    #   ε/Δx: 팁 3.4(2Δx floor 해제·0.25c 지배), 팁 chord 13.6셀 = light_tip5급
    #   해상을 45.3M셀/~18.6GB로 (137M/56GB 대비). 24GB GPU 적합.
    #   ⚠주의: (a)도메인 축소+L4 동시 변경이라 기존 light 결과와 절대 CT/FM 직접
    #   비교 불가(상류 0.75D·측면 1.25D는 호버 재순환/블로케이지 민감 — 재기준선
    #   필요), (b)wall clock ~3.3×(updates/coarse 131M→435M).
    #   셀: L0 0.67M + L1 4.17M + L2 9.37M + L3 13.68M + L4 17.43M = 45.3M.
    "light_slab5": (32,
                    [(0.65625, 1.84375, 1.15625),   # L1 max: x[3,83] y,z[3,77]
                     (0.4,     1.0,     0.84375),   # L2: x[12,56] lat ±27
                     (0.125,   0.25,    0.6875),    # L3: x[20,32] lat ±22
                     (0.03125, 0.0625,  0.53125)],  # L4 slab: x[23,26] lat ±17
                    {"nx_D": 3.25, "ny_D": 2.5, "nz_D": 2.5, "hub_x_D": 0.75}),
    # ★bench5 (2026-07-05): HPC 업그레이드용 초고속 baseline — light_slab5의
    #   5-level 토폴로지(로터 슬랩 L4 포함)를 D=16으로 축소, nesting 여백은
    #   절대 lu 기준 재배치(L0→L1 3lu, 레벨간 sep 3~4lu, 슬랩 두께 3lu=48
    #   fine cells로 ±3ε 지지 유지, lat ±9lu=±1.125R). 물리 정확도 목적이
    #   아니라 커널/MLG/ALM/sponge 전 경로의 회귀·성능 앵커(bit-identical +
    #   CT@2rev + MLUPS). 셀: L0 0.13M + L1 0.41M + L2 1.57M + L3 2.95M +
    #   L4 3.98M ≈ 9.05M / ~3.7GB. 503 steps/rev, 2rev ≈ 수 분(1×4090).
    "bench5": (16,
               [(0.8125, 1.1875, 1.25),     # L1: x[3,35]  lat ±20 (sponge x[36,56] 앞)
                (0.5625, 0.9375, 1.0),      # L2: x[7,31]  lat ±16
                (0.3125, 0.3125, 0.75),     # L3: x[11,21] lat ±12
                (0.0625, 0.125,  0.5625)],  # L4 slab: x[15,18] lat ±9
               {"nx_D": 3.5, "ny_D": 3.0, "nz_D": 3.0, "hub_x_D": 1.0}),
}
DEFAULT_PRESET = "light"   # 24GB GPU 안전. tip ε floor 제한(6.3 cell). 16-cell은 fine(DGX).
BYTES_PER_CELL = 410.0     # 실측(LBM/MLG; dyn_smag 포함 ~동일), CT 메모리모델 기준


# =============================================================================
# S3b. CONFIG BUILDER
# =============================================================================
def build_config(collective_deg, mtip=0.65, smoke=False,
                 smoke_sgs=False, preset=None, use_sgs=True,
                 eps_correction=None, prandtl_loss=True, run_tag="",
                 tip_sweep=False, sampling=None, smoke_levels=2,
                 n_rev=18, polar_source="neuralfoil",
                 marker_distribution="uniform", cosine_side="both",
                 radial_truncation=False, anisotropic=None):
    """HVAB hover config (주어진 collective, tip Mach).

    eps_correction : None|bool|dict  — viscous-core smearing 보정 (기본 off).
        production A/B 권장: eps_correction={"enabled":True,"target":"inviscid"}
        + prandtl_loss=False (보정이 물리적 팁 de-loading을 직접 해상하므로
        경험적 Prandtl과 중복; 둘 다 켜면 팁 이중 차감).
    prandtl_loss   : bool — 경험적 Prandtl tip/root loss (기본 True).
    tip_sweep      : bool — 팁 sweep(30°, 외측 5%) simple-sweep 보정 (기본 False).
        True면 swept 마커가 LE-법선 속도 u_rel·cosΛ로 폴라 Mach·Re·동압 계산
        → 팁 유효 Mach↓(drag-rise 완화)·하중↓. 직선 마커선이라 기하는 못 옮기나
        단면 공력은 cross-flow 원리로 반영. 기존 config 재현 위해 기본 off.
    sampling       : None|dict — 속도 샘플러 모드 (A/B 연구; patch_notes/almlbm_sampler_ab/).
        None 또는 {"mode":"gaussian"} = 현행 ±3ε Gaussian (bit-identical).
        {"mode":"point"}=B-i, {"mode":"mask_disk"}=B-iii,
        {"mode":"aniso","eps_r_factor":0.5}=B-ii. 팁 유입 결손의 샘플링 기여 격리용.
    n_rev          : int — production 회전수 (기본 18 → max_steps=n_rev·STEPS_REV).
        18-rev 케이스를 25로 늘리려면 새 config는 n_rev=25, 기존 결과는 restart로
        (--max-steps 25·STEPS_REV 또는 --extend (25-18)·STEPS_REV).
    polar_source   : "neuralfoil"(기본) | "nasa_overflow" — RC 익형 C81 덱 선택.
        기본=기존 자체 덱(기존 8-case 재현). "nasa_overflow"=NASA 공식 OVERFLOW 덱
        ([[reference_hvab_cfd_benchmarks]]). per-config라 기존 폴더와 충돌 안 함.
    """
    omega = mtip * C_S_PHYS / R_PHYS                  # [rad/s]
    rpm = omega * 60.0 / (2.0 * np.pi)
    tip_speed = omega * R_PHYS

    if smoke:
        # smoke_levels로 nesting 깊이 조절(기본 2=기존동작). 5-level 안정성 재현용.
        _SMOKE_EXT = [(0.5, 1.5, 1.0), (0.3, 0.6, 0.7),
                      (0.2, 0.4, 0.5), (0.12, 0.25, 0.4)]
        _nf = max(1, smoke_levels - 1)
        D, EXTENTS, DEVICE, N_REV = 16, _SMOKE_EXT[:_nf], "cpu", 1
        USE_SGS = smoke_sgs
        _DOM = None
    else:
        preset_name = preset or DEFAULT_PRESET
        _pv = GRID_PRESETS[preset_name]
        # len-3 preset = (D, EXTENTS, DOMAIN): DOMAIN overrides the default
        # 8D×5D×5D box (nx/ny/nz in units of D, hub_x_D from the inlet).
        if len(_pv) == 3:
            D, EXTENTS, _DOM = _pv
        else:
            D, EXTENTS = _pv
            _DOM = None
        DEVICE, N_REV, USE_SGS = "gpu", n_rev, use_sgs
    NUM_LEVELS = len(EXTENTS) + 1

    U_MAX_LU  = 0.1
    STEPS_REV = int(round(np.pi * D / U_MAX_LU))
    RHO_LU    = 1.0

    if not smoke and _DOM is not None:
        Nx = int(round(_DOM["nx_D"] * D))
        Ny = int(round(_DOM["ny_D"] * D))
        Nz = int(round(_DOM["nz_D"] * D))
        HUB_X, HUB_Y, HUB_Z = int(round(_DOM["hub_x_D"] * D)), Ny // 2, Nz // 2
    else:
        Nx, Ny, Nz = 8 * D, 5 * D, 5 * D
        HUB_X, HUB_Y, HUB_Z = 3 * D, Ny // 2, Nz // 2

    def _box(up, down, lat):
        return {"x_min": HUB_X - int(up * D),  "x_max": HUB_X + int(down * D),
                "y_min": HUB_Y - int(lat * D), "y_max": HUB_Y + int(lat * D),
                "z_min": HUB_Z - int(lat * D), "z_max": HUB_Z + int(lat * D)}
    levels = [{}] + [{"region": _box(*e)} for e in EXTENTS]

    simulation = {"device_mode": DEVICE, "precision": "float32", "dimension": 3,
                  "lattice_model": "D3Q27", "collision_model": "cumulant"}
    physics = {"rho": RHO_PHYS, "U_inf": 0.0, "nu": NU_PHYS,
               "L_char": D_PHYS, "flow_direction": [1.0, 0.0, 0.0]}
    grid = {"Nx": Nx, "Ny": Ny, "Nz": Nz, "resolution": D}
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

    # 다종 RC Mach C81 덱 — Mach-pass(M=u_rel/a)가 각 단면 Mach를 직접 전달.
    airfoil_polar = {
        "method": "multi",
        "default": "RC4-12",
        "airfoils": {
            "RC4-12":  {"method": "c81", "path": _c81("RC4-12", polar_source)},
            "RC4-10":  {"method": "c81", "path": _c81("RC4-10", polar_source)},
            "RC6-08":  {"method": "c81", "path": _c81("RC6-08", polar_source)},
            "RC6-08T": {"method": "c81", "path": _c81("RC6-08T", polar_source)},
        },
    }

    # Tip-sweep options (patch_notes/alm_tip_sweep_refine). Accepts:
    #   bool True  = FULL 3-element sweep (aero cosΛ + 기하 후퇴 + LE-법선 α)
    #   bool False = off
    #   dict {"aero","geometric","alpha_normal"} = 개별 A/B 토글
    # aero cosΛ는 sweep이 활성이면 항상 적용(기하/α는 그 위 추가). marker_sweep(Λ)은
    # sweep 활성 시에만 세팅.
    if isinstance(tip_sweep, dict):
        _sw_geom = bool(tip_sweep.get("geometric", False))
        _sw_alpha = bool(tip_sweep.get("alpha_normal", False))
        _sw_active = bool(tip_sweep.get("aero", True)) or _sw_geom or _sw_alpha
    else:
        _sw_active = bool(tip_sweep)
        _sw_geom = _sw_alpha = bool(tip_sweep)     # bare True → full 3-element

    actuator_line = {
        "enabled": True,
        "rotor": {
            "rpm": rpm, "radius": R_PHYS, "omega": omega, "n_blades": N_BLADES,
            "hub_center": [HUB_X, HUB_Y, HUB_Z], "rotation_axis": [1, 0, 0],
            "thrust_direction": [-1, 0, 0], "theta_0": 0.0,
            "blade": {"sections": [
                {"r": float(r), "chord": float(c), "twist": float(tw),
                 "airfoil": af, "active": act,
                 "sweep": float(sw) if _sw_active else 0.0}
                for r, c, tw, af, act, sw in _blade_sections(collective_deg)],
                # Sharp sweep (method 2): constant 30° straight LE from the break
                # to the tip → Λ step + exact linear back-set (TM 1.911in), not the
                # section-interpolated ramp. Only when sweep active.
                **({"sweep_break_rR": TAPER_START, "sweep_tip_deg": TIP_SWEEP_DEG}
                   if _sw_active else {})},
            "grid": {"n_radial": N_RADIAL,
                     "marker_distribution": marker_distribution,
                     "cosine_side": cosine_side},
        },
        "gaussian_cutoff": 3.0, "rho_ref": 1.0, "coeff_mode": "rotorcraft",
        "ramp_steps": STEPS_REV, "prandtl_loss": prandtl_loss,
        "sweep_geometric": _sw_geom, "sweep_alpha_normal": _sw_alpha,
    }
    if eps_correction is not None:
        actuator_line["eps_correction"] = eps_correction
    if sampling is not None:
        actuator_line["sampling"] = sampling
    _spreading = {}
    if radial_truncation:
        # Merabet 2021 tip/root radial truncation + renormalisation of the
        # Gaussian source (default off → bit-identical).
        _spreading["radial_truncation"] = True
    if anisotropic is not None:
        # Natelson 2026 Eq.7 / Churchfield 2017 anisotropic Gaussian force
        # projection. Per-axis width factors × the isotropic ε (c=chord,
        # t=thickness, r=radial); c=t=r=1 reduces exactly to isotropic.
        # anisotropic=True → recommended physical default (chord-optimal ε_c=ε_iso
        # ≈0.25c; thinnest-resolvable ε_t≈0.5ε_iso≈2Δx at the tip since physical
        # 0.25·thickness is sub-grid; span ε_r=ε_iso).
        if isinstance(anisotropic, dict):
            _spreading["anisotropic"] = {
                "enabled": True, "c": float(anisotropic.get("c", 1.0)),
                "t": float(anisotropic.get("t", 1.0)),
                "r": float(anisotropic.get("r", 1.0))}
        elif anisotropic:
            _spreading["anisotropic"] = {"enabled": True,
                                         "c": 1.0, "t": 0.5, "r": 1.0}
    if _spreading:
        actuator_line["spreading"] = _spreading

    mlg = {"enabled": True, "num_levels": NUM_LEVELS, "overlap_width": 2,
           "interpolation": "cubic", "filter_level": 1, "levels": levels}
    sgs = ({"enabled": True, "model": "dyn_smag"} if USE_SGS
           else {"enabled": False, "model": "off"})
    time = {"max_steps": N_REV * STEPS_REV, "output_interval": STEPS_REV,
            "logging_interval": max(1, STEPS_REV // 20),
            "checkpoint_interval": 5 * STEPS_REV,
            "conservation_interval": max(1, STEPS_REV // 2)}

    tag = "smoke" if smoke else "mlg%d_D%d_%s" % (
        NUM_LEVELS, D, (preset or DEFAULT_PRESET))
    if not smoke and not USE_SGS:
        tag += "_nosgs"
    if run_tag:
        tag += "_" + run_tag          # ALM/격자 변형이 baseline 폴더를 덮어쓰지 않도록
    folder = "result_hvab_hover_c%04.1f_M%03d_%s" % (
        collective_deg, round(mtip * 1000), tag)
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
    print("  [HVAB HOVER — AIAA HPW]")
    print("  R=%.4f m, c_ref=%.4f m, c_tip=%.4f m, %d blades, sigma=%.4f"
          % (R_PHYS, CHORD_REF, CHORD_TIP, N_BLADES, SIGMA))
    om = 0.65 * C_S_PHYS / R_PHYS
    print("  M_tip=0.65 → RPM=%.1f, V_tip=%.1f m/s, twist=%.2f deg/R @ %.2fR"
          % (om * 60 / (2 * np.pi), om * R_PHYS, TWIST_RATE, TWIST_REF))
    for src in ("neuralfoil", "nasa_overflow"):
        print("  RC decks [%s]: " % src + ", ".join(
            "%s=%s" % (n, os.path.exists(_c81(n, src)))
            for n in ("RC4-12", "RC4-10", "RC6-08", "RC6-08T")))
    print("\n  preset   D  lvls  dx_fine[mm]  chord_fine_tip[cells]  steps/rev")
    for name in ("light", "medium", "fine"):
        D, EX = GRID_PRESETS[name]
        nl = len(EX) + 1
        dx_fine = (D_PHYS / D) / (2 ** (nl - 1))
        print("  %-7s %3d  %3d  %9.3f  %14.1f        %5d"
              % (name, D, nl, dx_fine * 1000, CHORD_TIP / dx_fine,
                 int(round(np.pi * D / 0.1))))
