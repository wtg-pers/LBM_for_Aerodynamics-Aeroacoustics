# ALM 팁 sweep 정교화 — 단계별 계획 (patch_notes/alm_tip_sweep_refine)

배경: 현행 sweep은 aero-only(`u_aero=u_rel·cosΛ`를 Re/Mach/동압에 적용, 마커는
직선). 감사 결과(2026-07-06):
- 코딩 버그 없음(Λ=0 identity, cos 우함수, 일관 적용).
- 동압/Mach cos²Λ는 팁에서 u_n≪u_t라 0.1% 이내로 정확.
- **누락 1**: α를 LE-법선 평면으로 미투영. 올바른 독립원리 φ_n=atan2(u_n,u_t·cosΛ)는
  팁 φ를 0.3~0.5° 키움 → α 그만큼 작아져야(팁 de-load). 코드는 안 함.
- **누락 2**: 마커가 실제로 후퇴 안 됨(직선). 팁 offset ΔY_tip=1.911in ≈ 9 fine cell
  ≈ 2.2 ε_tip → 해상 가능, wake/팁와류 origin·방위각 위상 미반영.

목표: **기하 후퇴 + cosΛ 유지 + α LE-법선 투영** 세 요소를 동시에. 기본 OFF로
기존 task3 run은 byte-identical 유지(opt-in).

핵심 물리(로터 팁 sweep은 로터面 내):
- 축방향 u_n은 LE(로터面 내)에 항상 수직 → cosΛ 감소 대상 아님.
- 접선 rel 성분 u_t=(ωr−u_θ)만 cosΛ로 감소: u_t_n=u_t·cosΛ.
- V_n=√(u_n²+u_t_n²) (≈u_rel·cosΛ, 팁서 0.1% 이내), φ_n=atan2(u_n,u_t_n),
  α_n=twist−φ_n.

---

## Step 1 — 기하 마커 후퇴 (blade.py)  ← 이번 구현
- `generate_markers`: 누적 LE 접선변위 `marker_sweep_offset[j] =
  Σ_{k≤j} tan(Λ_k)·dr_k` [length] 계산·저장. Λ=0 내측은 0. 팁 ≈ tan30°·swept_span.
- `get_marker_positions(θ)`: `sweep_geometric` True면 base 위치에
  `(sweep_sign·offset_j)·ê_θ(θ)` 가산. `sweep_sign=−sign(ω)`(후퇴=−운동방향).
  샘플·spread은 이 위치를 자동 사용. 반경 성장 0.04%, 방위각 lag 1.6°(무시).
- Blade 속성 `sweep_geometric:bool=False`, `_sweep_sign:float=-1.0`. 기본 OFF.
- 검증: HVAB 블레이드 생성 → 팁 offset 크기 확인, OFF 시 위치 bit-identical.

### Step 1 상태: 완료 (2026-07-06, blade.py)
- `marker_sweep_offset` = cumsum(tan Λ·dr), `sweep_geometric`/`_sweep_sign` 추가.
- `get_marker_positions`에 `(sweep_sign·offset)·ê_θ(θ)` 가산.
- 검증(scratchpad/test_sweep_geom.py): OFF→bit-identical, offset∥−ê_θ(dot −1.0),
  tip 반경 +0.024%·방위각 lag −1.25°(무시), inboard offset=0.
- **발견**: tip offset 1.44in < TM 1.911in — Λ 선형보간이 sweep break를 매끄럽게
  함(0.95→0.975 램프). aero cosΛ와 self-consistent. **해결=Step 3 config에서
  break 근방(0.95023) 섹션 조밀 배치**로 near-step화 → 기하·aero 동시 sharpen.
- 미결(Step 3): Kleine/Dağ 자유후류가 마커 위치를 참조 → sweep_geometric과 병용 시
  후류 필라멘트 origin도 후퇴 위치 써야 하는지 확인.

## Step 2 — LE-법선 aero (actuator_line.py `_compute_bem_forces`)
- 헬퍼 `_le_normal(u_n, u_tan_rel, cosΛ, twist)`→(V_n, α_n_deg, φ_n_deg).
- `u_aero=u_rel·cosΛ` → `u_aero=V_n`, polar α=`alpha_deg`→`α_n`. 투영 φ도 φ_n 사용.
- smearing 재계산 경로(1311-1318)도 동일 헬퍼로.
- Λ=0 → V_n=u_rel, α_n=α, φ_n=φ → byte-identical.
- 반경방향 힘 성분(sinΛ)은 직선-마커 일관성상 무시(문서화).

### Step 2 상태: 완료 (2026-07-06, actuator_line.py)
- `_le_normal_resolve(u_n,u_rel,phi,twist,cosΛ)`→(V_n,φ_n,α_n) staticmethod 추가.
- `_sweep_alpha_normal` 플래그(기본 False). ON일 때만 초기 pass·smearing pass 모두
  LE-법선 해(u_aero=V_n, polar α=α_n, 투영 φ=φ_n) 사용. OFF=legacy cos²Λ·α불변.
- 검증(test_le_normal.py, test_bem_sweep.py):
  - Λ=0 identity V_n=u_rel(1e-8), α_n=α. inboard on-vs-off |ΔF_n|=7.6e-19(byte-id).
  - Λ=30° 팁: V_n≈legacy(<0.05%, 동압 불변), α_n −0.3~−0.5°, **F_n −5~−17%**(팁 de-load).
- 미결: `_kleine_w_corr` 내부 u_aero0는 여전히 legacy cosΛ form — alpha_normal+Kleine
  병용 시 일관성 Step 3에서 확인(희소 조합).

## Step 3 — config 배선 + 검증 (_hvab_hover_base.py, rotor/actuator factory)
- `tip_sweep`: 기존 bool(=aero-only, 하위호환) 유지 + dict 확장
  `{"aero":True,"geometric":True,"alpha_normal":True}`. bare True=현행.
- create_actuator_line_from_config / Rotor.from_config → blade.sweep_geometric,
  _sweep_sign(=−sign ω) 세팅.
- 초소형 smoke: task3 sweep run에서 팁 하중 A/B(현행 vs 3요소), Λ=0 identity.

### Step 3 상태: 완료 (2026-07-06)
- `_hvab_hover_base.build_config`: `tip_sweep` bool(True=full 3요소)/dict 파싱 →
  `sweep_geometric`·`sweep_alpha_normal` 키 + 섹션 sweep 게이트(`_sw_active`).
- `create_actuator_line_from_config`: `model._sweep_alpha_normal`, LU 블레이드에
  `sweep_geometric`/`_sweep_sign=−sign(ω)` 세팅(to_lattice_units가 플래그 미전달).
- **결정 반영**: task3_c*(=tip_sweep=True)는 자동으로 full 3요소 ON(파일 수정 불필요).
- 검증(test_step3_config.py): 5개 tip_sweep 형태 플래그 정확, model 플래그·LU offset
  0.433lu(1.44in)·위치 shift=offset 일치. 非task3(tip_sweep=False 기본) 무영향.

### Step 3b 상태: sharp-break 정합 완료 (2026-07-06, method 2)
- 사용자 확인: sweep은 0.95023→1.0을 잇는 **직선 LE, Λ=30° 일정**(램프 아님).
- TM 표(twist/chord/airfoil) 대조: **현재 config가 이미 정확 재현**(twist<0.01°,
  chord 정확, airfoil 9 station 전부 일치) → 기하 섹션 불변.
- 버그=sweep을 섹션 선형보간(0.95:0°,0.975:30°)해 0→30 램프로 뭉갬 → offset 1.44in.
- 수정: `sweep_break_rR`/`sweep_tip_deg`(blade 속성, config→from_config→to_lattice_units
  전달). generate_markers가 **Λ=step**(30° 일정) + **Δs=tan30·max(r−r_break,0) 정확
  선형**. 섹션 안 건드려 非sweep run byte-identical(검증: tip_sweep=False→break키 없음).
- 검증: swept 마커 Λ={30.0} 일정, 최외곽 마커(r/R 0.9922) offset 1.61in, r/R=1.0
  외삽 시 정확 **1.911in**(TM). (마커가 tip에 정확히 놓이려면 endpoint 분포)
- **Kleine+alpha_normal 일관성(보류)**: `_kleine_w_corr` 내부 u_aero0는 legacy cosΛ
  form. eps_correction=kleine와 alpha_normal 병용(희소)시만 관련.
- **task3 A/B 실행**: 클러스터 run은 사용자 몫(로컬은 config/build/초소형 smoke만).

## 순서 원칙
각 Step 후 검증·정리·체크인. 자동 진행 금지. sweep 3요소 완료 → 다음 ① 비등방 Gaussian.
