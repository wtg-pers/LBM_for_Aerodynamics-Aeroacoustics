# ALM/Wake canonical 단일축 리팩터 — 계획 & Step 1

설계 근거: `_jobs/0707_analysis_update_process/ALM_axis_convention_audit_kr.md`
동기: free-wake 팁 부호 확정 조사 중, "축방향"이 6곳에서 제각각·모순 fallback으로 유도됨을 발견.

## 결정된 설계 (사용자 합의, 2026-07-08)

- 사용자 입력은 **`rotation_axis`(기하·회전sense) + `disk_normal`(=thrust_direction, 축 부호 앵커)** 둘.
- 내부 canonical: **`rotor.axial_inflow_dir` = n̂_a = −disk_normal(=−thrust_axis)** = 후류/축유입 방향.
  모든 축·wake 방향의 **단일 소스**. 보정은 `+n̂_a`만 써서 마이너스 없음(부호 곡예 제거).
- `rotation_axis`는 **기하 전담**(θ, ê_r, ê_θ, sign(ω) 회전sense). 축 계산엔 안 씀.
- ★부호 관례 정정(사용자 지적): 관례적으로 rotation_axis는 추력-향, 후류=−rotation_axis.
  HVAB는 비관례적으로 rotation_axis가 후류-향. 구코드 `u_n=u·rotation_axis`가 이 비관례
  라벨링을 암묵 강제했음 → u_n을 n̂_a로 옮기면 라벨링 무관해짐. assert는 **부호 무관**이어야 함.

## 2단계 계획 (stepwise; 각 단계 HVAB bit-identical)

### ✅ Step 1 (완료, 이 커밋) — 보정 단일소스화
- `rotor.axial_inflow_dir` property 추가 (rotor.py, `compute_relative_velocity` 앞).
  `n̂_a = normalize(−thrust_axis)`; thrust 미설정이면 `rotation_axis`로 통일 fallback.
- 3개 보정 사이트를 `rotor.axial_inflow_dir`로 치환, 모순 fallback 삭제:
  - free-wake 투영 (`_kleine_w_corr`): `−thrust_axis‖−sign(ω)·rot` → `axis = rotor.axial_inflow_dir`
  - prescribed-helix (`_dag_prescribed_helix_wcorr`): `−thrust_axis‖+sign(ω)·rot` → `downwash = ...`
  - straight-wake VIZ (`_dag_straight_wake_viz`): 동일 → `dw = ...`
- **검증**: HVAB(thrust=[−1,0,0]) → n̂_a=[1,0,0]=−thrust_axis → **bit-identical**(스텁 대조 통과).
  구문 OK, orphan(td/_td) 없음. 구 fallback 모순(free −sign(ω)·rot vs helix +sign(ω)·rot) 통일.
- **미변경**: u_n은 아직 `u·rotation_axis`(Step 2에서 이동). 속도삼각형·힘투영·straight(1D) 불변.

### ✅ Step 2 (완료, 2026-07-08) — u_n 라벨링-불변화 + assert
- `Rotor._axial_sign` property 추가 = `sign(rotation_axis · n̂_a)` (∈ ±1, assert가 평행 보장).
- `compute_relative_velocity`: decompose 직후 `u_n = u_n * self._axial_sign` — 축속도를
  rotation_axis(라벨 임의) → n̂_a(후류-양) 기준으로 재참조. u_theta·삼각형 나머지 불변.
  `recompute_velocity_triangle`은 이미 n̂_a-framed u_n을 받으므로 미변경.
- `setup.py`(thrust 로그 직후): **sign-agnostic assert** `|rotation_axis · thrust_axis| > 1−1e-6`
  (샤프트 ∥ disk 법선). 위반 = 설정 오류 → 즉시 실패.
- **검증**: HVAB `_axial_sign=+1` → u_n 불변 = **bit-identical**. 관례 라벨링(rotation=추력향)
  `_axial_sign=−1` → u_n 부호교정(구코드는 여기서 깨졌음) = 이제 지원. assert: HVAB/관례 통과,
  수직(오류) config 즉시 실패. 구문·import OK.
- **효과**: 축의 shaft-label 임의성 완전 제거. u_n·힘투영·모든 wake 보정이 n̂_a 단일 프레임서 정합.

## 회귀 재검증 (Step 2)
Step 2는 코어 삼각형(u_n)을 건드리므로 bench5_kleine_free를 **다시** 돌려 Step 1 결과와 동일
(팁 signature peak@0.992·rolloff=1.00, CT 동일)한지 확인 권장. HVAB _axial_sign=+1이라 동일해야 함.

## 회귀 검증
- ✅ **PASSED (2026-07-08, bench5_kleine_free 3rev, 사용자 클러스터 실행)**: 클린 완료(NaN 無),
  free-wake 팁 signature **peak@r/R=0.992·rolloff=1.00**로 slab5_kleine_free와 동일(팁-stuck/
  up-load 보존). 부호 뒤집혔다면 straight처럼 de-load(rolloff<1)로 나왔을 것 → **무회귀 확정**.
  결과: `.../0708_axis_reg_kleine_straight_free/0708_D_grid_test/d16_bench5_axisreg_kleine_free_csv`.
- (미실행) dag_helix 경로 스모크 — 필요 시 `configs/0708_axis_regression/bench5_dag_helix.py`.
- Step 2 후 할 것: thrust_direction 일부러 rotation_axis와 평행 아니게 준 config가 assert로 즉시 실패하는지.
