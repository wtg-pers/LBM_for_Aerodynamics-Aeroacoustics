# P2 · Stage 3 — correction on-device (solve·polar·velocity-triangle GPU) — 2026-07-08

05 설계 S3. 목표: Kleine correction을 **완전 on-device** → A를 CPU로 안 내림(S2의
B D2H 제거), solve(10ms)·polar(3.8ms) GPU화. 게이트=CV-band(05 §5 결정).

## 현황 (S2 후, 클러스터 실측)
bem 44.6 = freewake 29.5(GPU) + **solve 10.2(CPU)** + **polar 3.8(CPU)** + sample 6.6(GPU) …
- S2는 B만 GPU→CPU 내려서 `A = B@E_used`·`correct_noniterative` **CPU**. 즉 substep마다
  freewake는 GPU, correction은 CPU → **B D2H + A·u_n·polar CPU** 잔존.
- S1서 `_C81Table` GPU bilinear·`cubic_slope_4pt` 준비됨(polar 코어는 GPU-ready).

## S3 범위 (3 파트)

### S3a — `smearing_correction.py` correction xp化
- `_triangle`(sqrt/arctan2) + `correct_noniterative`(A@Γ, `eye`, **`xp.linalg.solve`**,
  where) → xp-agnostic. cl_eval/dcl_eval 콜백은 cupy 반환(아래 S3b). xp는 A/r서 파생.
- `influence_matrix`(straight fallback)도 xp化(cold-start용).

### S3b — `actuator_line.py` polar xp化 + 배선
- `_lookup_cl_cd` xp化: `CL/CD=xp.zeros`, valid 마스크, airfoil 그룹 인덱싱(`flatnonzero`
  /`arange`), 배치 쿼리(qf=S1 GPU LUT). scalar-only 폴라(CSV/NeuralFoil)는 GPU 불가 →
  그 경우 correction만 CPU 유지(HVAB=C81이라 무관).
- `_kleine_w_corr`: freewake B를 **CPU로 안 내림** → `E_used`도 GPU(cache), `A=B@E_used`
  GPU. `u_n,u_tan,Gprev,r,chord,eps,twist,active` GPU 업로드(static는 1회 cache).
  `correct_noniterative` GPU 실행 → `w_corr` GPU. `ALM_CORR_GPU` 토글.

### S3c — velocity-triangle GPU (`rotor.py`)
- `compute_relative_velocity`/`recompute_velocity_triangle` xp化 → **u_markers GPU 유지**
  (sample D2H 제거의 선행). geometry(rotation frame) cupy 미러. 배치는 kinematics/polar에.
- (freewake는 per-blade 유지 — 후류가 블레이드별 상이.)

## 게이트 (CV-band)
- **로컬**: `correct_noniterative` GPU==CPU (합성 cl_eval, linalg.solve 포함) <1e-11;
  `_lookup_cl_cd` GPU==CPU (실 C81) <1e-12.
- **클러스터**: bench5 rev2 C_T **CV-band(±~3%)** + drift~0 + 안정성. solve→↓, polar→↓,
  bem↓, s/step↓, **util↑**(B D2H·CPU correction 소멸분). A/B `ALM_CORR_GPU=0`.

## 리스크
- **cupy.linalg.solve(n=48)**: 소형 → 런치비. per-blade 4회. 필요시 batch block-diag.
- **_lookup_cl_cd 그룹핑 GPU**: multi-airfoil 인덱스 gather. C81 매니저 쿼리가 cupy alpha
  받는지 확인(S1 `_C81Table` OK, 매니저 래퍼 점검).
- solve 10ms의 대부분이 폴라조회(5×)라 → polar GPU화가 solve도 같이 줄임.
- ★per-call ring 업로드(S2 `[cp.asarray(rg) for rg in rings]` = L~50 개별 H2D) 비효율
  발견 → **S4서 ring 상주로 제거**(또는 stack 1회 H2D 단기 개선).

## 진행

### S3a — correction xp化 ✅ (2026-07-08, 로컬 검증)
- `smearing_correction.py`: `_triangle`(sqrt/arctan2) + `correct_noniterative`
  (`xp.eye`/`xp.linalg.solve`/`where`, xp는 A→r서 파생) xp-agnostic. cl/dcl 콜백은
  cupy 반환 가정. `influence_matrix`(cold-start straight fallback)는 CPU 유지(GPU 경로선
  A 항상 전달, 미호출).
- 로컬 게이트 PASS (`scratchpad/s3a_correction_gpu_gate.py`, 실 influence_matrix A +
  linalg.solve): u_n_c/alpha_c/Gamma_new/w_corr GPU vs CPU **max|Δ| ~1e-13**
  (linalg reduction 순서차, CV-band 내). 
- ★**라이브 무영향**: `_kleine_w_corr`는 아직 numpy A 전달(S2의 `cp.asnumpy(B)`) →
  correct는 CPU 경로(xp=np, 기존과 동일). S3b서 A를 GPU 유지로 flip해야 win 발생.

### 다음 — S3b (polar xp + 배선) → S3c (velocity-triangle)
S3b가 A를 GPU 유지 + `_lookup_cl_cd` xp + correct GPU 실행으로 flip → 첫 cluster-testable
(solve↓, polar↓, B D2H 소멸). S3c(velocity-triangle)는 u_markers 상주(sample D2H 제거).
