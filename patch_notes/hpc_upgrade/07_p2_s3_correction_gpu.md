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

### S3b — polar xp + correction 배선 ✅ (2026-07-08, 로컬 검증)
- `c81_loader.py`: **`make_c81_polar_query`/`make_c81_polar_query_mach` 래퍼 xp-safe**
  (S1이 `_C81Table.__call__`만 고쳐 놓친 `np.asarray(mach/Re)` → `xp.asarray`; 안 고치면
  cupy 강제변환 TypeError). `_lookup_cl_cd` xp화(zeros/valid/그룹 인덱스 `xp.asarray(gidx)`/배치쿼리).
- `actuator_line.py` `_kleine_w_corr`: `_gpu_corr`(ALM_CORR_GPU 기본 on) 결정 → free-wake
  **B를 GPU 유지**(`A=B@cp.asarray(E_used)`, B D2H 제거) → A+per-marker 입력 `xpc.asarray`
  1회 업로드 → warm-start·closures·`correct_noniterative`·safety-net 전부 xpc → **w만 D2H**
  반환. `_smooth_active` xp-safe화. `_LINALG_ERR`(np/cupy) fallback. profiling sync(계측 정확).
- 로컬 게이트 PASS (RTX3090):
  - `_lookup_cl_cd` GPU vs CPU (실 C81): **2.2e-16**
  - **통합 스모크** `_kleine_w_corr` 전체 GPU 경로 end-to-end 실행·numpy D2H 반환·
    w_gpu vs w_cpu **4.4e-16** (`scratchpad/s3b_kleine_wcorr_smoke.py`) ★디바이스 배관 검증
  - S1/S2/S3a/dcl 전 게이트 무회귀.
- ★기본값이 이제 correction GPU(both freewake+corr on). A/B: `ALM_CORR_GPU=0`.

### 클러스터 게이트 (S3b, 사용자) — CV-band
```bash
PYTHONUNBUFFERED=1 ALM_PROFILE_BEM=1 python main.py --gpu 0 \
    --config config/hpc_bench/bench5_baseline.py 2>&1 | tee bem_prof_s3b.log
# A/B: ALM_CORR_GPU=0 (correction만 CPU=S2 상태)
```
기대: solve 10.2→↓(correction 폴라 GPU), **freewake~29.5·polar~3.8(main, S3c 대상) 유지**,
bem 44.6→소폭↓ 또는 upload 오버헤드로 중립(per-call geometry H2D 다수 — S4 상주로 제거),
**util↑**(correction CPU 직렬 소멸), C_T CV-band(±3%). 순 wall 이득은 S4서 확실.

### ★S3b 클러스터 결과 (2026-07-08) — 성능 실패, 기본 OFF로 되돌림

bench5 완주, rev 1.97 (ALM_CORR_GPU 기본 on이던 상태):

| 항목 | S2 | **S3b(GPU corr)** | |
|---|---|---|---|
| solve | 10.2 | **148.5 ms** | **14.5× 퇴행** |
| freewake | 29.5 | 28.6 | ~유지 |
| bem | 44.6 | **187.4** | 원본(191)급 |
| s/step | 1.52 | **3.81 s** | |
| 완주 | 21분 | **60분** | |
| C_T | 0.00897 | 0.00899 | CV-band OK, drift 0 |

- **원인**: correction=n≈48 **작은 배열**. `cupy.linalg.solve(48²)` + 5× C81조회(각 여러
  tiny 커널) + 호출당 D2H sync = 전부 **런치 오버헤드 지배**. CPU 10ms → GPU 148ms.
- 이는 05 §6 "포팅 역효과(하지 말 것): solve(n=48) GPU 순손해"가 실측 확증된 것.
- **조치**: `ALM_CORR_GPU` 기본 **'0'(OFF)**로 되돌림 → S2 성능(21분) 복귀. xp 코드는 보존
  (opt-in `=1`, 문서화된 negative result). 물리는 CV-band 내라 무해했음, 순전히 속도 문제.

### ★결론 — GPU 상주는 S2에서 멈춘다 (sweet spot)
- **freewake(360K)만 GPU 이득**, correction/velocity-triangle(작은 배열)은 CPU가 빠름.
- **S3c(velocity-triangle)·S4(완전 상주) 폐기**: 같은 작은-배열 문제 반복 → 동일 퇴행 예상.
  게다가 [[00_phase0]] 스케일 caveat: production(207M)선 ALM 자체가 소수라 ROI 없음.
- 유일한 잠재 대안(비추천): correction 전체를 **단일 fused CUDA 커널**로 → tiny-launch 문제
  회피. 그러나 CPU 10ms 비용 대비 대공사 + production서 무의미 → 안 함.
- **최종 채택 상태 = 패치04(dcl) + S1(polar LUT, correction OFF라 미사용) + S2(freewake GPU).
  누적 bench5 61→21분(2.9×), 물리 CV-band.**
