# P2 — BEM 전체 GPU 상주 설계 스펙 (design) — 2026-07-08

사용자 결정(2026-07-08): ALM BEM을 **전부 GPU 상주**로 → per-substep
**D2H/H2D 왕복 제거**(16×/coarse), util 10% stall 소멸. 상태: **설계(구현 전)**.
Phase 1a처럼 design-doc first → 단계별 게이트.

## 0. 현황 (감사 결과, 04 이후)

per-substep 흐름: sample(GPU)→**u_markers D2H**→**BEM 191ms 전부 CPU numpy**→
project(CPU)→**F_global H2D**→spread(GPU RawKernel). GPU화된 건 sample/spread(대형
격자 접촉)뿐, BEM 공력코어(~200마커·~2500세그, 작음)는 미포팅.

- bem 191ms = wall 84%. freewake 87(CuPy후보) + solve 99(04서 dcl 벡터화→~13) + polar 4.
- 병목이 CPU라 GPU가 substep당 191ms×16 유휴 → util 10%.

## 1. 목표 & 비목표

- **목표**: `_compute_bem_forces`·freewake·correct·polar·velocity-triangle·project를
  전부 xp(cupy) 상주. u_markers는 GPU에서 안 내려오고, F_global은 cupy로 spread에 직행.
  substep 내 `.get()`/H2D **0회**(진단 로깅만 배치 D2H).
- **비목표(메모리)**: 마커 geometry ~KB, wake ~수십KB, A행렬 48²×4 — **GPU 메모리
  무의미**. 이 작업은 **속도/sync 제거**용, LBM Phase 1a/1b(메모리)와 목적 다름.
- **비목표(과포팅)**: n=48 linalg.solve, 4×4 project 등 초소형은 GPU 런치비 > 계산비.
  단 **상주 유지가 목적**이라 전송 회피 위해 GPU에 두되, 커스텀 커널까지는 불요(CuPy 충분).

## 2. ★크럭스 = polar C81을 GPU LUT로 (검증됨: tractable)

`_C81Table`(`c81_loader.py:166`)은 비균일격자 **clamped bilinear**
(`RegularGridInterpolator(method='linear')`). GPU 이식 = searchsorted bilinear:

```
a=clip(α, aoa[0],aoa[-1]); M=clip(mach, m[0],m[-1])
ia=clip(searchsorted(aoa,a)-1, 0,n_a-2);  im=clip(searchsorted(mach,M)-1, 0,n_m-2)
ta=(a-aoa[ia])/(aoa[ia+1]-aoa[ia]);        tm=(M-mach[im])/(mach[im+1]-mach[im])
C = data[ia,im]*(1-ta)*(1-tm) + data[ia+1,im]*ta*(1-tm)
  + data[ia,im+1]*(1-ta)*tm  + data[ia+1,im+1]*ta*tm
```

cupy `searchsorted`+fancy index로 완전 벡터화. rectilinear multilinear이라
RegularGridInterpolator와 **수치 동일(~1e-15)**.

★**multi-airfoil은 GPU서 더 깔끔**: CPU의 airfoil별 그룹핑 루프 대신, 덱들을
**단일 텐서 `data[n_airfoil, n_aoa, n_mach]`**(공통 α/M 격자로 리샘플) + per-marker
`airfoil_idx` → gather 한 방. 격자가 덱마다 다르면 per-airfoil LUT dict + idx.

## 3. xp-dispatch 전략

sample/spread는 이미 `xp` 인자로 numpy/cupy 무관. BEM도 동일 패턴:
- `self._xp` = u_field 타입서 결정(cupy면 GPU 경로). env `ALM_GPU=0`으로 CPU 강제(A/B·게이트).
- **CPU numpy 경로 보존**(레퍼런스 게이트·fallback). 모든 신규 GPU 코드는 xp-agnostic로.
- 마커 geometry(r,chord,twist,eps,dr,active,sweep,airfoil_idx)는 **static** →
  __init__/regrid서 1회 xp 상주(H2D 1회, per-step 아님). rotor.advance의 위치갱신만 매step.

## 4. Impact surface

| 파일 | 변경 | 스테이지 |
|---|---|---|
| `c81_loader.py` | `_C81Table` GPU bilinear 경로(searchsorted) + 배치 텐서 LUT | **S1** |
| `polar_slope.py` | dcl **closed-form**(cupy엔 polyfit無): 4점 큐빅 도함수 = Fornberg `(c₋₃−27c₋₁+27c₁−c₃)/(24δ_rad)` (polyfit과 수학동일) | S1 |
| `rotor.py` | `compute_relative_velocity`/`recompute_velocity_triangle`/`project_all_forces` xp化 + **전블레이드 배치**(4×48→192); geometry xp 미러 | S2 |
| `actuator_line.py` | `_compute_bem_forces` xp+블레이드배치, `_kleine_w_corr` xp, `_convect_and_shed_wake` xp, **u_markers GPU 유지**, F_global cupy→spread | S2/S3/S4 |
| `smearing_correction.py` | `freewake_influence`/`_seg_vz_batch`/`segment_missing_theta`/`edge_operator`/`influence_matrix`/`correct_noniterative` xp化(erf→`cupyx.scipy.special.erf`) | S3 |
| `blade.py` | marker geometry xp 미러 접근자 | S2 |

## 5. 단계 & 게이트 (클러스터=사용자)

> ★**게이트 철학 (2026-07-08 결정, S2 결과 반영)**: freewake GPU화가 자유후류를 FP 수준
> 섭동 → 적분 C_T가 ~2% 이동(버그 아님, [[06_p2_s2_freewake_gpu]] A/B로 CPU=0.00915 확정).
> 자유후류의 고유 민감성이라 S3/S4도 reduction 순서로 C_T를 조금씩 움직임 → **bit-identical
> 게이트 폐기, CV-band(±~3%) + drift~0 + 정상성(안정 limit cycle)으로 판정**. bench5 C_T는
> 회귀 앵커일 뿐 물리 목표 아님(D=16 축소); 최종 정확도=production 격자+물리 레퍼런스.

- **S1 — GPU polar LUT + closed-form dcl.** 크럭스 먼저. 게이트: CPU RegularGridInterp
  대비 배치/스칼라 **max abs diff <1e-12** (로컬, 실 NASA 덱). dcl closed-form == polyfit.
- **S2 — BEM kinematics xp + 블레이드 배치, u_markers GPU 유지.** freewake 아직 CPU(또는
  Dağ/pure). 게이트: bench5 **pureALM 물리 bit-close** vs CPU(ALM_GPU=0). polar S1 사용.
- **S3 — freewake + correct_noniterative GPU.** Biot-Savart CuPy 상주(rings GPU),
  cupy.linalg.solve(n=48). 게이트: `_freewake_influence_loop` 레퍼런스 대비 수치 일치 +
  bench5 kleine 물리(rev2 CT ±CV).
- **S4 — 완전 상주.** substep 내 D2H/H2D 0. F_global cupy→spread(H2D 소멸), wake GPU 상주.
  게이트: `[ALM prof]` bem↓·**util↑**(주목표), bench5 물리, slab5-smoke 2차앵커.

## 6. 리스크 & 열린 결정

- ★**초소형 배열 GPU 런치비**: velocity-triangle·closed-form dcl·solve(48) 각 ~10-30µs
  런치 × 여러개 × 16substep. 순이득은 **sync 제거**서 나옴 — compute가 같아도 191ms CPU
  유휴가 사라짐. S4서 실측 확인, 필요시 op 융합. (bench5 過표집이나 util은 실측 가치.)
- **cupy.linalg.solve(48)**: CPU보다 느릴 수 있으나 전송회피가 우선. 대안=배치 4블레이드
  block-diag 1솔브.
- **multi-airfoil LUT**: 덱별 격자 상이 → per-airfoil LUT + marker idx gather(§2).
- **결정성**: BEM GPU는 atomics 無 → 결정적(spread atomicAdd만 기존대로 비결정적).
- **CPU 경로 보존 필수**: 레퍼런스 게이트(ALM_GPU=0)·비-CUDA 환경 fallback.
- **wake convect**는 CFD 속도 샘플(u_field GPU) → 이미 GPU 근처, 자연 이식.

## 7. ★스케일 caveat (재확인)

bench5(11M)는 ALM 과표집. Watanabe-fine(207M)선 BEM=격자무관 소수. → 본 작업 이득은
**(a) bench5 턴어라운드 (b) util stall 제거**. multi-GPU 핵심 레버는 여전히 Phase 1a
coupling. 단 util 제거는 규모 무관 유효(Amdahl 직렬구간 소멸).

## 8. 진행

### Stage 1 — GPU polar LUT + closed-form dcl ✅ (2026-07-08, 로컬 검증 완료)
- `c81_loader.py`: 모듈 `_array_module`(cupy 감지) + `_C81Table._bilinear`(xp searchsorted
  clamped bilinear, grid 1회 상주) + `__call__` cupy 분기. **CPU numpy 경로=RegularGridInterp
  그대로**(bit-identical 레퍼런스 보존).
- `polar_slope.py`: `cubic_slope_4pt`(Fornberg 4점 도함수, cupy엔 polyfit無 대체).
- `actuator_line.py`: 가드 `import cupy as cp`, `dcl_eval` xp-분기(numpy→polyfit / cupy→closed-form),
  `cubic_slope_4pt` import.
- **로컬 게이트 PASS** (`scratchpad/s1_gpu_polar_gate.py`, 실 NASA RC4-10, N=192, cupy 13.6 RTX3090):
  - GPU bilinear vs RegularGridInterp: **max|Δ|=4.4e-16** (기계정밀도)
  - numpy bilinear vs RegularGridInterp: 4.4e-16
  - closed-form dcl vs np.polyfit (CPU·GPU): 7.1e-14  (<1e-12 전부 통과)
- ★크럭스(polar GPU LUT) 수치 확정 → S2/S3 진행 가능. 아직 BEM 호출부는 numpy(S2서 xp 전환).

### Stage 2 — freewake Biot-Savart GPU ✅ (2026-07-08, 로컬 검증 완료) — 06 참조
★설계상 S3였으나 패치04 클러스터 결과(freewake=bem의 85%)로 **당겨서 실행**.
- `smearing_correction.py` freewake 파이프라인 전체 xp-agnostic(erf→cupyx), `actuator_line.py`
  `_kleine_w_corr` 호출부 GPU 배선(`ALM_FREEWAKE_GPU`, B만 D2H, correct는 CPU 유지).
- 로컬 게이트 PASS: **vec-cupy vs vec-numpy = 5.6e-16**(GPU==CPU), 레퍼런스 루프 5.6e-16,
  프루닝 0.0. 속도 블레이드당 24.8→4.05ms(6.1×, RTX3090).
- **클러스터 게이트 대기**: freewake 87.5→~20-40ms, bem 102.6→~40-55, C_T=0.00915 무회귀.

### 다음 — Stage 3 (correction on-device) → Stage 4 (완전 상주)
- S3: `correct_noniterative`(solve) + velocity-triangle(`rotor.py`) xp化 → A를 CPU로 안 내림.
  cl_eval/dcl는 S1서 GPU-ready. 전블레이드 배치는 kinematics/polar에 적용(freewake는 per-blade 유지).
- S4: FreeWake.rings GPU 상주(shed/convect) → per-call H2D 제거, F_global cupy→spread,
  substep sync 0 → util 최대화. 게이트: `[ALM prof]` util↑ + slab5-smoke.
