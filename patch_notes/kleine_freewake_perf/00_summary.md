# Kleine Phase 2 (free-wake) 성능 패치 — A+B+C

## 동기
`hvab_hover_c10_kleine_free.py`(Phase 2 free-wake)가 `..._kleine.py`(Phase 1
straight)보다 per-step ~3배 느림. 원인 진단(코드 리뷰):

- **무거운 수학은 이미 벡터화됨** (`smearing_correction.freewake_influence` /
  `_seg_vz_batch`는 numpy 배치). 벡터화 누락이 원인이 **아님**.
- 진짜 병목:
  1. **free-wake 영향행렬 A를 매 스텝·매 블레이드 전체 재빌드** (`freewake_influence`
     ~25ms × 4blade ≈ 100ms/step). Phase 1은 A를 캐시 → 거의 0. **3배의 주범.**
     비용은 wake 길이(segment 수 ≈ ne·L)에 비례 → L→n_w=50으로 차며 증가.
  2. **wake 이류 trilinear 샘플을 ring마다 개별 호출** (`FreeWake.convect`의
     `for i in range(len(rings))`) + GPU에서는 ring마다 별도 `asnumpy`(tiny D2H).
     스텝당 최대 n_w×n_blades≈200회 작은 런치/동기 → latency-bound.
  3. **`_gradient_matrix(r)`를 매 스텝 재계산** (r은 고정인데).
- 행렬 크기(n~50, ns~2500)가 작아 **커스텀 CUDA 커널은 불필요** — 이득은
  재빌드 throttle + 배치 + 캐시에서 나옴 (FLOPs 문제 아님).

## 패치 내용
세 가지 모두 `src/actuator/actuator_line.py` (+ factory 파싱):

- **A) A 재빌드 throttle** — `_kleine_w_corr` free 분기에서 `_kleine_A_free[k]`
  캐시 사용. `_kleine_rebuild_every` 스텝마다만 `freewake_influence` 재빌드, 사이엔
  캐시 재사용. 콜드/shape 불일치 블레이드는 항상 재빌드. **config 키
  `eps_correction.rebuild_every`** (factory `max(1, int(...))`), **기본 1 = 매 스텝
  (정확, 기존 Phase 2와 bit-identical)**. 스텝 카운터 `_kleine_wake_steps`는
  `_convect_and_shed_wake`에서 스텝당 1회 증가. *(>1은 근사 — wake가 천천히 수축하므로
  사이 스텝의 A 변화 작음. 근사 오차는 기존 free와의 편차로 확인.)*
- **B) 이류 배치화** — `_convect_and_shed_wake` 재작성: 모든 블레이드의 모든 ring 점을
  `(M,3)` 하나로 concat → **단일 `_sample_trilinear` 호출 + 단일 D2H** → reshape 후
  `+ dt·v`. per-point gather라 **bit-identical** (검증 통과). shed 순서/타이밍 불변.
- **C) `_gradient_matrix` 캐시** — `_kleine_G[k]` (r 고정 → blade별 1회). bit-identical.

`FreeWake.convect`(smearing_correction.py)는 더 이상 프로덕션 호출자 없음 — 레퍼런스/
교차검증용으로 보존(삭제 안 함).

## 신규 config
`configs/hvab/hvab_hover_c10_kleine_free_fast.py` — `..._kleine_free.py`와 물리 동일,
`rebuild_every=10`만 추가. 기존 free 결과와 A/B 비교용 (팁 φ/α·CT·FM 일치하면 근사 안전).

## 검증
- **B bit-identical**: `scratchpad/test_batch_convect.py` — 배치 vs ring별 convect
  `np.array_equal` 통과 (2 blade, 5+3 ring, 합성장).
- **end-to-end smoke**: smoke(D16/CPU/2-level) free + `rebuild_every=2`로 10스텝 →
  crash 無, drift 0%, T_lu/P_lu/C_T finite. 재빌드+재사용 분기 모두 통과.
- **K=1 회귀(논리)**: B·C는 bit-identical 변환, A는 K=1에서 매 스텝 재빌드 → 동일 A.
  따라서 패치 후 K=1 = 기존 Phase 2와 동일 → 사용자가 돌리는 기존 free 결과는 유효한
  baseline으로 유지됨 (다시 돌리면 같은 물리, B+C로 다소 빨라짐).

## 비교 가이드 (사용자 클러스터)
1. 기존 `hvab_hover_c10_kleine_free.py` (=K1, 현재 대기 중 결과) ← baseline
2. `hvab_hover_c10_kleine_free_fast.py` (=K10) 신규 실행
3. 팁 φ/α·CT·FM·fallback 빈도 비교 + per-step wall-time.
   - 일치 → K10 채택(또는 K25/50로 더 공격적). 편차 크면 K 낮춤.
