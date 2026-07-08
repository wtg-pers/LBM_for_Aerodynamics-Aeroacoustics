# P1a — ALM 'solve' 병목 제거: dcl_eval 배치 벡터화 — 2026-07-08

03_p1_alm_freewake_measure의 클러스터 측정으로 피봇. 측정이 결정 트리를
바꿈: freewake 단독 지배가 아니라 **solve ≈ freewake (반반)**, 그리고 solve의
정체는 linalg가 아니라 **유한차분 폴라 조회 루프**로 판명 → GPU 불필요, 순수
CPU 벡터화로 해결.

## 측정 (사용자, 클러스터 1×4090, bench5, 정상상태 n=16000)

```
[ALM prof]     sample=6.9  bem=191.0  spread=0.4  ms/call (per fine sub-step)
[ALM prof.bem] freewake=87.4  solve=98.6  polar=3.9  other=1.1 ms/call
               [nb=4 nc=48 ne=49 L=2 ns_full=98]
[freewake] ns_kept=2501 (after prune)  Biot-Savart nc*ns_kept*3 = 360144
```

- bem 191 ms = wall(3580)의 84%. 그 안: **solve 99 ≳ freewake 87** (52:46), polar 4.
- freewake: 이미 벡터 numpy(`_seg_vz_batch`), ns_kept=2501 → GPU "viable"하나 **#2**.

## 근본원인 — solve 99 ms는 `dcl_eval`의 마커별 Python 루프

`correct_noniterative`의 `np.linalg.solve`는 **n=48 → 마이크로초**. 99 ms는
전부 내부 `dcl_eval` = `lift_curve_slope_batch`(`polar_slope.py:119-130`):

```python
for j in range(n):                    # 마커별 Python 루프
    lift_curve_slope(...)             # 마커마다:
      #  _cl_at × 4   (스칼라 C81 조회 4회: ±1.5δ, ±0.5δ)
      #  np.polyfit(xs, cls, 3)       (마커마다 SVD 최소제곱)
```

sub-step당 **4블레이드 × 48마커 × 4오프셋 = 768 스칼라 폴라 dispatch + 192 polyfit**.
반면 `polar` phase(3.9ms)는 같은 조회를 `_lookup_cl_cd`로 **배치**함
(docstring: "~2×192 dispatch 제거, 배치 C81 == 스칼라 bit-identical",
`actuator_line.py:622-629`). `dcl_eval`만 옛 스칼라 경로에 방치돼 있었음.

## 수정 (`src/actuator/actuator_line.py`, `_kleine_w_corr`의 `dcl_eval`)

4개 오프셋을 **`cl_eval`의 배치 경로**로 조회(그룹/Mach-aware) + **단일 multi-RHS
`np.polyfit`**:

```python
offs = np.array([-1.5, -0.5, 0.5, 1.5])          # × δ=1.0 deg
cls  = np.stack([cl_eval(a_deg + o) for o in offs])   # (4, n)  ← 배치 4회
xs   = offs * (np.pi/180)
coeffs = np.polyfit(xs, cls, 3)                  # (4, n) 한 번
return np.where(active, coeffs[2], 0.0)          # p'(0) = 기울기 [1/rad]
```

- `ALM_DCL_BATCH=0` → 옛 스칼라 `lift_curve_slope_batch` 복귀(A/B·bit 확인용).
- multi-RHS polyfit은 컬럼 독립(공유 Vandermonde, RHS만 다름) → 마커별 polyfit과 동일.

## 로컬 검증 (완료)

- **수치 동일성** (`scratchpad/dcl_equiv_test.py`, 실제 NASA RC4-10 C81 덱, n=48):
  old(스칼라 per-marker) vs new(배치) **max abs diff = 0.00e+00 (바이트 동일)**,
  inactive→0 양쪽 일치.
- **속도** (동일 스크립트, nb=4·n=48, RTX3090 데스크톱 CPU): dcl per sub-step
  **38.3 → 0.81 ms = 47×**. (배치 C81 조회 + polyfit 1회 vs 768 스칼라 + 192 polyfit.)

## 예상 효과 (클러스터)

solve 99 → cl_eval(배치) + 소 linalg + 극소 dcl ≈ **~10-15 ms**.
→ bem 191 → **~103**, wall 3580 → **~2100 (≈1.7×)**. freewake 87이 새 #1.
GPU-util 10% stall(=이 CPU 직렬 폴라 루프)도 동반 상승 기대.

## 클러스터 게이트 (사용자 실행)

수치 바이트 동일이라 **bench5 물리(rev2 CT) 회귀 + 프로파일 재측정**:

```bash
ALM_PROFILE_BEM=1 python main.py --gpu 0 --config config/hpc_bench/bench5_baseline.py 2>&1 | tee bem_prof_p1a.log
```
- **물리 게이트**: rev2 tail C_T ≈ 0.00915(ref step975 저점)·drift ~0 유지.
- **속도 게이트**: `[ALM prof.bem]` **solve 99 → 대폭↓**, bem 191 → ~103, wall↓, util↑.
- 의심 시 A/B: `ALM_DCL_BATCH=0` 재현 → 동일 물리 확인(솔브만 느려짐).

## 클러스터 게이트 결과 (2026-07-08) — ✅ 통과

bench5 완주 (1006 step, 1×4090), 정상상태 rev 1.97, n=16000:

| 항목 | 이전 | 패치 04 후 | 배율 |
|---|---|---|---|
| solve | 98.6 | **10.2 ms** | 9.7× |
| freewake | 87.4 | 87.5 (불변, 새 #1) | — |
| bem 전체 | 191.0 | **102.6 ms** | 1.86× |
| s/step | 4.4 | **3.00 s** | 1.47× |
| bench5 완주 | 60:41 | **37:02** | **1.64×** |
| **C_T (물리)** | 0.00915 | **0.00915** | ★바이트 무회귀 |
| drift | ~0 | −0.000% | ✅ |

로컬 Δ=0 예측대로 **물리 완전 동일 + bench5 61→37분**. solve 10.2ms = cl_eval+dcl
배치조회(5× `_lookup_cl_cd`)가 잔여 대부분 → S2/S3 GPU polar가 더 줄일 여지.
**freewake 87.5 = bem의 85%로 단일 지배** → 다음 레버 확정.

## 다음 (P1b — freewake 87 ms)

이제 freewake가 #1. 두 레버:
1. **`eps_correction={"rebuild_every": N}`** — 영향행렬 rebuild를 N substep마다
   (config, 물리근사, 즉효, production용). freewake ÷N.
2. **CuPy 포팅** — `freewake_influence`/`_seg_vz_batch`/`segment_missing_theta`
   (erf→`cupyx.scipy.special.erf`) GPU 상주. ns_kept=2501·360K → viable. 매 스텝 exact 유지.
   설계: 03_p1 §"GPU 포팅 설계".

## ★스케일 caveat (재확인, 00_phase0 §F)

bench5(11M, ALM-heavy)는 ALM을 과대표집. **Watanabe-fine(207M, ×20)에선 LBM
재지배, ALM 소수** — ALM 비용은 격자무관 고정. → 이 P1a/P1b는 (a) bench5 반복
턴어라운드 1.7×+ 가속, (b) util stall 제거가 주목적. **multi-GPU 목표의 레버는
여전히 Phase 1a(coupling)**. ALM 최적화가 multi-GPU 우선순위를 바꾸지 않음.
```
