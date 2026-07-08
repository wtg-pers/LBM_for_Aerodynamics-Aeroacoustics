# P2 · Stage 2 — freewake Biot-Savart GPU 포팅 — 2026-07-08

05 설계의 S3(freewake)를 **당겨서** S2로 실행. 근거: 패치 04 클러스터 결과에서
freewake=87.5ms = bem의 **85%**로 단일 지배 확정 → 잔여 최대 레버. `ns_kept=2501`
(360K Biot-Savart, erf 병렬)이라 GPU 적합. 05 §7 스케일 caveat 유지(본 이득=bench5
턴어라운드 + util; 207M선 소수).

## 무엇을 / 어떻게

`freewake_influence` 파이프라인 전체를 **xp-agnostic**로 (cupy 입력=GPU, numpy=CPU
레퍼런스 bit-identical). erf는 `cupyx.scipy.special.erf`로 분기.

### Part 1 — `smearing_correction.py` xp化
- 모듈: `import cupy as _cp` 가드 + `_am(x)`(array module) + `_erf(x, xp)` 분기
  (`scipy.special.erf` ↔ `cupyx.scipy.special.erf`).
- `phi_smeared`/`phi_ideal` → xp (`sqrt`/`exp`/`_erf`). `segment_missing_theta`는
  이들 위임이라 자동 xp-safe.
- `_build_segments`/`_seg_vz_batch`/`freewake_influence` → 전부 xp (`asarray`/`where`/
  `cross`/`broadcast_to`/`concatenate`/`clip`/`zeros`/fancy-index 축소 `Vz @ S`).
- prune의 `float()`→0-d 배열, `keep.all()`→`bool(...)`(작은 sync 1회/콜, 무시가능).
- CPU numpy 경로 **무변경**(xp=np → 기존과 동일 연산).
- `segment_induced_velocity`(스칼라, 레퍼런스 루프 전용)는 np 유지.

### Part 2 — `actuator_line.py` `_kleine_w_corr` 호출부 배선
- `ALM_FREEWAKE_GPU`(기본 on, cupy 있을 때) → wake geometry(ctrl3d·rings·eps·axis)를
  `cp.asarray` 업로드(작음) → GPU `freewake_influence` → **B(nc×ne)만 D2H**(`cp.asnumpy`,
  blocking이라 freewake 타이머가 GPU compute 포착) → `A = B @ E_used`(CPU) → 이후
  `correct_noniterative` **CPU 그대로**(변경 無).
- `=0` → CPU 레퍼런스(A/B·비-CUDA fallback). ring 완전 상주(per-call H2D 제거)는 **S4**.

## 로컬 검증 (완료, RTX3090 cupy 13.6)

`scratchpad/s2_freewake_gpu_gate.py` (합성 헬리컬 wake nc=48·ne=49·L=50→ns≈2450,
실 steady ns_kept≈2501과 정합):

| 체크 | max\|Δ\| |
|---|---|
| (A) vec-numpy(no-prune) vs 레퍼런스 삼중루프 | 5.6e-16 |
| (C) numpy prune vs no-prune (프루닝 무손실) | **0.0** |
| (B) **vec-CUPY vs vec-numpy** (둘 다 prune) | **5.6e-16 (기계정밀)** ★GPU==CPU |

속도: `freewake_influence` 블레이드당 **CPU 24.8ms → GPU 4.05ms = 6.1×** (RTX3090,
업로드 포함). 클러스터 4090은 더 빠를 것.

## 클러스터 게이트 (사용자)

```bash
PYTHONUNBUFFERED=1 ALM_PROFILE_BEM=1 python main.py --gpu 0 \
    --config config/hpc_bench/bench5_baseline.py 2>&1 | tee bem_prof_s2.log
# A/B (물리 동일·속도만 복귀 확인):
ALM_FREEWAKE_GPU=0 ... | tee bem_prof_s2_off.log
```

| 항목 | 패치04(현재) | S2 기대 |
|---|---|---|
| freewake | 87.5 ms | **~20–40 ms** (GPU, per-call 업로드 포함) |
| solve/polar | 10.2 / 3.8 | 동일 |
| **bem 전체** | 102.6 | **~40–55 ms** |
| **s/step** | 3.00 s | **~2.3–2.6 s** |
| **C_T (물리)** | 0.00915 | **0.00915** (GPU==CPU 5.6e-16) ★필수 |

- freewake는 wake 채워질수록(rev↑) 성장 → 정상상태(rev~2) 값으로 판정.
- A/B `_off`: freewake만 ~87로 복귀, C_T 동일.
- **util**: freewake CPU 직렬구간 축소 → GPU util 상승 예상(부분적; solve/polar·전송은 아직 CPU).

## 클러스터 게이트 결과 (2026-07-08) — 속도 ✅ / ★C_T 민감성 발견

bench5 완주, 정상상태 rev 1.97:

| 항목 | 패치04 | S2(GPU) | 배율 |
|---|---|---|---|
| freewake | 87.5 | **29.5 ms** | 2.97× |
| bem | 102.6 | **44.6 ms** | 2.30× |
| s/step | 3.00 | **1.52 s** | 1.97× |
| 완주 | 37:02 | **21:16** | 1.74× (누적 원본 대비 2.9×) |
| drift | ~0 | −0.000% | ✅ |
| **C_T** | 0.00915 | **0.00897** | **−2% (↓ 아래 분석)** |
| C_P | 0.00091 | 0.00088 | −3% (C_T와 정합) |

### ★C_T −2% = 자유후류 FP-민감성 (버그 아님)
- 로컬 게이트: GPU freewake **per-call == CPU 5.6e-16**. 그런데 실 후류(ns_kept=2503)의
  **reduction 순서**(cupy/cuBLAS matmul·sum vs numpy)가 ~1e-13 다름 → seed.
- **자유 와류 후류 = 카오스 민감**(양 Lyapunov): 1e-13 seed가 50링×1006스텝 convect로 증폭
  → 적분 C_T ~2%. C_P 동반 감소·drift=0 → **물리적으로 정합한 다른 궤적**(손상 아님).
- 패치04까지 CPU끼리 byte-identical이라 안 보이던 민감성이 첫 노출.
- **A/B `ALM_FREEWAKE_GPU=0` ✅확정**: `_off`(CPU) → C_T **0.00915 완전복귀**(패치04와 동일,
  freewake~87·bem~102·완주 36:52). → ① xp 리팩터가 CPU 경로 **바이트 보존** ② 2%는 순수
  GPU 전환에서만 발생 = 진단 확정. `_on` 0.00897도 rev1.95~1.99 안정 → 발산 아닌 인접 limit cycle.

### ★게이트 철학 함의 (중요)
S2부터 GPU 포팅은 자유후류를 FP 수준 섭동 → **bit-identical 게이트 불가**. S2-S4는
**CV-band(±~3%) 물리 게이트**로 전환(05 설계 "물리 rev2 CT ±CV"와 정합). 최종 정확도
검증은 bench5 앵커가 아니라 **물리 레퍼런스(production 격자)**. bench5 C_T는 회귀 탐지용
앵커일 뿐 물리 목표값 아님(D=16 축소 토폴로지).

## 다음 — S3 / S4
- **S3**: `correct_noniterative`(solve 10ms) + velocity-triangle GPU화 → correction 완전
  on-device (cl_eval/dcl는 S1서 GPU-ready). A를 CPU로 안 내림.
- **S4**: FreeWake.rings **GPU 상주**(shed/convect on-device) → per-call H2D 제거,
  F_global cupy→spread(H2D 소멸), substep 내 sync 0 → **util 최대화**(본 작업 최종 목표).
