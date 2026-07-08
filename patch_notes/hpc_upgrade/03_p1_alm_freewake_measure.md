# P1 — ALM free-wake: 측정 먼저, 그 다음 GPU 포팅 — 2026-07-06

Phase 1a에서 피봇(00_phase0 §권장 P1). 근거: bench5 wall의 **90.6%가 L4.advance=ALM**,
그리고 사용자 관측 **GPU-util 10% 초반**(= ALM free-wake의 CPU numpy 직렬 구간 동안 GPU 유휴).
bench5 wall/util을 실제로 움직이는 유일 레버.

## ★측정 먼저 (Stage A 교훈)

"bem = 188 ms/call"은 **phase 전체**입니다. `_compute_bem_forces`는 **per-blade 루프**로,
블레이드마다:
- `_lookup_cl_cd` **2회** (보정 전/후) — C81 Mach 보간
- `_kleine_w_corr`: `freewake_influence`(Biot-Savart) + `correct_noniterative`
  (**solve**; 내부에서 `cl_eval`/`dcl_eval`로 폴라 **재조회**)

즉 188 ms가 freewake인지·폴라인지·solve인지 **단정 불가**. 게다가 GPU 포팅이 이득이 되려면
Biot-Savart 배열(`nc × ns_kept`)이 **충분히 커야** 함 — hover wake는 대부분 blade에서 멀어져
`prune_factor=6ε`로 잘려나가므로 `ns_kept`가 작으면 GPU 포팅은 Stage A처럼 무효(런치/전송
지배). **먼저 측정해서 (a) 병목이 freewake인지 (b) 배열이 큰지 확인 후 결정.**

성장 시그니처(bem 184.6→188.4, wake 채워질수록↑)는 freewake를 지목하지만(유일한 wake-길이
의존항), 확증 필요.

## 계측 (이번 세션 추가, env `ALM_PROFILE_BEM`)

- `actuator_line.py`: bem phase를 **freewake / solve / polar / other** 4-way 분해
  (`_pf_acc` 확장, ALM_PROFILE 출력에 `[ALM prof.bem]` 행 추가). `solve`는 내부 폴라 포함.
- `smearing_correction.py`: `freewake_influence`가 **`nc, ne, ns_kept`(prune 후) 크기**를
  8000콜마다 출력(`[freewake] …`) — GPU viability 판정용. wake 채워지며 성장 관찰.
- 전부 **inert when off**(env 없으면 `_pf_bem=False`, 계측 0).

## 측정 런 (사용자, 클러스터)

```bash
ALM_PROFILE_BEM=1 python main.py --config configs/hpc_bench/bench5_baseline.py 2>&1 | tee bem_prof.log
```
- **마지막** `[ALM prof.bem]` 행(정상상태) + `[freewake]` 크기 행을 봐 주세요.
- (짧게 보려면 rev 2까지 = wake가 차야 freewake가 대표값. 기존 bench5 런에 env만 추가.)

## 결과별 결정 트리

| 측정 결과 | 해석 | 다음 |
|---|---|---|
| **freewake 지배 + ns_kept 큼(≳수천)** | Biot-Savart가 병목, 병렬성 충분 | → **`_seg_vz_batch`/`_build_segments` CuPy 포팅** (아래 설계) |
| **freewake 지배 + ns_kept 작음** | 병목이나 배열 작음 → GPU 무효(Stage A) | → `rebuild_every` throttle / 알고리즘(캐시·프루닝 개선) |
| **polar 지배** | C81 Mach 보간이 병목 | → 폴라 조회 벡터화/배치/GPU/캐시 (freewake 아님) |
| **solve 지배** | 내부 폴라(cl_eval/dcl_eval) 또는 linalg | → dcl 유한차분 폴라콜 수↓ / 폴라 배치 |

## GPU 포팅 설계 (freewake 지배 & 큼일 때만)

`_seg_vz_batch`는 `(nc, ns, 3)` **elementwise + 축소** → CuPy 직역 가능. 핵심 = **전송 최소화**:
- `FreeWake.rings`를 **CuPy 배열로 상주**(shed/convect도 GPU) → rebuild마다 H2D 불요.
- `freewake_influence` GPU 실행, 결과 **`A`(n×n)만 D2H**(작음) → solve는 CPU 유지(작은 linalg).
- prune·`_build_segments`도 GPU(불리언 마스크). `segment_missing_theta`(erf)는 `cupyx.scipy.special.erf`.
- 게이트: `_freewake_influence_loop`(레퍼런스) 대비 수치 일치 + bem `freewake` 항 타이밍↓ + util↑.
- ⚠️ ALM은 L4 substep당 1회(16×/coarse) → CuPy stream/graph로 런치 amortize 고려.

## 병행 저비용 레버 (측정과 무관하게 유효)

`eps_correction={"rebuild_every": N}` — 영향행렬 rebuild를 N substep마다(기본 1=매번).
wake는 천천히 convect하므로 근사 양호. **물리 근사**라 bit-gate엔 부적합하나, 실사용 런에서
bem을 ~N배 줄이는 즉효 노브(00_phase0 §E). GPU 포팅과 독립·병용 가능.

## 상태

측정 훅 구현·컴파일 완료. **클러스터 측정 런 대기** → 결정 트리로 GPU 포팅 vs 대안 확정.
