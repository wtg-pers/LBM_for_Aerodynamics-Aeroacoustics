# HPC 업그레이드 계획 — 속도 · 메모리 · multi-GPU (2026-07-05)

참고: Holzer 2024 학위논문(`to_claude/ref_papers/high_computing/`) — 상세 분석·페이지
인용은 `patch_notes/memory_multigpu_roadmap/ROADMAP.md`(Phase 1a/1b/2/3 순서 결정본)에
있음. 본 문서는 그 순서를 계승하되 **속도 트랙을 통합**하고 **baseline 앵커(bench5)에
게이트를 고정**한 실행 계획. multi-GPU 선행조건(Task 1)은 완료 상태
(`patch_notes/alm_multigpu/HANDOFF.md`: CUDA-aware OpenMPI+UCX, thread_level=serialized
필수, device-direct PASS, 4×4090 PCIe).

## 0. 현재 성능 앵커 (실측, 1×4090, testA 15rev)

| run | updates/coarse-step | MLUPS | wall/15rev |
|---|---|---|---|
| light 4-level, Dağ straight | 131.1M | **81.8** | 6.7 h |
| light 4-level, Kleine free (K1) | 131.1M | 65.1 | 8.4 h |

**★핵심 진단(가설)**: 81.8 MLUPS × 216 B/cell(D3Q27 fp32 2-array) ≈ **18 GB/s =
4090 대역폭(~1008 GB/s)의 ~2%**. LBM은 본질적으로 bandwidth-bound인데 우리는
그 근처에도 못 가 있음 → 지배 항은 대역폭이 아니라 **커널 런치/CPU-GPU 동기화/
MLG 재귀 substep 오버헤드 + ALM CPU 직렬 구간**. (nvidia-smi util 97-99%는
kernel-resident 시간 비율일 뿐 효율이 아님. 기존 발견 "util oscillation=동기화
stall"과 정합.) → **Phase 0에서 확정 후, 사실이면 속도 레버 1순위는 esoteric이
아니라 런치/동기화 제거(Phase 1c).**

이론 천장(참고): 4090 ~1TB/s ÷ 216 B ≈ 4.6 GLUPS(단일격자). MLG/ALM 감안해도
수백 MLUPS~GLUPS급이 합리적 목표 (50×+ headroom).

## 1. Baseline 프로토콜 (모든 Phase 공통 게이트)

**bench5** (`configs/hpc_bench/bench5_baseline.py`, preset `bench5`): light_slab5의
5-level 토폴로지(로터 L4 슬랩 포함)를 D=16으로 축소. 9.04M셀/~3.5GiB,
503 steps/rev × 2rev = 1006 steps → **분 단위 턴어라운드**. 전 경로 재현:
D3Q27 cumulant fused, 5-level MLG, dyn_smag, ALM+NASA 덱, sponge.
**ALM 보정 = Kleine free wake** (2026-07-05 모델 결정: 물리성 서열
kleine_free ≈ pure ALM > Dağ; 또한 free wake가 가장 무거운 ALM 경로
(per-step solve·exact-Φ·convect D2H·rebuild)라 Phase 1c 오버랩 작업의
회귀 대상 그 자체). LBM-only 회귀는 Phase 0 pure-LBM 토글 런이 커버.

게이트 3종 (reference 대비):
1. **bit-identical** — `checkpoints/checkpoint_00001005.npz` sha256 비교
   (reference sha256 `ac910ff9…b3b3e61`, 상세 `00_phase0_measurement.md` §A).
   (알고리즘 보존 변경: 인덱싱, 런치 배치, CUDA Graph, halo 등. 동일 GPU/env 전제).
2. **물리** — rev2 tail CT가 reference thrust-CV 이내
   (알고리즘 변경: cell-centred 결합, esoteric, 도메인 분할 등).
3. **속도** — performance.csv MLUPS, 동일 하드웨어 비교. bench5(회귀·반복용)와
   **slab5-smoke(45.3M, 실전 규모 성능 측정용 2차 앵커)** 둘 다 기록.

운영: reference 결과는 `hpc_bench_ref/` 등에 체크포인트+csv 아카이브(덮어쓰기
금지). 모든 런은 클러스터(로컬 대형격자 금지 — CPU 온도). 단계별 기록은 이
폴더에 `NN_*.md` 패치노트로 (stepwise 규칙).

## 2. Phases

### Phase 0 — 계측 (짧고 먼저)
- bench5 reference 런(클러스터 1×4090) → 게이트 기준값 확보.
- **pure-LBM 토글 런**(bench5에 `actuator_line.enabled=False`) → ALM 직렬화 vs
  LBM 자체 오버헤드 분리 (diag_pure_lbm의 bench5판).
- nsys(또는 CUDA events)로 coarse-step 1개 분해: 레벨별 커널시간 / 런치 갭 /
  D2H 동기화 / ALM CPU 구간. **판정: 갭이 지배(예상)면 Phase 1c 조기 착수,
  커널이 지배면 ROADMAP 순서 유지.**
- 산출: 분해 표 + "어디에 1.6s/step이 쓰이는가".

### Phase 1a — cell-centred coupling (f_prev 제거) [메모리]
ROADMAP대로 (Holzer p.74, explosion/coalescence eq.5.3-5.4; 스트리밍패턴 독립이라
더블버퍼 유지한 채 결합만 교체). 대상 `src/grid/coupling.py`,
`multi_level_grid.py`. **게이트: bench5 물리 + 질량보존 + 메모리 실측
(410 → ~300 B/cell 기대).**

### Phase 1b — esoteric in-place cumulant (f_post 제거) [메모리]
단일버퍼 esoteric pull/twist, cumulant용 신규 커널(기존 esoteric_d3q27은 BGK 전용).
단일격자 → MLG 순 검증. BC/ALM ordering 대응. **게이트: bench5 물리 + 메모리
(~200 B/cell). 효과: slab5 18.6→~9 GB, light_tip5(137M)급 27GB→24GB 경계,
per-GPU 수용 셀 ~2×.**

### Phase 1c — per-step 오버헤드 제거 [속도]
Phase 0 판정에 따라 1a/1b와 순서 교환 가능(launch-bound 확정 시 먼저).
- **CUDA Graphs**: MLG 재귀 substep의 커널 시퀀스(레벨별 collide/stream/결합)를
  그래프 캡처 → 런치 오버헤드 일괄 제거. ALM force 주입은 그래프 밖 또는
  graph update로. ([[project_su2_coupling_direction]]의 "per-step 속도" 투자와 동일.)
- **ALM 오버랩**: force 1-step lag + async(샘플→BEM→spread를 GPU 스트림/CPU와
  중첩) 또는 BEM GPU 포팅. 폴라 배치화(완료)의 다음 단계.
- 동기화 제거: 스텝 내 불필요 `.get()`/sync 소탕, conservation/logging 배치화.
- **게이트: bit-identical(1-step lag 제외) + bench5·slab5-smoke MLUPS.
  목표: slab5 기준 ≥5×(보수) / launch-bound 확정 시 10×+ 기대.**

### Phase 2 — multi-GPU (HANDOFF Task 4 흡수; 최종 단일-GPU 구조 위에)
- 4a 단일레벨 halo: 방향별 pack/unpack RawKernel + CUDA-aware device-direct
  (UCX, `mpi4py.rc.thread_level='serialized'` 런처에 고정). bench5 단일레벨
  변형으로 검증(단일=다중 bit 일치).
- 4b MLG 분산: coarse 분산, fine 분산 vs 복제 결정; GPU 경계 coarse↔fine
  bitmask coalescence(eq.5.7, branchless); 레벨별 부하분산(fine가 도메인 ~1.4%에
  일 ~80%, Holzer Table 9.4) → 분할축은 로터 중심 slab 회피.
- 4c ALM across-GPU: 마커 gather/scatter, hub_center 글로벌 L0 LU 좌표 주의
  ([[feedback_alm_hub_center_lu]]). non-blocking overlap(Algorithm 4).
- **게이트: bench5 1↔2↔4 rank 물리 일치 + slab5 4×4090 실측(강스케일링) +
  Watanabe-fine급(207M) 4분할 수용 확인.**
- 부하 불균형·PCIe(NVLink 無) → halo 부피 최소화, 오버랩 필수.

### Phase 3 (선택) — δf 저장 + FP16 coarse 레벨
ROADMAP대로 (plain FP16 절단 심함 p.146 → δf+FP16, coarse 한정). Phase 1b 후
메모리가 더 필요할 때만.

## 3. 성공 기준 (전체)

| 항목 | 현재 | 목표 |
|---|---|---|
| slab5 15rev wall (1×4090) | ~22h 추정(3.3×light) | Phase 1c 후 ≤5h |
| bytes/cell | 410 | Phase 1b 후 ~200 |
| per-GPU 최대 셀(24GB) | ~58M | ~120M |
| multi-GPU | 없음 | 4×4090서 단일=다중 일치 + fine급(≥207M) 수용 |

기록: 각 Phase 완료 시 bench5/slab5-smoke 앵커 표를 이 문서에 갱신.
