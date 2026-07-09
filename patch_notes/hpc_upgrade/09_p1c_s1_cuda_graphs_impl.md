# Phase 1c S1 — CUDA Graphs 구현 + ★cuBLAS 캡처 차단막 발견 — 2026-07-08

08 설계의 S1(pure-LBM whole-coarse-step graph)을 구현하고 로컬 RTX3090에서
검증하던 중, **08/PoC가 놓친 근본 차단막**을 실측으로 발견. 아래 §B가 이번 세션의
핵심이며, 진행 방향은 **사용자 결정 대기**(§D).

---

## A. 구현 완료 (MLG CUDA graph capture/replay)

`src/grid/multi_level_grid.py`:
- `advance()`를 **디스패처**로, 기존 재귀 본문을 `_advance_eager()`로 분리.
- `MLG_CUDA_GRAPH=1`이면: 정적 전제 검사(`_graph_precheck`) → **N(기본3) eager
  warmup**(lazy 버퍼 할당·RawKernel JIT·풀 포인터 안정) → **whole-step 캡처**
  (`Stream.begin_capture`→`_advance_eager`→`end_capture`) → 이후 `graph.launch` replay.
- **replay 부기**(`_graph_replay_bookkeeping`): graph는 GPU 커널만 재실행하므로
  host 카운터는 수동 전진 — 레벨 k는 coarse당 2^k substep → `lev.step_count += 1<<k`,
  `self._step_count += 1`.
- **캡처 record run은 실행이 아님**(기록만) → 카운터는 record 때 1회 증가, 실제 상태는
  이어지는 `launch()`가 반영 → 일관. 실패 시 **카운터 snapshot 복원 + 영구 eager
  폴백**(`_graph_failed`) + 클린 eager 1스텝.
- 전제(`_graph_precheck`, 하나라도 위반 시 eager): cupy 백엔드 / `MLG_PROFILE`·
  `MLG_NVTX` OFF(섹션 sync가 캡처 파괴) / 전 레벨 `al_model is None`(ALM=CPU-in-loop,
  S1 제외) / 전 레벨 `nan_trap` OFF(host `.get()` sync).
- env: `MLG_CUDA_GRAPH`(기본0), `MLG_CUDA_GRAPH_WARMUP`(기본3).

기본 OFF·순수 add-on이라 기존 런 무영향. **폴백 경로는 게이트로 정상 검증됨**(§C).

## B. ★근본 발견 — whole-step 캡처는 cuBLAS(einsum)이 막는다

로컬 게이트 실행 결과 캡처가 즉시 실패:
```
NotImplementedError: calling cuBLAS API during stream capture is currently unsupported
```
cupy 13.6은 **stream capture 중 임의의 cuBLAS 호출을 무조건 금지**(격리 재현 완료).
순수-LBM coarse-step에서 cuBLAS를 부르는 지점을 전수 추적:

| 경로 | cuBLAS? | 근거 |
|---|---|---|
| fused macro+collision | ❌無 | RawKernel(`_fused_kernel`, `_macro_kernel`, `_wale_kernel`) |
| streaming | ❌無 | RawKernel(`streaming_d3q27`) |
| **domain BC (eq/neumann)** | ❌無(실측) | eq+상수타깃 → **`_cached_f_eq` 재사용 + early return**(face_bc.py:118,150). 매 스텝 einsum 안 탐 |
| sponge | ❌無 | elementwise (einsum 없음) |
| cubic coupling 보간 | ❌無 | RawKernel(`CubicInterpolationKernel3D`) |
| **coupling rescaling** | ✅**有** | `coupling.py:312` `einsum('dq,q...->d...')` (momentum), `:319` `einsum('dq,d...->q...')` (cu). C2F/F2C 매 호출 |

→ **차단막 = coupling rescaling의 einsum 2곳이 전부.** L0.advance(BC=캐시)는 통과하고
첫 `C2F.L1` rescaling에서 abort. (collision `cumulant.py`의 einsum은 **fused 경로에선
미사용**=RawKernel이 대체, 무관.)

### 대체의 수치 영향 (로컬 실측, fp32 D3Q27)
`einsum` → cuBLAS-free 축약(broadcast-sum 또는 RawKernel)으로 바꾸면:
- **캡처 가능**해짐(broadcast.sum 캡처 성공 확인).
- 그러나 **bit-identical 아님**: `max|Δ|≈2.1e-7`(fp32 ε급 반올림 잡음), 값의 ~77%가
  last-bit 상이. maxrel은 momentum≈0(대칭 상쇄) 지점에서 커 보이나 절대오차는 2e-7.
- fp64 누산 후 downcast도 fp32 einsum과 불일치(오히려 더 정확) → **어느 방식이든
  기존 fp32 GEMM과 bit-일치 불가** = **참조 sha256 재기준선 불가피**(coupling은
  ALM 런과 공유 → bench5_baseline `ac910ff`도 이동). 물리는 보존(2e-7).

### 왜 이게 중요한가 (Phase 0와 연결)
Phase 0: pure-LBM 391ms 중 **coupling 300ms(C2F.L4 158ms)가 launch-bound 병목**,
advance(collision+stream+BC) 91ms는 floor 1.8–3.6×(양호). 즉 **graph의 실이득은
coupling에 있는데, 그 coupling이 einsum 때문에 캡처 불가.** advance만 graph(=아래 S2)
하면 bit-safe지만 이득 상한이 작다.

## C. 로컬 게이트 (`gates/p1c_s1_graph_gate.py`, RTX3090)

bench5_pure_lbm MLG를 실제 setup 경로로 빌드 → 초기 f 스냅샷 → K=8 coarse eager vs
graph(동일 초기상태 복원) → **레벨별 f bit 비교 + 캡처성공 확인**.
- **현재 결과**: 전 레벨 `max|Δ|=0`(폴백이 eager와 완전 일치), 카운터 8/8 정확,
  **capture=False**(cuBLAS 폴백). → **폴백 경로 정상성 PASS, 캡처는 §B 차단으로 미도달**.
- einsum de-cuBLAS(Option A) 적용 시 이 게이트가 capture=True + max|Δ|=0(graph==신-eager)로
  전환되어야 함 = S1 성공 판정.

## D. ★진행 결정 (사용자 대기) — Phase 1c 재조정

| 옵션 | 내용 | 수치 | 이득 상한 | 리스크 |
|---|---|---|---|---|
| **A**(권장) | coupling einsum 2곳 de-cuBLAS(작은 RawKernel/broadcast) → **whole-step S1 graph 활성** | last-bit 이동(2e-7) → **sha256 재기준선**(물리=CV-band) | **큼**(launch-bound coupling 300ms 직격) | 재기준선(방법론 영향) |
| **B** | per-level `_advance_fused`만 graph(§B서 전부 cuBLAS-free), coupling eager 유지 | **bit-exact**(sha256 보존) | **작음**(advance 91ms만, floor 1.8–3.6×) | 낮음. 캡처 구조 per-level 재작성 |
| **C** | Phase 1a(cell-centred coupling) 먼저 — f_prev/시간보간 제거 겸 graph-friendly·de-cuBLAS를 그 재작성에 내장(의도된 1회 재기준선) | 재기준선(구조 변경) | 큼(구조적: 메모리+multi-GPU halo+속도) | 중. 범위 큼 |

권장 = **A**: launch-bound 프라이즈(coupling)를 물리 무의미한 last-bit 변화로 직접
개방. CV-band 게이트는 S2 freewake GPU에서 이미 채택된 관례([[project_hpc_upgrade]] §A).
단 참조 sha256을 이동시키므로 **사용자 확인 후 진행**.

## E. 변경 파일 (이번 세션, uncommitted)
`src/grid/multi_level_grid.py`(graph capture/replay + eager 분리),
`patch_notes/hpc_upgrade/09_p1c_s1_cuda_graphs_impl.md`(본 파일),
`patch_notes/hpc_upgrade/gates/p1c_s1_graph_gate.py`.
+ **Option A 실측용 임시**: `src/grid/coupling.py` einsum→broadcast(§F에서 판정 후
거취 결정).

---

## F. ★Option A 실측 — launch-bound 가설 **반증**, S1 이득 ≈ 0 (2026-07-08)

사용자 승인(Option A)으로 coupling einsum 2곳을 cuBLAS-free broadcast 축약으로
교체 → **whole-step 캡처 성공**. 로컬 RTX3090서 bench5_pure_lbm(11.3M) 전면 측정:

### F.1 정확성 (capture 성공, 하지만 bit-불가)
- 단일스텝 격리(동일 시작상태): **capture+launch vs eager = 7.4e-7**, replay vs eager
  = 7.4e-7. → 물리적 무의미(fp32 ε급)하나 **bit-identical 아님**.
- **replay vs replay = 6e-8** (run-to-run **비결정**). eager는 완전 결정적(Δ=0, 2회
  대조)인데 graph는 아님 → 캡처영역 어떤 reduction이 graph서 비결정 순서가 됨(내가 넣은
  broadcast `.sum`은 격리 테스트서 결정적=범인 아님; 스텝 내 타 reduction). → **graph
  경로는 원천적으로 bit-재현 불가 → 게이트는 CV-band만 가능**.

### F.2 ★성능 (steady-state 25-step 평균, warmup/capture 제외)
```
  eager        :  961.6 ms/coarse-step
  graph replay :  955.7 ms/coarse-step
  speedup      :  1.006x   (6 ms/961 = 0.6%)
```
→ **whole-step graph가 전 launch/Python 오버헤드를 제거했는데도 0.6%만 빠름 =
스텝의 99.4%가 실제 GPU 커널 시간 = compute/bandwidth-bound.** Phase 0가 nsys로
확정하려다 못 한 판별을 **graph 실험이 대신 확정: pure-LBM은 launch-bound 아님.**
(Phase 0의 "391ms≫floor 25-50ms=launch-bound"는 오판 — cumulant collision·cubic
interp의 compute intensity와 실 DRAM 트래픽을 저평가. floor가 순수 2-array streaming
가정이라 과소.) 큰 격자(207M)일수록 커널↑ → launch 오버헤드 더 amortize → graph 더 무의미.

### F.3 결론 & 재조정
**Phase 1c CUDA Graphs = 종료(이득 없음).** 구현은 정확·안전(기본 OFF, 폴백 검증)하나
enable할 이유 없음. 두 병목 재확인:
- bench5(ALM-heavy): 84.8% = L4 BEM(CPU). 이미 S2(freewake GPU)까지 최적화, S3 기각 →
  ALM-on-CPU 전제선 near-floor.
- production(207M, LBM지배): **compute/bandwidth-bound** → 레버 = **Phase 1a
  cell-centred coupling**(f_prev+시간보간 제거 = DRAM 트래픽·배열수↓, bandwidth 직격)
  또는 **multi-GPU(Phase 2, scale-out)**. launch 감소는 레버 아님.

거취(사용자 결정): coupling einsum 변경은 graph 전용이었고 이득 0 → **되돌려 bit-참조
보존** 권장(graph scaffolding은 OFF로 dormant 유지 or 제거). 다음 = Phase 1a 또는 Phase 2.
