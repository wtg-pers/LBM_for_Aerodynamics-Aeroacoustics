# Phase 1c — CUDA Graphs 설계 스펙 (design) — 2026-07-08

PLAN §1c. Phase 0 판정(**launch-bound**: coarse-step 3620ms ≫ 대역폭 floor 25-50ms,
70×; coupling 300ms/collision 91ms 모두 실 커널시간은 작음)에 근거. MLG advance =
coarse-step당 **~150-300 커널 런치**(2⁵-1=31× sim.advance + ~30 coupling, 각 다중 커널)
→ 런치/드라이버 오버헤드 지배. CUDA Graph로 시퀀스 캡처·replay하면 일괄 제거.
추천순서 ②(①Stage A bit-gate → ②본 작업 → ③multi-GPU). multi-GPU per-step 속도 투자와 동일
([[project_su2_coupling_direction]]).

## 0. 메커니즘 검증 (PoC 완료, RTX3090 cupy 13.6)

`scratchpad/p1c_cuda_graph_poc.py`:
- **replay == direct: max|Δ|=0** (bit-identical).
- 120 small-launch: direct 0.887 → graph **0.341 ms = 2.60×** (런치 오버헤드 실재; 실 MLG는
  런치 수 많고 heavier 커널이라 배율 TBD).
- **캡처 중 할당 WORKS** (cupy 13.6 풀이 graph-safe) → Stage A coupling의 slab 할당 문제없음.
- API: `Stream.begin_capture()`/`end_capture()`→`cp.cuda.Graph`, `graph.launch(stream)`.

## 1. 캡처 가능성 분석 (코드 실측)

| 대상 | 경로 | 버퍼 안정성 | host sync | 캡처 |
|---|---|---|---|---|
| **L0-L3 advance** | `_advance_fused` (ALM無) | `self.f/_f_post/rho/u/nu_t` 재사용, WALE는 `_rho_buf/_u_buf/_nu_t_in` | nan_trap OFF시 無 | **✅ 가능** |
| **coupling** C2F/F2C | `coupling.*` | slab 할당(Stage A) | 無(순 GPU) | **✅**(alloc-in-capture OK) |
| **L4 advance** | `_advance_fused_with_alm` | ①`macroscopic.compute`가 rho/u **재할당** ②중간 **CPU BEM** | BEM=CPU | **❌ 통째 불가** |
| pure-LBM 전체 advance | ALM 완전 OFF (`bench5_pure_lbm`) | 전 레벨 fused, 안정 | 無 | **✅ 통째 1-graph** |

★핵심 제약:
- **nan_trap 반드시 OFF**(host `.get()` sync 있으면 캡처 깨짐 — 기본 OFF, [[project_gpu_perf]]).
- **ALM(L4)은 CPU-in-loop** → whole-coarse-step graph 불가(pure-LBM 제외). 힘 주입은 graph 밖.
- 캡처는 **non-default stream**에서. 첫 스텝 캡처→이후 replay(고정 시퀀스 전제: 격자·BC·레벨 불변 ✓).

## 2. 스테이징 (bit-identical 게이트 — algorithmic 보존이라 sha256 유지 기대)

- **S1 — pure-LBM whole-coarse-step graph** (가장 clean·production 직결). `bench5_pure_lbm`
  (ALM OFF)는 `mlg.advance()` 전체가 순 GPU → 1 graph 캡처, 매 coarse-step replay.
  게이트: pure-LBM 물리·질량 bit + coarse-step wall↓. **LBM launch-bound 이득의 순수 측정.**
- **S2 — per-level `_advance_fused` graph** (ALM 런에도 적용). L0-L3 advance를 레벨별 graph
  캡처(2^k× replay). 경계·coupling은 Python 유지. L4는 un-graphed(ALM). 게이트: bench5 bit + wall↓.
- **S3 — coupling graph + L4 post-ALM segment**. C2F/F2C graph화; L4는 BEM(CPU) 뒤
  (collide+stream+BC)만 graph(body_force 포인터 고정, rho/u 버퍼 안정화 필요=`_advance_fused_with_alm`
  를 재할당 없는 in-place 버전으로). 게이트: bench5 bit + wall/util.
- (S4 stretch) 비-ALM subtree 통합 graph.

## 3. 구현 노트
- MLG에 graph 캐시(`self._graphs`), 첫 advance서 capture, 이후 replay. `MLG_CUDA_GRAPH` env
  토글(기본 off→검증 후 on). capture 실패(sync 감지 등) 시 자동 eager 폴백.
- 버퍼 포인터 고정 감사: fused 경로 rho/u/f/f_post/nu_t 재사용 확인(S2). `_advance_fused_with_alm`
  의 `macroscopic.compute` 재할당 → in-place 버전 필요(S3).
- graph 무효화 조건: 격자 리매핑/레벨 수 변경/dtype 변경 없음(정적 런) → 1회 capture 유효.

## 4. 리스크
- capture 중 숨은 host sync(로깅·conservation·nan) → 캡처구간서 배제. conservation/logging은
  coarse-step 경계(graph 밖)라 무관.
- alloc-in-capture는 PoC서 OK지만 대용량서 풀 단편화 가능 → S1서 실측.
- 실 이득 배율 미지(heavier 커널). Phase 0가 launch-bound라 판정했으니 유의미 기대, S1로 확증.
- ALM 런은 L4가 여전히 eager → 이득은 L0-L3+coupling에 국한(S2). production(207M, LBM지배)선
  pure-LBM-형 이득이 대표.

## 5. 진행
PoC ✅. 다음: **S1 구현** — MLG.advance에 graph capture/replay + `bench5_pure_lbm` bit/타이밍 게이트.
