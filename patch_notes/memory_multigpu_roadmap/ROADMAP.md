# Roadmap: 메모리 절감 + Multi-GPU (robust 순서)

근거 레퍼런스: Holzer 2024 (lbmpy/waLBerla, exascale). 우리 스택과 동일(D3Q27 K17
cumulant → CUDA → CuPy). 보고서: 본 세션 분석 + `to_claude/ref_papers/2024_Markus_Holzer_*.pdf`.

## 결정된 순서 (robust)
**esoteric+cell-centred(단일 GPU) → multi-GPU.** MPI halo/경계결합이 스트리밍패턴+결합방식
위에 지어지므로, 최종 구조를 단일 GPU에서 확정 후 분산. (현재 구조 위에 MPI 먼저 = 재작업.)
원칙: ①한 번에 어려운 것 하나 ②MPI는 최종 구조 위에 ③구조변경은 단일 GPU 검증 후 분산.

### 두 "값싼 win" 처리
- **f_prev strip-only**: 버림 — Phase 1a(cell-centred)가 f_prev 통째 제거 → throwaway.
  (Phase 1a 지연 시 임시 bridge로만 고려.)
- **δf(zero-centred) 저장**: 보류 — FP16 전용 가치. esoteric/multi-GPU엔 불필요(메모리 0,
  FP32 정확도 미미). 현 cumulant는 δf 미사용 확인됨(bgk FP16 커널은 사용). FP16 Phase의
  첫 단계로 편입.

## Phases

### Phase 1a — cell-centred explosion/coalescence 결합 (f_prev 제거)
- Lagrava cubic+temporal 보간 → cell-centred volumetric (explosion eq.5.3 / coalescence
  eq.5.4, Holzer p.74). "explosion이 2행 유효 fine cell 생성 → 비동기 step서 시간보간 불필요"
  → **f_prev 전부 제거**. 2 ghost layer.
- 결합은 스트리밍패턴 독립(p.71) → **더블버퍼(f_post) 유지한 채** 결합만 교체, 단독 디버깅.
- 대상: `src/grid/coupling.py`(coarse_to_fine/fine_to_coarse), `src/grid/multi_level_grid.py`
  (f_prev 할당/copyto 제거), interpolation 커널.
- gate: 질량/운동량 보존, light(4-lvl)+fine_mini(5-lvl) 물리 tol 내 일치, 메모리 실측(f_prev 사라짐).

### Phase 1b — cumulant in-place esoteric 스트리밍 (f_post 제거)
- Esoteric Pull/Twist 단일버퍼. cumulant in-place 커널 신규(옛 esoteric_d3q27는 BGK전용).
- in-place는 pull 대비 처리량 수% 이내(Fig 9.7) → 속도 손실 ≈0.
- 단일격자 먼저 검증 → MLG(Phase 1a 결합과 결합). BC/ALM은 esoteric ordering 대응 필요
  (보간벽/IBB/q-fraction은 implicit BB 혜택 없음 — 명시 처리).
- 대상: 새 cumulant esoteric 커널, `simulation.py`(f_post 제거), `stream.py`.
- gate: 단일격자 bit/물리 일치 → MLG 회귀 → 메모리 실측(f_post 사라짐, 3배열→~1).

### Phase 2 — Multi-GPU (multi-node)
- CUDA-aware MPI(mpi4py + device 포인터 직송, host staging 無).
- 방향별 halo pack/unpack = CuPy RawKernel(스트리밍패턴 인지, face 넘는 PDF만).
- 레벨별 부하분산(fine 레벨이 ~1.4% 도메인에 ~80% 작업, Holzer Table 9.4).
- GPU-경계 coarse↔fine = branchless **bitmask coalescence**(eq.5.7, warp divergence 회피).
- 재귀 timestep + non-blocking MPI overlap(Algorithm 4).
- gate: 단일=다중 결과 일치, weak-scaling.

### Phase 3 (선택) — δf → FP16 coarse 레벨
- δf 저장 먼저(전제). FP16 저장/FP32 연산, coarse 레벨 한정(plain FP16은 절단 심함, p.146).

## 진행 로그
- [ ] Phase 1a 설계 (현 coupling.py 구조 분석 → cell-centred 스펙)
- [ ] Phase 1a 구현 + gate
- [ ] Phase 1b 구현 + gate
- [ ] Phase 2 구현 + gate
