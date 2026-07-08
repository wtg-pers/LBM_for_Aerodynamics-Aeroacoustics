# Phase 1a — cell-centred coupling 설계 스펙 (design) — 2026-07-06

PLAN.md §1a + ROADMAP Phase 1a 실행. 상태: **설계(구현 전)**. 구조 변경이라
ROADMAP대로 단일 GPU 검증 후 분산. Phase 0 측정(00_phase0)이 근거를 정밀화함.

## 0. 목적 & Phase 0 근거

- PLAN 목표: **f_prev 제거**(메모리 410→~300 B/cell) + Holzer **cell-centred**
  (multi-GPU halo-ready 구조).
- Phase 0 측정: **coupling 300 ms = pure-LBM wall의 77%**, **C2F.L4 158 ms**(9.9 ms/call
  ×16) 단일 최악. fprev copy 9 ms. → coupling이 LBM측 병목.

## 1. 현재 스킴 = vertex-centred node-coincident (Lagrava)

- **`fine_shape = 2·fdx + 1`** (`overlap_manager.py:217-225`) — **"+1" = 노드공유 지문**.
  coarse 노드마다 fine 노드가 겹침(짝수 인덱스) + 홀수는 보간.
- **C→F** (`coupling.py:160-204`): coarse 서브볼륨 추출 → **half-step 시간보간
  `0.5·(f_prev+f_now)`**(:189) → f=f_eq+f_neq 분해 → τ-rescale → `_upsample_to_fine`
  (짝수배치+홀수 cubic 보간, 3 RawKernel) → **6면 경계 strip만 기록**.
- **F→C** (`coupling.py:210-254`): `f_fine[:, 0::2,0::2,0::2]` 일치노드 추출 → filter →
  τ-rescale → excised 기록.
- **rescale = τ 비** (`level_scaling.py:194-233`, `τ_f/2τ_c`, `2τ_c/τ_f`) — 물리, **정렬
  무관 불변**(에이전트 확인).
- 초기화: 각 레벨 자기 fine_shape의 f_eq(ρ₀,u₀), coarse 복사 아님 → **index-map 무관**.

## 2. ★Phase 0가 드러낸 두 개의 분리가능한 비효율

| | 정체 | 처방 | 성격 |
|---|---|---|---|
| **I1 (속도, ~158ms)** | C→F가 fine_domain **전체볼륨** upsample(할당·zero·3커널)하지만 경계 strip만 사용. L4=5M셀 생성→~1M 경계 사용, 540MB zero×16/step | **boundary-only** upsample (strip+stencil 여유만) | vertex/cell **무관**, **bit-identical** |
| **I2 (메모리+구조)** | 비동기 서브사이클 중간시각 coarse 데이터 필요 → `f_prev` 시간보간 | cell-centred **explosion**(2행 유효 fine)으로 제거 | 구조변경, multi-GPU halo 정렬 |

**핵심**: C2F.L4 158 ms의 큰 부분(전체볼륨 할당·zero·보간)은 **cell-centred 여부와
무관하게** boundary-only로 제거 가능. f_prev 제거(I2)와 독립.

## 3. 목표 스킴 = Holzer cell-centred explosion/coalescence

- 노드공유 없음; fine 셀 중심이 coarse 대비 **half-cell offset**; **`fine_shape = 2·fdx`**
  (+1 제거) → **모든 fine 레벨 shape 변경**.
- **C→F = explosion** (Holzer eq.5.3): coarse 1셀 → 자식 8셀 volumetric 분배.
- **F→C = coalescence** (eq.5.4): 자식 8셀 평균 → coarse 1셀.
- 시간보간·f_prev **제거**. **2 ghost layer**.
- **경계-scoped explosion으로 구현하면 I1+I2 동시 해결 + multi-GPU halo 구조 그 자체.**

## 4. Impact surface (Explore 에이전트 매핑, `file:line`)

| 파일 | 변경 | 리스크 |
|---|---|---|
| `grid/coupling.py`(+`_2d`) | `_upsample_to_fine`(:282), `fine_to_coarse`(:210), C→F half-step 분기(:181) | 핵심 |
| `grid/overlap_manager.py`(+`_2d`) | `fine_shape +1→×2`(:221), `coarse_to_fine`/`fine_to_coarse`/`is_coarse_coincident`(:244-318), 경계 slice(:364-434), validation ≥1→≥2(:479) | 높음(shape 파급) |
| `grid/interpolation.py`, `kernels/interpolation_d3q27.py`(+`d2q9`) | even-known/odd-filled cubic → explosion 커널 교체 | 중 |
| `grid/multi_level_grid.py` | `_f_prev` 리스트·save/use 전체 제거(:114-120,145-166,212-253) | 중(내가 계측한 fprev 섹션도 제거) |
| `grid/grid_level.py`(+`_2d`) | `_f_prev` vestige(:176,239-264) **무콜러=저위험** + `physical_to_index`/`index_to_physical`(:355-387) half-cell | 낮음 |
| `solver/setup.py` | `_mlg_level_origins`(:1149), fine obstacle map(:1319), ALM hub-in-fine(:1253) **half-cell shift** | 중(ALM/geometry) |
| `io/mlg_vtk_writer.py`(+`_2d`) | per-level origin `+0.5·dx` (ParaView 오버레이) | 낮음 |
| `solver/initializer.py` | **로직 무변경**, fine_shape만 반영 | 없음 |
| `grid/level_scaling.py` | **무변경** (τ비 확인됨) | 없음 |
| **미영향(검증됨)** | ALM spread/sample(물리좌표·Gaussian, parity 무관), body force=L0, checkpoint I/O(parity 무관) | — |

## 5. 게이트 계획 (단계별, 클러스터=사용자)

1. **2D testbed 먼저** (`to_claude/test_coupling_2d.py` C1-C4, `test_mlg_2d_poiseuille.py`
   V1-V2). ⚠️ **C3(even/odd 보간 :127)는 vertex 전용 → cell-centred용 재작성**.
   질량보존 assert 추가 권장(현재 테스트는 analytic/macro만, mass/bit 미검사).
2. **단일격자** explosion/coalescence bit/물리 → **fine_mini 5-level** MLG 회귀.
3. **bench5**: 물리(rev2 CT ±CV) + **질량보존** + (알고보존이면)bit + **C2F/F2C 타이밍
   재측정**(MLG_PROFILE). 목표: coupling 300→대폭↓, f_prev 메모리 실측 감소.
4. slab5-smoke(45.3M) 2차 앵커.

## 6. 권장 단계 (결정: **옵션 1**, 2026-07-06)

> **진행**: Stage A **구현·로컬 검증 완료**(bit-identical 12/12, 3.0× — 상세
> `02_phase1a_stageA_boundary_only.md`). 클러스터 bit(sha256)+타이밍 게이트 대기.
> Stage B는 그 다음.


- **옵션 1 (추천): Stage A → Stage B 분리.**
  - Stage A = **boundary-only C→F** (I1). vertex 유지, **bit-identical**, 저위험, 빠름
    → C2F 158 ms 대폭 절감을 먼저 확정·검증.
  - Stage B = **cell-centred + f_prev 제거** (I2). 구조·메모리·multi-GPU. 격리된 큰 변경.
  - 장점: 검증된 속도 win 조기 확보, 구조 리스크 격리, 각 단계 독립 게이트.
- **옵션 2: 곧장 cell-centred를 boundary-scoped로 한 번에.** I1+I2 동시, ROADMAP 정합,
  단 단일 대형 변경(2D→단일→MLG 게이트로 완화).
- **옵션 3: PLAN 문자 그대로 cell-centred 전면(전체볼륨 유지).** I1 방치 → **비추천**.

## 7. 리스크 & 열린 결정

- ★**시간보간 제거의 정당성**: explosion+2ghost가 정말 비동기 서브사이클서 시간보간
  불요인지 = **coupling ordering 문제**(현재 "C→F after advance"). **2D testbed서
  최우선 검증**. 실패 시 → vertex+f_prev 유지하고 I1(boundary-only)만 취함.
- **fine_shape 축소**(각 레벨 −1 노드)의 물리/region 영향 → region 재정의 필요.
- **half-cell 정렬**: VTK origin·setup origin·ALM hub-in-fine 판정 정확도.
- **2D↔3D 이중 유지보수**: 모든 coupling/overlap 변경 ×2(`_2d`).
- REFINE_RATIO=2 하드코딩 6곳(참고, 변경 안 함).
