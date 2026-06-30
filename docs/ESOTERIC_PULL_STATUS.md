# Esoteric Pull Implementation Status

**Date:** 2026-04-12
**Status:** 전체 완료 (BGK + Cumulant + MLG + RegBC + ALM + D2Q9)

---

## 1. 구현 완료 항목

### Esoteric BGK D3Q27 Kernel

- **파일:** `src/kernels/esoteric_d3q27.py`
- **방식:** Lehmann (2022) Esoteric Pull — single buffer, even/odd alternation
- **방향 순서:** Paired opposites (i, i+1) for i=1,3,...,25
  - 표준 D3Q27 → Esoteric 매핑: `_STD_TO_ESO`
- **Race-free:** 각 메모리 주소를 읽는 thread = 쓰는 thread (동일)
- **BC 통합:** kernel 내부 node_type flag
  - NODE_FLUID (0): collision
  - NODE_SOLID (1): skip (implicit bounce-back)
  - NODE_EQ_BC (2): equilibrium BC (f = f_eq at target rho, u)
  - NODE_NEUMANN (3): no collision, passthrough
  - NODE_SPONGE (4): collision + blending toward target
- **MEM Force:** kernel 내부 atomicAdd (post-collision fhn에서 계산)

### Esoteric D2Q9 BGK Kernel

- **파일:** `src/kernels/esoteric_d2q9.py`
- **구조:** D3Q27과 동일 — 4 pairs (E/W, N/S, NE/SW, NW/SE), 2D indexing
- **BC 통합:** FLUID, SOLID, EQ_BC, NEUMANN, SPONGE, REG_INLET, REG_OUTLET
- **유틸:** gather/scatter, convert std↔eso, init_f
- **Simulation dispatch:** `len(domain_shape)==2` + BGK → 자동 2D Esoteric
- **검증:** 평형 보존, gather roundtrip (diff=0), 2D Poiseuille parabolic (대칭 1.9e-7)

### ALM 통합 (Esoteric)

- **방식:** 2-pass — Pass 1: gather physical f → macroscopic → ALM, Pass 2: kernel(body_force)
- **`_advance_esoteric_with_alm()`**: esoteric_gather_physical → macroscopic.compute → _compute_body_force → kernel launch
- **Dispatch:** `advance()`에서 `al_model is not None` 체크 후 분기
- **Cumulant 커널:** body_force 파라미터 이미 구현 (Guo correction + source term + sign-flip)
- **BGK 커널:** body_force 미지원 (ALM은 주로 Cumulant와 사용)
- **검증:** constant Fx body force → Poiseuille parabolic profile, 대칭 확인

### Regularized BC (Esoteric)

- **노드 타입:** `NODE_REG_INLET = 5` (velocity Dirichlet), `NODE_REG_OUTLET = 6` (pressure Dirichlet)
- **bc_normal 배열:** (int8) 면 법선 방향 인코딩 (0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z)
- **커널 내부 동작:**
  1. bc_normal에서 interior neighbor 인덱스 계산
  2. Interior neighbor의 f를 Esoteric LOAD로 27개 수집
  3. Interior에서 rho_int, u_int, f_eq_int 계산
  4. Pi_neq = Σ f_neq * c_ia * c_ib (stress tensor)
  5. Target (rho_bc, u_bc) 결정: inlet은 prescribed, outlet은 interior u + relaxation K
  6. f = f_eq(rho_bc, u_bc) + f^(1)(Pi_neq) 재구성
- **Outlet:** bc_rho에 target 밀도, bc_ux에 relaxation K 저장
- **BGK + Cumulant 커널 모두 동일 블록 삽입**
- **simulation.py:** 'regularized'/'regularised' method 자동 감지, node_type/bc_normal 설정
- **검증:** Poiseuille 채널 200 steps, parabolic profile, wall ux=0, NaN 없음

### MLG Coupling (Esoteric-aware)

- **구현:** `esoteric_gather_physical()` / `esoteric_scatter_physical()` in `esoteric_d3q27.py`
- **원리:** Esoteric 메모리에서 물리적 f를 추출 (self + neighbor 슬롯에서 parity 기반 gather)
- **gather:** LOAD 연산 복제 — `xp.roll()`로 이웃 접근, 13쌍 × 2방향
- **scatter:** gather의 역연산 — `xp.roll(shift=+c_i)`로 이웃 슬롯에 쓰기
- **roundtrip:** gather + scatter = identity (검증 완료, max_diff = 0.0)
- **MultiLevelGrid 수정:**
  - `_get_physical_f(sim)`: Esoteric → standard 변환 (non-Esoteric은 copy)
  - `_set_physical_f(sim, f)`: standard → Esoteric scatter (non-Esoteric은 copyto)
  - `_save_f_prev(k)`: physical layout으로 f_prev 저장 (temporal interpolation 호환)
  - `_coupling_c2f()` / `_coupling_f2c()`: gather → coupling → scatter 래핑
- **coupling 코드 변경 없음:** coupling은 항상 standard-ordered f를 받음
- **dtype 버그 수정:** CUDA interpolation kernel이 float64를 float32로 읽던 문제 → dtype 체크 추가
- **검증:** 2-level BGK Esoteric MLG, 50 coarse steps, rho 오차 < 3e-6

### Esoteric Cumulant D3Q27 Kernel

- **파일:** `src/kernels/esoteric_cumulant_d3q27.py`
- **방식:** Esoteric Pull + Cumulant collision (Geier 2015) in one kernel
- **Collision:** Forward Chimera → Cumulant transform → Relax (Galilean) → Backward Cumulant → Backward Chimera
- **방향 매핑:** Esoteric paired ordering → K[cx+1][cy+1][cz+1] (bijective, 검증 완료)
- **Relaxation:** omega_1 (shear), omega_bulk, omega_high (3rd-10th order)
- **Guo forcing:** body_force 파라미터 추가, velocity correction + source term
- **BC 통합:** BGK 버전과 동일 (FLUID, SOLID, EQ_BC, NEUMANN, SPONGE)
- **Simulation dispatch:** CumulantCollision 감지 시 자동 Esoteric Cumulant 커널 선택
- **검증:** 표준 cumulant kernel과 post-collision f 완전 일치 (max_diff = 0.00e+00)
- **안정성:** Taylor-Green vortex 100 steps @ omega=1.8 정상 감쇄

### Implicit Bounce-back

- Solid 노드가 kernel skip → slot swap 미발생 → 자동 방향 반전
- Poiseuille parabolic profile 검증 완료 (wall ux=0)

### Checkpoint 지원

- `esoteric_step` (parity) checkpoint extra_data에 저장
- Restart 시 f가 이미 Esoteric layout → 이중 변환 방지
- `_esoteric_f_already_set` flag로 제어

### MEM Force

- Kernel 내부에서 post-collision `fhn[]`로 force 계산
- `needs_bounce` 배열을 Esoteric 방향 순서로 변환
- atomicAdd로 global force 누적

### Sponge BC

- NODE_SPONGE: collision 후 f_target으로 blending
- sigma는 bc_uz 배열에 저장 (재사용)
- Sponge layer의 노드별 sigma 프로파일 설정

---

## 2. 성능

| Config | Baseline | Phase 3 | Esoteric | 총 가속 |
|--------|----------|---------|----------|---------|
| Poiseuille 16K | 0.96 | 1.52 | **205** | **213x** |
| Sphere 1.46M | 13.69 | 149 | **830** | **61x** |

### 성능 향상 원인

1. **Single buffer:** f_post 제거 → memory bandwidth 절반
2. **BC kernel 내부:** Python BC overhead 완전 제거 (이전 병목 80%)
3. **Implicit bounce-back:** 별도 HWBB pass 불필요
4. **1 kernel launch per step:** collision + streaming + BC 통합

---

## 3. 메모리

```
이전 (Phase 3):
  f(108) + f_post(108) + rho(4) + u(12) = 232 B/node + streaming indices

Esoteric:
  f(108) + rho(4) + u(12) + node_type(1) + bc_arrays(16) = 141 B/node
  (f_post 제거, streaming indices 제거)

43M nodes:
  이전:  ~20 GB
  Esoteric: ~6 GB (추정)
```

---

## 4. 검증 결과

| 검증 | 결과 |
|------|------|
| Poiseuille drift | -0.0016% |
| Sphere Cd | 1.1260 (baseline 1.1251, 오차 0.08%) |
| Parabolic profile | 대칭, wall에서 u=0 |
| Checkpoint restart | 정상 (parity 복원, CSV 연속) |
| Conservation | 안정적 |

---

## 5. 최적화 (2026-04-12)

### Cumulant 3D Esoteric 통합 ✓

- Cumulant collision이 이제 Esoteric Pull 경로 사용 (BGK와 동일)
- `set_distribution()`: `isinstance(self.collision, (BGKCollision, CumulantCollision))`
- 별도 `streaming.compute()` 호출 제거 — single kernel launch
- block_size 최적화: 128 → 96 (80 regs/thread, 50% occupancy, 3075 MLUPS)

### ALM CUDA 파이프라인 ✓

- `src/actuator/alm_cuda_kernels.py`: Interpolation + BEM RawKernel
- Interpolation: CuPy array ops → RawKernel (4.3x speedup, atomicAdd)
- BEM: Python loop → CUDA kernel (25.1x speedup, GPU polar lookup)
- 전체 ALM step: 2.72 ms → 0.33 ms (8.1x speedup)
- 수치 정확도: rel_err < 1e-14 (machine precision)

### Multi-GPU Domain Decomposition 설계 ✓

- `src/parallel/`: Partition, HaloExchange (P2P/NCCL), DomainDecomposition
- x축 분할, D3Q27 halo = 1 cell, CuPy + NCCL 통신
- 43M nodes (360×360×330) on 4 GPUs: ~10.7M/GPU
- MLG 호환: per-level halo exchange, coarse level halo = 2 cells
- 설계 문서: `docs/MULTI_GPU_DESIGN.md`

---

## 6. 이전 완료 항목

### ~~Cumulant Esoteric Kernel~~ ✓ 완료

- ~~Chimera transform을 Esoteric kernel의 collision 블록에 삽입~~
- ~~Esoteric 방향 순서에서 K[3][3][3] 매핑~~
- ~~omega_bulk, omega_high 파라미터 추가~~

### ~~MLG Coupling~~ ✓ 완료

- ~~Esoteric f에서 macroscopic/f_eq/f_neq 추출 시 parity 고려~~
- ~~C→F: coupling 시점의 parity에 따라 slot 매핑~~
- ~~F→C: 동일~~
- ~~f_prev: physical (standard) layout으로 저장~~

### ~~D2Q9 Esoteric Kernel~~ ✓ 완료

- ~~D3Q27과 동일한 구조, 방향 수만 다름~~
- ~~9 directions, 4 pairs~~

### ~~Regularized BC~~ ✓ 완료

- ~~Esoteric에서 f_neq 추출: parity에 따라 slot↔direction 매핑~~
- ~~f_neq / (1-omega) 보정 (post-collision에서 추출 시)~~

### ~~ALM 통합~~ ✓ 완료

- ~~Macro → ALM → collision 순서~~
- ~~Esoteric kernel에서 2-pass 구조 또는 별도 macro kernel~~

---

## 6. 초기화 주의사항

Esoteric f의 메모리 레이아웃은 step parity에 의존:

```python
# Fresh start (t=0, even):
f_eso = convert_f_std_to_esoteric(xp, f_std)  # 방향 재배열
f_mem = init_f_esoteric(xp, f_eso, t_start=0)  # slot swap for even

# Checkpoint restart:
f_mem = checkpoint['f']  # 이미 Esoteric layout
sim._esoteric_f_already_set = True  # 이중 변환 방지
sim._esoteric_step = checkpoint['esoteric_step']  # parity 복원
```

---

## 7. 파일 목록

### 신규

```
src/kernels/esoteric_d3q27.py           — Esoteric BGK D3Q27 kernel + utilities + gather/scatter
src/kernels/esoteric_cumulant_d3q27.py  — Esoteric Cumulant D3Q27 kernel
src/kernels/esoteric_d2q9.py            — Esoteric BGK D2Q9 kernel + utilities
```

### 수정

```
src/solver/simulation.py        — _init_esoteric(), _advance_esoteric(), dispatch (BGK+Cumulant)
src/solver/initializer.py       — Esoteric restart parity restore
src/solver/output_manager.py    — Esoteric force output, checkpoint extra
src/streaming/stream.py         — CUDA streaming dispatch (index 배열 미할당)
src/grid/multi_level_grid.py    — Esoteric-aware coupling (gather/scatter 래핑)
src/grid/coupling.py            — CUDA interp float64 dtype guard
src/kernels/interpolation_d3q27.py — float32 dtype 검증 추가
```
