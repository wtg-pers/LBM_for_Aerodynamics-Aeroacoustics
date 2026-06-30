# Phase 3 Result: Fused CUDA Kernel (BGK + Cumulant)

**Date:** 2026-04-10
**Status:** Complete
**Branch:** master

---

## 1. 변경 내용

### 신규 파일

| 파일 | 내용 |
|------|------|
| `src/kernels/__init__.py` | 모듈 초기화 |
| `src/kernels/bgk_d3q27.py` | D3Q27 BGK fused collision CUDA kernel |
| `src/kernels/cumulant_d3q27.py` | D3Q27 Cumulant fused collision CUDA kernel |

### 수정 파일

| 파일 | 변경 |
|------|------|
| `src/solver/simulation.py` | fused kernel dispatch 로직 추가 (`_advance_fused`, `_advance_fused_with_alm`) |

### Fusion 범위

```
현재 advance() 흐름:
  Step 1: macro(f → rho, u)        ← ┐
  Step 2: ALM(u → force)            ←  │ ALM 없으면 Step 1+3+4 fused
  Step 3: u += F/(2ρ)               ← ┘
  Step 4: collision(f → f_post)     ← ┘ FUSED into 1 CUDA kernel
  Step 5: streaming(f_post → f)     ← 기존 CuPy 유지
  Step 6: BC(f, f_post)             ← 기존 코드 유지
```

ALM 없을 때: **1 kernel launch** (macro + Guo + collision)
ALM 있을 때: macro(1) + ALM + **1 kernel launch** (Guo + collision)
Streaming + BC: 기존 CuPy 코드 (변경 없음)

### Dispatch 로직

```python
# simulation.py — set_distribution()
if GPU and D3Q27:
    if BGK:       → BGKCollideKernelD3Q27 (RawKernel)
    if Cumulant:  → CumulantCollideKernelD3Q27 (RawKernel)
else:
    → _advance_default() (기존 CuPy path, CPU fallback)
```

---

## 2. Kernel 설계

### BGK Kernel (`bgk_d3q27.py`)

```
Per thread (1 node):
  1. Load f[27] from global memory     → 108B read
  2. Compute rho, u in registers       → 0B
  3. Guo correction (if force)         → +12B read
  4. BGK collision in registers         → 0B
  5. Guo source term (if force)        → 0B
  6. Write f_post[27], rho, u          → 108+4+12 = 124B write

Registers: ~34/thread
Block size: 256
```

### Cumulant Kernel (`cumulant_d3q27.py`)

```
Per thread (1 node):
  1. Load f[27] → registers            → 108B read
  2. Compute rho, u                     → 0B
  3. Guo correction                     → +12B read (if force)
  4. Reshape to K[3][3][3] (= f_local) → 0B (register reuse)
  5. Forward Chimera (3 × 1D transform) → 0B
  6. Forward cumulant transform          → 0B
  7. Relax cumulants + Galilean          → 0B
  8. Backward cumulant transform         → 0B
  9. Backward Chimera                    → 0B
  10. Write f_post[27], rho, u          → 124B write

Registers: ~90/thread
Block size: 128 (lower due to register pressure)
```

전체 Cumulant collision (Chimera forward/backward + cumulant transform + relaxation)이
**register에서 완결** — global memory 접근은 처음 read와 마지막 write뿐.

---

## 3. 성능 결과

### Benchmark: Empty Box 128³ (2.1M nodes)

| Collision | Baseline (CuPy) | Phase 3 (CUDA) | 가속 |
|-----------|-----------------|----------------|------|
| BGK | 33.86 MLUPS | **111.79 MLUPS** | **3.3×** |
| Cumulant | 27.20 MLUPS | **112.10 MLUPS** | **4.1×** |

**확인 파일:**
- BGK: `bench_empty_box_bgk/csv/performance.csv`
- Cumulant: `bench_empty_box_cumulant/csv/performance.csv`

### Benchmark: Sphere Drag

| Config | Baseline | Phase 3 | 가속 |
|--------|----------|---------|------|
| Sphere Single (BGK) | 13.69 MLUPS | **41.23 MLUPS** | **3.0×** |
| MLG Sphere (Cumulant, 3-lv) | 17.43 MLUPS | **35.31 MLUPS** | **2.0×** |

**확인 파일:**
- Sphere Single: `val_sphere_single/csv/performance.csv`
- MLG Sphere: `bench_sphere_mlg/csv/performance.csv`

### 핵심 발견: BGK vs Cumulant 속도 동일화

| | Baseline | Phase 3 |
|---|---|---|
| BGK MLUPS | 33.86 | 111.79 |
| Cumulant MLUPS | 27.20 | 112.10 |
| **Cumulant/BGK 비율** | **0.80×** (20% 느림) | **1.00×** (동일!) |

CuPy에서 Cumulant이 느렸던 이유는 FLOPs가 아닌 **kernel launch 횟수**
(80+ vs 15)였음. Fused kernel로 전환하면서 이 overhead가 완전히 제거됨.
LBM의 memory-bound 특성상 Cumulant의 추가 연산(~430 FLOPs)은
메모리 대기 시간에 완전히 숨겨짐.

---

## 4. 검증 결과

### 4.1 Poiseuille BGK

```
Metric: mass conservation drift (final step)
Baseline: +0.3653%
Phase 3:  +0.3653%
Result:   동일
```

**확인 파일:**
- Baseline: `validation_baseline/poiseuille_bgk_csv/mass_conservation.csv`
- Phase 3: `val_poiseuille_single_bgk/csv/mass_conservation.csv`

**Note:** 중간 step의 drift가 소수점 이하에서 미세하게 다름 (FP32 연산 순서 차이).
최종 값은 동일.

### 4.2 Poiseuille Cumulant

```
Metric: mass conservation drift (final step)
Baseline: +0.3538%
Phase 3:  +0.3538%
Result:   동일
```

**확인 파일:**
- Baseline: `validation_baseline/poiseuille_cumulant_csv/mass_conservation.csv`
- Phase 3: `val_poiseuille_single_cumulant/csv/mass_conservation.csv`

### 4.3 Sphere Drag Single BGK (Re=100)

```
Metric: Cd (last step)
Baseline: 1.12482
Phase 3:  1.12505
Relative error: 0.02% (< 0.1% tolerance)
Result:   PASS
```

**확인 파일:**
- Baseline: `validation_baseline/sphere_single_csv/force_history.csv`
- Phase 3: `val_sphere_single/csv/force_history.csv`

### 4.4 정밀도 기준

CUDA kernel과 CuPy array ops의 FP32 연산 순서가 다르므로 bit-exact 일치는
불가능. 허용 기준:

| 항목 | 기준 | 결과 |
|------|------|------|
| Conservation drift (최종값) | 동일 | ✅ |
| Cd 상대 오차 | < 0.1% | ✅ (0.02%) |
| 정성적 유동장 | ParaView 목시 확인 | ✅ |

---

## 5. CPU Fallback 호환성

GPU가 없거나 numpy 사용 시 자동으로 기존 CuPy path로 fallback:

```python
# simulation.py — set_distribution()
if xp.__name__ == 'cupy' and len(domain_shape) == 3:
    # BGK → BGKCollideKernelD3Q27
    # Cumulant → CumulantCollideKernelD3Q27
else:
    # _advance_default() — 기존 Python/CuPy 코드 (변경 없음)
```

기존 collision/*.py 코드는 삭제하지 않고 유지.

---

## 6. MLG 호환성

각 level의 Simulation이 독립적으로 advance() 호출하므로 fused kernel이
자동 적용됨. MLG coupling은 kernel 외부에서 동작하여 변경 없음.

- Level별 다른 shape → kernel 파라미터 N으로 처리 (shape-agnostic)
- Level별 다른 tau → omega_1 파라미터로 전달
- f_prev 저장 → advance() 전에 수행 (변경 없음)
- C→F, F→C coupling → f 배열에 대해 동작 (변경 없음)

---

## 7. 메모리 절감 (Phase 1+3 누적)

```
Original:        340 B/node (f + f_post + f_new + rho + u)
Phase 1:         232 B/node (f_new 제거)
Phase 3:         232 B/node (변경 없음 — Phase 3은 속도 최적화)

59.3M nodes: 13.1 GB (Phase 1 수준 유지)
```

Phase 3은 **속도 최적화**이므로 메모리 사용량은 Phase 1과 동일.
메모리 추가 절감은 Phase 4 (FP16 storage)에서 진행.

---

## 8. 남은 최적화 가능성

### Streaming fusion (다음 단계)

현재: fused collision kernel (1 launch) + streaming (1 CuPy op) = 2 launches
Streaming fusion 시: pull + collision이 단일 kernel = 1 launch

예상 추가 효과:
- Memory traffic: 232B → 108+124 = 232B (변화 없음, 이미 최소)
- Kernel launch overhead: 2 → 1 (marginal)
- 실질적 효과: streaming의 CuPy advanced-indexing overhead 제거

### 현재 위치

```
                   MLUPS    대비 이론 최대(4,300)
Original CuPy:     ~30         0.7%
Phase 3 Fused:    ~112         2.6%
Streaming Fusion:  ~200?       4.7% (예상)
Phase 4 FP16:     ~400?       9.3% (예상)
```

RTX 3090 이론 최대 대비 아직 격차가 크지만, 현재 bottleneck은
fused kernel 자체가 아닌 streaming + BC + CuPy 오버헤드.

---

## 9. 다음 단계

Streaming fusion: collision kernel에 pull streaming을 통합하여
단일 kernel에서 read(neighbors) → collision → write(current node) 수행.

이를 위해 f_post/BC 호환 문제를 해결해야 함:
- HalfwayBounceBack이 f_post를 필요로 함
- BC는 별도 pass로 유지하되, f_post를 kernel에서 기록
