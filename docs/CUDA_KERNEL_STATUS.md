# CUDA Kernel 변환 현황

**Date:** 2026-04-11
**Status:** 진행 중

---

## 1. 완료된 CUDA Kernel

### Collision Kernel (BGK + Cumulant)

| 파일 | 함수 | 원래 | 효과 |
|------|------|------|------|
| `src/kernels/bgk_d3q27.py` | `BGKCollideKernelD3Q27` | CuPy ~15 launches | 1 launch |
| `src/kernels/cumulant_d3q27.py` | `CumulantCollideKernelD3Q27` | CuPy ~80 launches | 1 launch |

Macroscopic + Guo correction + collision을 register에서 수행.
Cumulant과 BGK의 속도 차이 소멸 (112 vs 112 MLUPS).

### Streaming Kernel

| 파일 | 함수 | 원래 | 효과 |
|------|------|------|------|
| `src/kernels/streaming_d3q27.py` | `StreamingKernelD3Q27` | CuPy advanced indexing | 1 launch + 인덱스 배열 제거 (-324 B/node) |

Precomputed index 배열 (src_x, src_y, src_z, q_idx) 불필요.

### HalfwayBounceBack Kernel

| 파일 | 함수 | 원래 | 효과 |
|------|------|------|------|
| `src/kernels/bounce_back_d3q27.py` | `HWBBKernelD3Q27` | Python loop Q=27회 | 1 launch (apply + reset) |

3가지 모드: apply(separate f_post), apply_inplace(f_post=None), reset_solid.

### MEM Force Kernel

| 파일 | 함수 | 원래 | 효과 |
|------|------|------|------|
| `src/kernels/mem_force_d3q27.py` | `MEMForceKernelD3Q27` | Python loop dim×Q=78회 | 1 launch (atomicAdd reduction) |

---

## 2. 성능 누적

### Sphere Single Grid (BGK, 180×90×90, Re=100)

| 단계 | MLUPS | 가속 (vs baseline) |
|------|-------|-------------------|
| Baseline (CuPy) | 13.69 | 1.0× |
| + Fused Collision | 41.23 | 3.0× |
| + CUDA Streaming + HWBB | 68.61 | 5.0× |
| **+ CUDA MEM Force** | **71.26** | **5.2×** |

### Empty Box (128³)

| Collision | Baseline | 현재 | 가속 |
|-----------|----------|------|------|
| BGK | 33.86 | 113.55 | 3.4× |
| Cumulant | 27.20 | 119.86 | 4.4× |

### Step 내부 시간 분배 (Sphere 180×90×90)

```
Collision CUDA kernel:  0.55 ms  (3%)
Streaming CUDA kernel:  0.43 ms  (2%)
BC (Python/CuPy):      16.90 ms (94%)
  └ Python overhead:   ~14.7 ms (87%)
  └ f_eq 계산:          ~2.2 ms  (13%)
```

**현재 병목: BC의 Python overhead (94%)**
격자가 커지면 BC 비율이 줄어듦 (BC는 O(N²), bulk은 O(N³)).

### MLG Sphere (2-level, Cumulant)

```
MLG advance: 37.14 ms
  Collision+Streaming: ~2 ms (L0+L1)
  BC:                 ~17 ms
  C→F coupling:       10.21 ms (28%)
  F→C coupling:        2.88 ms (8%)
  Coupling total:     13.10 ms (35%)
```

---

## 3. 미완료 CUDA Kernel

### BC (Regularized, Sponge, Corner, Neumann)

```
병목 비율:   94% (소규모 격자), ~30% (대규모 격자)
CUDA 난이도: HIGH (6면 × 다른 BC type × corner 처리)
병목 원인:   Python overhead (face slice 접근, 함수 호출, ~30 CuPy launches)
             f_eq 계산 자체는 전체의 13%만
상태:        보류 — 격자 커지면 비율 줄어듦
```

### MLG Coupling (C→F, F→C)

```
병목 비율:   35% (MLG에서)
핵심 연산:   C→F interpolation (cubic, 3축 순차)
CUDA 난이도: MEDIUM-HIGH
상태:        진행 예정
```

---

## 4. 코드 구조

```
src/kernels/
├── __init__.py
├── bgk_d3q27.py              # Collision (FP32) + Stream-collide (미사용)
│                                + FP16S collision (미사용)
├── cumulant_d3q27.py          # Collision (FP32) + Stream-collide (미사용)
├── streaming_d3q27.py         # Pull streaming
├── bounce_back_d3q27.py       # HWBB apply + reset_solid
└── mem_force_d3q27.py         # MEM force (atomicAdd reduction)
```

### Dispatch in simulation.py

```python
set_distribution():
    if GPU + D3Q27:
        BGK → BGKCollideKernelD3Q27
        Cumulant → CumulantCollideKernelD3Q27
        StreamingKernelD3Q27 (항상)
        HWBBKernelD3Q27 (obstacle 있을 때)

_advance_fused():
    fused_collision(f → f_post)          # CUDA kernel
    streaming(f_post → f)                # CUDA kernel
    bc_manager.apply_all(f, f_post)      # Python/CuPy (기존)
    hwbb.apply(f, f_post)               # CUDA kernel (obstacle)

_advance_fused_with_alm():
    macro(f → rho, u)                   # CuPy (ALM needs u first)
    ALM(u → force)                      # Python/CuPy
    fused_collision(f, force → f_post)  # CUDA kernel
    streaming(f_post → f)               # CUDA kernel
    BC + HWBB                           # 위와 동일
```

### CPU Fallback

GPU 없거나 2D → `_advance_default()` (기존 Python/CuPy 코드 그대로).

---

## 5. 주의사항 (indexing)

CUDA kernel에서 3D 인덱스 ↔ 1D 인덱스 변환:

```c
// CuPy C-contiguous: f[q, x, y, z], z가 가장 빠르게 변함
// 올바른 변환:
int ix = idx / (Ny * Nz);
int rem = idx - ix * Ny * Nz;
int iy = rem / Nz;
int iz = rem - iy * Nz;
int linear = ix * Ny * Nz + iy * Nz + iz;

// 주의: iz * Nx * Ny + iy * Nx + ix 는 잘못됨!
```

Streaming kernel에서 이 버그를 발견하여 수정 완료.
Stream-collide kernel (미사용)에는 아직 미수정 — 활성화 시 수정 필요.

---

## 6. 검증 결과

모든 CUDA kernel은 baseline과 비교하여 검증 완료:

| 검증 | Baseline | CUDA | 일치 |
|------|----------|------|------|
| Poiseuille drift | +0.3653% | +0.3653% | bit-exact |
| Sphere Cd | 1.1251 | 1.1251-1.1253 | 상대 오차 < 0.02% |

FP32 연산 순서 차이로 인한 미세한 차이만 존재 (MEM Force atomicAdd).
