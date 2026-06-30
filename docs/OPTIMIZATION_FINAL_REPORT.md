# Optimization & Acceleration: Final Report

**Date:** 2026-04-11
**Hardware:** NVIDIA RTX 3090 (24 GB, 936 GB/s)
**Solver:** LBM D3Q27, BGK + Cumulant, Multi-Level Grid

---

## 1. 최종 성과

### 성능

| Config | Baseline | 최종 | 가속 |
|--------|----------|------|------|
| Sphere BGK (reg. inlet) | 13.69 | **149 MLUPS** | **10.9×** |
| Sphere BGK (eq. inlet) | 50.80 | **259 MLUPS** | **5.1×** |
| Sphere MLG 2-level | 24.19 | **129 MLUPS** | **5.3×** |
| Empty Box BGK 128³ | 33.86 | **209 MLUPS** | **6.2×** |
| Empty Box Cumulant 128³ | 27.20 | **197 MLUPS** | **7.2×** |

### 메모리

```
Before: 340 B/node  (f + f_post + f_new + rho + u)
After:  232 B/node  (f + f_post + rho + u) + streaming index 배열 제거
절감:   -32% (+ index 배열 -324 B/node)
```

### 핵심 달성

- **BGK와 Cumulant 속도 차이 소멸** (112 vs 112 MLUPS → CuPy launch overhead 제거)
- **MLG에서 5.3× 가속** (coupling interpolation CUDA kernel화 포함)
- **모든 검증 통과** (Cd 상대 오차 < 0.05%, conservation drift 동일)

---

## 2. 적용된 최적화 목록

### Phase 1: Buffer Reduction [완료]

| 항목 | 내용 |
|------|------|
| 변경 | f_new 버퍼 제거 (3→2 buffers) |
| 파일 | `src/solver/simulation.py` |
| 효과 | 메모리 -32% (340→232 B/node) |
| 검증 | Cd bit-exact, conservation drift 동일 |

### Phase 3: Fused CUDA Collision Kernel [완료]

| 항목 | 내용 |
|------|------|
| 변경 | Macroscopic + Guo correction + collision → 단일 CUDA kernel |
| 파일 | `src/kernels/bgk_d3q27.py`, `src/kernels/cumulant_d3q27.py` |
| 효과 | BGK 3.3×, Cumulant 4.1× 가속 |
| 검증 | Cd 상대 오차 0.02% (FP32 연산 순서 차이) |

Cumulant kernel: Chimera forward/backward + cumulant transform + Galilean correction을
모두 register에서 수행 (~90 registers/thread).

### CUDA Streaming Kernel [완료]

| 항목 | 내용 |
|------|------|
| 변경 | CuPy advanced indexing → CUDA pull-streaming kernel |
| 파일 | `src/kernels/streaming_d3q27.py` |
| 효과 | 추가 ~10% 가속 + precomputed index 배열 제거 (-324 B/node) |
| 검증 | CuPy 결과와 bit-exact 일치 |

**발견된 버그:** 초기 구현에서 3D 인덱스 변환이 잘못됨
(`iz*Nx*Ny + iy*Nx + ix` 사용 → CuPy C-contiguous는 `ix*Ny*Nz + iy*Nz + iz`).
단위 테스트에서 발견하여 수정 완료.

### CUDA HalfwayBounceBack Kernel [완료]

| 항목 | 내용 |
|------|------|
| 변경 | Python Q-loop (27회) → 단일 CUDA kernel |
| 파일 | `src/kernels/bounce_back_d3q27.py` |
| 효과 | Sphere에서 +66% 추가 가속 (obstacle이 있는 케이스) |
| 검증 | Cd bit-exact |

3가지 kernel: apply(separate f_post), apply_inplace(f_post=None), reset_solid.
D3Q27 opposite 인덱스 하드코딩 (lattice.opp와 일치 확인).

### CUDA MEM Force Kernel [완료]

| 항목 | 내용 |
|------|------|
| 변경 | Python dim×Q loop (78회) → 단일 CUDA kernel (atomicAdd) |
| 파일 | `src/kernels/mem_force_d3q27.py` |
| 효과 | +3.9% 추가 가속 (force 계산 step에서) |
| 검증 | Cd 상대 오차 0.02% (atomicAdd FP32 누적 순서 차이) |

### CUDA MLG Interpolation Kernel [완료]

| 항목 | 내용 |
|------|------|
| 변경 | CuPy cubic interpolation 3축 순차 → 3× CUDA 1D kernel |
| 파일 | `src/kernels/interpolation_d3q27.py` |
| 효과 | MLG에서 C→F coupling 가속 (전체 MLG 5.3× 가속에 기여) |
| 검증 | CuPy 결과와 max diff < 1.12e-8 |

### CornerBC 캐싱 [완료]

| 항목 | 내용 |
|------|------|
| 변경 | 12 edges + 8 corners의 f_eq를 init 시 미리 계산, apply 시 write만 |
| 파일 | `src/boundary/corner_bc.py` |
| 효과 | BC 시간 54% 절감 → Sphere에서 2.1× 추가 가속 |
| 검증 | Cd 0.02% 이내, conservation drift 미세 차이 (rho_dummy 사용) |

상수 target(velocity×velocity, velocity×sponge 등)인 edge/corner만 캐싱.
Extrapolation 필요한 경우(wall×wall with no prescribed rho) 기존 runtime 경로 유지.

### Face BC Equilibrium 캐싱 [완료]

| 항목 | 내용 |
|------|------|
| 변경 | Equilibrium 모드 VelocityDirichletBC의 f_eq를 init 시 캐싱 |
| 파일 | `src/boundary/face_bc.py` |
| 효과 | Equilibrium inlet 사용 시 face BC 시간 대폭 절감 |
| 검증 | Cd 동일 |

Regularized 모드는 매 step Pi_neq 추출 필요 → 캐싱 불가, 그대로 유지.

### BC Save-then-write [완료]

| 항목 | 내용 |
|------|------|
| 변경 | Bounce-back BC에 f_post=None 시 in-place 안전 처리 추가 |
| 파일 | `wall.py`, `domain_wall.py`, `face_bc.py` |
| 효과 | 향후 streaming fusion 시 활용 가능 |
| 검증 | 기존 경로 bit-exact 유지 |

---

## 3. 시도했으나 보류된 최적화

### Streaming Fusion (collision + streaming 통합)

**3회 시도, 모두 실패.**

| 시도 | 문제 | 결과 |
|------|------|------|
| 1차 | BC가 f_post=f(같은 배열)에서 방향쌍 충돌 | Cd=3.17 |
| 2차 | BC save-then-write로 방향쌍 해결 | Cd=3.17 (다른 원인) |
| 3차 | Eq+Neumann BC만 사용 (f_neq 미사용) | Cd=3.16 (여전히 실패) |

**근본 원인:** Pull-collide kernel에서 `f_dst[i, x]`의 물리적 의미가 다름.

```
현재:   f_post[i, x] = 노드 x에서 방향 i로 나가는 post-collision 분포
Fusion: f_dst[i, x]  = 방향 i에서 온 값이 x에서 충돌한 결과 ≠ 나가는 분포
```

HBB가 기대하는 "방향 i로 나가려던 분포가 반사"가 성립하지 않음.

**추가 문제:**
- BC가 쓴 값이 다음 step pull의 source → 경계값이 이웃에 확산
- Post-collision에서 f_neq = (1-ω)·f_neq_pre → regularized BC, MLG coupling 영향

**해결 경로:**
- Esoteric Pull/Push (별도 아키텍처)
- Collide-then-push (scatter write, GPU 비효율)
- Flag-based in-kernel BC (kernel 내부에서 경계 처리)

**코드 상태:** Stream-collide kernel은 `src/kernels/`에 유지 (삭제하지 않음).
활성화하려면 BC 아키텍처 전면 재설계 필요.

상세: `docs/STREAMING_FUSION_ISSUES.md`

### FP16 Shifted Storage

**1회 시도, 실패.**

| 문제 | 내용 |
|------|------|
| sim.f 의미 변화 | 실제 f → shifted (f-w) |
| 외부 접근 오류 | macro.compute(shifted f) → rho ≈ 0 → conservation 발산 |
| BC 비호환 | BC가 실제 f를 쓰지만 FP16S 배열에 unshifted 값 기록 |

**근본 원인:** `sim.f`가 20+곳에서 직접 접근됨. 내부 표현 변경 시 모든 접근점 수정 필요.

**검증 결과:** Conservation drift -70728% (완전 발산)

**해결 경로:** sim.f property 추상화 (read 시 FP32 변환, write 시 FP16S 변환).
또는 streaming fusion 완료 후 f 상태가 고정되면 추상화 용이.

**코드 상태:** FP16S kernel은 `src/kernels/bgk_d3q27.py`에 유지. 단위 테스트 통과 (kernel 자체는 정확).

상세: `docs/FP16S_ISSUES.md`

---

## 4. 발견된 버그 및 수정 사항

### Streaming kernel 3D 인덱스 순서

```c
// 잘못됨 (z-major):
int iz = idx / (Nx * Ny);
int iy = (idx - iz*Nx*Ny) / Nx;
int ix = idx - iz*Nx*Ny - iy*Nx;
int linear = iz * Nx * Ny + iy * Nx + ix;

// 올바름 (x-major, CuPy C-contiguous):
int ix = idx / (Ny * Nz);
int iy = (idx - ix*Ny*Nz) / Nz;
int iz = idx - ix*Ny*Nz - iy*Nz;
int linear = ix * Ny * Nz + iy * Nz + iz;
```

Streaming kernel에서 수정 완료.
Stream-collide kernel(미사용)에는 아직 미수정 — 활성화 시 수정 필요.

### Force CSV restart 데이터 유실

```
문제: setup 단계에서 force_mgr.initialize()가 CSV를 'w'로 덮어씀
      → initializer에서 start_step을 알아도 이미 데이터 삭제됨
수정: initialize()를 info 출력만, open_csv(start_step)를 별도 메서드로 분리
      → initializer에서 start_step 확정 후 open_csv() 호출
영향: force_history.csv, rotor_performance.csv 모두 수정
검증: restart 후 290행 연속 확인 (이전 190행 보존 + 새 100행)
```

### MLUPS 계산 오류 (MLG)

```
문제: Level 0 노드 수만 카운트, fine level 무시
수정: updates_per_step = Σ(N_k × 2^k) for k=0..M-1
      (level k는 coarse step당 2^k번 advance)
```

### Lattice validation FP32 실패

```
문제: lattice weights를 FP32로 저장하면 isotropy check 통과 불가
      (8/27의 FP32 표현 오차 > tol=1e-12)
수정: validation을 항상 FP64 참조 lattice로 수행
      + validate_lattice config 옵션으로 선택적 실행
```

### Unicode encoding 에러 (서버)

```
문제: setup_log.txt에 τ, ν 문자 포함 → ASCII locale 서버에서 UnicodeEncodeError
수정: open(..., encoding='utf-8') 추가
```

---

## 5. 현재 코드 구조

### CUDA Kernels

```
src/kernels/
├── __init__.py
├── bgk_d3q27.py              # BGK collision (FP32)
│                                + Stream-collide (미사용, 보류)
│                                + FP16S collision (미사용, 보류)
├── cumulant_d3q27.py          # Cumulant collision (FP32)
│                                + Stream-collide (미사용, 보류)
├── streaming_d3q27.py         # Pull streaming
├── bounce_back_d3q27.py       # HWBB apply + reset_solid
├── mem_force_d3q27.py         # MEM force (atomicAdd)
└── interpolation_d3q27.py     # Cubic interpolation 3×1D (MLG coupling)
```

### Dispatch in Simulation.advance()

```
GPU + D3Q27:
  _advance_fused():
    collision CUDA kernel → streaming CUDA kernel → BC(Python+cached) → HWBB CUDA kernel

  _advance_fused_with_alm():
    macro(CuPy) → ALM(Python) → collision CUDA → streaming CUDA → BC → HWBB CUDA

CPU or 2D:
  _advance_default():
    macro(CuPy) → collision(CuPy) → streaming(CuPy) → BC(Python)
```

### BC 캐싱 전략

```
CornerBC (corner_bc.py):
  Init: 상수 target edge/corner → f_eq 미리 계산
  Apply: 캐싱된 f_eq를 slice write만 수행
         extrapolation 필요한 경우만 runtime 계산

VelocityDirichletBC (face_bc.py):
  Init: equilibrium 모드 → f_eq 미리 계산 (_cached_f_eq_3d)
  Apply: 캐싱된 f_eq를 face slice에 write만
         regularized 모드 → 매 step Pi_neq 계산 (캐싱 불가)
```

---

## 6. Step 내부 시간 분배

### Sphere (180×90×90, regularized inlet)

```
                Before          After
Collision:      ~8 ms (50%)     0.55 ms (6%)     ← CUDA kernel
Streaming:      ~2 ms (13%)     0.43 ms (5%)     ← CUDA kernel
HWBB:           ~3 ms (19%)     ~0.1 ms (1%)     ← CUDA kernel
CornerBC:       ~9 ms           ~0.5 ms           ← 캐싱
Face BC (reg):  ~7.5 ms         ~7.5 ms (80%)     ← 변경 없음 (regularized)
Sponge:         ~0.2 ms         ~0.2 ms           ← 변경 없음
MEM Force:      ~1 ms/10step    ~0.1 ms/10step    ← CUDA kernel
```

**현재 병목: Regularized Face BC (80%).** Equilibrium으로 전환하면 해소됨.

### Sphere (180×90×90, equilibrium inlet)

```
Collision:      0.55 ms (14%)
Streaming:      0.43 ms (11%)
Face BC (eq):   ~1 ms (26%)     ← 캐싱 적용
CornerBC:       ~0.5 ms (13%)   ← 캐싱 적용
HWBB:           ~0.1 ms (3%)
기타:           ~1.3 ms (33%)
```

훨씬 균형잡힌 분배. 특정 단일 병목 없음.

---

## 7. ALM 모듈 호환성

모든 최적화는 ALM과 독립적:

| 최적화 | ALM 영향 | 이유 |
|--------|----------|------|
| Buffer reduction | 없음 | ALM은 u만 사용 |
| Fused collision | 없음 | ALM 경로는 별도 (macro→ALM→collision) |
| CUDA streaming | 없음 | streaming은 ALM 이후 |
| CUDA HWBB | 없음 | obstacle BC는 ALM과 독립 |
| CUDA MEM force | 없음 | force 계산은 f_post에서 |
| MLG interpolation | 없음 | coupling은 level 간 전달 |
| BC 캐싱 | 없음 | BC는 ALM 이후 |

`_advance_fused_with_alm()`에서 ALM이 u를 필요로 하므로:
macro를 먼저 계산(CuPy) → ALM → fused collision kernel.
이 2-pass 구조는 유지됨.

---

## 8. 미해결 과제

### Streaming Fusion

Pull-collide 구조의 본질적 한계로 보류.
해결에는 Esoteric Pull 또는 kernel 내부 BC 통합 필요.
상세: `docs/STREAMING_FUSION_ISSUES.md`

### FP16 Storage

sim.f 인터페이스 추상화 필요.
Streaming fusion 해결 후 함께 적용이 효율적.
상세: `docs/FP16S_ISSUES.md`

### Esoteric Pull (Phase 2)

모든 보류 최적화의 전제조건.
BC 전면 재설계 + f 상태 전환 + parity 관리 필요.
가장 큰 잠재적 이득 (메모리 추가 -50%, streaming fusion 가능).

### BC Python Overhead (Regularized)

Regularized BC가 매 step Pi_neq를 계산 — CUDA kernel화 어려움.
대안: equilibrium BC 사용 (외부 유동에서 충분), 또는 Pi_neq 계산을 CUDA kernel화.

### Multi-GPU

독립적으로 적용 가능. Domain decomposition + halo exchange.

---

## 9. 검증 요약

| 검증 | Baseline | 최종 | 일치 |
|------|----------|------|------|
| Poiseuille BGK drift | +0.3653% | +0.3664% | 미세 차이 (CornerBC rho_dummy) |
| Sphere Single Cd | 1.1251 | 1.1253 | 상대 오차 0.02% |
| Sphere MLG Cd | 1.0598 | 1.0603 | 상대 오차 0.05% |
| Checkpoint restart | 290행 연속 | 정상 | ✅ |

모든 차이는 FP32 연산 순서 (CUDA kernel vs CuPy) 또는
atomicAdd 누적 순서에 의한 것으로, 물리적으로 무의미.

---

## 10. 파일 변경 목록

### 신규 파일

```
src/kernels/__init__.py
src/kernels/bgk_d3q27.py
src/kernels/cumulant_d3q27.py
src/kernels/streaming_d3q27.py
src/kernels/bounce_back_d3q27.py
src/kernels/mem_force_d3q27.py
src/kernels/interpolation_d3q27.py
```

### 수정 파일

```
src/solver/simulation.py         — buffer 감소, fused kernel dispatch
src/solver/initializer.py        — CSV open_csv(start_step) 분리
src/solver/output_manager.py     — MLUPS 계산 수정, performance.csv
src/solver/setup.py              — precision config, lattice validation 선택적
src/utilities/force_calculator.py — CUDA kernel dispatch, open_csv 분리
src/utilities/flux_utils.py      — CSV restart 보존
src/grid/coupling.py             — CUDA interpolation dispatch
src/boundary/wall.py             — save-then-write bounce-back
src/boundary/domain_wall.py      — save-then-write bounce-back
src/boundary/face_bc.py          — save-then-write + equilibrium 캐싱
src/boundary/corner_bc.py        — f_eq 캐싱
src/lattice/d3q27.py             — dtype 파라미터
src/lattice/d2q9.py              — dtype 파라미터
src/lattice/__init__.py          — get_lattice dtype 전달
src/collision/bgk.py             — lattice.dtype 사용
src/collision/cumulant.py        — lattice.dtype 사용
src/io/mlg_vtk_writer.py         — solid_mask 출력, velocity_magnitude 제거
```

### 문서

```
docs/OPTIMIZATION_STRATEGY.md
docs/OPTIMIZATION_RISK_ANALYSIS.md
docs/OPTIMIZATION_STATUS.md
docs/OPTIMIZATION_FINAL_REPORT.md    (이 문서)
docs/CUMULANT_MEMORY_ANALYSIS.md
docs/CUDA_KERNEL_STATUS.md
docs/STREAMING_FUSION_ISSUES.md
docs/FP16S_ISSUES.md
docs/PHASE1_RESULT.md
docs/PHASE3_RESULT.md
docs/ALM_MLG_COMPATIBILITY.md
```

---

## 11. 참고 문헌

- Lehmann, "Esoteric Pull and Esoteric Push", Computation 10(6), 2022
- Lehmann et al., "Accuracy and performance of the LBM with 64/32/16-bit", Phys. Rev. E 106, 2022
- Geier & Schoenherr, "Esoteric Twist", Computation 5(2), 2017
- FluidX3D: https://github.com/ProjectPhysX/FluidX3D
- Autodesk/AGAL, "GPU Grid Refinement for LBM", IPDPS 2024
- CuPy RawKernel: https://docs.cupy.dev/en/stable/user_guide/kernel.html
