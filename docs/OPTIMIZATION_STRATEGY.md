# Memory Optimization & Computation Acceleration Strategy

**Date:** 2026-04-09 (revised)
**Status:** Design Phase
**Target Hardware:** NVIDIA RTX 3090 (24 GB, 936 GB/s bandwidth)

---

## 1. Current Performance Profile

### 1.1 Solver Architecture
- **Array layout:** SoA `f[Q, Nx, Ny, Nz]` (correct for GPU)
- **Streaming:** Pull scheme with precomputed indices (correct for GPU)
- **Computation:** CuPy array operations (einsum, broadcasting)
- **Precision:** Configurable float32/float64 (float32 default)

### 1.2 Memory Footprint (D3Q27, float32)

```
Per node: 27 directions × 4 bytes = 108 bytes

Current buffer allocation:
  f       : 108 bytes/node    (current distribution)
  f_new   : 108 bytes/node    (post-streaming buffer)
  f_post  : 108 bytes/node    (post-collision buffer)
  rho     :   4 bytes/node    (density)
  u       :  12 bytes/node    (velocity 3D)
  ─────────────────────────────
  Total   : 340 bytes/node

For 59.3M nodes: 340 × 59.3M = 19.2 GB
```

### 1.3 Computation Bottleneck
- LBM arithmetic intensity: ~2.4 FLOP/byte (memory-bound)
- CuPy array ops: BGK ~15, Cumulant ~80+ kernel launches per step
- Each launch reads/writes f from global memory → redundant bandwidth
- **Estimated current: ~100-300 MLUPS**
- **RTX 3090 theoretical max (D3Q27): ~4,300 MLUPS**

### 1.4 BGK vs Cumulant Memory (동일 격자 기준)

```
(27, N) 크기 배열 동시 존재 개수:
  BGK:       f + f_post + f_new + cu + f_eq       = 5개 × 108 B/node
  Cumulant:  f + f_post + f_new + input + output   = 5개 × 108 B/node
                                   (chimera pair)

→ 메모리 peak는 거의 동일 (Chimera sequential 분해 덕분)
→ 차이는 Cumulant relaxation의 (N,) 소형 배열 ~28개 ≈ 112 B/node (부차적)
```

---

## 2. Optimization Phases

### Phase 1: Buffer Reduction — f_new 제거 (Memory)

**Goal:** 3 f-buffers → 2 f-buffers

**Current:**
```
collision(f → f_post)     # f_post = post-collision
streaming(f_post → f_new) # f_new = post-streaming + BC
swap(f, f_new)            # f becomes new state
```

**Proposed:**
```
collision(f → f_post)     # f_post = post-collision
streaming(f_post → f)     # overwrite f directly (pull reads from f_post)
# No swap needed — f is updated in-place
```

**Impact:**
```
Before: 340 bytes/node × 59.3M = 19.2 GB
After:  232 bytes/node × 59.3M = 13.1 GB  (-32%)
```

**Risk:** LOW — pull streaming의 source(f_post)와 dest(f)가 다른 배열이므로 충돌 없음.
HalfwayBounceBack 인터페이스만 소폭 수정 필요.

**모듈 호환성:**

| 모듈 | 영향 | 안전 |
|------|------|------|
| Macroscopic, Collision (BGK/Cumulant) | 없음 | ✅ |
| Streaming (Pull) | dest를 f_new→f로 변경 | ✅ |
| HalfwayBounceBack | 인터페이스 수정 (f_new→f) | ⚠️ 소폭 |
| Regularized/Sponge BC | 대상만 f_new→f | ✅ |
| MEM Force | f_post 유지 | ✅ |
| ALM | u만 사용 | ✅ |
| MLG coupling | f_prev는 advance() 전 저장 | ✅ |
| Checkpoint | f 저장 동일 | ✅ |

---

### Phase 2: Esoteric Pull Streaming (Memory + Speed) — 마지막 적용

**Goal:** f 배열 1개만으로 collision + streaming 수행.

**Reference:** Lehmann (2022), Geier & Schoenherr (2017)

**Principle:** Even/odd 스텝에서 읽기/쓰기 패턴 교대.
```
Even step: f[i](x) ← collide( f[i](x - c_i) )   # pull + collide
Odd step:  f[i](x + c_i) ← collide( f[i](x) )    # collide + push
```

**Impact:**
```
Phase 1 이후:  232 bytes/node
Phase 2 적용:  124 bytes/node  (f_post도 제거)
59.3M nodes: 7.0 GB  (-64% from original)
```

**Key advantage:** HalfwayBounceBack이 implicit (별도 BC pass 불필요).

**위험도: HIGH — 모든 Phase 중 가장 높음**

단일 격자에서는 비교적 단순하지만, MLG에서 적용하려면 아래 과제 해결 필요:

#### MLG 적용을 위한 과제

**과제 1: C→F 시간 보간과 even/odd parity**

현재 C→F coupling은 `f_prev`(half-step)와 `f`(full-step)를 시간 보간한다.
Esoteric에서는 even/odd 스텝의 f 메모리 패턴이 다르므로:
- f_prev와 f의 parity가 다를 수 있음
- 보간 전에 parity를 일치시키는 변환이 필요

```
해결 방안:
  A. Coupling 시점을 항상 even step으로 강제
     → nested time-stepping에서 coarse step = 2 fine steps이므로
       fine의 2번째 step 후(= even) coupling → parity 일치
  B. Parity-aware interpolation
     → 보간 함수가 source의 parity를 인자로 받아 방향 매핑 변환
```

**과제 2: F→C coupling의 f 기록**

F→C는 fine.f를 읽어 coarse.f의 excised region에 기록한다.
Esoteric에서는 기록 대상의 parity에 맞춰 방향 인덱스를 매핑해야 한다.

```
해결 방안:
  C. Coupling 전용 temporary buffer 사용
     → coupling 시에만 일시적으로 standard layout으로 변환
     → 메모리 절감 효과 감소하지만 coupling 정확도 보장
  D. Parity-aware write
     → 기록 시 even/odd에 따라 방향 인덱스 flip
```

**과제 3: MEM Force 계산**

f_post가 별도 저장되지 않으므로 momentum exchange 계산 불가.

```
해결 방안:
  E. Collision 내부에서 force 계산
     → fused kernel에서 collision 후 바로 f_post에서 force 누적
     → global memory에 f_post를 쓰지 않고 register에서 처리
  F. Force 계산 step에서만 f_post 임시 저장
     → force_interval(10 step마다)에서만 추가 buffer 사용
```

**과제 4: 비자명 BC (Regularized, Sponge)**

Even/odd에 따라 face의 f 방향 인덱스 해석이 달라짐.

```
해결 방안:
  G. BC 방향 매핑 테이블을 even/odd 2벌 준비
     → advance() 호출 시 현재 parity에 맞는 테이블 선택
  H. BC를 fused kernel에 통합 (flag array 기반)
     → Phase 3의 fused kernel과 결합하면 자연스러움
```

**Phase 2 적용 전제조건:**
- Phase 1, 3이 완료된 상태에서 적용
- Fused kernel이 even/odd dispatch를 내부적으로 처리
- MLG coupling의 parity-aware 로직 검증 완료

---

### Phase 3: Fused CUDA Kernel (Speed)

**Goal:** Collision + streaming을 단일 RawKernel로 통합.

**Reference:** FluidX3D, Autodesk/AGAL (2024)

**Fused pull-collide kernel:**
```python
kernel_code = r'''
extern "C" __global__
void collide_stream_d3q27(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    float* __restrict__ rho_out,      // macroscopic output
    float* __restrict__ u_out,        // macroscopic output
    const float* __restrict__ force,  // body force (or NULL)
    const int Nx, const int Ny, const int Nz,
    const float omega
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny * Nz) return;
    int N = Nx * Ny * Nz;

    // 1. Pull: f_local[27] ← f_in[neighbors]  (1× global read)
    // 2. Macroscopic: rho, u from f_local       (register)
    // 3. Guo correction: u += F/(2rho)          (register, if force)
    // 4. Collision: f_local → f_post_local      (register)
    // 5. Write: f_out[idx] ← f_post_local       (1× global write)
    // 6. Write: rho_out[idx], u_out[idx]         (macroscopic output)
}
'''
```

**Performance:**
```
Memory traffic:  27 reads + 27 writes = 54 × 4 bytes = 216 bytes/node
RTX 3090:        936 GB/s → 936e9 / 216 = 4,333 MLUPS theoretical
Expected:        ~3,000-3,500 MLUPS (10-30× speedup)
```

**Register pressure:**
```
BGK:       ~34 registers (well within limit)
Cumulant:  ~90 registers (occupancy 감소 가능, __launch_bounds__ 튜닝)
```

**ALM과의 호환 — 2-pass 구조:**
```
현재:   macro(f→rho,u) → ALM(u→force) → collision(f,force→f_post) → streaming
Fused:  ALM이 u를 필요로 하므로 단순 1-pass 불가

해결: 2-pass
  Pass 1: Macroscopic kernel (f→rho,u)  [lightweight]
  Pass 2: ALM.step(u→force)             [CPU/GPU hybrid]
  Pass 3: Fused collision+streaming(f,force→f_out)

ALM이 없으면: 1-pass (macro도 kernel 내부에서 계산, rho/u global write)
```

**MEM Force 호환 — 조건부 f_post 기록:**
```python
if step % force_interval == 0:
    fused_kernel_with_fpost(f, f_out, f_post, ...)  # f_post도 기록
else:
    fused_kernel(f, f_out, ...)                      # f_post 생략 (성능 최적)
```

**모듈 호환성:**

| 모듈 | 영향 | 안전 |
|------|------|------|
| Macroscopic | kernel에 통합 또는 별도 pass | ✅ |
| BGK/Cumulant | kernel로 대체 (Python fallback 유지) | ✅ |
| Streaming | kernel에 통합 | ✅ |
| HalfwayBounceBack | 초기: 별도 pass 유지. 이후: flag array로 kernel 통합 | ⚠️ |
| Regularized/Sponge BC | 별도 pass 유지 (face slices만 처리) | ✅ |
| MEM Force | 조건부 f_post 기록 | ⚠️ |
| ALM | 2-pass 구조 | ⚠️ |
| MLG coupling | kernel 결과 f에 대해 동작 — 변경 없음 | ✅ |
| Checkpoint | 동일 | ✅ |

---

### Phase 4: FP16 Shifted Storage (Memory + Speed)

**Goal:** f를 FP16으로 저장, 연산은 FP32.

**Reference:** Lehmann et al., Phys. Rev. E 106, 015308 (2022)

```
Storage:    f_stored[i] = float16(f[i] - w[i])
Arithmetic: f[i] = float32(f_stored[i]) + w[i]
```

**Impact:**
```
f storage: 27 × 2 bytes = 54 bytes/node (vs 108 for FP32)
Phase 1+4:  54(f) + 54(f_post) + 4(rho) + 12(u) = 124 bytes/node
59.3M nodes: 7.0 GB  (vs 19.2 GB original → -64%)
Bandwidth:   108 bytes/node/step → theoretical ~8,667 MLUPS
```

**MLG coupling 주의:**
```
f_neq = f - f_eq  (f ≈ f_eq이므로 뺄셈에서 유효 자릿수 손실 위험)

해결: coupling 연산만 FP32에서 수행
  1. f_stored(FP16) → f(FP32) 복원
  2. FP32에서 f_neq 계산, 보간, rescaling
  3. 결과를 FP16으로 다시 저장
```

**Fused kernel과의 자연스러운 결합:**
Phase 3의 fused kernel 내부에서 FP16↔FP32 변환을 처리하면
외부 모듈은 정밀도를 의식할 필요 없음. **Phase 3 이후에 적용.**

**모듈 호환성:**

| 모듈 | 영향 | 안전 |
|------|------|------|
| Collision (fused kernel 내부) | FP16 read → FP32 연산 → FP16 write | ✅ |
| BC (Regularized, Sponge, BB) | FP32로 BC 계산 후 FP16 저장 | ⚠️ |
| MEM Force | FP16→FP32 복원 후 계산 | ⚠️ |
| ALM | u는 FP32 유지 | ✅ |
| MLG coupling | **FP32에서 수행 필수** (정밀도) | ⚠️ |
| Checkpoint | FP16 또는 FP32로 저장 선택 | ⚠️ |

---

### Phase 5: MLG-Specific Optimizations (Speed)

**5a. Fused coupling kernels**

C→F, F→C의 interpolation + rescaling을 단일 kernel로 통합.
Reference: Autodesk/AGAL, IPDPS 2024 (1.3-2× speedup).

**5b. Skip computation in excised regions**

L0의 excised region에서 collision 생략 (F→C가 덮어쓰므로).
Streaming은 경계 이웃을 위해 유지.

**5c. Asynchronous level execution (CUDA streams)**

```
Stream 1: L0.advance()
Stream 2:                  L1.advance() (overlapped)
Sync:     C→F, F→C coupling (synchronization barrier)
```

---

### Phase 6: Advanced Optimizations (Future)

- **6a. FP16C (compressed non-equilibrium):** f_neq만 FP16으로 저장
- **6b. GPU-native AMR (Autodesk approach):** octree 기반
- **6c. Multi-GPU:** domain decomposition + halo exchange

---

## 3. Implementation Roadmap (Revised)

```
Phase 1: Buffer Reduction              [1-2 days, LOW RISK]
  → f_new 제거
  → 메모리 -32% (19.2 → 13.1 GB)
  → 검증 후 Phase 3으로

Phase 3: Fused CUDA Kernel             [5-7 days, MEDIUM RISK]
  → BGK kernel 먼저, Cumulant kernel
  → ALM 2-pass 구조, MEM Force 조건부 분기
  → 속도 10-30× 향상

Phase 4: FP16 Shifted Storage          [3-5 days, MEDIUM RISK]
  → Fused kernel 내부에서 FP16↔FP32 변환
  → MLG coupling은 FP32 유지
  → 추가 메모리 -50%, 속도 +80%

Phase 5: MLG Coupling Fusion           [3-5 days]
  → C→F, F→C kernel 통합
  → Excised region skip

Phase 2: Esoteric Pull                 [5-7 days, HIGH RISK — 마지막]
  → Phase 1+3+4 완료 후 적용
  → MLG parity-aware coupling 구현
  → f_post 완전 제거 (MEM Force는 kernel 내부 처리)
  → 추가 메모리 절감 (f 1개만 유지)

Phase 6: Advanced                      [Future]
  → Multi-GPU, GPU-native AMR
```

---

## 4. Phase별 검증 항목

### Phase 1 검증

| 항목 | 방법 | 기준 |
|------|------|------|
| **단일 격자 BGK** | Poiseuille 3D, parabolic profile 비교 | L2 error < 1% (이전과 동일) |
| **단일 격자 Cumulant** | 동일 Poiseuille 3D | L2 error < 1% |
| **Sphere drag (single)** | Cd at Re=100 | 이전 단일 격자 결과와 bit-exact |
| **MLG Poiseuille** | 2-level MLG Poiseuille 3D | L2 error < 1% (이전 MLG 결과와 동일) |
| **MLG Sphere drag** | 3-level MLG Re=100k | Cd가 이전 결과와 일치 |
| **Checkpoint restart** | 5000 step 저장 → restart → 10000 step | 연속 실행과 f가 bit-exact |
| **Force CSV 연속성** | restart 후 force_history.csv | 이전 데이터 보존, 이어쓰기 확인 |
| **Conservation** | mass drift 확인 | 이전과 동일한 drift 수준 |

### Phase 3 검증

| 항목 | 방법 | 기준 |
|------|------|------|
| **BGK kernel 정확도** | Poiseuille 3D, CuPy 결과와 비교 | 최대 상대 오차 < 1e-5 (FP32 precision) |
| **Cumulant kernel 정확도** | 동일 비교 | 최대 상대 오차 < 1e-5 |
| **BGK kernel MLUPS** | Empty box 128³ 벤치마크 | > 2,000 MLUPS (현재 대비 10×+) |
| **Cumulant kernel MLUPS** | 동일 | > 1,500 MLUPS |
| **Sphere drag Cd** | Re=100 single grid | CuPy 결과와 0.1% 이내 |
| **ALM 호환** | NTNU BT1 config (있으면) | C_T, C_P가 CuPy 결과와 일치 |
| **MLG coupling** | 3-level sphere drag | Cd가 CuPy 결과와 일치 |
| **BC 정상 동작** | Regularized inlet/outlet + sponge | 유동장 정성적 확인 (ParaView) |
| **MEM Force 분기** | force_interval step에서 Cd, 비-interval step에서 skip 확인 | CSV 출력 정상 |
| **CPU fallback** | numpy로 실행 | CuPy와 동일 결과 (fused kernel 대신 Python path) |

### Phase 4 검증

| 항목 | 방법 | 기준 |
|------|------|------|
| **FP16 정밀도** | Poiseuille Cd, FP32 결과와 비교 | 상대 오차 < 0.1% |
| **FP16 안정성** | Re=100k sphere 30000 steps | divergence 없음 |
| **MLG coupling 정밀도** | 3-level sphere drag, FP32 vs FP16 | Cd 차이 < 1% |
| **Conservation** | mass drift, FP32 vs FP16 | drift 증가 < 10× |
| **메모리 절감 확인** | GPU memory 사용량 측정 | 예상값과 일치 (124 B/node) |
| **MLUPS 향상 확인** | 벤치마크 | FP32 대비 50%+ 향상 |

### Phase 5 검증

| 항목 | 방법 | 기준 |
|------|------|------|
| **Coupling fusion 정확도** | MLG Poiseuille, 기존 coupling과 비교 | L2 error 동일 |
| **Excised skip** | MLG sphere drag, skip 전후 비교 | Cd 동일, 속도 향상 확인 |
| **Async streams** | MLG 3-level 벤치마크 | 동기 대비 속도 향상 측정 |

### Phase 2 검증

| 항목 | 방법 | 기준 |
|------|------|------|
| **Even/odd 정확도** | Poiseuille 3D, standard streaming과 비교 | L2 error 동일 |
| **Implicit bounce-back** | Sphere drag, 기존 HWBB와 비교 | Cd 동일 |
| **Parity-aware C→F** | MLG Poiseuille | L2 error 동일 |
| **Parity-aware F→C** | MLG sphere drag | Cd 동일 |
| **Non-trivial BC** | Regularized inlet + sponge, even/odd 전환 | 유동장 연속성 확인 |
| **MEM Force in kernel** | Cd 비교 | 기존 MEM과 동일 |
| **Checkpoint parity** | even step 저장 → restart → 검증 | 연속 실행과 동일 |
| **MLG 전체 통합** | Geier config 5-level sphere drag | Cd 수렴, 안정성 |

---

## 5. Expected Results After Full Optimization

### Memory

```
Current:                  340 bytes/node → 19.2 GB (59.3M nodes)
Phase 1 (f_new 제거):     232 bytes/node → 13.1 GB
Phase 4 (FP16S):          124 bytes/node →  7.0 GB
Phase 2 (Esoteric, 최종):  70 bytes/node →  3.9 GB

→ Same GPU (24 GB) can handle 5.5× more nodes
→ 59.3M nodes at 3.9 GB → room for 300M+ nodes
```

### Speed

```
Current (CuPy ops):       ~100-300 MLUPS
Phase 3 (fused FP32):     ~3,000-3,500 MLUPS
Phase 4 (fused FP16S):    ~5,000-7,000 MLUPS

→ 10-50× speedup over current implementation
```

### Combined Impact

```
59.3M nodes, current:
  Memory: 19.2 GB, Speed: ~200 MLUPS → ~300 sec/1000 steps

59.3M nodes, fully optimized (Phase 1+3+4):
  Memory: 7.0 GB, Speed: ~5,000 MLUPS → ~12 sec/1000 steps

59.3M nodes, with Phase 2:
  Memory: 3.9 GB, Speed: ~5,000+ MLUPS → ~12 sec/1000 steps
```

---

## 6. Design Principles

1. **Backward compatibility:** CuPy array-ops path remains as CPU/debug fallback.
   Fused kernels are selected automatically when running on GPU.

2. **Incremental validation:** Each phase is independently testable.
   Results must match the previous phase within floating-point tolerance.

3. **Separation of concerns:**
   - `src/kernels/` for CUDA kernel source code
   - `src/collision/` keeps the Python interface unchanged
   - `Simulation.advance()` dispatches to fused or default path

4. **Physics first:** Optimization must not compromise physical accuracy.
   Cd convergence test is the primary validation metric.

5. **MLG compatibility:** All optimizations must work with multi-level grids.
   Phase 2 (Esoteric) requires parity-aware coupling — applied last.

---

## 7. Key References

- Lehmann, "Esoteric Pull and Esoteric Push", Computation 10(6), 2022
- Lehmann et al., "Accuracy and performance of the LBM with 64-bit, 32-bit,
  and customized 16-bit number formats", Phys. Rev. E 106, 2022
- Geier & Schoenherr, "Esoteric Twist", Computation 5(2), 2017
- FluidX3D: https://github.com/ProjectPhysX/FluidX3D
- Autodesk/AGAL, "GPU Grid Refinement for LBM", IPDPS 2024
- Autodesk, "GPU-Native AMR", arXiv:2308.08085, 2025
- NVIDIA, "Shared Memory Register Spilling", Developer Blog, 2024
