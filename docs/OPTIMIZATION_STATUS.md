# Optimization Status Report

**Date:** 2026-04-11
**Current State:** Phase 1 + Phase 3 (collision-only fusion) 완료

---

## 1. 적용 완료된 최적화

### Phase 1: Buffer Reduction (f_new 제거)

```
변경: f, f_post, f_new (3 buffers) → f, f_post (2 buffers)
원리: Pull streaming에서 source(f_post)와 dest(f)가 다른 배열 → f_new 불필요
효과: 메모리 -32% (340 → 232 B/node)
검증: Cd bit-exact 일치
```

### Phase 3: Fused CUDA Collision Kernel (BGK + Cumulant)

```
변경: CuPy array ops (15-80 kernel launches) → 단일 CUDA kernel (1 launch)
원리: macroscopic + Guo correction + collision을 register에서 수행
효과: BGK 3.3×, Cumulant 4.1× 속도 향상
      Cumulant과 BGK의 속도 차이 소멸 (112 vs 112 MLUPS)
검증: Cd 상대 오차 0.02% (FP32 연산 순서 차이)
```

### BC Save-then-write (Phase 3 부산물)

```
변경: HalfwayBounceBack 등 bounce-back BC에 f_post=None 시 in-place 처리 추가
원리: 같은 배열에서 읽고 쓸 때 방향쌍 충돌 방지 (outgoing 값 미리 저장)
효과: 기존 호환성 유지 + 향후 streaming fusion 시 활용 가능
검증: 기존 경로(f_post 전달) bit-exact 동일
```

### 현재 성능

```
              Baseline    현재        가속
BGK 128³:     33.86      111.79 MLUPS  3.3×
Cumulant 128³: 27.20      112.10 MLUPS  4.1×
Memory:       340        232 B/node    -32%
RTX 3090 이론 최대: ~4,300 MLUPS (달성률 2.6%)
```

---

## 2. 시도했으나 보류된 최적화

### Streaming Fusion (collision + streaming 통합)

```
목표: 2 launches → 1 launch (collision + streaming을 단일 kernel로)
구현: Pull-collide kernel (BGK/Cumulant 모두) + ping-pong buffer
```

**문제:**

| # | 이슈 | 위험도 |
|---|---|---|
| 1 | **f 상태 변화**: post-streaming → post-collision | 근본적 |
| 2 | **BC 값의 전파**: BC가 쓴 값이 다음 step의 pull source로 이웃에 확산 | 근본적 |
| 3 | **f_neq 스케일링**: post-collision에서 f_neq = (1-ω)·f_neq_pre | HIGH |
| 4 | Regularized BC의 f_neq 추출이 잘못됨 | HIGH |
| 5 | MLG coupling의 f_neq rescaling 공식 변경 필요 | HIGH |

**검증 결과:**
- Poiseuille drift: +0.3653% → +0.0118% (값 다름, 벽 BC는 정상)
- Sphere Cd: 1.125 → 3.168 (완전 실패 — inlet BC 전파 문제)

**해결 조건:**
- 단순 BC 재설계로는 불가 — pull-collide 구조의 본질적 한계
- f_dst[i,x]가 "방향 i로 나가는 분포"가 아닌 "방향 i에서 온 값의 collision 결과"
- 해결에는 아키텍처 변경 필요: Esoteric Pull, collide-push, 또는 kernel 내 BC 통합

**수정 필요 범위:**

| 모듈 | f_neq 사용 | 수정 내용 |
|------|-----------|----------|
| Regularized BC | ✅ | f_neq_post / (1-ω)로 보정 |
| MLG coupling | ✅ | rescaling 공식 변경 |
| Sponge BC | ❌ | target 조정 (소폭) |
| Bounce-back | ❌ | 이미 동작 (post-collision에서 자연스러움) |
| Neumann/Corner | ❌ | 변경 없음 |
| Macroscopic | ❌ | rho, u는 동일 |
| ALM | ❌ | u만 사용, 영향 없음 |

**보류 사유:** 실제 수정 범위는 2곳이지만, BC 인터페이스에 omega 전달 +
물리적 검증이 필요하여 충분한 설계 후 진행이 바람직.

**코드 상태:** Stream-collide kernel은 `src/kernels/`에 유지 (삭제하지 않음).

---

### FP16 Shifted Storage

```
목표: f 배열을 float16(f - w)로 저장, 메모리/대역폭 -50%
구현: FP16S CUDA kernel (BGK), FP16↔FP32 변환 kernel 내부 처리
```

**문제:**

| # | 이슈 | 위험도 |
|---|---|---|
| 1 | **sim.f의 의미 변화**: 실제 f → shifted (f-w) | 근본적 |
| 2 | f에 접근하는 모든 외부 코드가 오작동 (20+곳) | HIGH |
| 3 | macro.compute(sim.f)가 shifted 값으로 rho 계산 → 발산 | HIGH |
| 4 | BC가 실제 f를 쓰지만 FP16S 배열에 unshifted 값 기록 | HIGH |
| 5 | MLG coupling에서 f_eq/f_neq 분해 시 precision 손실 | MEDIUM |

**검증 결과:**
- Conservation drift: -99.6% → -70728% (완전 발산)
- 원인: finalize()에서 macro.compute(shifted f) → rho ≈ 0

**해결 조건:**
- sim.f 인터페이스 추상화 (property로 외부에 FP32 반환)
- 또는 f에 접근하는 모든 코드에 shift/unshift 변환 추가

**보류 사유:** sim.f가 read+write 양방향으로 사용되어 property 추상화가 어려움.
Streaming fusion과 동일한 아키텍처 수준의 변경 필요.

**코드 상태:** FP16S kernel은 `src/kernels/bgk_d3q27.py`에 유지.

---

### Esoteric Pull (Phase 2, 최초부터 후순위)

```
목표: f 배열 1개로 축소 (even/odd step 교대 접근)
```

**예상 문제:** Streaming fusion의 모든 문제 + even/odd parity 관리.
Streaming fusion이 해결되지 않는 한 시도 불가.

---

## 3. 미적용 최적화 (향후 가능)

### Phase 5: MLG Coupling Fusion

```
목표: C→F, F→C interpolation + rescaling을 단일 CUDA kernel로
현재 상태: CuPy array ops로 동작 중
기대 효과: coupling 1.3-2× 향상 (Autodesk/AGAL 참고)
의존성: 없음 (독립적으로 적용 가능)
```

### Streaming CUDA Kernel화 (fusion 아님)

```
목표: CuPy advanced indexing → custom CUDA streaming kernel
현재 상태: streaming은 1회 CuPy 연산
기대 효과: launch overhead 소폭 감소
의존성: 없음
```

### Multi-GPU

```
목표: domain decomposition + halo exchange
의존성: 없음 (독립)
```

---

## 4. 의존 관계 맵

```
                        ┌─── Phase 5: MLG Coupling Fusion (독립)
                        │
Phase 1 (완료) ─────────┤
                        │
Phase 3 (완료) ─────────┼─── Streaming CUDA Kernel화 (독립)
                        │
                        │         ┌─── FP16S ←── sim.f 추상화
                        │         │
                        └─── Streaming Fusion
                              │
                              ├─── Regularized BC 수정 (f_neq / (1-ω) 보정)
                              ├─── MLG coupling 수정 (f_neq rescaling)
                              ├─── BC 인터페이스에 omega 전달
                              └─── Sponge target 조정
                              │
                              └─── Esoteric Pull (모든 위의 것 + parity 관리)
```

---

## 5. 우선순위 재검토

### 즉시 가능 (의존성 없음)

| 작업 | 효과 | 난이도 |
|------|------|--------|
| **Phase 5: MLG coupling fusion** | MLG 속도 1.3-2× | 중 |
| Streaming CUDA kernel화 | 소폭 속도 향상 | 저 |
| Multi-GPU | 대규모 도메인 | 고 |

### Streaming Fusion 선행 작업 (순서대로)

| 순서 | 작업 | 수정 범위 |
|------|------|----------|
| 1 | BC 인터페이스에 omega 전달 | BC constructor + apply() |
| 2 | Regularized BC: f_neq 보정 | `f_neq / (1-ω)` 추가 |
| 3 | MLG coupling: f_neq rescaling | coupling.py 공식 변경 |
| 4 | Sponge: target 조정 | sponge.py |
| 5 | Streaming fusion 재활성화 | simulation.py |
| 6 | 전체 검증 | validation configs |

### FP16S 선행 작업

| 순서 | 작업 | 수정 범위 |
|------|------|----------|
| 1 | sim.f 접근 추상화 설계 | Simulation 클래스 |
| 2 | 외부 f 접근점 20+곳 수정 | 전체 솔버 |
| 3 | FP16S kernel 통합 | simulation.py |

---

## 6. 판단

### 현재 달성한 것

```
메모리:  340 → 232 B/node (-32%)
속도:    ~30 → ~112 MLUPS (3.3-4.1× 가속)
Cumulant = BGK 속도 동일 달성
```

### 추가 최적화의 비용 대비 이득

| 작업 | 이득 | 비용 | 비용/이득 |
|------|------|------|----------|
| MLG coupling fusion | MLG 1.3-2× | 중간 (독립) | **좋음** |
| Streaming fusion | 1 launch 절약 (~10-20% 향상) | 높음 (BC+coupling 수정) | 보통 |
| FP16S | 메모리 -50%, 대역폭 -50% | 매우 높음 (f 추상화) | 보통 |
| Esoteric Pull | 메모리 추가 -50% | 매우 높음 (전체 재설계) | 낮음 |

### 추천

1. **MLG coupling fusion** 먼저 (독립적, 즉시 가능, MLG에서 효과 큼)
2. **Streaming fusion** 다음 (Regularized BC + MLG coupling 2곳만 수정)
3. **FP16S** 이후 (streaming fusion 완료 후 f 상태가 고정되면 추상화 용이)
4. **Esoteric Pull** 최후 (모든 것이 안정된 후)
