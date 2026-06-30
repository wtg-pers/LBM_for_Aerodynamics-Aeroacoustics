# Streaming Fusion: 시도 및 실패 기록

**Date:** 2026-04-10
**Status:** 보류 — BC 재설계 필요
**Rollback:** collision-only fusion (Phase 3)으로 복원

---

## 1. 시도한 것

### 목표

Collision-only fused kernel (2 launches: collision + streaming)을
Pull-collide fused kernel (1 launch)로 통합.

```
Before:  fused_collision(f → f_post)  +  streaming(f_post → f)  =  2 launches
After:   fused_pull_collide(f_src → f_dst)                      =  1 launch
```

### 구현

Ping-pong 버퍼 방식:
- f_src: 이전 step의 post-collision + BC 결과 (read only)
- f_dst: 이번 step의 post-collision 결과 (write)
- 매 step마다 f와 f_post 역할 교대

Kernel 내부:
```
Thread at node x:
  1. Pull from neighbors: f_local[q] = f_src[q, x-c_q]  (streaming)
  2. Macro: rho, u from f_local
  3. Collision: f_local → f_post_local
  4. Write: f_dst[q, x] = f_post_local
```

---

## 2. 발생한 문제

### 문제 1: f의 의미 변화

| | Collision-only fusion | Streaming fusion |
|---|---|---|
| f에 저장된 상태 | post-streaming + BC | **post-collision + BC** |
| 다음 step 시작 시 | collision input | **pull(streaming) input** |
| BC 적용 대상 | post-streaming f | **post-collision f_dst** |

**모든 BC가 post-streaming 상태를 전제로 설계되어 있음.**
Post-collision 상태에 BC를 적용하면 물리가 달라짐.

### 문제 2: Bounce-back BC의 self-reference

```python
# BC 호출: 같은 배열을 source와 destination으로 전달
self.bc_manager.apply_all(f_dst, f_dst)
```

Bounce-back: `f_dst[opp_q, x] = f_dst[q, x]`
방향 쌍 (q, opp_q)에서 한쪽을 수정하면 다른 쪽이 오염됨.

### 문제 3: 검증 실패

| Config | Baseline Cd | Streaming Fusion Cd | 판정 |
|--------|------------|-------------------|------|
| Poiseuille drift | +0.3653% | +0.0118% | 값 다름 |
| **Sphere Cd** | **1.125** | **3.168** | **❌ 완전히 틀림** |

Sphere Cd가 3배 이상 높게 나옴 — BC가 제대로 작동하지 않음.

---

## 3. 근본 원인

현재 BC 모듈들은 **post-streaming 상태의 f**를 전제로 설계됨:

- **Regularized BC**: post-streaming f에서 f_neq를 추출하여 재구성
- **Sponge BC**: post-streaming f를 target으로 blending
- **HalfwayBounceBack**: post-streaming f에서 invalid 값 교체 + f_post에서 source 읽기
- **DomainWallBC**: post-streaming f에서 bounce-back

Streaming fusion은 f에 post-collision 상태를 저장하므로,
이 모든 BC가 잘못된 입력을 받게 됨.

---

## 4. 해결에 필요한 작업

Streaming fusion을 정상 동작시키려면 **모든 BC를 post-collision 상태에서
동작하도록 재설계**해야 함:

| BC | 필요한 변경 |
|---|---|
| HalfwayBounceBack | f_dst 자체에서 bounce (self-reference 해결 필요) |
| Regularized inlet/outlet | post-collision f에서 f_neq 추출 방식 변경 |
| Sponge | blending target 재정의 |
| DomainWallBC | post-collision 상태에서 bounce-back |
| CornerBC | 동일 |

이는 사실상 **Phase 2 (Esoteric Pull)에서 예상했던 BC 전면 재설계**와 동일한 작업.

---

## 5. 판단

### 비용 대비 이득

```
Streaming fusion 이득:
  - Kernel launch: 2 → 1 (marginal)
  - Streaming CuPy 연산 제거
  - 메모리: 변화 없음 (232 B/node 동일)

Streaming fusion 비용:
  - 모든 BC 재설계 (5+ BC 클래스)
  - 검증 전면 재수행
  - MLG coupling 호환성 검증
```

**비용 >> 이득. 현 단계에서는 보류가 합리적.**

### 향후 재시도 조건

1. Phase 2 (Esoteric Pull) 구현 시 BC 재설계가 불가피 → 그때 streaming fusion도 함께 적용
2. 또는 BC를 post-collision 호환으로 리팩토링하는 별도 작업을 먼저 수행

---

## 6. 코드 상태

- Streaming-fused kernel 코드는 **유지** (삭제하지 않음):
  - `src/kernels/bgk_d3q27.py`: `BGKStreamCollideKernelD3Q27` 클래스
  - `src/kernels/cumulant_d3q27.py`: `CumulantStreamCollideKernelD3Q27` 클래스
- `simulation.py`: collision-only fusion으로 **복원 완료**
- 향후 BC 재설계 후 활성화 가능

---

## 7. 2차 시도 (2026-04-11): BC save-then-write 재설계 후 재시도

### 수정 내용
- HalfwayBounceBack, MovingWallBounceBack, DomainWallBounceBack,
  DomainBounceBackBC의 apply()에 save-then-write 패턴 추가
- f_post=None 시 f 자체를 source로 사용, 방향쌍 충돌 방지

### 결과
```
Poiseuille drift: +0.0118% (baseline +0.3653% — 값 다름)
Sphere Cd:        3.1677   (baseline 1.125 — 완전히 틀림)
```

### 새로 발견된 근본 원인

BC의 in-place 처리는 정상이지만, **더 근본적인 문제**가 존재:

```
현재 (collision → streaming → BC):
  BC가 f[inlet face]에 값을 씀
  다음 step: macro(f) → inlet의 BC값이 macro에 직접 반영
             collision → streaming → inlet face는 이웃의 f_post를 pull
             BC가 다시 inlet face를 덮어씀
  → BC값은 inlet face에만 존재, streaming이 덮어쓴 후 BC가 다시 설정

Streaming fusion (pull+collision → BC):
  BC가 f_dst[inlet face]에 값을 씀 → swap → f에 저장
  다음 step: pull from f → inlet face의 BC값이 이웃 노드로 streaming!
             이웃이 inlet BC값을 가져감 (의도하지 않은 전파)
  → BC값이 streaming source로 사용되어 이웃에 확산
```

**핵심 차이:**
- 현재: BC가 쓴 값은 macro에만 사용, streaming에서 덮어써짐
- Fusion: BC가 쓴 값이 다음 step의 pull source → 이웃으로 전파

이건 bounce-back뿐 아니라 **모든 Dirichlet-type BC** (regularized inlet 포함)에
해당하는 문제. BC가 설정한 경계값이 streaming을 한 번 더 거침.

---

## 8. BC 근본 재설계 방향

Streaming fusion이 작동하려면, BC가 쓰는 값이 **"이웃이 pull했을 때
올바른 물리적 값을 받도록"** 계산되어야 한다.

### 현재 BC의 동작 (post-streaming 기준)

Regularized inlet 예시:
```
BC가 쓰는 값: f[q, inlet] = f_eq(ρ_target, u_target) + f_neq
의미: "이 노드의 분포함수를 목표 상태로 설정"
다음 step: macro가 이 값을 직접 사용 → ρ ≈ ρ_target, u ≈ u_target
```

### Streaming fusion에서 필요한 BC (post-collision 기준)

```
BC가 쓰는 값: f[q, inlet] = ???
의미: "이웃 노드가 pull할 때 올바른 값을 받도록 준비"
다음 step: 이웃이 f[q, inlet - c_q]를 pull
           → 이웃의 rho, u가 올바르게 계산되어야 함
```

이건 단순히 f_eq를 쓰는 것이 아니라, **"이 경계 노드에서 이웃 방향으로
streaming되었을 때 올바른 물리적 값이 되는 f"**를 계산해야 함.

### 구체적 재설계

**Regularized inlet (velocity Dirichlet):**
- 현재: f = f_eq(ρ, u_target) + regularized f_neq
- Fusion: f = collision(f_eq(ρ, u_target) + regularized f_neq)
  즉, **BC 값에 collision을 한번 더 적용**하여 post-collision 상태로 만들어야 함.
  이웃이 pull하면 "streaming + collision이 된" 상태를 받게 되므로,
  BC 노드에서 미리 collision을 적용해놓는 것.

**Bounce-back:**
- 현재: f[opp_q] = f_post[q] (post-collision outgoing → reflected incoming)
- Fusion: f[opp_q] = f[q] (post-collision → post-collision, 이미 올바름!)
  Bounce-back은 이미 post-collision 상태에서 자연스럽게 동작.
  (2차 시도에서 Poiseuille drift가 개선된 이유 — 벽 BC는 올바르게 작동)

**Sponge:**
- 현재: f = (1-σ)f + σ·f_target (post-streaming f를 blending)
- Fusion: f = collision((1-σ)f_pre + σ·f_target)
  또는 post-collision 상태에서 blending 후 다시 collision

### FluidX3D의 접근법 (참고)

FluidX3D는 모든 BC를 **"이웃이 pull할 값을 계산"**하는 방식으로 구현:
- Velocity BC: anti-bounce-back 방식으로 incoming 분포 설정
- Pressure BC: equilibrium + anti-bounce-back
- 각 BC가 "streaming-aware"하게 설계되어 별도 streaming pass 불필요

이 접근법으로 모든 BC를 재작성하면 streaming fusion이 가능해짐.
하지만 이는 사실상 **전체 BC 모듈의 재구현**에 해당.

---

## 9. 현재 상태

```
Collision-only fusion (현재 활성):
  BGK:      111.79 MLUPS (baseline 33.86 → 3.3×)
  Cumulant: 112.10 MLUPS (baseline 27.20 → 4.1×)

  advance() = fused_collision(1 launch) + streaming(1 CuPy op) + BC
```

BC의 save-then-write 수정은 유지 (기존 호환성 + 향후 활용 가능).
Streaming fusion은 BC 근본 재설계가 완료될 때까지 보류.

---

## 10. 3차 시도 (2026-04-11): Eq+Neumann BC (f_neq 미사용) 조합

### 가설
Regularized BC를 제거하면 f_neq 문제가 사라져 streaming fusion이 작동할 것.

### 결과
```
Eq+Neumann Sphere (streaming fusion):  Cd = 3.163 ← 여전히 틀림
Eq+Neumann Sphere (collision-only):    Cd = 1.125 ← 정상
```

### 새로 발견된 근본 원인: HBB의 물리적 불일치

Streaming fusion에서 `f_dst[i, x]`의 물리적 의미가 다름:

```
현재 (collision → streaming → HBB):
  f_post[i, x] = 노드 x에서 방향 i로 나가는 post-collision 분포
  HBB: f[ī, x] = f_post[i, x]  (나가려던 분포가 반사)

Streaming fusion (pull+collision → HBB):
  f_dst[i, x] = 노드 x-c_i에서 온 값이 x에서 충돌한 결과
               = 이웃에서 온 분포를 현재 노드에서 collision시킨 것
               ≠ 노드 x에서 방향 i로 나가는 분포!
```

Pull-collide kernel에서 각 thread는 27개 이웃에서 pull한 값으로 collision.
`f_dst[i, x]`는 방향 i에서 pull된 값의 collision 결과이지,
방향 i로 나가는 값이 아님. 이 두 개는 물리적으로 다른 양.

**이 문제는 BC 재설계가 아닌, pull-collide 구조의 본질적 한계.**
FluidX3D 등은 이를 "collide at source, push to destination" 또는
Esoteric twist/pull 방식으로 해결하지만, 이는 전체 아키텍처 변경.

### 결론

Streaming fusion은 다음 중 하나로만 가능:
1. **Esoteric Pull/Push** — 별도 아키텍처 (Phase 2)
2. **Collide-then-push** — scatter write (GPU 비효율)
3. **Flag-based in-kernel BC** — kernel 내부에서 경계 처리

현재의 pull-collide + 외부 BC 방식으로는 원리적으로 불가.
