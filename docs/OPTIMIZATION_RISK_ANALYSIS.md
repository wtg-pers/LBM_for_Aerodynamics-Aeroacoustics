# Optimization Risk Analysis: 모듈별 영향 평가

**Date:** 2026-04-09
**Purpose:** 각 최적화 Phase가 솔버의 모든 모듈에 정상 적용 가능한지 사전 검토

---

## 1. 현재 f 배열 의존성 맵

```
                    ┌─────────┐
                    │    f    │ (현재 분포함수)
                    └────┬────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    Macroscopic    MLG f_prev      Checkpoint
    (rho, u)       (.copy())       (save/load)
          │
          ▼
    ALM → body_force
          │
          ▼
    ┌──────────┐
    │ Collision │ (BGK or Cumulant)
    │ f → f_post│
    └────┬─────┘
         │
         ├──────────────┐
         ▼              ▼
    Streaming      MEM Force
    f_post→f_new   (reads f_post)
         │
         ▼
    ┌──────────┐
    │    BC    │ (Regularized, Sponge, HalfwayBB)
    │ on f_new │
    └────┬─────┘
         │
         ▼
    Buffer Swap: f ← f_new
         │
         ▼
    MLG Coupling (C→F, F→C)
```

---

## 2. Phase 1: f_new 제거 (3 buffer → 2 buffer)

### 변경

```
현재: collision(f→f_post) → streaming(f_post→f_new) → BC(f_new) → swap(f, f_new)
제안: collision(f→f_post) → streaming(f_post→f)     → BC(f)     → (swap 불필요)
```

### 모듈별 영향

| 모듈 | 영향 | 안전 여부 | 이유 |
|------|------|----------|------|
| **Macroscopic** | 없음 | ✅ | advance() 시작 시 f에서 rho,u 계산 — 변경 없음 |
| **BGK Collision** | 없음 | ✅ | f→f_post는 동일 |
| **Cumulant Collision** | 없음 | ✅ | f→f_post는 동일 |
| **Streaming (Pull)** | 대상 변경 | ✅ | f_post(source)→f(dest): pull이므로 source와 dest가 다른 배열이면 충돌 없음 |
| **HalfwayBounceBack** | 대상 변경 | ⚠️ | `f_new[opp] = f_post[i]` → `f[opp] = f_post[i]`. f와 f_post가 다른 배열이므로 OK. 단, `apply_with_reset`이 f_new를 인자로 받으므로 인터페이스 수정 필요 |
| **Regularized BC** | 대상 변경 | ✅ | f_new → f로 대상만 바뀜 |
| **Sponge BC** | 대상 변경 | ✅ | 동일 |
| **MEM Force** | 없음 | ✅ | f_post는 여전히 존재 |
| **ALM** | 없음 | ✅ | u만 사용, f 배열 무관 |
| **MLG f_prev** | 없음 | ✅ | advance() 전에 f.copy() → f는 아직 이전 상태 |
| **MLG C→F** | 없음 | ✅ | f_prev와 advance 후 f를 사용 — 동일 |
| **MLG F→C** | 없음 | ✅ | fine.f → coarse.f 기록 — 대상은 f |
| **Checkpoint** | 없음 | ✅ | f를 저장/복원 — 동일 |
| **Conservation** | 없음 | ✅ | rho만 사용 |
| **VTK** | 없음 | ✅ | rho, u만 사용 |

### 위험 요소

1. **Buffer swap 제거:** `f, f_new = f_new, f` → 더 이상 필요 없음.
   실수로 swap이 남아있으면 데이터 꼬임.

2. **HalfwayBounceBack 인터페이스:** `apply_with_reset(f_new, f_post)` →
   `apply_with_reset(f, f_post)`. 파라미터 이름만 변경.

### 결론: ✅ 안전하게 적용 가능 (인터페이스 소폭 수정)

---

## 3. Phase 2: Esoteric Pull Streaming (1 buffer)

### 변경

```
현재:  f, f_post 2개 배열
제안:  f 1개만 사용, even/odd 스텝에서 읽기/쓰기 패턴 교대

Even step: f[i](x) ← collide( f[i](x - c_i) )   # pull + collide
Odd step:  f[i](x + c_i) ← collide( f[i](x) )    # collide + push
```

### 모듈별 영향

| 모듈 | 영향 | 안전 여부 | 이유 |
|------|------|----------|------|
| **Macroscopic** | 변경 필요 | ⚠️ | even/odd에 따라 f의 의미가 다름. 어느 방향으로 pull된 상태인지 인지해야 함 |
| **BGK Collision** | 통합 필요 | ⚠️ | 독립 collision → fused collision+streaming으로 변경 |
| **Cumulant Collision** | 통합 필요 | ⚠️ | 동일 — Chimera를 fused 내부에서 수행 |
| **Streaming** | **폐기** | 🔄 | 별도 streaming 단계가 사라짐 (collision에 통합) |
| **HalfwayBounceBack** | **자동화** | ✅ | Esoteric streaming에서 bounce-back이 implicit — 별도 BC 불필요! |
| **Regularized BC** | 수정 필요 | ⚠️ | even/odd에 따라 f face slice의 해석이 달라짐. 방향 인덱스 매핑 변경 필요 |
| **Sponge BC** | 수정 필요 | ⚠️ | 동일 — f 값의 의미가 step parity에 의존 |
| **MEM Force** | **재설계** | ❌ | f_post가 더 이상 별도 저장 안 됨. collision 중에 force를 계산하거나, 별도 패스 필요 |
| **ALM** | 없음 | ✅ | u만 사용 |
| **MLG f_prev** | 수정 필요 | ⚠️ | f의 의미가 even/odd에 따라 다름. 저장 시점의 parity 기록 필요 |
| **MLG C→F** | 수정 필요 | ❌ | 시간 보간이 f_prev(half) + f(full)인데, esoteric에서는 두 시점의 f가 다른 메모리 패턴 — 보간 로직 재설계 |
| **MLG F→C** | 수정 필요 | ⚠️ | fine.f → coarse.f 기록 시 parity 일치 필요 |
| **Checkpoint** | 수정 필요 | ⚠️ | even/odd 상태도 함께 저장해야 복원 가능 |
| **Conservation** | 없음 | ✅ | rho만 사용 |
| **VTK** | 없음 | ✅ | rho, u만 사용 |

### 위험 요소

1. **MEM Force 재설계 필수:** f_post가 사라지므로 momentum exchange method의
   기반이 바뀜. collision 내부에서 force를 계산하거나, 별도 f_post 저장 패스 추가.
   → 별도 패스를 추가하면 메모리 절감 효과 감소.

2. **MLG Coupling 재설계:** C→F 시간 보간이 even/odd parity에 의존하게 됨.
   MLG와 Esoteric의 조합은 학술적으로도 잘 연구되지 않은 영역.

3. **모든 BC 재검토:** Regularized inlet/outlet, sponge 등 비자명 BC는
   f의 방향 인덱스 해석이 even/odd마다 달라짐. 각 BC를 2벌로 작성하거나
   방향 매핑 테이블을 동적으로 전환해야 함.

4. **디버깅 난이도:** even/odd 교대 패턴에서 버그 발생 시 원인 추적이 어려움.
   단일 스텝 결과가 이전과 달라 bit-exact 비교 불가.

### 결론: ❌ 고위험. MLG와의 조합이 특히 위험. Phase 3(Fused Kernel) 이후로 연기 권장.

---

## 4. Phase 3: Fused CUDA Kernel (collision + streaming)

### 변경

```
현재: CuPy array ops (15-80 kernel launches)
제안: 단일 RawKernel (1 launch, register에서 모든 연산)

Fused pull-collide kernel:
  1. Pull: f_local[27] ← f[neighbors]    (global read)
  2. Macroscopic: rho, u from f_local     (register)
  3. Collision: f_local → f_post_local    (register)
  4. Write: f_out[current] ← f_post_local (global write)
```

### 모듈별 영향

| 모듈 | 영향 | 안전 여부 | 이유 |
|------|------|----------|------|
| **Macroscopic** | kernel 내부로 이동 | ⚠️ | rho, u가 kernel 내부에서 계산되지만 다른 모듈에서도 필요 → 별도 global write 필요 |
| **BGK Collision** | kernel으로 대체 | ✅ | Python 버전은 fallback으로 유지 |
| **Cumulant Collision** | kernel으로 대체 | ⚠️ | Chimera를 register에서 수행. ~90 registers → occupancy 감소 가능 |
| **Streaming** | kernel에 통합 | ✅ | pull 접근이 kernel 첫 단계 |
| **HalfwayBounceBack** | **별도 패스 유지** | ⚠️ | fused kernel에 통합하려면 flag array 필요. 초기에는 별도 패스로 유지하는 게 안전 |
| **Regularized BC** | 별도 패스 유지 | ✅ | face slices만 처리 — kernel과 독립 |
| **Sponge BC** | 별도 패스 유지 | ✅ | slab region만 처리 — kernel과 독립 |
| **MEM Force** | 수정 필요 | ⚠️ | fused kernel이 f_post를 global memory에 안 쓸 수 있음. 옵션: (A) kernel이 f_post도 기록, (B) force 계산을 kernel 내부에 통합 |
| **ALM** | **순서 문제** | ❌ | 현재: macro→ALM→collision. Fused kernel은 macro+collision이 통합. ALM의 body_force를 kernel에 전달하려면 ALM이 kernel 전에 실행되어야 하지만, ALM은 rho,u가 필요. → **2-pass 구조 필요** |
| **MLG f_prev** | 없음 | ✅ | f.copy()는 kernel 호출 전 — 변경 없음 |
| **MLG C→F** | 없음 | ✅ | f 배열에서 읽기만 — kernel 결과 사용 |
| **MLG F→C** | 없음 | ✅ | f 배열에 쓰기만 — kernel과 순차 |
| **Checkpoint** | 없음 | ✅ | f 저장 — 동일 |

### 위험 요소

1. **ALM과의 순서 의존:**
   ```
   현재:   macro(f→rho,u) → ALM(u→force) → collision(f,force→f_post) → streaming
   Fused:  ALM needs u, but u is computed inside the fused kernel
   ```
   해결: **2-pass 구조**
   ```
   Pass 1: macro kernel (f→rho,u를 global memory에 기록)
   Pass 2: ALM.step(u→force)
   Pass 3: fused collision+streaming kernel (f,force→f_out)
   ```
   이렇게 하면 3 kernel launches이지만, collision+streaming 통합의 핵심 이점은 유지.
   ALM이 없으면 Pass 1+3 → 2 launches. Macro를 kernel 내부에서 계산하되
   rho,u도 global에 기록하면 1 launch (ALM 없을 때).

2. **MEM Force 계산:**
   f_post를 global memory에 기록하면 메모리 절감 효과 감소.
   대안: force 계산 주기가 10 step이므로, force 계산 step에서만 f_post 기록.
   ```python
   if step % force_interval == 0:
       fused_kernel_with_fpost(f, f_out, f_post, ...)  # f_post도 기록
   else:
       fused_kernel(f, f_out, ...)  # f_post 생략
   ```

3. **Cumulant register pressure:**
   D3Q27 Cumulant에서 ~90 registers/thread. RTX 3090은 thread당 최대 255이지만,
   64 registers 초과 시 occupancy 감소 → warp scheduling 효율 저하.
   `__launch_bounds__` directive로 튜닝 필요.

4. **CPU fallback 유지:**
   numpy 사용 시 CUDA kernel 실행 불가. 기존 CuPy array ops 코드를
   fallback으로 유지해야 함. → dispatch 로직 필요.

### 결론: ✅ 적용 가능하지만 ALM 2-pass 구조와 MEM Force 분기 필요

---

## 5. Phase 4: FP16 Shifted Storage

### 변경

```
현재: f를 float32로 저장/연산
제안: f를 float16(f - w)로 저장, 연산 시 float32로 복원

저장: f_stored[i] = float16(f[i] - w[i])
복원: f[i] = float32(f_stored[i]) + w[i]
```

### 모듈별 영향

| 모듈 | 영향 | 안전 여부 | 이유 |
|------|------|----------|------|
| **Macroscopic** | 변환 필요 | ⚠️ | f_stored(FP16)에서 rho,u 계산 시 FP32 복원 필요. 또는 fused kernel 내부에서 처리 |
| **BGK Collision** | 변환 필요 | ⚠️ | 입출력이 FP16이지만 내부 연산은 FP32 |
| **Cumulant Collision** | 변환 필요 | ⚠️ | 동일 |
| **Streaming** | 변환 필요 | ⚠️ | FP16 배열에서 pull → FP16으로 기록. bit 조작 주의 |
| **HalfwayBounceBack** | 변환 필요 | ⚠️ | f[opp] = f_post[i] — FP16 간 복사. shifting (w 빼기/더하기) 처리 |
| **Regularized BC** | 변환 필요 | ⚠️ | FP32로 BC 계산 후 FP16으로 저장 |
| **Sponge BC** | 변환 필요 | ⚠️ | 동일 |
| **MEM Force** | 변환 필요 | ⚠️ | f_post를 FP32로 복원 후 force 계산 |
| **ALM** | 없음 | ✅ | u(FP32)만 사용 |
| **MLG C→F** | 정밀도 주의 | ❌ | 보간 + f_neq rescaling에서 FP16→FP32→FP16 반복 변환. 보간 정밀도 저하 위험 |
| **MLG F→C** | 정밀도 주의 | ❌ | f_neq = f - f_eq 계산에서 catastrophic cancellation 위험 (FP16에서 f ≈ f_eq) |
| **Checkpoint** | 형식 변경 | ⚠️ | FP16으로 저장 시 checkpoint 호환성 변경. 또는 FP32로 변환 후 저장 |
| **Conservation** | 정밀도 주의 | ⚠️ | rho = sum(f) — FP16 복원 후 FP32 합산 필요 |

### 위험 요소

1. **MLG Coupling에서 정밀도 손실:**
   ```
   f_neq = f - f_eq  (f ≈ f_eq이므로 값이 매우 작음)
   ```
   FP16에서 f와 f_eq가 비슷한 값 → 뺄셈에서 유효 자릿수 손실.
   **해결:** coupling 연산만 FP32에서 수행. f를 FP32로 복원 후 coupling.

2. **모든 BC가 FP16 I/O 인터페이스 필요:**
   현재 BC들은 FP32 f를 직접 읽고 씀. FP16 shifted 배열에서는
   읽기: `f32 = fp16_to_fp32(f_stored) + w` / 쓰기: `f_stored = fp32_to_fp16(f32 - w)`.
   모든 BC 파일에 변환 로직 추가 필요.

3. **Fused kernel과 조합 시 자연스러움:**
   Phase 3의 fused kernel 내부에서 FP16↔FP32 변환을 처리하면
   외부 모듈은 변환을 의식하지 않아도 됨. **Phase 3 이후에 적용하면 위험 감소.**

### 결론: ⚠️ Phase 3(Fused Kernel) 이후에 적용 권장. MLG coupling에서 정밀도 주의.

---

## 6. 종합 위험도 매트릭스

```
              Phase 1      Phase 2       Phase 3        Phase 4
              (f_new제거)   (Esoteric)   (Fused CUDA)   (FP16)
              ──────────   ──────────   ────────────   ──────────
Macroscopic      ✅           ⚠️            ⚠️            ⚠️
BGK              ✅           ⚠️            ✅             ⚠️
Cumulant         ✅           ⚠️            ⚠️            ⚠️
Streaming        ✅           🔄(폐기)      ✅(통합)       ⚠️
HalfwayBB        ⚠️(소폭)    ✅(자동!)      ⚠️            ⚠️
Regularized BC   ✅           ⚠️            ✅             ⚠️
Sponge BC        ✅           ⚠️            ✅             ⚠️
MEM Force        ✅           ❌            ⚠️            ⚠️
ALM              ✅           ✅            ❌(순서)       ✅
MLG f_prev       ✅           ⚠️            ✅             ⚠️
MLG C→F          ✅           ❌            ✅             ❌
MLG F→C          ✅           ⚠️            ✅             ❌
Checkpoint       ✅           ⚠️            ✅             ⚠️
──────────────────────────────────────────────────────────────────
총 위험도       LOW          HIGH          MEDIUM         MEDIUM
```

---

## 7. 권장 적용 순서

```
Phase 1: f_new 제거                    [LOW RISK]
  → 거의 모든 모듈에 안전
  → HalfwayBounceBack 인터페이스만 소폭 수정
  → 즉시 착수 가능

Phase 3: Fused CUDA Kernel            [MEDIUM RISK]
  → Phase 2(Esoteric)를 건너뛰고 직행
  → ALM 2-pass 구조로 해결
  → MEM Force는 조건부 f_post 기록
  → CPU fallback 유지

Phase 4: FP16 Storage                 [MEDIUM RISK]
  → Phase 3의 fused kernel 내부에서 변환 처리
  → MLG coupling만 FP32 유지
  → Phase 3 완료 후 적용

Phase 2: Esoteric Pull                [HIGH RISK — 보류]
  → MLG, MEM Force와의 호환 문제 심각
  → Phase 3+4로 이미 충분한 최적화 달성
  → 향후 단일 격자 전용 옵션으로 고려
```

### Phase 2 보류 근거

Phase 1+3+4만으로도:
- 메모리: 340 → ~70 B/node (**4.8× 절감**)
- 속도: ~200 → ~5,000 MLUPS (**25× 가속**)

Phase 2 추가 시:
- 메모리: ~70 → ~35 B/node (추가 2× — 점점 수확 체감)
- MLG/MEM Force/BC 전면 재설계 비용 >> 이득

---

## 8. Phase 1 착수 전 체크리스트

- [ ] Pull streaming에서 source(f_post)와 dest(f)가 다른 배열인지 확인
- [ ] HalfwayBounceBack.apply_with_reset() 인터페이스 수정
- [ ] Simulation.advance()에서 buffer swap 제거
- [ ] set_distribution()에서 f_new 할당 제거
- [ ] MLG f_prev가 advance() 전에 저장되는지 확인
- [ ] 검증: Poiseuille 3D (단일 격자 + MLG)
- [ ] 검증: Sphere drag Cd (단일 격자 + MLG)
- [ ] 검증: Checkpoint restart 후 결과 연속성
