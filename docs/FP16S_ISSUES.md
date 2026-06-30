# FP16 Shifted Storage: 시도 및 실패 기록

**Date:** 2026-04-10
**Status:** 보류 — f 배열 의미 변경으로 인한 전면 수정 필요
**Rollback:** FP32 collision-only fusion (Phase 3)으로 복원

---

## 1. 시도한 것

### 목표

f 배열을 FP16 shifted로 저장하여 메모리/대역폭 절반 감소.

```
저장: f_stored = float16(f - w)     // 0 근처로 이동 → FP16 정밀도 최대화
복원: f = float32(f_stored) + w     // collision 전에 FP32로 복원
```

### 구현

- BGK FP16S CUDA kernel 작성: `bgk_collide_d3q27_fp16s`
  - FP16 read → FP32 collision → FP16 write
  - 단위 테스트 통과 (rho 오차 1.1e-5, f_post 상대 오차 5.6e-5)
- Simulation.set_distribution()에서 f를 FP16S로 변환
- BC 전후에 FP32 unshift/reshift 추가

### 기대 효과

```
메모리: 232 → 124 B/node (-47%)
대역폭: kernel read/write 47% 감소
MLUPS: ~80% 향상 예상
```

---

## 2. 발생한 문제

### 핵심 문제: f 배열의 의미 변경

FP16S에서 `sim.f`는 실제 분포함수가 아니라 **shifted 값** `(f - w)`를 저장한다.
솔버의 모든 모듈이 `sim.f`를 **실제 분포함수**로 가정하고 있으므로,
f에 직접 접근하는 모든 코드가 오작동한다.

### 영향받는 코드 목록

| 위치 | 접근 패턴 | 문제 |
|------|----------|------|
| **output_manager.py:336** | `macro.compute(sim.f)` | shifted f로 rho 계산 → 잘못된 conservation |
| **initializer.py:74** | `macro.compute(sim.f)` | 초기 rho가 ~0 → conservation M0 오류 |
| **checkpoint.py** | `f = sim.f` 저장 | shifted 값 저장 → restart 시 해석 불일치 |
| **multi_level_grid.py** | `f_prev = sim.f.copy()` | shifted 값 복사 → C→F 보간 오류 |
| **coupling.py** | f_eq + f_neq 분해 | shifted f에서 f_eq 계산 → f_neq 오류 |
| **output_manager.py** finalize | `macro.compute(sim.f)` | 최종 rho 오류 |

### 검증 실패

```
val_poiseuille_single_bgk:
  Baseline conservation: +0.3653%
  FP16S conservation:    -99.6%  ← 완전 발산

원인: finalize()에서 macro.compute(sim.f)가 shifted f를 받아
      rho ≈ 0을 계산 → conservation drift 폭발
```

---

## 3. 근본 원인

**`sim.f`가 솔버의 공개 인터페이스**이며, 수십 개의 모듈이 이를 직접 접근한다.
f의 의미를 바꾸면(실제 값 → shifted 값) 모든 접근점을 수정해야 한다.

이는 streaming fusion의 BC 문제(f의 의미: post-streaming → post-collision)와
**동일한 종류의 문제** — 내부 표현 변경이 외부 인터페이스를 깨트림.

---

## 4. 해결에 필요한 작업

### Option A: f 접근을 property로 추상화

```python
@property
def f(self):
    if self._use_fp16s:
        return self._f_fp16s.astype(float32) + self._w_broadcast
    return self._f_raw
```

**장점:** 외부 코드 수정 불필요
**단점:** f 접근 시마다 FP32 변환 → 임시 배열 생성 → 메모리/속도 불이익

### Option B: 모든 f 접근점 수정

`sim.f` 대신 `sim.get_f()` 또는 `sim.f_real`을 사용하도록 전체 코드 수정.

**장점:** 명시적
**단점:** 수십 곳 수정 필요, 리팩토링 규모 큼

### Option C: Kernel 내부에서만 FP16, 외부는 FP32 유지

Kernel read/write만 FP16을 사용하되, global memory의 f는 FP32로 유지.
실질적 메모리 절감 없음 → **Phase 4의 목적에 맞지 않음.**

### Option D: Streaming fusion + FP16S 동시 구현

Streaming fusion이 되면 f가 kernel 내부에서만 존재하고,
외부에는 post-collision 상태만 노출. 이때 FP16S를 적용하면
외부 접근 문제가 줄어듦. 단, streaming fusion 자체가 BC 문제로 보류 중.

---

## 5. 판단

FP16S는 **f 내부 표현 변경**이라는 점에서 streaming fusion과 동일한
아키텍처 수준의 변경이 필요. 현재 솔버의 `sim.f` 공개 인터페이스에
의존하는 코드가 너무 많아 부분 적용이 불가능.

### 재시도 조건

1. `sim.f` 접근을 property로 추상화하는 리팩토링 선행
2. 또는 streaming fusion + BC 재설계가 완료된 후 함께 적용
3. 또는 FP16S를 kernel 내부 최적화로만 사용 (외부 FP32 유지, 메모리 절감 없음)

---

## 6. 코드 상태

- FP16S kernel 코드는 **유지** (삭제하지 않음):
  - `src/kernels/bgk_d3q27.py`: `_BGK_D3Q27_FP16S_KERNEL`, `BGKCollideKernelD3Q27_FP16S`
- `simulation.py`: FP32 collision-only fusion으로 **복원 완료**
- 단위 테스트는 통과 (kernel 자체는 정확):
  - rho 오차: 1.1e-5
  - f_post 상대 오차: 5.6e-5
  - 메모리: 50% 절감 확인

---

## 7. 현재 상태 (Phase 3 collision-only fusion)

```
BGK:      111.79 MLUPS (baseline 33.86 → 3.3×)
Cumulant: 112.10 MLUPS (baseline 27.20 → 4.1×)
Memory:   232 B/node (Phase 1 수준)
```
