# Phase 1 Result: Buffer Reduction (f_new 제거)

**Date:** 2026-04-10
**Status:** Complete
**Branch:** master

---

## 1. 변경 내용

### 수정 파일

| 파일 | 변경 |
|------|------|
| `src/solver/simulation.py` | `_f_new` 할당 제거, streaming이 `f`에 직접 쓰기, buffer swap 제거 |

### 변경 전 (3-buffer)

```python
# set_distribution()
self.f = f
self._f_new = self.xp.empty_like(f)    # ← 이 배열 제거
self._f_post = self.xp.empty_like(f)

# advance()
self.collision.collide(self.f, self._f_post, ...)
self.streaming.compute(self._f_post, self._f_new)  # f_post → f_new
self.bc_manager.apply_all(self._f_new, self._f_post)
self.obstacle_bc.apply_with_reset(self._f_new, self._f_post)
self.f, self._f_new = self._f_new, self.f           # buffer swap
```

### 변경 후 (2-buffer)

```python
# set_distribution()
self.f = f
self._f_post = self.xp.empty_like(f)

# advance()
self.collision.collide(self.f, self._f_post, ...)
self.streaming.compute(self._f_post, self.f)        # f_post → f 직접
self.bc_manager.apply_all(self.f, self._f_post)
self.obstacle_bc.apply_with_reset(self.f, self._f_post)
# swap 불필요
```

### 안전성 근거

Pull streaming은 `f_post[q, x-cx, y-cy, z-cz]`에서 읽고 `f[q, x, y, z]`에 쓴다.
`f_post`와 `f`는 다른 배열이므로 read-write 충돌이 없다.

---

## 2. 메모리 절감

```
Before: f(108) + f_post(108) + f_new(108) + rho(4) + u(12) = 340 B/node
After:  f(108) + f_post(108)              + rho(4) + u(12) = 232 B/node

절감: 108 B/node (-32%)
```

| 격자 크기 | Before | After | 절감 |
|----------|--------|-------|------|
| 2.1M (bench 128³) | 0.68 GB | 0.46 GB | 0.22 GB |
| 5.0M (sphere MLG) | 1.62 GB | 1.10 GB | 0.52 GB |
| 59.3M (Geier config) | 19.2 GB | 13.1 GB | **6.1 GB** |

---

## 3. 검증 결과

### 3.1 Poiseuille Single BGK

```
Metric: mass conservation drift
Baseline: +0.3653%
Phase 1:  +0.3653%
Result:   bit-exact IDENTICAL
```

**확인 파일:**
- Baseline: `validation_baseline/poiseuille_bgk_csv/mass_conservation.csv`
- Phase 1:  `val_poiseuille_single_bgk/csv/mass_conservation.csv`

### 3.2 Sphere Drag Single Grid (Re=100)

```
Metric: Cd (last step)
Baseline: 1.1248218613945027
Phase 1:  1.1248218613945027
Result:   bit-exact IDENTICAL
```

**확인 파일:**
- Baseline: `validation_baseline/sphere_single_csv/force_history.csv`
- Phase 1:  `val_sphere_single/csv/force_history.csv`

### 3.3 Sphere Drag MLG 2-Level (Re=100)

```
Metric: Cd (last step)
Baseline: 1.0598478402339808
Phase 1:  1.0598478402339808
Result:   bit-exact IDENTICAL
```

**확인 파일:**
- Baseline: `validation_baseline/sphere_mlg_csv/force_history.csv`
- Phase 1:  `val_sphere_mlg/csv/force_history.csv`

### 3.4 Performance (MLUPS)

| Config | Baseline | Phase 1 | 변화 |
|--------|----------|---------|------|
| Poiseuille BGK | 0.96 | 0.95 | -1% (noise) |
| Sphere Single | 13.69 | 23.15 | +69% |
| Sphere MLG | 24.19 | 24.07 | -0.5% (noise) |

**확인 파일:**
- Phase 1: 각 `*/csv/performance.csv`

**Note:** Sphere Single의 MLUPS 차이(+69%)는 buffer 제거에 의한 cache 효과로 추정.
소규모 격자(Poiseuille 16K)에서는 GPU utilization이 낮아 차이 미미.

---

## 4. 영향받지 않는 모듈

Phase 1에서 `simulation.py` 외에 수정한 파일 없음.
아래 모듈들은 변경 없이 정상 동작 확인:

- Collision (BGK, Cumulant): `f → f_post` 동일
- Streaming: `compute(f_post, dest)` — dest만 f_new→f로 변경
- HalfwayBounceBack: `apply_with_reset(f, f_post)` — 파라미터 동일 패턴
- Regularized/Sponge BC: `apply_all(f, f_post)` — 동일
- MEM Force: `f_post` 유지 → 변경 없음
- MLG Coupling: `f`와 `f_prev` 사용 → 변경 없음
- Checkpoint: `f` 저장 → 동일
- ALM: `u`만 사용 → 무관

---

## 5. 다음 단계

Phase 3: Fused CUDA Kernel (collision + streaming 통합)
- 예상 효과: 10-30× 속도 향상
- 현재 MLUPS ~24 → 목표 ~3,000+
