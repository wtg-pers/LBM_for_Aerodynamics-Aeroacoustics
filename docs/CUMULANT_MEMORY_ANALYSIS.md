# Cumulant vs BGK: 메모리 및 연산 비교

**Date:** 2026-04-09

---

## 1. 핵심 결론 (요약)

- **메모리:** BGK와 Cumulant은 거의 동일하다 (forcing 없을 때).
- **속도 차이의 원인:** FLOPs가 아니라 CuPy kernel launch 횟수 (15 vs 80+).
- **Fused CUDA kernel로 전환하면 속도 차이가 사라진다.**

---

## 2. 메모리의 핵심: (27, N) 크기 배열이 지배

격자점 수 `N = Nx × Ny × Nz`를 포함하는 배열은 3가지 크기뿐이다:

| 방향 차원 | 차원 | bytes/node (float32) | 예시 |
|-----------|------|---------------------|------|
| Q = 27 | `(27, Nx, Ny, Nz)` | **108** | f, f_post, f_new, f_ijk, kappa |
| dim = 3 | `(3, Nx, Ny, Nz)` | 12 | u, body_force |
| scalar | `(Nx, Ny, Nz)` | 4 | rho, inv_rho, Dxu, k220_orig 등 |

**메모리는 `(27, N)` 배열이 몇 개 동시에 살아있느냐로 결정된다.**
`(N,)` 크기 배열은 `(27, N)`의 1/27이므로 무시 가능.

---

## 3. BGK: (27, N) 배열 카운트

```
영구:    f, f_post, f_new                    → 3개
임시:    cu, f_eq  (collision 중 동시 존재)  → 2개
────────────────────────────────────────────────
Peak:                                        → 5개 × 108 B/node = 540 B/node
```

BGK collision 흐름:
```
cu = einsum(c, u)                → NEW (27,N)  ← 4번째
f_eq = w * rho * (1+3cu+...)     → NEW (27,N)  ← 5번째
f_out[:] = f - omega*(f - f_eq)  → in-place on f_post
                                    cu, f_eq 해제 → 다시 3개
```

---

## 4. Cumulant: (27, N) 배열 카운트

```
영구:    f, f_post, f_new                     → 3개
임시:    chimera input + output (동시 2개)    → 2개
────────────────────────────────────────────────
Peak:                                         → 5개 × 108 B/node = 540 B/node
```

**BGK와 동일한 5개.** 이유는 Chimera의 sequential 분해:

```
Chimera forward (모멘트 변환):
  f_ijk(NEW)  + kz(NEW)    = 2개 동시  → f_ijk 해제
  kz          + ky(NEW)    = 2개 동시  → kz 해제
  ky          + kappa(NEW) = 2개 동시  → ky 해제
  → kappa만 남음 (1개)

Relaxation:
  kappa에 in-place 연산
  중간값은 모두 (N,) 크기 = 4B/node (무시 가능)

Chimera backward (분포함수 복원):
  kappa       + kx(NEW)       = 2개 동시  → kappa 해제
  kx          + ky(NEW)       = 2개 동시  → kx 해제
  ky          + f_star(NEW)   = 2개 동시  → ky 해제
  → f_star만 남음 → f_out에 복사 후 해제
```

**어느 시점에서든 (27, N) 임시 배열은 최대 2개만 동시 존재.**

---

## 5. Chimera Sequential 분해란?

D3Q27의 27개 분포함수를 중심 모멘트로 변환할 때,
27×27 행렬 연산 대신 **3축 각각 독립적인 3×3 변환 3번**으로 분해:

```
f_{ijk} --[z축 3×3]--> κ_{ij|γ} --[y축 3×3]--> κ_{i|βγ} --[x축 3×3]--> κ_{αβγ}
```

각 1D 변환의 수식 (속도 v 기준 중심 모멘트):
```
m₀ = d₋₁ + d₀ + d₊₁              (0차: 합)
m₁ = -d₋₁ + d₊₁ - v·m₀           (1차: 속도 보정된 차이)
m₂ = d₋₁ + d₊₁ - 2v·m₁ - v²·m₀  (2차)
```

**메모리 이점:** 입력 `(3,3,3,N)` 하나를 읽어 출력 `(3,3,3,N)` 하나를 쓰고,
입력을 즉시 해제. 동시에 2개만 유지.

---

## 6. Relaxation 단계의 추가 배열

Cumulant relaxation (Step 4)에서 생성되는 중간 배열들:

| 변수 | 차원 | 개수 | bytes/node | 용도 |
|------|------|------|-----------|------|
| inv_rho | `(N,)` | 1 | 4 | 1/ρ |
| Dxu, Dyv, Dzw | `(N,)` | 3 | 12 | 속도 기울기 (Galilean correction) |
| Gal_xx/yy/zz | `(N,)` | 3 | 12 | Galilean 보정항 (shear) |
| Gal_bulk_xx/yy/zz | `(N,)` | 3 | 12 | Galilean 보정항 (bulk) |
| diff_xy, diff_xz, trace | `(N,)` | 3 | 12 | 2차 대각 복원 |
| 3차 sum/dif | `(N,)` | 6 | 24 | 3차 대칭/반대칭 |
| comb1/2/3 | `(N,)` | 3 | 12 | 4차 traceless |
| 6 originals (k220_orig 등) | `(N,)` | 6 | 24 | forward transform 보존 |

**합계:** ~28개 × 4 B/node ≈ **112 B/node**

다만 이들은 순차 생성/해제되므로 동시 peak는 ~50 B/node.
`(27, N)` = 108 B/node의 절반 이하 — 전체 메모리에서 부차적.

---

## 7. 전체 비교표

### 메모리 (forcing 없음)

| | BGK | Cumulant | 비율 |
|--|-----|---------|------|
| 영구 (27,N) 배열 | 3개 (324 B) | 3개 (324 B) | **1.0×** |
| 임시 (27,N) peak | 2개 (216 B) | 2개 (216 B) | **1.0×** |
| 임시 (N,) 배열 | ~1개 (4 B) | ~28개 (112 B) | — |
| **총 peak** | **~544 B/node** | **~652 B/node** | **1.2×** |

메모리 차이의 대부분은 relaxation의 `(N,)` 소형 배열.
`(27, N)` 대형 배열 기준으로는 **동일**.

### 연산

| | BGK | Cumulant | 비율 |
|--|-----|---------|------|
| FLOPs/node | ~570 | ~1,004 | 1.8× |
| Array 할당 횟수 | ~3 | ~50 | **17×** |
| CuPy kernel launches | ~15 | ~80+ | **5×** |

### 실제 속도 (CuPy 기반)

| | BGK | Cumulant | 원인 |
|--|-----|---------|------|
| 추정 MLUPS | ~200-300 | ~50-100 | kernel launch overhead |
| 비율 | 기준 | **3-5× 느림** | 80+ launches vs 15 |

### Fused CUDA kernel 전환 후

| | BGK | Cumulant | 원인 |
|--|-----|---------|------|
| Kernel launches | 1 | 1 | 동일 |
| Registers/thread | ~34 | ~90 | Chimera 중간값 |
| Temp memory | 0 B | 0 B | 모두 register |
| 추정 MLUPS | ~3,500 | ~3,000 | register pressure ~15% |
| **비율** | 기준 | **1.1-1.2× 느림** | **거의 동일** |

---

## 8. 59.3M 노드 기준 (현재 Geier config)

| | BGK | Cumulant |
|--|-----|---------|
| 영구 배열 | 19.2 GB | 19.2 GB |
| Collision temp peak (최대 level 기준) | +5.7 GB | +5.7 GB |
| + relaxation (N,) 배열 | — | +0.3 GB |
| **총 peak** | **~24.9 GB** | **~25.2 GB** |
| RTX 3090 | 24.0 GB | 24.0 GB |

**두 모델 모두 24GB에 빡빡하다.** f_new 제거(Phase 1 최적화)가 필수.
f_new 제거 시 영구 배열 19.2 → 12.8 GB (-6.4 GB).
