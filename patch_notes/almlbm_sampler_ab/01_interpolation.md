# 01 — interpolation.py (새 샘플러 §7)

**파일:** `src/actuator/interpolation.py` (말미 §7 신규, §1–§6 불변)

## 추가된 함수 (전부 xp-generic: numpy CPU / cupy GPU, 반환 numpy)

| 함수 | 역할 |
|------|------|
| `sample_velocity_alt(mode, u_field, pos, eps, xp, n_cut, hub, axis, radius, eps_r_factor)` | dispatch (point/aniso/mask_disk). "gaussian"은 라우팅 안 함(§6 유지) |
| `_alt_stencil(...)` | 공유 (N,S,S,S) 스텐실 (gx_c,gy_c,gz_c, dx,dy,dz, d_sq, valid, pos, eps). §6.2 미러 |
| `_alt_gather(u_field, st, w, xp)` | 정규화 가중합 u_j=Σw·u/Σw |
| `_sample_trilinear(...)` | **B-i**: 8노드 trilinear, ε 무관 |
| `_sample_aniso(...)` | **B-ii**: 비등방, w=exp(−d_r²/ε_r² − d_⊥²/ε²), ε_r=factor·ε |
| `_sample_mask_disk(...)` | **B-iii**: 등방 Gaussian + cyl(node)≤R 마스크 |

## 핵심 수식 (코드 충실)

- 스텐실 sizing: `half = ceil(n_cut·max(ε)) + 1` (§6.2와 동일, floor/ceil 정합).
- d_r (B-ii) = d·ê_r, ê_r = normalize((p−hub) − ((p−hub)·n̂)n̂) per marker.
- cyl² (B-iii) = |g−hub|² − ((g−hub)·n̂)², g = pos + d (unclipped grid coord).
- 컷오프: 전부 `|d| ≤ n_cut·ε` 구형. B-iii는 추가로 cyl≤R, B-ii는 비등방 arg.

## 정합/안전

- 정규화상수 1/(π^{3/2}ε³)는 분자·분모 약분 → exp(...)만 계산 (baseline과 동일 관례).
- d_⊥² = max(d_sq − d_r², 0), er_norm = max(|er|,1e-30), W = max(Σw,1e-30) — 0division/음수 방어.
- "gaussian" 모드는 이 §7을 전혀 타지 않음 → baseline bit-identical.
