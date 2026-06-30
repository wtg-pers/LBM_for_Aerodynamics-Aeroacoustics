# 01 — Phase 0: lift-curve slope ∂C_l/∂α (완료)

**파일:** `src/actuator/polar_slope.py` (신규) / 설계 `00_design.md` §5 Phase 0.

## 무엇
Kleine 비반복 선형화(식 A5/A6, 5.14)에 필요한 **∂C_l/∂α**를, 기존 polar query에 대한
**중심 차분(central FD)**으로 제공. 폴라 로더(C81/csv/flat_plate) 변경 불필요.

## API
- `lift_curve_slope(polar_query, alpha_deg, Re, name=None, mach=None, delta_deg=1.0) -> dCl/dα [1/rad]`
- `lift_curve_slope_batch(polar_query, alpha_deg[], Re[], active[], multi_airfoil, marker_airfoil, mach[], delta_deg)`
  → `(N,)` per-marker slope, `_lookup_cl_cd`와 동일한 dispatch/multi-airfoil/Mach-pass 미러.
- 반환 단위 **per-radian** (Kleine α는 radian).
- `delta_deg=1°` 중심차분이 bilinear C81 deck의 국소 kink를 평활화 (Kleine §6.1의 smooth slope 취지).

## 검증 (PASS)
- 평판류 `Cl=2π·sinα`에서 FD slope vs 해석 `2π·cosα`: `|err|~3e-4` (α=0/±5/10/−3°).
- 4가지 polar_query 시그니처 `(α,Re)|(α,Re,mach=)|(α,Re,name)|(α,Re,name,mach=)` 모두 정상 dispatch.
- batch: inactive 마커=0, multi-airfoil+mach 정상.

## 다음 (Phase 1)
prescribed-wake 비반복 보정: Φ 커널(식 3.20) + 영향계수 A + `[I−diag(b)A]ΔΓ=Γ†−Γⁿ⁻¹`
(`numpy.linalg.solve`) → 보정 α → 힘. 본 slope를 `b_y/b_z`(식 A5/A6)에 사용. 4-level light에서 동작.
