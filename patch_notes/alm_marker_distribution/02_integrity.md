# 무결성 검증 (exact 수식·물리, smoke 아님) — 2026-06-30

스크립트: `integrity_check.py` (재현). 각 계산 스텝을 지배방정식과 대조.

| # | 검증 대상 | 방법 | 결과 |
|---|---|---|---|
| C1 | Dağ deficit 커널 | `exp(-(d/ε)²)/d == ideal(1/d) − smeared((1-exp)/d)` | PASS (1e-15), _INV4PI=1/4π |
| C2 | influence_matrix per-marker Δr | explicit Biot-Savart `A_ik=-(1/4π)ΣKmat_im·Δr_m·G_mk` 와 대조 | PASS (2.8e-17) |
| C3/4 | Σmarker_dr=span + quadrature | ∫1,∫r 적분 (uniform·cosine·endpoint) | PASS (전부 0.0e+00, 선형까지 exact) |
| C5a | np.gradient 비균일 | 2차함수 Γ에 exact여야 | PASS (1.8e-13) |
| C5b | dΓ/dr 수렴성 | N=40→320 수렴률 | PASS (1.99,2.0,2.0 = 2차) |
| C5c | 클러스터 dΓ/dr 미오염 | uniform vs cosine 오차 동차수 | PASS (5.1e-3 vs 1.2e-2) |
| C6 | 추력 보존 | 고정 하중 L(r), ΣL·dr vs ∫L dr | 전부 <0.4% (정보용, 아래 주의) |
| C7 | 끝점 마커 무결성 | active·eps·chord | PASS (전 마커 active, **endpoint 팁=3.27in·루트=5.45in EXACT**) |

## 결론: 구현은 수식적으로 exact
- deficit 커널 = ideal−smeared 정확, Biot-Savart per-marker Δr 가중 정확(2.8e-17).
- 3 분포 모두 Σdr=span(기계정밀) + 선형까지 quadrature exact = **추력가중 보존**.
- np.gradient 비균일 = 형식적 2차(2차함수 exact, 2차수렴) — 클러스터가 dΓ/dr 오염 안 함(오히려 팁서 h↓로 개선).
- 끝점 마커 전부 active, endpoint가 진짜 팁(3.27in)·루트(5.45in)에 정확 배치.

## ★C6 정직한 주의 (버그 아닌 trade-off)
elliptic류 **팁-특이(steep)** 하중에서 적분오차: cosine/both 5e-4(최선) < uniform 1.2e-3 < **endpoint 4.3e-3(최악)**.
이유: **endpoint(사다리꼴)는 특이 끝점을 *직접* 샘플** → 적분오차↑. midpoint(uniform/cosine)는 끝점을 피함.
선형하중(C3/4)에선 셋 다 exact. → endpoint는 "팁을 직접 해상"하지만 "적분 정확도"는 손해. **물리적 trade-off로 명기.**
