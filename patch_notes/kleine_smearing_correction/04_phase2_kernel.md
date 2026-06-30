# 04 — Phase 2 step 1: exact-Φ 커널 (완료·검증)

**파일:** `src/actuator/smearing_correction.py` (`phi_smeared`, `phi_ideal`, `segment_missing_theta`).

## ★전사 오류 정정 (중요)
한글 요약(`docs/papers_kr/2022_...kr.md`)의 **Eq 3.20이 틀렸었다**: 2번째 항에 잉여 `Z/ε`가 있어
`(Z/ε)exp(-r²/ε²)erf(Z/ε)` → Z→±∞에서 **발산**. 원문 PDF **p.11** 확인 → 올바른 항은
`exp(-r²/ε²)erf(Z/ε)`. (요약·`smearing_correction.py` 모두 정정.)

```
Φ(r,Z) = (1/r)[ -Z/√(r²+Z²)·erf(√(r²+Z²)/ε) + exp(-r²/ε²)·erf(Z/ε) ]      (3.20)
Φⁱ(r,Z) = (1/r)(-Z/√(r²+Z²))                                              (3.21)
u_m = (Γ/4π)[(Φⁱ(Z+)-Φⁱ(Z-)) - (Φ(Z+)-Φ(Z-))]   (ideal-smeared, deficit) (3.23)
```

## 검증 (기계정밀도 PASS)
1. **Eq 3.19 수치적분 일치**: `Φ(Z+)-Φ(Z-)` == `r·∫ g(s)/s³ dz'` (g=erf(s/ε)-2s·η(s)), 4 케이스 relerr~1e-16.
2. **반무한 극한**: z=0,zj-=0,zj+→∞ → u^v=Γ/(4πr)[1-exp(-r²/ε²)] (Eq 3.24) + deficit=Γ/(4πr)exp(-r²/ε²)
   (Lamb-Oseen), 3 케이스 relerr~1e-14.

## 남은 Phase 2 (step 2-3) — free-vortex wake

**step 2 — free-wake 기하 (stateful, per-blade):**
- 패널 edge(마커 사이 r_{j+1/2})에서 trailed 와류 shed, 강도 = Γ_j−Γ_{j+1}(dΓ/dr).
- 매 스텝 새 wake 행 shed + 기존 점들을 **샘플 CFD 속도로 이류**(Euler). 최근 n_w(=50) 유지,
  d_w(=ε/2) 미만 융합. wake = (edge)×(age) 점격자, per-blade state(`_kleine_wake[k]`).
- 보정속도 계산 전에 미리 이류(현재 Γ가 시트 기하에 영향 안 줌 → Phase 1 선형화와 양립).

**step 3 — 영향행렬(free-wake) + 비반복 통합:**
- 각 control point j × 각 wake 세그먼트: 일반기하 Biot-Savart (perp dist r, 축 Z±, 방위방향)
  로 `segment_missing_theta` 누적 → **축방향 성분**(u_n에 작용)이 A[j,k] (Γ_k 선형).
- 이 A를 Phase 1 `correct_noniterative`에 투입(나머지 동일). → 팁 rollup/수축 포착 → deficit↑.

**기대**: Phase 1(직선 semi-inf, 팁≈Dağ)와 달리 **곡선 free-wake가 팁 유도를 더 크게** 잡아
팁 결손 회복이 7%를 의미있게 넘을 가능성. step 2가 핵심 작업량.

## 다음
step 2(free-wake 기하) 집중 구현 → step 3(영향행렬) → HVAB A/B.
