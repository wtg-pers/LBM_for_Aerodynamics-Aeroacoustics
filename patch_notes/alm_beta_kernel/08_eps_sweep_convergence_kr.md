# 08 — 유한-ε 스윕 / 보정 수렴도 (2026-07-20~)

## 1. 질문 (β 트랙 결론에서 파생)

커널 형태가 기각되면 잔여 +15%(피크 0.85–0.97R)의 다음 후보는 **유한-ε
스미어링 스케일**. 이론(Martínez-Tossas FLLC 2019): 보정이 수렴하면
결과는 **ε-무관**해야 한다. 그러므로 결정 질문 하나:

> **우리 archB 보정은 이미 수렴했는가 (= 피크영역이 ε에 둔감한가)?**

- ε-무관 → 보정 수렴 → +13%는 ALM+2D폴라의 **수렴된 근본 답** →
  유한-ε 배제 → 원인은 모델 외적(팁 손실 = Shen 방향 정당화).
- ε-의존 → 보정 미완성(과소 회복) → 유한-ε가 살아있는 레버.

## 2. 설계 — 한 변수(ε)만 변화

**동결**: D40 farfield40, gaussian 커널, KSAS덱, n_radial=64, 4-blade,
prandtl OFF, archB(kleine-straight + radial-trunc + soft-start).
**변화**: `epsilon_chord_factor` (ε_base = factor·chord). 코드 노브 신설
(blade.py, 기본 0.25 = 원본 chord/4 bit-identical; rotor/config 관통).

| factor | ε_tip/Δx (D40) | ε/chord | 비고 |
|---|---|---|---|
| 0.20 | ~3.3 | 0.20 | 하한(격자해상 ε≥3Δx 유지) |
| 0.25 | ~4.2 | 0.25 | 현행 baseline(재런 불요, gaussian archB 결과 보유) |
| 0.30 | ~5.0 | 0.30 | |
| 0.35 | ~5.8 | 0.35 | 상한(1.75× 스팬) |

동시에 **pure(무보정)도 극단 2점(0.20/0.35)** 실행 = "무보정 ε-민감도"
기준선. 판정 지표 = **archB 기울기 / pure 기울기** (보정 효능).

지지 스팬 확인: factor 0.20에서 tip ε=3.3, gaussian 3ε 컷=9.9 > δr=3.74
→ 마커 지지 여전히 겹침(연속 선 유지). n_radial 변경 불요(동결 유지).

## 3. 판정 규칙 (피크영역 M²cₙ 스팬곡선)

- archB 곡선들이 ε 전 범위에서 **붕괴(collapse, Δpeak ≲ 1–2%)** →
  보정 수렴 → **유한-ε 배제 확정 → Shen 트랙 GO**.
- archB 곡선이 ε따라 **부채꼴(fan)** → 보정 미완성 → 유한-ε 레버 有
  (수렴점=ε→해상한계 방향; free-wake 보정 재고).
- pure는 강하게 부채꼴이어야 정상(대조군). archB/pure 기울기비가
  보정이 ε-민감도를 얼마나 죽였는지 정량.

## 4. Shen 이중계산 경고 (판정과 직결)

Shen(팁손실)은 모델에 더하는 팩터. archB wake 보정이 이미 팁 relief를
일부 제공(pure→archB 피크 −4.5점)하고 Prandtl은 OFF(prandtl_loss=False).
**wake 보정 위에 Shen을 얹으면 이중계산.** 이 스윕이 "보정이 이미
얼마나 relief를 만드나"를 정량화 → Shen 추가 여지/양을 알려줌. Shen 켜기
전 이 진단 선행이 정도(正道).

## 5. 실행 (클러스터, 사용자)

configs: `hvab_hover_c10_farfield40_eso_epssweep_{archb,pure}_eps{20,25,30,35}.py`
- 각 15 rev(피크는 전역 CT보다 빨리 정착; ~5h/런 2-rank).
- **단계적 권장**: 먼저 archB 극단 2점(eps20, eps35) → 붕괴/부채꼴이
  명확하면 조기 판정, 애매하면 중간점(eps30)+pure 극단 추가.
- 분석: `analyze_eps_sweep.py` (스팬 붕괴 오버레이 + 피크 기울기 표).
- 25rev 도장 런과 GPU 경합 주의(도장 우선).
