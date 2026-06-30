# HVAB 팁 해상도 요구 — light에서 얼마나 더 올려야 하나

작성: 2026-06-26 / 근거: `to_claude/tip_sampling_diag.py`, `configs/hvab/_hvab_hover_base.py` GRID_PRESETS,
문헌(Kleine 2022 App.B, Shives&Crawford 2013, Martínez-Tossas, Merabet-Laurendeau 2021)
관련: [[docs/almlbm_paper_analysis_kr.md]] §0.5, [[docs/papers_kr/2022_noniterative_vortex_smearing_correction_kr]]

## 결론 (한 줄)

**light는 팁에서 ~2.5× 부족하다.** 팁 chord가 6.3 cell뿐이라 **ε이 물리 최적값 0.25c가 아니라
2Δx에 floor(과스미어)되고, Gaussian 코어도 미해상(ε/Δx=2)**. 문헌 타깃(ε=0.25c **및** ε/Δx≈4)을
동시에 만족하려면 **fine preset(팁 chord≈16 cell, dx 2.5× finer, cells/R 320, DGX 128GB)**이 필요하다.

## 두 가지 해상도 타깃 (문헌)

1. **최적 ε = 0.25c** (Martínez-Tossas; Merabet 로터 ALM도 0.25c) — ε이 2Δx floor에 안 걸리려면
   `c/4 ≥ 2Δx` → **팁 chord ≥ 8 cell**.
2. **Gaussian 코어 해상** (Kleine 2022 App.B: ε=2Δx면 코어 미표현→u 오차 0.1%, **Δx≤ε/4(ε≥4Δx)면
   0.01%**; Shives&Crawford: AoA 오차 줄이려면 ε≈4Δx) — ε=0.25c와 동시 만족하려면 `0.25c ≥ 4Δx`
   → **팁 chord ≥ 16 cell**.

→ **binding 타깃 = 팁 chord ≥ 16 cell** (이때 ε=0.25c=4Δx, 두 조건 동시 충족).

## preset별 현황 (HVAB: R=1.689 m, 팁 chord=3.27 in=0.0831 m)

| preset | dx_fine[mm] | 팁 chord[cell] | cells/R | cells/D | ε/Δx | ε 분기 | 판정 |
|---|---|---|---|---|---|---|---|
| **light**(DEFAULT,24GB) | 13.20 | **6.3** | 128 | 256 | **2.0** | 2dx | ε floor(과스미어)+코어 미해상 |
| light_wide(24GB) | 13.20 | 6.3 | 128 | 256 | 2.0 | 2dx | 동일(반경 여유만 다름) |
| medium(~32GB) | 10.56 | 7.9 | 160 | 320 | 2.0 | 2dx | ε floor 직전, 코어 미해상 |
| **fine**(DGX 128GB) | 5.28 | **15.7** | 320 | 640 | **3.9** | c/4 | **ε≈0.25c OK, 코어 ε/Δx≈4 (타깃)** |

(dx_fine = (D_PHYS/D_on_L0)/2^(레벨−1). ε = max(c/4, 2Δx). 표는 팁 마커 기준.)

## 해석

- **light/medium**: 팁 chord < 8 cell → `c/4 < 2Δx` → ε이 **2Δx로 floor**. 즉 팁이 **물리 최적(0.25c)보다
  과하게 smear**(ε=2Δx=0.32·c_tip)되고, ε/Δx=2라 **코어도 격자에 미표현** → smearing-보정을 켜도 보정
  자체의 정확도가 떨어짐(Kleine App.B의 0.1% 영역).
- **fine**: 팁 chord 15.7 cell → ε=c/4=3.9Δx. **ε=0.25c(Martínez-Tossas/Merabet) + ε/Δx≈4(Kleine/Shives)
  동시 충족**. cells/R=320(Natelson 112보다 충분, Diaz "팁와류 D/48"의 13배). → Merabet 레시피의 ③
  "tip vortex 직접 해상" 성립 가능.
- **light→fine 비용**: dx 13.2→5.3 mm(**2.5× finer = +1.35 refinement level**), 셀수 ~30M→183M, 24GB→DGX 128GB.

## 시사점 (왜 ABC가 7%였는지와 연결)

ABC(light, pure-ALM)에서 샘플러를 바꿔도 팁 α가 7%만 회복된 핵심 이유:
**light 팁이 ε=2Δx로 과스미어 + 코어 미해상**이라 유동장에 애초에 팁 다운워시가 안 실린다.
샘플링은 "없는 걸 못 읽고", smearing-보정도 코어 미해상 영역이라 약하다. → **해상도가 1차 레버.**

## 권고

1. **fine preset로 ABC(또는 baseline) 1점 재확인** — 팁 chord 16/ε=0.25c/cells-R 320에서
   팁 φ·α가 운동량(~3°) 쪽으로 회복되는지. (Merabet 가설: 보정 없이도 상당 회복 기대.)
2. fine이 DGX 전용이라 비용 큼 → **중간 단계**로 medium(팁 7.9, ε는 여전히 2Δx floor 직전)은
   큰 개선 기대 어려움. 팁만 국소 refine(팁 영역 +1 level)이 비용 대비 효율적일 수 있음(검토).
3. fine으로도 잔차가 남으면 **Kleine 2022 비반복 vortex 보정**(코어 해상된 상태라 보정 정확도↑).

## 재현
```
python to_claude/tip_sampling_diag.py     # preset별 팁 ε/Δx·off-disk
```
또는 본 표 계산은 `_hvab_hover_base.py`의 GRID_PRESETS + ε=max(c/4,2Δx) 공식 그대로.
