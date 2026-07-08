# 01 — testC light 결과: 반경 절단이 팁 과부하를 못 고침 (2026-07-06)

데이터: `260706_5level_test/testC_light_pure_radtrunc_csv`(9.2rev) vs off baseline
`260630/260630_results_nasa_c81/pureALM_csv`(25rev). light 4-level, pure ALM.
그림 `testC_radtrunc_light_spanwise_compare.png`.

## 결과 — 예상대로 큰 차이 없음

| 지표 | off | radtrunc | |
|---|---|---|---|
| CT (6-9rev 동일창) | 0.01019 | 0.01014 | 사실상 동일 |
| M²Cn 팁 | 0.393 | 0.400 | 불변(실측 0.146 대비 2.7×) |
| 팁 α° | 5.96 | 5.57 | −0.4°(미미) |
| 팁 u_n | +0.00113 | +0.00184 | 소폭↑ |
| collapse 반경 r/R | 0.914 | 0.977 | ↑(외측 다운워시 더 유지) |
| 팁 retention | 11.3% | 18.5% | ↑ |

**판정: 반경 절단은 방향은 맞으나(외측 밴드 다운워시 소폭 회복, collapse 반경
0.91→0.98) 팁 과부하를 못 고침.** dCT_n 팁 하중·M²Cn·적분 CT 모두 사실상 불변.
spanwise 곡선 전체가 팁 포함 거의 포개짐. 부차: 중간스팬(r/R 0.3-0.6) α·CL 소폭
하락(root+팁 재분배 효과, 팁 이슈와 무관).

## 해석 — 가설 약화, but light는 청정조건 아님

- **outboard-leak이 팁 과부하의 지배 원인은 아닌 듯.** light는 ε이 2Δx floor(최대
  over-smear)라 leak이 가장 큰 격자인데도 팁이 안 내려감 → leak 제거만으론 부족.
  과부하는 더 근본적으로 **ε-smearing이 팁 vortex를 너무 퍼뜨려 집중 다운워시를 못
  만드는 것**에 기인(판별 A/three-way의 "보편적 팁붕괴"와 정합).
- **Merabet 재해석**: 그들의 성공은 4요소 중 **B-R급 조밀격자로 tip vortex 직접
  해상**(=해상도)이 지배적이고, 반경 절단은 부차적이었을 가능성. → 진짜 레버는
  여전히 **해상도**(testB: 감당가능 범위서 부분적, 비쌈).
- **주의(청정조건)**: light는 ε=2Δx floor로 confound. slab5(ε=0.25c, 팁 chord
  13.6셀, floor OFF)가 Merabet 조건에 맞는 청정 테스트. 단 three-way서 slab5 field
  ≈ light field였으므로 slab5서도 큰 개선 기대난망. 그래도 값싸니 확인 권장.

## 다음

1. **slab5 청정 확인**(`testC_slab5_pure_radtrunc.py`, ε=0.25c) — 절단 효과가
   조밀격자서 커지는지 마지막 확인. 기대는 낮게.
2. slab5도 미미하면 → **반경 절단 가설 기각**, 판정 = 팁 과부하는 ε-smearing/해상도
   본질. 경로 재확정: Prandtl surrogate(공학적) or SU2 근접장 or 훨씬 공격적 해상도.
3. 구현물(반경 절단)은 config off 기본이라 보존; Merabet 재현 옵션으로 남김.
