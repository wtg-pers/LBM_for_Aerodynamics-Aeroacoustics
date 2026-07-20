# 09 — Path A: Shen 팁손실 (wake 보정 대체) (2026-07-20)

## 0. 결정 (사용자, 2026-07-20)

β 트랙 3단계 계획 확정:
- **A(지금)**: straight FLLC를 한계로 수용 → OFF → **Shen 팁손실 모델로
  대체**, 결과 확보. (wake 보정 완전 미적용.)
- **C(다음)**: 보정 재검토 (hover에 맞는 계열).
- **B(그다음)**: C 결과 보고 free-wake 제대로 재구현 or prescribed-helix.

배경(05~08): 커널 형태 기각(wendland≈gaussian), Kleine-straight 보정이
큰 ε에서 과보정·ε-의존(08 ε-스윕). FLLC/Dağ/Kleine 셋 다 **축류
검증뿐, hover 검증 전무** → hover용 스미어링 보정 부재가 진짜 gap.

## 1. 구현

`prandtl_loss` 기계를 재사용 (Shen = Prandtl의 f에 계수 g):
- Prandtl: F=(2/π)arccos(exp(−f)), f=(B/2)(R−r)/(r sinφ)
- **Shen 2005: f → g·f** (g<1이면 롤오프가 안쪽으로 넓어짐).
  Shen g = exp(−c1(Bλ−c2))+0.1 (c1=0.125, c2=21) → **hover(λ=ΩR/U∞→∞)
  극한 g=0.1**.
- config: `prandtl_loss={"model":"shen","g":0.3,"tip":True,"root":False,
  "eps_offset":False}`. **g=1.0 = Prandtl (bit-identical)** — `_tip_loss_g`
  기본 1.0, 검증됨. 코드 변경 = actuator_line.py만 (init+factor+parse).

## 2. ★핵심 발견 — g=0.1은 과공격적, 보정 필요

팁손실 factor 프로파일(φ≈8°):

| r/R | Prandtl g=1 | Shen g=0.5 | Shen g=0.1 |
|---|---|---|---|
| 0.72 | 0.997 | 0.95 | **0.46** |
| 0.91 | 0.84 | 0.67 | **0.25** |
| 0.96 | 0.64 | 0.48 | **0.17** |

g=0.1(교과서 hover 극한)은 **0.72R 중간 스팬 하중까지 절반으로** 깎음 —
물리적으로 과함. Shen g 공식이 풍력용 보정이라 λ→∞ 외삽 시 부작용
(Dağ가 경고한 "경험식, 형상 무관하게 유효하지 않음"의 실례). **g는
스윕으로 보정 필요.**

## 3. 검증

- g=1.0 → Prandtl factor와 bit-identical (오프라인).
- bench5 pure+Shen(g=0.3) 120스텝 e2e: 정상 완주, **팁 F_n 매끄러운
  롤오프**(0.082→0.051), 음수 튐 없음(과보정 병리 부재 — 05~07의 wake
  보정 병리와 대조적).

## 4. g-스윕 (클러스터, 사용자)

configs: `hvab_hover_c10_farfield40_eso_shen_pure_g{100,050,030,010}.py`
(pure ALM + KSAS + Shen, radial-trunc 없음 = 팁 처리 중복 방지, 15rev).

- 판정: healthy peak M²cₙ(0.85–0.97R)이 **한 자릿수 %로 내려오는 g** 탐색.
  pure+KSAS 기준선 피크 ~+17% → 목표 단자릿수.
- 예상: g=1(Prandtl)이 이미 강한 롤오프 → 아마 g=0.5~1 부근이 적정,
  g=0.1은 과보정으로 피크 붕괴 예상. **g=1.0/0.5/0.3부터** 권장.
- ★caveat(이중계산): pure ALM도 resolved-wake 팁 relief 일부 있음 → Shen을
  얹으면 소폭 이중계산 불가피(BEMT 팁손실을 ALM에 쓰는 본질적 긴장).
  g 보정이 그걸 흡수하는 셈. 이건 Path A의 알려진 한계로 수용.
- 산출물: analyze_shen_gsweep.py (데이터 후 작성).
