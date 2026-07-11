# Merabet & Laurendeau 2021 (CASI) — 헬기 로터 ALM — 요약 (2026-07-12)

> 원문: "ALM for helicopter rotors" (CASI). `to_claude/ref_papers/alm/2021_Merabet_...pdf`.
> 우리 ALM 레시피(ε=0.25c, radial truncation+renorm, integral sampling)의 원류 계보 중 하나.

## 1. 무엇을 했나
- **Star-CCM+ URANS(k-ω SST)**, 2차 FV, dual-time 1°/step. ALM은 User Coding(C, MPI) 외부 라이브러리.
- 3케이스: 호버=**S-76**(rect tip, Mtip 0.6, AIAA Hover Prediction Workshop 소재), 전진비행, 지면효과/밀폐(NACA0012 로터).
- **B-R(blade-resolved overset)과 동일 배경격자(10-20M셀)** 공유 — ALM vs B-R 정면대조. B-R은 블레이드당 +5-10M셀.

## 2. ALM 레시피
- BET + **integral velocity sampling**(그들 ref[16,29,30]; [30]=자체 샘플링 파라메트릭 연구)
- Gaussian spreading **ε≈0.25c** ("best results" [30-33]) + **radial truncation & re-normalization**
- ★**FLLC류 smearing 보정 없음, tip-loss 모델 없음**

## 3. 핵심 결과 (호버, trim CT/σ=0.09, vs B-R)
- 추력 예측 우수(고 collective에서 ALM 소폭 과소), 토크 소폭 과대 → FM 소폭 저하(팁 과잉 드래그).
- 스팬 하중: **80%까지 일치 우수, 92%+에서 상하 진동** — 원인을 명시적으로 **선행 블레이드 팁 와류
  조우(BVI)의 과잉 up/down-wash**로 귀속. 팁 와류 위치 추적도 실험과 양호.

## 4. Integral velocity sampling — 그들 vs 우리
개념 동일: point sampling은 자기 힘의 국소 교란에 오염 → **스프레딩과 같은 Gaussian으로 속도장 가중적분**.
우리 구현(`interpolation.py:400`): u(x_j)=Σu·η_ε/Ση_ε, ε=0.25c 등방, **±3ε 절단+이산 재정규화**(절단오차
분모 상쇄). 그들: 동일 원리(세부는 ref[30]에 위임), 절단/정규화 명시 없음.
⇒ **샘플링은 사실상 동일**. 차이는 주변 레시피(그들=보정 無 / 우리 archB=+FLLC-straight)와 유동장의 질.

## 5. ★우리 ⑤(ILES/SGS) 가설과의 연결 (2026-07-12 판독)
Merabet이 **보정 없이도** 팁이 맞은 기전 = **선행 블레이드 팁 와류가 살아남아 BVI 다운워시를 공급**.
재료(sampling/ε/trunc)가 같은 우리가 팁 과부하였다면 남는 변수 = **와류 생존성(소산)**.
우리 설정 = cumulant **ω_high=1.0**(ILES성 감쇠 최대, 코드 기본값·HVAB 미override 확인) + dyn_smag 중첩:
팁 와류 조기확산 시 ① BVI 다운워시 상실→팁 과부하(pure 서명) ② 후류 순환 약화→유도유입 과소→
**전 스팬 offset**(D40 잔존 +15%의 성격) — 한 기전이 두 증상 설명. → CASE5/6(SGS-off A/B)로 검증.

## 6. 한계/주의
- URANS k-ω SST 자체 소산도 큼 — "그들이 팁을 맞춘 건 저소산이라서"라고 단정 불가(배경격자 10-20M로
  로터 근방 해상이 좋았던 점, 1°/step 시간해상 병행). A/B는 우리 솔버 안에서 해야 결론 가능(=CASE5/6).
- S-76은 HVAB와 다른 로터(rect tip, Mtip 0.6) — 정량 이식 불가, 기전 비교만.
