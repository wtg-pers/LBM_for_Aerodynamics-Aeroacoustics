# Kleine, Hanifi & Henningson 2022 — 비반복(non-iterative) vortex smearing 보정 ALM — 리뷰 (2026-07-10)

> 원문: "Non-iterative vortex-based smearing correction for the actuator line method"
> (arXiv:2206.05448, KTH/ITA). `to_claude/ref_papers/alm/`.
> ★★우리 코드와의 관계: **`eps_correction method="kleine"`이 이 논문의 구현**
> (`actuator_line.py` `_kleine_wake_mode ∈ {"straight","free"}`). archB(CASE 3/4)의 이론 기반.

## 1. 계보와 기여
- 계보: Dağ&Sørensen 2020 / Martínez-Tossas&Meneveau 2019(**missing velocity** 개념: Gaussian으로
  스미어된 와도가 만드는 유도속도와 이상 와류의 차이를 해석적으로 보정) → Meyer Forsting FLLC(반복법)
  → **본 논문 = 반복 제거**.
- 핵심 아이디어: lifting line을 **전 시간스텝 순환 Γⁿ⁻¹ 주위로 선형화** → 매 스텝 보정이 **작은 선형계
  직접해**(relaxation 반복 불필요, 결정적·안정). Cl(α)가 비선형이어도 Γⁿ⁻¹이 좋은 선형화점이라 성립.
  **Cl 기울기(dCl/dα)** 필요 — shape-preserving piecewise cubic으로 공급.
- 부가 기여: ① **스미어드 와류 세그먼트의 유도속도 해석식**(보정함수 근사 제거) ② CFD 속도로 이류되는
  **free-vortex wake**(tracing particle, 필라멘트 융합 n_w=50/nnw=10/d_w=ε/2 → 팁 근방 wake 길이 ~20ε,
  1차 Euler, quasi-steady) ③ ALM↔비선형 LL의 **수학적 동치 증명**(セグ먼트 상수 순환 가정 하).

## 2. 검증·결과
- 솔버: **Nek5000 스펙트럴 요소**(7차) + AMR(액추에이터 주변 강제 최대세분). 병진 날개(AR=10,
  Re_c=1e4, 이상 익형 Cl=2πα) + **NREL 5MW, 전단 유입**(비정상 테스트, TSR 7.55).
- 날개: LL 대비 유도속도 차 **~1e-4**(문헌 최고 수준; Dağ·Meyer Forsting 보고치보다 훨씬 작음).
  반복 vs 비반복 차이 무시가능(비정상 로터에서도).
- **ε-불감성 달성**: ε=3.5Δx vs 7Δx 힘 거의 동일(보정의 목적 그 자체). 단 완전 제거는 아님 — near-wake는
  여전히 ε 영향(Meyer Forsting 2019b와 일치), 힘 차이는 ALM 근사 오차 수준.
- ★**Appendix B — ε=2Δx 경고**: 관행적 최솟값 ε≈2Δx에선 **Gaussian 와류 코어가 격자에 미해상** →
  보정 이론(코어의 완전한 Gaussian 표현 가정)이 깨져 오차 ~10×(1e-4→1e-3). 같은 ε라도 격자 세분 시 회복.
  즉 **ε/Δx ≥ 3~3.5 권장**(그들의 검증도 3.5Δx 기준).
- 한계(자인): 3D drag 보정 없음, 비정상 shed vorticity(스팬방향) 무시(quasi-steady), blade-resolved 대조 미완.

## 3. 우리 구현과의 대조 (실무 포인트)
1. **비반복 선형계 = 우리 경로**: patch04 "dcl 벡터화 solve 99→10ms"가 정확히 이 선형계. Dağ
   influence_matrix(`wake="straight"` = 반무한 직선 필라멘트) = 선형계 행렬의 straight 버전.
2. ★**wake 선택의 긴장**: 논문은 **free-vortex를 일반성 근거로 권장**(직선/나선 처방은 값싼 대안으로만).
   그러나 우리 bench5/slab5 실험은 **free가 팁 ~1/12 과소보정 → 폐기, straight 채택**. 모순이 아니라
   조건 차이로 해석 가능: 그들의 검증은 **강한 축류**(병진 날개·풍력 TSR7.55) — wake가 빠르게 하류로
   이류되어 ~20ε 길이·quasi-steady·1차 Euler로 충분. **호버는 자유류 0** — 나선 wake가 로터 근방에
   머물러 유도속도가 지배, 짧은 wake·융합·quasi-steady 가정이 모두 악화. 우리 결론(straight)은 논문
   권고의 이탈이지만 물리적으로 방어 가능. (그들 free-wake의 CFD-속도 이류 = 우리가 관측한 fp-카오스
   민감성의 근원이기도 — CV-band 게이트 정당화.)
3. ★**Appendix B ↔ 우리 격자 설계**: 우리 D-sweep의 ε_lu=2 고정 caveat(메모리)가 정확히 Appendix B의
   경고 지점. **D40 팁 ε=0.25c ≈ 4.2Δx**(c_tip/dx=16.67) → Kleine 권장영역(≥3.5Δx) 안 — farfield40
   설계가 이 논문 기준으로 건전함을 확인. 조계면(코어스 레벨)에선 ε/Δx가 반감되므로 L4 슬랩이 ALM
   지지역을 전부 덮는 우리 설계가 중요.
4. **힘 계산 속도 선택**: 그들은 보정된 속도로 힘 계산(모호성은 Kleine et al. 2023에서 해소, 2차 오차).
   우리 구현도 corrected velocity 사용 — 점검 시 target="inviscid" 옵션과 함께 재확인 가치.
5. **walberla-wind와의 연결**: walberla-wind가 미해결로 남긴 "격자 세분 시 힘 감소"(compact 커널 무보정)의
   정답이 바로 이 보정 계열 — 우리 β단계(새 커널로 FLLC/smearing 보정 재유도)의 위치가 두 논문 사이 공백.

## 4. 비판적 노트
- 검증이 **저하중·강축류** 케이스 중심(이상 익형 날개, NREL 전단). 호버 로터(자유류 0, 팁 하중 집중,
  압축성 팁)로의 외삽은 미검증 — 우리 4-케이스가 사실상 그 실험.
- Nek5000 스펙트럴(저산일)이라 LBM 대비 수치 산일 환경이 다름 — 보정 자체는 솔버-불가지론적이지만
  vortex core 표현 오차(Appendix B)는 스킴 의존일 수 있음.
- 계산비용 분석을 의도적으로 유보("future studies") — 우리 S1/S2 GPU 최적화가 그 공백을 실무로 채운 셈.

## 5. 액션
1. 4-케이스 archB(CASE 3/4) 해석 시: ε_tip/Δx=4.2가 Kleine 권장영역임을 명시(보정 이론 유효 전제 충족).
2. free vs straight 결과 논의 시 §3-2의 "호버=자유류 0" 논거로 논문 권고 이탈을 정당화.
3. β단계(커널 노벨티) 포지셔닝: walberla-wind(무보정 compact, 미해결) ↔ Kleine(Gaussian+보정) 사이에서
   "compact 커널 + 재유도된 보정"으로.
