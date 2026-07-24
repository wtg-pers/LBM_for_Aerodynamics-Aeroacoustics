# 22 — cumulant 3차 완화율 파라미터화(K17L) + 스윕 인프라 (2026-07-22)

## 근거 (Geier 3편 정독, to_claude/ref_papers/cumulant/)
- **2017 Part I (JCP 348:862)**: 4차 정확 확산 파라미터화 — ω₃,ω₄,ω₅를
  ω₁,ω₂의 닫힌형으로(식 111–113). **전부 (ω₁−2) 인자 → 우리 운전점
  (τ≈0.5, ω₁→2)에서 ω₃=ω₄=ω₅→0** = 우리 기본값 all-one(매 스텝 3차
  cumulant 평형화=최강 필터링)과 정반대. 최적점은 all-one보다 불안정 →
  **limiter 식(116)** 필수: ω_lim = ω + (1−ω)|C|/(ρλ+|C|), λ=0.01 canonical.
  A,B(4차 cumulant 평형 보정, 식 114–115)는 완전 4차 정확도용 — 미구현
  (소산 레버가 목적이지 형식적 4차가 아님).
- **2021 TGV**: K17L이 under-resolved/LES 워크호스. limiter는 "WALE보다
  훨씬 가파른 low-pass"(최고 파수만 감쇠=코어 보존). Re1600 미해상에서
  WALE 생략이 오히려 개선(우리 SGS-off 방향 지지). K17이 안정성도 개선
  (Re 3000 vs all-one류 1980 @64³). 주의: Re 160k서 스펙트럼 bottleneck.
- **2017 Part II**: K17L을 **nested grid**(구 항력위기)서 검증 = MLG 호환 선례.

## 구현
- CUDA(esoteric_cumulant_d3q27.py): 시그니처에 omega_3/4/5·lambda_lim 추가,
  w3/w4/w5 분리(w6–10은 omega_high 유지), 3차 완화 7슬롯(합3·차3·C111)에
  per-cumulant limiter(λ≤0=off). 래퍼 kwargs 기본 None→omega_high 폴백.
- 플럼: CumulantCollision(omega_3/4/5·limiter kwargs, omega_high 속성 신설)
  → setup(sim_params omega_3/4/5·cumulant_limiter 읽기+에코 출력) →
  eso 메타데이터 _eso_omega_345/_eso_lambda(BGK-안전 폴백) → from_level →
  LocalLevel.advance → 커널. 단일-GPU eso 호출부도 동일 전달.
- ★기존 _eso_omega_high가 collision.omega_3에서 읽던 conflation 수정
  (omega_high 속성으로 분리 — ω₃=0 설정 시 w6–10까지 0이 되던 함정 예방).

## 검증
- 컴파일 ✓, gbeta0 gate PASS, eso_sgs_alm_gate PASS.
- **bit-호환 주의**: 기본 경로가 fast_math 재결합으로 ~3e-8(상대 2e-6)
  변화(로직 불변, 코드 추가→명령 재배열). 게이트 허용오차(fp32 1e-6급)
  내 — 저장소 기준 충족으로 수용, 문서화.
- K17L 스모크(bench5, SGS off, 300스텝): 안정·NaN 無·완화율 실활성
  (all-one 대비 Δ 6.5e-7 = 노이즈 20배) ✓. SGS 프리패스 부재로
  0.209 s/step(기존 0.254).

## 스윕 설계 (mlg4_nosgs 기준선 = all-one 263°)
| config | ω₃₄₅ | λ | 판정 |
|---|---|---|---|
| mlg4_nosgs (완료) | 1.0 | — | half-life 263° 앵커 |
| **mlg4_k17l** | **0.0 (Geier 극한)** | 0.01 | 본명제: 84% 바닥이 3차 필터링인가 |
| mlg4_w3mid (선택) | 0.5 | 0.01 | 응답 형상(선형/문턱) — k17l 효과 크면 |
전부 SGS off·mlg4 격자·20rev. 지표: in-level half-life/코어/궤적(하중인과
예측=불변)/CT. limiter λ 민감도는 필요시 후속(0.001/0.1).
