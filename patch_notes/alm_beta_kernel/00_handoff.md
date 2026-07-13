# β Kernel Handoff — ALM 정규화 커널 노벨티 (다음 세션 시작점)

작성: 2026-07-13. 전제: HPC/멀티GPU 트랙 완결(4-rank 0.442 s/step,
25-rev ≈ 3.9h — 하루 2개 가설 실험 가능 체제). 이 문서 하나로 다음
세션이 ①문제를 완전히 파악하고 ②설계 결정을 내리고 ③구현·검증 plan을
확정할 수 있어야 한다.

---

## 1. 무엇을 위해 β kernel을 만드는가

**HVAB rigid hover CT 과대예측의 마지막 물리 레버.**

- 목표량: GT rigid CFD CT = 0.00832 (@collective 10°, 동일 케이스 Fluent).
- 우리 최선(case4 = archB+KSAS덱): **+14.8%**. 나머지 전 시도가 이 잔여를
  못 줄였고, 원인이 **Gaussian 정규화 커널의 finite-ε 편향**으로 수렴했다.
- 스팬 분해: 팁(r/R≥0.9)은 archB로 해결됨(로컬 M²cₙ dip 0.217 vs KSAS
  rigid 0.215). **잔여는 피크영역 r/R 0.85–0.97에서 +13~17%** — 하중이
  가장 큰 곳이 체계적으로 과대.

## 2. 우리 ALM의 한계 — 시도했고 기각된 가설 전부 (재탕 금지 목록)

| 가설 | 시험 | 결과 → 판정 |
|---|---|---|
| 격자 해상도 | D20~80 sweep (2026-07-08) | CT 단조↓ p≈1.5, **외삽 CT(h→0)=0.0097~0.0100 = 여전히 +17~20%** → 격자 artifact 아님. caveat: ε_lu=2 고정이라 grid+ε_phys 혼재 |
| 마커 수/간격 | D40·N24↔D40base, D80·N96↔D80base | RMS 0.0015~0.0017 = **마커수 무관** (crossover 가설 반증) |
| C81 덱 수준 | NASA↔KSAS/psu 덱 교차 | 덱 시프트 −5.4%는 있으나 잔여 설명 불가. 사용자 다수 확인으로 종결 |
| trim/기하 정의 | 다수 확인 | 종결 (재검토 금지) |
| rigid 가정 | rigid CFD(GT/KSAS)와 비교 | rigid끼리 비교로 배제 |
| SGS 소산 | case1↔5, 4↔6 (SGS on/off) | ΔCT +0.16%/−0.39% = **무관**. 3중 기각: ①CT 불변 ②와류코어가 이미 ε-커널 한계(pure FWHM 1.4–1.7ε ≈ 이론 2ε√ln2=1.67ε; archB 0.72ε 자체첨예화) ③ν_t 국소지배(3ν_t~7e-4 ≫ τ−0.5)인데도 하중 무영향 |
| Type2 팁 보정 | 부호 의심 포함 점검 | 약함 |
| 비등방 ε / ring 스무딩 | task 시험 | 전부 약함 |
| ε_r 간격 | δr≤ε_c,tip 겹침 수정(alm_eps_r_spacing) + n_radial=64 | blob 방지엔 필수였으나 잔여 미해결 |
| 팁 Gaussian 반경절단+재정규화 부재 | 구현·재실행(archB radial_truncation) | **팁 roll-off는 해결** — 그러나 보정이 하중을 inboard로 재분배해 피크영역은 오히려 유지/상승 |

**결론적 진단**: Gaussian ε가 유도속도장을 ε 스케일로 뭉개 마커 위치의
유효 받음각을 피크영역에서 과대평가한다. Kleine 보정(비반복 smearing
correction)이 이걸 일부 상쇄(−2.8%)하지만, **Gaussian이라는 커널 형태
자체의 잔여 편향**이 +15%로 남는다.

## 3. 문헌 좌표 — novelty가 성립하는 지점

- **Kleine 2022** (`docs/papers_kr/2022_Kleine_noniterative_smearing_kr.md`):
  Gaussian + 해석적 비반복 보정. ε≥3.5Δx 필요(와류코어 해상). 우리
  archB가 이 계열(wake="straight", 우리 tip ε=0.25c=4.2Δx 충족).
- **waLBerla-wind 2023** (`docs/papers_kr/2023_walberla_wind_kr.md`):
  compact-support 커널 사용, **보정 없음 = 명시된 open problem**.
  (lattice 128셀/직경급에서 결과 양호 — compact의 실전성 방증.)
- **Merabet 2021**: integral velocity sampling — 우리 방식과 동계열
  (±3ε 절단 + 이산 재정규화).
- **β kernel의 novelty = compact 커널 + 그 커널용 보정의 해석적
  재유도(결합)**. 어느 쪽 문헌도 이 결합을 갖고 있지 않다.

## 4. 철학과 논리 (설계를 지배할 원칙)

1. **물리-코드 일치**: 커널은 수학적 대상이다 — 스프레딩 η(r), 샘플링
   가중, 그리고 보정(그 커널의 해석적 자기유도속도)이 **같은 η에서
   유도**되어야 한다. 지금 코드는 Gaussian이 세 곳(스프레딩 RawKernel,
   샘플링 RawKernel, Kleine 보정식)에 각각 박혀 있다 — β 작업의 첫
   단계는 이 세 곳의 **커널 추상화**다.
2. **한 번에 한 변수**: 커널 형태를 바꿀 때 ε(폭), 절단, 정규화 규약을
   동결하고 A/B. 6-케이스 매트릭스가 이 규율의 전례.
3. **검증 사다리 재사용**: 새 커널은 물리 변경이므로 bit 기준이 아니라
   **등가·수렴·물리 게이트**로: 단위(모멘트/정규화) → 고립와류 →
   bench5 → D40. 단, 코드 치환 자체(추상화 리팩터)는 Gaussian 선택 시
   기존과 **bit-identical**해야 한다(리팩터 게이트).
4. **실험 체제**: D40 4-rank 3.9h/런(`--dist-init`, runbook 참조).
   케이스당 스팬 M²cₙ + CT 수렴 + 팁와류 FWHM을 표준 산출물로.

## 5. 설계 공간 — 다음 세션에서 결정할 것들

### 5.1 커널 후보 (η(r), 3D 정규화)
- **현행**: 절단 Gaussian exp(−r²/ε²), r≤3ε, 이산 재정규화.
- **후보 A — Wendland/β-spline류 compact**: C² 이상 매끄러움 + 정확한
  compact support. waLBerla-wind 계열.
- **후보 B — Peskin discrete delta**: 이산 모멘트 조건(Σw=1, Σw·x=0...)
  을 격자상에서 정확히 만족 — 이산 보존성 관점 최강.
- **후보 C — 저차 모멘트 보존 Gaussian 변형**(음의 로브 포함 고차 커널):
  smearing 편향 자체를 커널 모멘트로 상쇄하는 계열.
- 결정 기준: ①해석적 자기유도속도 u_ind(r; η)가 닫힌형/수치적분으로
  유도 가능한가(보정식의 전제) ②이산 모멘트 조건 ③support 크기
  (셀 수 = 계산비용, 샘플링 커널 S³) ④음향 목표와의 정합(고주파 잡음).

### 5.2 보정 재유도
- Kleine 비반복식의 구조: u_corr = f(Γ분포, η의 자기유도). Gaussian이면
  닫힌형. β 커널이면: (a) 닫힌형 재유도 시도 (b) 1D 수치적분 사전표
  (ε/Δx별 LUT) (c) FLLC(Meyer-Forsting) 이식 중 택1.
- 주의: 보정의 wake 모형은 "straight" 유지(분산 지원·production 검증됨).

### 5.3 구현 지점 (코드 지도)
- `src/actuator/spreading.py`: 스프레딩 RawKernel(_spread_rawkernel_gpu)
  + radial truncation 스케일(compute_radial_scales_batch) — exp 하드코딩.
- `src/actuator/interpolation.py`: `alm_sample_markers` RawKernel —
  exp(−d²/ε²) 하드코딩, half=ceil(n_cut·ε_max)+1 박스.
- `src/actuator/actuator_line.py`: `_kleine_w_corr`(보정식),
  `_eps_correction` config 파싱.
- 추상화 제안: config `actuator_line.kernel = {"type": "gaussian"|"wendland"|...,
  "eps": ..., "n_cut": ...}` → 커널별 (η 계수 테이블, support 반경,
  보정 파라미터)를 한 곳에서 생성해 세 RawKernel 소스에 베이크.
  RawKernel은 ASCII 전용([[feedback_cuda_kernel_ascii]]) + 고정순서 누적
  유지(분산 결정성).

### 5.4 검증 plan 골격 (다음 세션에서 확정)
1. **G-β0 리팩터 게이트**: kernel="gaussian"에서 기존 경로와 bit
   (스프레딩/샘플링/보정 셋 다). 이것 전에는 물리 실험 금지.
2. **G-β1 단위**: 커널 모멘트(0차=1, 2차 등), 이산 재정규화, 등방성,
   support 경계 연속성 — numpy 대조.
3. **G-β2 고립 와류**: 단일 고정 마커 링/직선 와류의 유도속도 vs 해석해;
   FWHM vs 이론(Gaussian 1.67ε 상당의 β-커널 값 유도 포함).
4. **G-β3 bench5 A/B**: gaussian↔β, CT 궤적 CV-band + 스팬 분포 비교
   (물리 차이는 기대되는 것 — 게이트는 안정성/보존성).
5. **D40 매트릭스**: case1'(β pure)·case4'(β+보정) — 표준 산출물로
   피크영역 M²cₙ과 CT vs GT. 성공 기준: **피크영역 +13~17% → 한 자릿수
   %, CT +15% → +5% 이내**(팁 dip 0.215 유지 확인 필수 — 보정 재분배로
   팁이 되살아나지 않는지).

## 6. 자산 지도

- 6-케이스 결과·분석: `aeromechanics_workshop/HVAB/0711_gpu_optimum/`
  (analyze_4case.py, analyze_vtk_tipvortex.py, summary_6case.csv)
- 스팬/팁 분해 산출물: 0708_D_grid_test/, spanwise_*.png, tip_overload_decomposition.png
- 논문 요약: docs/papers_kr/ (Kleine, waLBerla-wind, Merabet)
- 실행: docs/SIMULATION_RUN_GUIDE_kr.md — 4-rank D40 = 3.9h, 마커 VTP에
  마커별 alpha/CL/F_n 16종(스팬 분석이 CSV 없이도 가능)
- 관련 메모리: project_hvab_overpred_decomp, project_hvab_grid_convergence,
  project_alm_natelson_sweep, project_next_session

## 7. 다음 세션 첫 액션

1. 이 handoff 정독 → §5.1 커널 후보와 §5.2 보정 전략에 대한 **설계
   결정**(사용자와 합의) → §5.4를 구체 plan으로 확정(01_design.md).
2. G-β0(커널 추상화 + gaussian bit 게이트)부터 착수 — 물리 실험은
   그 다음.
