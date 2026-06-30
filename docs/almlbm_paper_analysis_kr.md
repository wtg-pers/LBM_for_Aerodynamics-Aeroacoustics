# ALM-LBM 구현 ↔ 참고논문 5편 정밀 대조 분석

작성: 2026-06-26 세션 / 대상: 우리 ALM-LBM 솔버 (`src/actuator/`) vs `to_claude/ref_papers/lbm_alm/` 5편
핵심 질문: **HVAB 팁 유입 결손(tip φ→0, α≈기하피치)이 구현 결함인가, smeared ALM 본질 한계인가?**

관련 기존 문서: `docs/alm_epsilon_theory_kr.md`, `docs/alm_mach_pass_theory_kr.md`,
`docs/alm_tip_overprediction_record_kr.md`, `aeromechanics_workshop/HVAB/HVAB_ALM_results_report_kr.md`

---

## 0. 결론 요약 (Executive Summary)

1. **팁 유입 결손은 본질적으로 smeared-ALM 고유 현상이며, 5편 모두 "구현 버그"가 아니라 "ε·해상도·속도샘플링이 지배하는 가변(tunable) 오차"로 규정한다.** 단, "완전히 못 고치는 한계"는 아니고 **ε 축소 + 로터를 finest level에 배치 + 속도샘플링 보정**으로 줄일 수 있다고 일치한다.

2. **우리 솔버에는 5편 어디에도 없는 고유 요소가 정확히 하나 있다 — Gaussian-weighted(±3ε) 속도 샘플링.** 5편 전부 point/trilinear(Asmuth·Diaz·Watanabe) 또는 ring-average-linear(Natelson) 샘플링을 쓴다. 우리만 spreading과 동일 커널로 ±3ε 구를 평균한다 (`actuator_line.py:317` → `interpolate_velocity_batch_gpu`). 팁에서 이 구가 disk 바깥(induction이 거의 없는 영역)을 평균에 섞어 downwash를 희석 → u_n 과대 → φ→0 → α≈기하피치. **메모리의 "팁 외측 upwash 섞임" 가설과 기계적으로 일치하며, dx·마커·extent·margin이 모두 기각된 뒤 남은 "ALM 고유" 항목의 가장 유력한 정체다.**

3. **즉, 팁 결손은 "ALM 본질 + 우리 샘플링 방식이 증폭" 의 복합이다.** 본질 부분(smearing de-induction)은 Dağ/Meyer-Forsting류 보정으로, 증폭 부분(±3ε 평균의 off-disk 오염)은 **샘플링을 point 또는 비등방/내측편향 커널로 바꾸는 A/B**로 분리·검증 가능하다. 이 A/B가 다음 결정적 실험.

4. **부차 발견**: (a) production task3 config가 ALM에 **Prandtl tip-loss**를 켜는데, Diaz는 "ALM에 Prandtl/Glauert는 해상된 induction을 이중계산 → ALM엔 variable-ε를 써라"고 명시 → 방법론적으로 재검토 필요. (b) Asmuth 두 편이 지목하는 정석 팁보정은 **Meyer-Forsting(2019) vortex-based tip/smearing correction**인데 우리는 그 사촌격인 Dağ&Sørensen만 구현(메모리상 1/13로 불충분) → Meyer-Forsting 검토 가치. (c) 우리 Mach-pass + sweep cosΛ는 **5편 전부보다 앞서는** 강점(결함 아님).

---

## 0.5 ★중대 정정·보강 (2026-06-26 갱신 — `alm/` 라이브러리 반영 + ABC 실험 결과)

> 본 분석은 처음 `to_claude/ref_papers/**lbm_alm**/` 6편만으로 작성됐다. 이후 **형제
> 디렉토리 `ref_papers/alm/`**(스미어링-보정 계보 + 로터크래프트 ALM)를 누락했음을 발견,
> 그리고 ABC 샘플러 실험을 실제로 돌려 아래를 **정정**한다. (`docs/papers_kr/`에 한글 요약 존재.)

**누락했던 핵심 논문** (`alm/`):
- **Merabet & Laurendeau 2021** (헬리콥터 로터 ALM, S-76 hover = **HVAB에 가장 가까운 검증**)
- **Kleine et al. 2022** (KTH, arXiv 2206.05448, JFM) — **비반복 vortex smearing 보정**(최신·최강)
- Martínez-Tossas 2017/2019(filtered LL 이론 기반), Dağ 2020(우리 보정 출처), Rullaud 2018·Luca 2024(2D ALM-LBM)

**정정 1 — §0.2 "우리만의 Gaussian 샘플링 = 유력 주범" 은 틀렸다.**
Gaussian-weighted(integral/g-가중) 속도 샘플링은 **anomaly가 아니라 표준 우수 관행**이다.
Merabet-Laurendeau(로터 ALM)가 **정확히 우리와 같은 integral 샘플링**을 쓰며, "force와 같은 커널로
속도를 평균해 self-induction을 일관 처리 → ε 민감도↓"가 **장점**이라고 명시. 5편(lbm_alm)이
point였던 건 그 표본 편향이었을 뿐, 광의 ALM 문헌(Sørensen-Shen 계보)은 integral이 정석.

**정정 2 — ABC 실험이 샘플링 가설(b)을 대체로 반증.** HVAB c10 pure-ALM에서
gaussian/mask_disk/point 비교 결과 **팁 α 7.31→6.72→7.00°**(결손 4.3° 중 **7%만 회복**), φ도 거의
불변. 즉 **off-disk 희석은 기하적으론 ~30%지만 실제 영향은 미미**(under-induced 팁엔 희석할
다운워시 자체가 없음). 내 합성 smoke가 step-downwash를 가정해 과대예측했던 것. → 팁 결손의
지배 원인은 **(b)샘플링이 아니라 (a)유동장의 본질적 under-induction + 해상도**.

**정정 3 — 정석 팁모델은 Meyer-Forsting이 아니라 Kleine 2022(우리가 이미 보유).**
§0.4(b)/§4 P3의 "Meyer-Forsting 검토, 우린 Dağ만"은 부정확. Kleine 2022가 더 진보(비반복
선형화 + 정확 해석 유도속도식 + free-vortex wake, ALM↔LLT 0.01%). 우리 `_viscous_core_correction`은
이 계보의 **최단순 멤버**(직선 근사·single-pass)라 7%였던 것. ([[2022_noniterative_vortex_smearing_correction_kr]])

**정정 4 — 1차 레버는 "해상도"다 (Merabet 레시피).** Merabet의 "tip 보정 불요" 비결 =
**① ε≈0.25c + ② integral 샘플링 + ③ tip vortex를 직접 해상하는 조밀 격자(10–20M, B-R급)**.
우리 light preset은 팁 chord 6.3 cell이라 **ε이 0.25c가 아니라 2Δx에 floor(과스미어) + 코어 미해상
(ε/Δx=2)**. → 해상도가 binding. 필요량은 **fine preset(팁 chord≈16, ε/Δx≈4, cells/R 320; light 대비
dx 2.5× finer; DGX)**. 상세: `docs/hvab_resolution_requirement_kr.md`.

**갱신된 판정**: 팁 결손 = **(a) smeared-ALM 본질 under-induction + 해상도 지배**. (b)샘플링은
부차(~7%, 반증). 경로 = **① 해상도↑(fine, Merabet 레시피) → ② 잔차에 Kleine 2022 비반복
보정**. (Prandtl-on-ALM은 여전히 surrogate로만; §4 ④ 유지.)

---

## 1. 논문 개요 (lbm_alm 6편 + alm 스미어링-보정 라이브러리)

| # | 논문 | 솔버/스킴 | 로터 모델 | 응용 | 우리 케이스 근접도 |
|---|------|-----------|-----------|------|-----|
| 1 | **Asmuth 2019** (LBM Frameworks) | ELBE, **Cumulant** "AllOne", D3Q27(추정) | ALM | NREL 5MW 풍력 | 방법론 정석 |
| 2 | **Asmuth 2020** (wakes; WES 5,623) | ELBE, **Cumulant** parameterized, D3Q27, Smagorinsky C_s=0.08 | ALM | NREL 5MW wake | 방법론+wake 정석 |
| 3 | **Diaz 2023** (WES 8,363) | **OpenFOAM/SOWFA LES**(★LBM 아님) | ADM & ALM, 3 force법 | 풍력 wake farm | **우리 Stage 설계 출처** |
| 4 | **Natelson 2026** (VFS Forum 82) | in-house GPU LBM, **MRT**, D3Q19, Smagorinsky C_s=0.042, wall-model | ALM & ADM (blade-resolved 아님) | **XV-15류 proprotor+wing, cruise/conversion** | ★로터 최근접 |
| 5 | **Watanabe 2026** (TAML) | in-house GPU LBM, **Cumulant**, D3Q27, implicit-LES, FP32 | ALM | 풍력 farm real-time | 최신·최상세 방법서술 |

> 주의: Diaz는 폴더상 lbm_alm에 있으나 **FV-LES(SOWFA)** 논문. 가치는 flow solver가 아니라 **force-calc taxonomy(airfoil/analytical/numerical = 우리 Stage)** 와 **ALM tip 처리 철학**.

---

## 2. 차원별 대조

### 차원 ① Velocity Sampling (속도 샘플링) — **★핵심 차이**

| 항목 | 논문 권장/관행 | 우리 구현 (file:line) |
|------|----------------|----------------------|
| 방식 | **point/trilinear** (Asmuth1·2, Watanabe Eq. 인접 8노드 trilinear; Diaz 8-cell linear) / **ring-average linear** 20점 (Natelson p.4) | **Gaussian-weighted 정규화 평균**, 커널 `exp(-d²/ε²)`, r_cut=3ε, **spreading과 동일 커널** (`interpolation.py:148-163`, `actuator_line.py:317`, n_cut=3.0) |
| self-induced 처리 | 제거 안 함 (전부). Watanabe: 큰 ε→self-induction↓→u_n↑→팁 과대(p.14) | 제거 안 함. ±3ε 평균이 self-induction을 공간평균 |
| 팁 전용 처리 | 없음 (전부) | 없음 (등방 ±3ε) |

**차이·리스크·개선안:**
- **우리만 Gaussian 적분 샘플링.** 이게 팁에서 양날이다. 장점: Martínez-Tossas류 "projection-consistent" 샘플로 lift 정합 의도. **단점(우리 증상의 핵심)**: 팁 마커의 ±3ε 구가 **disk 바깥(induction 거의 0, hover면 외측 재순환/정지류)** 을 평균에 포함 → 팁 downwash 희석 → 샘플 u_n이 freestream/정지류 쪽으로 치우침 → φ→0, α 고정.
- HVAB 정량: 충분 해상도에서 ε_tip=0.25·c_tip(chord/4 우세, base config 주석 L125-138), c_tip≈3.27in≈0.083 m → ε_tip≈0.021 m → **3ε≈0.062 m ≈ 0.037R**가 팁 바깥으로 새어나감. 팁 부근 induced-velocity 스팬 구배가 가파르므로 이 정도 외측 평균도 비물리 결손 유발 가능.
- Watanabe(point)의 팁 과대는 "큰 ε→self-induction 손실"이 원인이고 ε를 줄이면 회복(p.12-16). 우리는 **point가 아니라 평균**이라 동일 ε에서도 팁 오염이 더 클 수 있음.
- **개선안(결정적 A/B)**: 팁(또는 전 스팬) 샘플을 **(i) 순수 point/trilinear**, **(ii) 비등방 커널(radial ε_r 축소)**, **(iii) 내측편향/단측(one-sided, disk 안쪽만) 커널**로 바꿔 C_T·tip-φ 비교. dx·마커·extent·margin이 모두 기각된 상태에서 **샘플링 방식이 마지막 미검증 ALM 고유 변수**이므로 최우선.

---

### 차원 ② ε (Projection / Regularization Width)

| 항목 | 논문 권장 | 우리 구현 |
|------|-----------|-----------|
| 값 | Asmuth1: ε≥2Δx(하한), ε=3Δx 양호, **Δx<ε/4면 C_T·C_P<1%**. Asmuth2: ε=2.5Δx(=0.078D). Watanabe: **ε=2.5Δx**(1Δx까지 안정). Diaz: ADM 2Δx, **ALM 2Δx→tip 1Δx 테이퍼**. Martínez-Tossas 최적 **ε≈0.25c** | **ε=max(c/4, 2·Δx)** (`blade.py:359-362`). tip_taper 모드시 r/R>0.7부터 max(factor·2Δx, 2Δx)로 선형 블렌드 |
| 스팬 변화 | 대부분 상수. Diaz ALM만 팁 테이퍼 | default=상수형(max), tip_taper 옵션 |
| 차원 | 전부 3D 등방 (Natelson만 비등방) | 3D 등방 |

**차이·리스크·개선안:**
- **우리 ε=max(0.25c, 2Δx)는 5편보다 오히려 잘 근거가 있다.** `0.25c`=Martínez-Tossas 최적 ε(이론), `2Δx`=Asmuth/Watanabe 격자 안정 하한. 둘의 max라 두 기준 동시 만족. **이 자체는 결함 아님.**
- 다만 **팁에서 어느 항이 우세한지**가 팁거동을 좌우: HVAB는 충분 해상도면 0.25c_tip 우세(작은 chord) → ε_tip이 작아져 좋음. 해상도 부족이면 2Δx 바닥에 걸림.
- Diaz의 ALM 권장(팁 1Δx로 테이퍼)과 우리 tip_taper는 같은 철학. **단 production task3는 legacy(테이퍼 없음)** 사용 중 → 팁 테이퍼 ON A/B 가치.
- 리스크: ε이 작을수록 팁 회복엔 좋으나 격자 안정/Mach 민감(Asmuth1 p.7: ε↓→body force↑→압축성 오차 O(u³)). HVAB M_tip=0.65라 ε 과소시 LBM 압축성 주의.

---

### 차원 ③ Force Projection / Spreading 커널

| 항목 | 논문 | 우리 구현 |
|------|------|-----------|
| 형태 | 3D 등방 Gaussian η_ε=exp(-(d/ε)²)/(π^{3/2}ε³) (Asmuth2 Eq.18, Watanabe Eq.20, Diaz Eq.12). **Natelson만 비등방 (ε_c,ε_t,ε_r)** Churchfield식 | η_ε=exp(-d²/ε²)/(π^{3/2}ε³) (`spreading.py:27,92,127`) — **동일** |
| 정규화 | 1/(π^{3/2}ε³) (Watanabe·Diaz·Natelson). Asmuth는 1/(π^{3/2}ε²)=힘/길이 [N/m]용 | 1/(π^{3/2}ε³) — Watanabe와 **정확히 일치** (이산 힘 [N] 분배) |
| cutoff | Watanabe **3.5ε**, 나머지 명시 없음 | **3ε** (n_cut=3.0) |
| sampling 커널과 관계 | 전부 **다름**(sampling=point) | **같음**(sampling도 Gaussian) — 우리 고유 |

**차이·리스크·개선안:**
- spreading 자체는 **표준이며 Watanabe와 사실상 동일**(3D, 1/ε³, cutoff만 3ε vs 3.5ε). 결함 없음.
- 유일한 개선여지: **Natelson 비등방 커널(ε_r 별도)**. 팁의 radial 방향 over-smear를 ε_r 축소로 직접 억제 → 팁 결손 완화의 정공법 중 하나. 단 구현비용 큼(현 등방 → 3축).
- cutoff 3ε vs 3.5ε는 무시 가능(exp(-9)≈1e-4 잘림).

---

### 차원 ④ 팁 유입 / 팁손실 처리 — **★판정 핵심**

| 논문 | tip-loss(Prandtl/Glauert) | smearing/viscous-core 보정 | 팁 거동 보고 |
|------|---|---|---|
| Asmuth1 | **없음** | 없음(구현). Martínez-Tossas[47], **Meyer-Forsting[49]** 지목 | 팁 F_t 과대; 3D 커널이 팁 너머로 smear + 인접 projection overlap + **샘플속도가 trailed-vortex induction에 오염**(p.6). "smeared ALM 본질적 + 보정으로 완화" |
| Asmuth2 | **없음** | 없음. p.629 "deviation = force smearing + **correction model 부재**(Meyer-Forsting 2019)" | BEM 대비 팁 편차를 smearing+보정부재로 귀속. LBM 결함 아님(NS-ALM 재현) |
| Diaz | ADM=**Glauert**, **ALM=Glauert 안 씀** | ALM=**variable-ε**(Dağ&Sørensen 2020) | **"tip 보정에도 마지막 10% 스팬 fₙ·fθ 과대 잔존"(p.371)**. D/48+ 라야 팁와류 해상. "부분적 본질" |
| Natelson | 없음 | 없음 | 팁와류는 해상됨(Fig7d-e). 오히려 **wake edge 난류 과대**. **로터를 finest level에 co-locate 안 하면 downwash 소산(p.8)** |
| Watanabe | **없음** | **없음** | 큰 ε→self-induction↓→**팁 과대**. ε 줄이면 회복. "ε/해상도가 지배, tunable" |

| 우리 구현 | tip-loss | smearing-corr | 사용 현황 |
|---|---|---|---|
| `actuator_line.py` | **Prandtl tip/root** `_compute_prandtl_factor` (L364-424), R_tip_eff=R_tip−ε_tip | **Dağ&Sørensen viscous-core** `_viscous_core_correction` (L456-499), w=−(1/4π)Σ Γw/d·K, target inviscid/opt, relax | task3 production=**Prandtl ON + eps_corr OFF**; epscorr config=**eps_corr ON + Prandtl OFF** |

**차이·리스크·개선안:**
- **방법론 경고(Diaz)**: production task3가 **ALM에 Prandtl tip-loss를 적용**한다. Diaz는 명시적으로 "ALM은 팁와류·induction을 유동에서 해상하므로 유한익수 tip-loss factor를 또 곱하면 **이중계산**; ALM엔 variable-ε를 써라"(p.369). → **task3의 Prandtl-on-ALM은 재검토 대상.** (단 우리 해상도에서 팁와류가 충분 해상 안 되면 surrogate로 정당화 여지 있음 — Diaz도 D/48+ 라야 팁와류 해상이라 했고 우리 MLG fine 해상도가 그에 못 미치면 Prandtl이 임시 surrogate로 기능. 이건 "정석"이 아니라 "보충"임을 문서·발표에 명기 권장.)
- **정석 보정 누락**: Asmuth 두 편이 공통 지목한 팁보정은 **Meyer-Forsting et al. 2019 "vortex-based tip/smearing correction"**. 우리는 사촌격 Dağ&Sørensen만 구현했고 메모리상 1/13로 불충분. → **Meyer-Forsting 검토 가치 높음** (특히 샘플속도의 trailed-vortex 오염을 직접 보정하는 접근이라 우리 ±3ε 오염 문제와 직결).
- **수렴된 판정 근거**: 5편 모두 팁편차를 "smearing 본질 + ε/해상도/샘플링이 magnitude 지배"로 본다. **누구도 LBM 결함이나 단순 버그로 보지 않으며, 누구도 ALM에 Prandtl 권장 안 함.**

---

### 차원 ⑤ Wake 해상 / MLG (Refinement)

| 논문 | 해상도 | refinement |
|------|--------|-----------|
| Asmuth1 | Δx=1/64·1/32·1/16 D | factor-2 telescoping nest(Filippova-Hänel) |
| Asmuth2 | D/16·D/24·D/32 | **균일(비교 위해)**, AMR 없음 |
| Diaz | production **D/32** (팁와류는 **D/48+ 라야 발달**) | nested D/8..D/64 zone |
| Natelson | finest Δx=1.7mm, **146 cell/chord=112 cell/R** | **5-level nest, 로터+wing 둘 다 finest에 co-locate**(안 하면 downwash 소산) |
| Watanabe | D/32·D/48·D/64 | **균일**(AMR=future work) |

| 우리 구현 | MLG L3(또는 L4/L5) fine region, 로터를 finest level에 ALM 연결 (`setup.py` _create_fine_level_alm) |
|---|

**차이·리스크·개선안:**
- **Natelson의 교훈이 우리 증상과 직결**: "fine region이 로터를 충분히 안 감싸면 **downwash가 소산**된다." 우리는 margin(radial, ≥0.5D) 가설을 이미 기각했지만, **hover 축방향 수축 wake가 finest level을 벗어나 coarse로 내려가며 downwash 소산되는지**(radial margin이 아닌 **축방향 아래쪽 fine extent**)는 별도 점검 가치. 팁 결손이 "induced inflow 소산"이면 Natelson 메커니즘과 동일.
- **Diaz: 팁와류는 D/48+** 라야 발달. 우리 HVAB fine 해상도를 **cell/R, cell/tip-chord**로 환산해 Diaz/Natelson(112 cell/R)과 비교 권장. base config 주석이 "tip chord ≥16 cell" 목표(5-level)라 한 만큼, 3-level이면 팁와류 미해상 → 팁 induction 미회복 → Prandtl surrogate 불가피.
- 즉 **팁 결손의 일부는 해상도 한계(팁와류 미해상)** 일 수 있고, 이는 5편이 말하는 "본질+해상도 의존" 그대로.

---

### 차원 ⑥ Velocity Triangle / Mach / Sweep — **우리가 앞서는 강점**

| 항목 | 논문 | 우리 구현 |
|------|------|-----------|
| 삼각형 | Watanabe Eq.8-12: u_n,u_θ,u_rel=√(u_n²+(rω−u_θ)²), φ=atan2, α=φ−γ | **동일 구조** (`rotor.py:485-502`), ω부호 분리, **α=twist−φ(프로펠러 관례)** |
| 힘 투영 | turbine: F_n=L cosφ+D sinφ, F_θ=L sinφ−D cosφ | **propeller: F_n=L cosφ−D sinφ, F_θ=L sinφ+D cosφ** (`actuator_line.py:657-658`) |
| Mach | Asmuth·Watanabe: 수치 Mach만(압축성 보정 없음). Natelson: tip M≈0.40, 명시적 보정 없음(테이블에 암묵) | **Mach-pass**: M=V_n·(dx/dt)/a, polar에 mach 인자 전달 (`actuator_line.py:599-606`) |
| Sweep | 전부 **없음**(Diaz가 Nathan 2015 인용만) | **cross-flow cosΛ**: u_aero=u_rel·cosΛ가 q·Re·Mach 구동 (`actuator_line.py:581-585`) |

**차이·리스크·개선안:**
- 삼각형은 Watanabe와 **수식 일치**(프로펠러 부호 관례만 다름, 내부 정합). **결함 없음.**
- **Mach-pass + sweep cosΛ는 5편 전부에 없는 우리 강점.** HVAB M_tip=0.65 천음속 팁엔 필수. Natelson(M_tip≈0.40, 소형)조차 명시 보정 없음 → 우리가 더 적절.
- **부차 점검(경미)**: `_compute_bem_forces` 메서드 docstring(L520-521)은 turbine 관례(F_n=L cosφ**+**D sinφ)를 적어 **실제 코드(L657, −)와 불일치**. 코드 자체는 프로펠러 관례로 내부 정합하나, **docstring 갱신** 권장(혼동 방지).

---

## 3. 핵심 질문 판정: 팁 유입 결손 = 구현 결함 vs ALM 본질 한계?

**판정: "ALM 본질 한계(smearing de-induction + 해상도) 위에, 우리 고유의 Gaussian 속도샘플링이 팁 오염을 증폭한 복합 원인." 순수 LBM 버그도, 단일 코딩 실수도 아니다.**

근거:
1. **본질 부분** — 5편 일치: 유한 ε smeared ALM은 팁에서 trailed-vortex de-induction이 부족해 팁 과대/φ 과대가 구조적으로 발생(Asmuth1·2, Watanabe). Diaz는 variable-ε로도 마지막 10% 스팬 과대 잔존을 인정 = **부분적 본질**. → 우리 팁 결손의 baseline은 정상적인 ALM 거동 범위.
2. **해상도 부분** — Diaz "팁와류 D/48+", Natelson "112 cell/R, finest co-locate 안 하면 downwash 소산". 우리 fine 해상도가 이에 못 미치거나 hover wake가 fine을 벗어나면 팁 induction 미회복. → 메모리의 dx/margin 기각은 **radial** 위주였으므로 **축방향 fine extent / cell-per-R 환산** 재점검 필요.
3. **증폭 부분(우리 고유)** — 5편 중 누구도 Gaussian ±3ε 속도샘플링을 안 쓴다. 우리만 쓰며, 팁에서 off-disk 저-induction 영역을 평균에 섞어 u_n을 과대평가 → φ→0 직접 증폭. **이것이 "dx·마커·extent·margin 모두 기각 후 남은 ALM 고유"의 정체로 가장 유력.**

**따라서 "불가피한가?"** → **아니오, 부분적으로 완화 가능.** 본질·해상도 부분은 정석상 한계가 있으나, **증폭 부분(샘플링)은 우리가 선택한 방식이라 바꿀 수 있다.** Prandtl-on-ALM은 정석이 아니므로(Diaz), 발표/논문에는 "해상도 한계에 대한 surrogate"로 명기하는 것이 정직.

---

## 4. 우선순위 개선안 (실험 계획)

| 우선 | 실험 | 가설 | 비용 | 산출 |
|------|------|------|------|------|
| **P1** | **속도샘플링 A/B**: (i) point/trilinear, (ii) 비등방 ε_r↓, (iii) 내측편향 단측 커널 — 동일 config에서 tip-φ·α·C_T 비교 | ±3ε 평균의 off-disk 오염이 팁 결손 증폭. point면 φ 회복? | 중(샘플러 1종 추가) | 증폭 부분 분리·정량 |
| **P2** | **fine extent / cell-per-R 환산** 점검 + 축방향 아래쪽 fine 확장 A/B | hover downwash가 fine 벗어나 소산(Natelson) | 저(분석)~중(런) | 해상도 부분 분리 |
| **P3** | **Meyer-Forsting 2019 보정** 검토/구현 (Dağ 대체) | trailed-vortex 샘플오염 직접보정이 Dağ보다 우월 | 고 | 정석 팁보정 확보 |
| **P4** | **tip_taper ON** (Diaz ALM식 팁 ε→2Δx) A/B | 팁 ε 축소로 induction 회복 | 저(옵션 존재) | ε 부분 정량 |
| **P5** | **task3 Prandtl-on-ALM 재검토**: surrogate 명기 or variable-ε 전환 | Diaz: ALM엔 Prandtl 이중계산 | 저(문서/config) | 방법론 정합 |

> P1이 결정적. dx·마커·extent·margin이 모두 기각된 지금, **샘플링 방식이 마지막 미검증 ALM 고유 변수**이며 5편 대비 우리만의 차이이므로 가장 먼저 끊어야 한다.

---

## 5. 부록: 논문별 핵심 수치 (대조용 빠른 참조)

- **Asmuth1 (LBM Frameworks)**: ε≥2Δx 하한, ε=3Δx 양호, **Δx<ε/4 → C_T·C_P<1%**. C_T,ref=0.86, C_P,ref=0.55. forcing=cumulant central-moment shift `u+=Δt/2ρ·F`. tip 보정 없음, Meyer-Forsting[49]·MT[47] 지목.
- **Asmuth2 (wakes, WES5,623)**: D3Q27, Smagorinsky C_s=0.08, ε=0.078D={1.25,1.875,2.5}Δx, 64 pt/blade, **균일격자 D/16·D/24·D/32**, Ma=0.1. forcing=Δt/2 force를 1차 cumulant에 가산. "deviation=smearing+correction model 부재".
- **Diaz (WES8,363; SOWFA LES)**: ADM ε=2Δx+**Glauert**, **ALM ε=2Δx→tip1Δx+NO Glauert**(Dağ&Sørensen). root: 𝓕=1−exp(−2.335(x/0.07)⁴). production D/32, 팁와류 D/48+. **"tip보정에도 마지막 10% 과대 잔존."** 3 force법=airfoil(속도삼각형 有)/analytical(無)/numerical(無) = **우리 Stage**.
- **Natelson (proprotor)**: MRT D3Q19, Smagorinsky C_s=0.042, **비등방 Gaussian (ε_c,ε_t,ε_r)** Churchfield, **ring 20점 linear 샘플**, ε 수치 미공개. 5-level nest, **146 cell/chord=112 cell/R, 로터+wing finest co-locate(안 하면 downwash 소산)**. tip M≈0.40(소형). ALM이 C_T 3%·C_Q 5~10% 이내(tip-loss 보정 없이). hover(θ=90°) 제외.
- **Watanabe (TAML real-time)**: Cumulant D3Q27, implicit-LES, FP32, **trilinear 샘플**, **ε=2.5Δx**(1Δx까지 안정), η=exp/(π^{3/2}ε³) cutoff **3.5ε**, 1/ε³(이산 힘[N]). 균일 D/32~D/64. 삼각형 Eq.8-12. CFL=0.02. **"큰 ε→self-induction↓→팁 과대; ε 줄이면 회복"**. tip 보정 전무. 8308 MLUPS/GPU(H100), real-time(<1) for 5/10/15MW@D/32.

---

## 6. 우리 구현 file:line 인덱스 (재현·검증용)

- 속도샘플링: `actuator_line.py:317` (`interpolate_velocity_batch_gpu`, n_cut=3.0=`actuator_line.py:204`); 커널 `interpolation.py:79-80,148-163`
- ε 선택: `blade.py:336-374` (`set_lattice_spacing`, ε=max(c/4,2Δx), tip_taper)
- spreading: `spreading.py:62-141` (η_ε, 1/π^{3/2}ε³, r_cut=3ε)
- 속도삼각형: `rotor.py:429-518` (`compute_relative_velocity`, `recompute_velocity_triangle`)
- BEM/Mach/sweep/보정: `actuator_line.py:505-688` (`_compute_bem_forces`); sweep `:581-585`; Mach `:599-606`; eps_corr `:613-633`
- viscous-core(Dağ): `actuator_line.py:456-499`; Prandtl: `:364-424`
- config 토글: `_eps_corr/_eps_corr_target/_eps_opt_factor/_eps_corr_relax` `:238-241`, `prandtl_loss` `:226`; loader `:1193-1212`
