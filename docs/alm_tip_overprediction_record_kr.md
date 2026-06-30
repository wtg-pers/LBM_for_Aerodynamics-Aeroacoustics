# ALM Hover 추력 과대예측 — 조사 기록 & Dağ 보정 설계

> Caradonna-Tung hover (θ=8°, M_tip=0.877, D32 light, exp C_T=0.00473)에서 ALM+LBM이
> 추력을 과대예측하는 문제의 전체 조사 기록과, 다음 단계인 Dağ viscous-core de-induction
> 설계. 작성 2026-06-23. 관련: `docs/alm_epsilon_theory_kr.md`, `patch_notes/eps_taper_stageB/`.

---

## 1. 문제 정의

ALM+LBM hover 솔버가 로터 추력을 **과대예측**한다. 진단(이전 세션): 코드 버그·격자 dx·SGS·
미수렴 아님 → **smeared-ALM 팁 유도 다운워시 결손**. 유한 ε Gaussian 힘 투영이 진짜
lifting-line 유도속도를 재현 못함 → 팁이 거의 기하 받음각을 봄(φ→0) → 팁 과하중 → C_T 과대.

문헌 정렬(`to_claude/ref_papers/` 조사): 우리 ε=chord/4≈5.3셀=Martínez-Tossas 이론최적
(0.2~0.25c) 부합. tip-loss는 고전 ALM에서 보편적이나, 현대 ALM-LES(MT2017/JFM2019)는
compact kernel이면 불필요하다 주장하고, 보정 쓰는 쪽(Diaz2023 ε-taper+Glauert, Dağ2020
viscous-core)은 물리 기반. 핵심 미보유 레퍼런스: **Asmuth 2020 (3D HAWT ALM-LBM, WES 5:623)**.

---

## 2. 시도 1 — Stage B: 가변 ε 팁 테이퍼 (Diaz 2023 §2.1.4)

**아이디어**: 팁 쪽 ε을 floor(2Δx)까지 좁혀 팁 와류 과확산을 줄임. config `epsilon_mode=tip_taper`.
구현: `blade.py` set_lattice_spacing 분기 + 디커플 속성, 커널 무수정(per-marker ε 이미 지원).
기본 off=bit-identical. (`patch_notes/eps_taper_stageB/01-06`)

**결과**: 테이퍼 메커니즘은 **검증됨** — Prandtl OFF에서 ΔC_T=−0.00042(−8.8%p), 팁에 분포적
(팁>0.9 ΣF_n −0.113), 팁 φ 회복(0.99: 1.89→4.10°), artifact 없음. **그러나 단독으론 부족**
(−5~9%p로 +35~48% 못 닫음).

**부산물 발견**: Prandtl ON에서 테이퍼가 오히려 추력↑(+4.2%p) → 조사해보니 Prandtl 커플링
artifact (아래).

---

## 3. 시도 2 — Prandtl 구현 감사

**동기**: tip-loss는 ALM에서 보편적인데 우리만 +17% 남음 → Prandtl 구현 의심.

**결과**: 공식은 **표준 정확**(ALM `_compute_prandtl_factor` = BEMT `bemt_hover.py` = 교과서
`F=(2/π)arccos(exp(−f))`, `f=B(R_tip−r)/(2r·sin|φ|)`). **유일 비표준 = `R_tip_eff = R_tip − ε_tip`**
(표준/BEMT는 R). 영향(R≈128lu, B=2):

| r/R | 표준 R | ALM R−5.33(baseline ε) | ALM R−2.11(taper) |
|---|---|---|---|
| 0.95 | 0.71 | 0.32 | 0.60 |
| 0.97 | 0.63 | **0.00** | 0.44 |

`f=B·max(R_tip_eff−r,0)/…`라 **r>R_tip_eff 외측 ε밴드를 hard-zero**(ε=5.33→외측 4.2% 완전제거).
→ Prandtl을 과격하게 만들고 ε에 커플링(=테이퍼 역효과 근원: 테이퍼가 hard-zero 밴드를
4.2%→1.6%로 줄여 마커 un-zero → 추력↑).

**재평가**: R−ε는 순수 hack 아님 — Gaussian이 팁 너머로 force 투영(Churchfield: "blade appears
more efficient")하는 over-projection을 상쇄하는 방어가능 보정. 단 ε커플링+hard-zero가 문제.

---

## 4. 시도 3 — Prandtl 표준화 (`eps_offset` flag) + 디커플

**변경**: `prandtl_loss` dict에 `eps_offset` 키. 기본 True=레거시 R−ε(재현성), False=표준 R.
(`patch_notes/eps_taper_stageB/07`, 스모크 회귀 0.080959 bit-identical 통과)

**결과**: 디커플 **완벽 작동** — 표준 Prandtl 위 테이퍼가 깨끗하게 추력↓(ΔC_T −5.3%p,
F_n==0 마커 0개, 0.97 un-zero 스파이크 소멸). 하지만 표준화로 **baseline +16.9%→+35.1%**
폭로 → 레거시 R−ε가 몰래 −18%p 보정하던 강한 crutch였음.

---

## 5. 결과 비교 매트릭스 (CT M0.877 D32 light, exp 0.00473)

| Prandtl | taper | C_T | vs exp | 비고 |
|---|---|---|---|---|
| legacy R−ε | ✗ | 0.00553 | **+16.9%** | 기존 "최선" (tuned crutch) |
| legacy R−ε | ✓ | 0.00573 | +21.1% | un-zero artifact |
| 표준 R | ✗ | 0.00639 | **+35.1%** | 정직한 baseline |
| 표준 R | ✓ | 0.00614 | **+29.8%** | 깨끗한 taper −5.3%p ✓ |
| OFF | ✗ | 0.00700 | +47.9% | raw |
| OFF | ✓ | 0.00658 | +39.1% | 깨끗한 taper −8.8%p |

**해석**: 표준 물리 raw 과대예측 +35~48%. 테이퍼 −5~9%p(검증·일관). 레거시 R−ε −18%p(강하나
crude/coupled). 격차 ~18%p = "팁 보정 강도" 문제. **딜레마**: best 일치=legacy(+17%, tuned) vs
best 정직=std+taper(+30%, clean).

---

## 6. 진행 중 — extended (wake extent 진단, DGX)

`ct_hover_t08_m088_extended[_noprandtl].py`: dx 동일(D32), finest 박스만 팁와류 따라 확장
(하류 0.25D→1.0D). **판정**: C_T가 내려가고 팁 φ 회복 → wake extent 한계(격자로 해결);
거의 안 변함 → ALM 팁 모델 본질(→ §7 Dağ). ~80M/33GB, DGX 대기.

**왜 중요한가**: 우리 ε=0.25c≈ε_opt라 §7 sectional 보정은 작을 수 있음. 그러면 과대예측의
주범은 **wake 수치확산/extent**(팁와류가 0.25c로 태어나도 거친 격자에서 하류로 퍼지거나 박스
이탈 → 유도 결손). extended가 이걸 판별.

---

## 7. 설계 — Dağ & Sørensen (2020) viscous-core de-induction

### 7.1 물리 원리

Gaussian ε로 smear된 (trailed) 와류는 **viscous core = ε인 Lamb-Oseen 와류**처럼 유도한다:

```
 w_i(r) = Γ/(4πr) · [1 − exp(−(r/ε)²)]        (Dağ Eq.15)  ← ALM이 실제 유도하는 양
          └─inviscid─┘   └──viscous core 결손──┘
```

이상적(point) 와류 `Γ/(4πr)` 대비 **누락된 유도 = viscous part**:

```
 w_corr(r) = Γ/(4πr) · exp(−(r/ε)²)            (Dağ Eq.16)  ← 되더할 보정
```

bound 와류는 받음각에 기여 안 함(점이 중심) → 보정은 **trailing(wake) 와류**에만 적용,
Biot-Savart로 받음각 보정.

### 7.2 이산식 (near-wake 직선 trailed sheet — 우리가 쓸 형태)

스팬을 따라 actuator 점 i에서:

```
 Γ(i)    = ½ · c(i) · CL(i) · u_rel(i)                       (Eq.19, Kutta-Joukowsky)
 Γw(j)   = Γ(j) − Γ(j−1)   (팁: 가장 끝 bound Γ)             (Eq.18, trailed 와류 강도)
 d(i,j)  = r(i) − y_w(j)    (y_w = 스팬 station 중간점)        (Eq.20)
 w_corr(i) = Σ_j  Γw(j)/(4π·d(i,j)) · exp(−(d(i,j)/ε)²)      (Eq.17)
 α(i)    = α_g − sin⁻¹[ (−u_z(i) + w_corr(i)) / u_rel(i) ]    (Eq.21)
```

**핵심 통찰 (hover 적합성 해결):** `exp(−(d/ε)²)`는 |d|≳3ε에서 ~0. 따라서 w_corr는 **국소
(점 i 근방 스팬 station)에서만** 나온다 → 먼 나선 wake·BVI·다중 회전은 기여 ≈0. 그래서:
- Dağ의 **무거운 3D 나선 wake 적분(Eq.22-23)은 불필요** → 값싼 1D 스팬 합(Eq.17)으로 충분.
- 이전 우려("DS 2-rev 절단이 hover서 부정확")는 **이 국소 형태에선 무의미**(먼 wake가 애초에
  기여 안 함). near-wake 직선 sheet 가정이면 됨.

### 7.3 우리 코드 적용 지점

`_compute_bem_forces` (`actuator_line.py` ~442-554), per blade, 속도삼각형
(`rotor.py::compute_relative_velocity` 485-502) 직후:

1. 1차 polar lookup: 샘플 속도 → `u_n, u_theta, u_rel, φ, α` → `CL(α)`.
2. `Γ(i) = ½ c_i CL_i u_rel_i` (스팬 배열).
3. `Γw(j) = Γ(j)−Γ(j−1)`; 팁/루트 = 끝 bound Γ.
4. `w_corr(i) = Σ_j Γw(j)/(4π d_ij)·exp(−(d_ij/ε_i)²)` — 스팬 NxN 합(N~40, 값쌈).
   특이점: 마커는 cell-center r(i), trailed는 edge y_w → |d|≥dr/2>0 유한. i=j(bound) 제외.
5. **보정을 속도로 적용**(α-only 패치 아님): `u_n_corr = u_n + w_corr` (다운워시 증가 방향).
   → `φ,α,u_rel` 재계산 → polar 재lookup → 힘. (rotor.py 속도삼각형 헬퍼 재사용 권장.)
6. **수렴**: 스텝 내 inner 고정점 루프 대신 **스텝간 under-relaxation**(JFM f_u≈0.1):
   `w_corr ← (1−f)·w_corr_prev + f·w_corr_new`. 진동 위험 회피.

진단: pre/post α, w_corr를 blade_diagnostics에 추가(eps_lu 옆).

### 7.4 보정 목표 core: inviscid(Dağ) vs ε_opt(JFM) — **중요 파라미터**

- **Dağ**: inviscid까지 보정(`exp(−(d/ε)²)`). trailed wake 와류는 실제 core가 작으니 타당.
- **JFM-2019**: ε_opt=0.25c까지만 보정(`exp(−(d/ε)²) − exp(−(d/ε_opt)²)`). 실제 와류는 유한
  core라 inviscid 보정은 **과보정** 위험.
- **우리 상황**: ε_LES=chord/4≈ε_opt → JFM식이면 보정≈0(이미 최적). 그런데도 +35% 과대 →
  **과대예측이 bound-ε sectional 결손이 아니라 wake-확산일 가능성** 시사. → §6 extended가 먼저
  판별해야 하는 이유. 구현 시 `eps_target`(inviscid | 0.25c | effective) 파라미터로 둘 것.

### 7.5 Prandtl과의 관계

Dağ는 Prandtl을 **대체**(명시 거부). eps_correction ON이면 **Prandtl OFF 권장**(둘 다 같은 팁
결손 겨냥, 이중계산). 단 우리 R−ε crutch가 +17%를 내던 걸 보면, 제대로 된 ε-인지 보정(Dağ)이
효과 있을 여지. config는 독립 플래그로 둬서 A/B/matrix 가능하게.

### 7.6 구현 범위 (제안 config 키)

```python
"eps_correction": False,
# dict: {"enabled": True, "eps_target": "inviscid"|"opt", "eps_opt_factor": 0.25, "relax": 0.1}
```
편집: `actuator_line.py`(__init__ 플래그, `_compute_bem_forces` 보정 패스, 새 메서드
`_viscous_core_correction`, BEMResult에 α_uncorrected·w_corr), `rotor.py`(속도삼각형 override
헬퍼 리팩터), 진단 3곳. 커널 무수정.

---

## 8. 가설 elimination 계층 (싼 수치원인부터 제거 → 남으면 모델보정)

팁 과하중 = 팁 유도속도 결손. 기여 인자를 비용 순으로 제거:

| # | 가설 | 상태 | 테스트 | 비용 |
|---|---|---|---|---|
| (a) | dx (fine region 내 격자 곱기) | **이미 제거** (D32→D36 loading invariant) | 완료 | — |
| (1) | **마커수 n_radial 부족** (cell-center 최외측=0.99R → 팁스트립 이산화 과하중 + 스팬 순환구배 해상) | 미확인 | `prtipR_n60/n80` vs prtipR(0.00639) | **쌈**(격자 동일, ALM만 미미↑) |
| (2) | **wake refine 영역(extent) 부족** (팁와류 박스이탈/확산→유도결손) | 진행중 | `extended[_noprandtl]` (DGX) | 큼(80M/33GB) |
| (3) | **유도속도 보정 필요**(smeared-ALM 모델한계) | (1)(2) 후 잔차일 때만 | Dağ §7 구현 | 중(구현) |

**순서**: (1) 마커수 먼저(제일 쌈, 같은 격자) → (2) extent(DGX 진행중) → 둘 다 해결 후
잔차 남으면 (3) Dağ. (1)(2)는 같은 메커니즘(유도결손)의 **수치 기여**, (3)은 **모델 잔차**.

**caveat**: 우리 ε≈ε_opt(0.25c)라 (3)의 sectional 보정은 JFM 논리상 작을 수 있음. 그러면
(1)(2) 후 남는 잔차의 정체는 **wake 수치확산**(=넓게 보면 (2)의 일종, 가용 메모리로 한계)
→ 그땐 effective-core Dağ 보정 또는 calibration. 즉 **(1)+(2)가 핵심 lever**.

```
(1) 마커수 수렴?
 ├─ C_T 더 내려가 수렴   → 마커수 기여 (해결)
 └─ 거의 불변            → 마커수 아님
       ↓
(2) extended (extent)?
 ├─ C_T↓ & 팁φ 회복      → wake extent 한계 (격자로 해결, HART2는 큰 fine-region or calibration)
 └─ 거의 불변            → (3) ALM 모델/wake확산 → Dağ §7 (Prandtl OFF) or calibration
                            + Asmuth2020 입수해 3D LBM-ALM 정합 기준 확보 권장
```

### 테스트 config (준비됨)
- 마커수: `ct_hover_t08_m088_prtipR_n60.py`, `..._n80.py` (표준 Prandtl, no-taper, n=60/80)
- extent: `ct_hover_t08_m088_extended[_noprandtl].py`
- 비교: `compare_taper_ab.py --A result_..._prtipR --B result_..._prtipR_n80 --la n40 --lb n80`

**잠정 권고**: extended 결과 확인 → 그 후 Dağ 구현 여부/형태 확정. HART2 마감(6/30) 급하면
legacy R−ε(+17%)를 "ε-aware tip-projection 보정"으로 문서화해 잠정 제출, 물리 개선은 후속.

---

## 9. 구현 완료 + 검증 (2026-06-25)

§7 설계를 구현. **구현 중 단위 테스트가 §7.2 edge-기반 이산식의 팁 끝단 결함을 잡아내
JFM-2019 control-point 형식으로 변경**한 것이 핵심 기록.

### 9.1 설계 → 구현 변경 (중요)
§7.2의 **edge-기반 + 특이 커널** `Γw(j)/(4π·d)·exp(−(d/ε)²)` (trailed 와류를 panel edge에
배치, exp/d는 d→0에서 특이)을 처음 구현했으나, 합성 하중 단위 테스트에서 **팁 마커가 부호
반대(w_corr<0 = upwash, 팁 α↑)** + 내부 비정상 평탄이 나옴. 원인: 팁 근처 인접 edge들의
ΔΓ 불균형 + exp/d 특이성 → 이산 끝단 artifact (정확히 우리가 고치려는 팁 영역).

→ **JFM-2019 control-point 형식**으로 교체:
```
 gradΓ(j) = dΓ/dr  (np.gradient, 끝단 one-sided → 팁 구배 포착)
 K(d) = exp(−(d/ε)²)                                  target="inviscid" (i=j 자기항 제외)
      = exp(−(d/ε)²) − exp(−(d/ε_opt)²)               target="opt"  (d→0서 자연히 0, 정규)
 w_corr(i) = −1/(4π) · Σ_j gradΓ(j)·K(d_ij)/d_ij · dr   (d=r_i−r_j, control→control)
```
control point에서 평가, i=j 항은 자기유도=0(Dağ). 정규 커널이라 끝단 특이점 없음.
단위 테스트 결과(합성 elliptic 하중): **팁 w_corr=+0.00101 (양수·스팬 최대) = 다운워시
회복=팁 α↓**, 물리적으로 올바른 방향. (opt는 ε=chord/4=ε_opt 우연 일치 케이스라 합성서 ≈0;
실제론 팁 ε가 floor 제한이라 ε<ε_opt면 nonzero.)

### 9.2 단일 패스 (스텝간 closure)
§7.6 `relax` under-relaxation 대신 **스텝당 1패스**(inner 루프 없음, LBM 타임스텝 루프가
closure 제공). `relax` 파라미터는 보존(기본 1.0). 1차 lookup→Γ→w_corr→u_n+=w_corr→속도삼각형
재계산(`rotor.recompute_velocity_triangle`)→Re/Mach 재계산→polar 재lookup.

### 9.3 코드 변경
- `actuator_line.py`: `BEMResult`(+`w_corr`,`alpha_uncorrected`); `__init__`(+`_eps_corr`,
  `_eps_corr_target`,`_eps_opt_factor`,`_eps_corr_relax`); `_lookup_cl_cd`(polar lookup 추출,
  multi+Mach 동일); `_viscous_core_correction`(신규, 위 식); `_compute_bem_forces`(보정 패스);
  factory `eps_correction` 파싱(bool-or-dict). 커널 무수정.
- `rotor.py`: `recompute_velocity_triangle(blade_idx,u_n,u_theta)` 신규(속도삼각형 미러).

### 9.4 검증 (smoke, CPU)
| 테스트 | 결과 | 판정 |
|---|---|---|
| compile (actuator_line, rotor) | OK | ✓ |
| 단위 테스트 (합성 하중 부호) | 팁 +0.00101 (양수·최대), 경고 없음 | ✓ 올바른 방향 |
| CT smoke OFF (회귀) | T_lu=**0.080959** (bit-identical) | ✓ inert |
| CT smoke ON (inviscid, prandtl 동일) | T_lu=0.080701 (**−0.32%**) | ✓ 추력↓ 올바른 방향 |
| HVAB smoke ON (multi+Mach+corr) | 4 RC polar 로드 + T_lu=0.119934 유한 | ✓ 전체 경로 |

coarse D16(ε floor)이라 효과 작음(−0.32%); 해상된 HVAB light/fine에선 클 것 → 사용자 클러스터 A/B로 정량화.

### 9.5 config 키 (최종)
```python
"eps_correction": {"enabled": True, "target": "inviscid", "eps_opt_factor": 0.25, "relax": 1.0}
# bool 도 가능: True == {"enabled":True} (나머지 기본). 기본 미지정=off=bit-identical.
```

### 9.6 클러스터 A/B (준비됨)
- ON: `configs/hvab/hvab_hover_c{06,08,10,1276}_epscorr.py` (inviscid + **prandtl_loss=False**, §7.5)
- baseline: 기존 `hvab_hover_c{...}.py` (prandtl ON, 보정 OFF) — 사용자 보유 결과
- smoke gate: `ct_hover_smoke_epscorr.py`, `hvab_hover_smoke_epscorr.py`
- 해석: ON vs baseline → "물리 viscous-core 보정이 경험적 Prandtl보다 CT/CP 팁 과예측을
  더 잘 완화하고 FM peak를 회복하는가". 후처리는 §부록 hover_fm_post + spanwise_post.

---

## 부록 — 재현/비교 도구

- **통합 비교**(모든 케이스 한 표/플롯): `python src/utilities/consolidate_ct_results.py`
  (기본 glob `result_ct_t08.0_M877_*` 자동탐색 → C_T 표 + `ct_consolidated.csv` +
  C_T 막대그래프 + spanwise F_n/phi 오버레이. 폴더명에서 prandtl/taper/n_radial/extent
  자동 파싱. 미실행 폴더는 "데이터 없음"으로 표시 → 런 완료될수록 채워짐.)
- 2개 비교: `python src/utilities/compare_taper_ab.py --A <folder> --B <folder> --mtip 0.877 --la .. --lb ..`
  (C_T + ε확인 + 팁표 + spanwise 오버레이 PNG)
- 결과 폴더 규약: `result_ct_t08.0_M877_mlg4_D32_<preset>[_taper][_prtipR][_noprandtl][_extended]`
- 임시 cleanup: `rm -f configs/caradonna_tung/_tmp_*.py && rm -rf result_ct_smoke_*`
