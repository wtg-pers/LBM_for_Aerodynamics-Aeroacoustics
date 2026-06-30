# HVAB 로터 — 기하/조건 (ALM 셋업용, 출처 명시)

> **모든 수치의 1차 출처**: NASA/TM–2020–5002153, *Hover Validation and Acoustic Baseline
> Blade Set Definition* (NTRS, 미국 정부 저작물=공개).
> PDF: https://ntrs.nasa.gov/api/citations/20205002153/downloads/NASA-TM-2020-5002153.pdf
> 인용: https://ntrs.nasa.gov/citations/20205002153
> (아래 값은 해당 TM의 **Table 1**(planform)·**Table 2**(chord/twist/sweep/airfoil 분포)에서
> pdftotext로 추출. 사용자 cross-check 권장 — 표 페이지 직접 대조.)
> 시험 조건 보조 출처: [AIAA HPW HVAB](https://www.aiaa-hpw.org/hvab-rotor),
> NASA HVAB 데이터 저장소 https://rotorcraft.arc.nasa.gov/HVAB/

## Planform (Table 1)
- 블레이드 수 N_b = **4**
- 반경 R = **66.50 in = 1.6891 m**
- Geometric solidity σ = **0.1033** (root cutout 무시, root chord 5.45in이 중심까지 연장 가정)
- 기준 chord = **5.45 in (0.13843 m)**, 기준 airfoil = RC-series (아래)
- TE 두께(design) 0.0350 in (blunt TE)

## Chord / Twist / Airfoil 분포 (Table 2, as-designed)

| r/R | twist [deg] | chord [in] | airfoil |
|---|---|---|---|
| 0.13985 (root, r=9.30in) | 8.20 | 5.45 | RC(4)-12 |
| 0.16722 | 8.20 | 5.45 | RC(4)-12 |
| 0.25188 (fairing 끝/공력 시작) | 7.01 | 5.45 | RC(4)-12 |
| 0.65128 | 1.40 | 5.45 | RC(4)-12 |
| 0.70120 | 0.70 | 5.45 | RC(4)-10 |
| 0.80075 | −0.70 | 5.45 | RC(4)-10 |
| 0.85053 | −1.40 | 5.45 | RC(6)-08 |
| 0.95023 (sweep break) | −2.80 | 5.45 | RC(6)-08 |
| 1.00000 (tip) | −3.50 | 3.27 | RC(6)-08T |

**해석 (핵심):**
- **Twist**: r/R=0.25188(7.01°)부터 팁(−3.50°)까지 **선형 ≈ −14.05°/R** (메모리의 "−14° linear twist" 확인). root 내측(0.14~0.167)은 fairing이라 ~8.2°로 유지.
  - 검증: 7.01 − 14.05·(r/R − 0.25188) → 0.651→1.40, 0.801→−0.70, 0.95→−2.80, 1.0→−3.50 모두 일치.
- **Chord**: **r/R≈0.95까지 5.45in 등현(constant)**, 그 후 팁까지 **3.27in로 테이퍼** → 즉 **"전체 테이퍼"가 아니라 외측 ~5%만 tapered tip**. (단 팁이 물리적으로 가장 중요.)
- **Airfoil**: RC(4)-12(root~0.65) → RC(4)-10(0.70~0.80) → RC(6)-08(0.85~0.95) → RC(6)-08T(팁). 3종 RC + 팁 tab. (TM에 blend 구간 명시; ALM은 마커별 최근접 순수 단면 배정 가능.)
- **Tip sweep**: LE 따라 r/R=0.95023에서 sweep break (HPW 페이지: 30°@95%R). 0.5in radius.
- **Root cutout**: 구조 root 9.30in(r/R=0.140); 공력/fairing 끝 r/R=0.25188 → **ALM 공력 활성 스팬 ~0.25~1.0** (root fairing은 무양력).

## 시험 조건 (hover)
출처: AIAA HPW HVAB 페이지 + NASA General Info Readme
- 팁 Mach: **0.60 / 0.65 / 0.675** (주력 0.65)
- RPM: **1250.39** (M0.65, SLS); 변형데이터용 1160/1250/1310
- Collective(HPW 지정): **6° / 8° / 10° / 12.76°** (시험은 4~15° 1° 간격)
- 조건: standard day sea level (a≈340.3 m/s, ν≈1.46e-5). Re_tip ≈ 1–2 M (ref chord 기준 ~2.0e6, tip chord 기준 ~1.2e6) → SGS 필요.

## ALM 셋업 함의
- **테이퍼+sweep tip**: 등현 Re→M 트릭이 팁(외측 5%)에서 깨짐 → **ALM Mach-pass 필요**(per-element 국소 Mach). 30° sweep은 ALM 마커선으로 근사(직선) — 한계로 명시.
- **3종 RC 익형**: multi-airfoil + Mach C81 덱 필요 (`c81_from_neuralfoil.py`로 RC(4)-12/RC(4)-10/RC(6)-08[T] Mach 덱 생성). RC 좌표 출처 확인 필요.
- **grid**: tip chord 3.27in(=0.0831m)이 가장 작음 → "tip chord 기준 ≥16 cell"(ε=0.25c & ε≥4Δ) 만족하도록 fine region 사이징. (등현 CT보다 tip 해상이 더 빡빡.)

## RC 익형 polar 덱 (생성 완료, `data/airfoils/RC*.C81`, Mach 0–0.8)

`c81_from_neuralfoil.py`(NeuralFoil+Viterna)로 생성. provenance/출처:

| 익형 | 덱 | provenance | 출처 / cross-check |
|---|---|---|---|
| **RC(4)-10** | `RC4-10.C81` | **authentic** | aerosandbox `rc410.dat` (= NASA-TP-3009, Noonan 1990) |
| **RC(6)-08** | `RC6-08.C81` | **authentic** | **NASA-TM-4264 Table I** (Noonan 1991), NTRS https://ntrs.nasa.gov/citations/19930020261 — 좌표→.dat 변환, **cross-check 통과**(두께 8.0%·camber +0.69% = TM 본문 "reflexive camber 0.7%c" 일치) |
| **RC(4)-12** | `RC4-12.C81` | **PLACEHOLDER** | 보고서 없음(TM 명시) → RC(4)-10(rc410)을 12% 두께로 스케일(같은 series-4 family 근사). **production은 FileShare plot3d로 교체 권장** |
| **RC(6)-08T** | `RC6-08T.C81` | **근사** | RC(6)-08 재사용 (tip tab=TE 0.035in 미세수정 무시). production은 FileShare plot3d as-built 권장 |

좌표 .dat도 `data/airfoils/RC*.dat`에 저장(RC6-08은 NASA-TM-4264 설계좌표, RC4-12는 스케일).

## TBD 해소 (2026-06-26, TM Table 1/2 원문 직접 대조)
TM PDF(pdftotext) Table 2 = `r/R | Twist | c | Sweep | ΔX | ΔY | ΔZ | Airfoil`:
- **Anhedral/droop: 없음** — 전 station **ΔZ = 0** (블레이드는 면내 평면). → 우리 미모델 정확.
- **Precone: 블레이드 기하 입력 아님** — ΔZ=0(평면), flap/lead-lag 힌지만 허브중심 3.5in.
  precone은 허브/시험 셋업값으로 Table 1/2에 없음 → 우리 `PRECONE_DEG=0` 타당(강체 ALM, coning 미포착=소).
- **Tip sweep = 30°**, **팁 station(1.0)에만** 표기(0.95023 break부터), 팁 0.25c **ΔY=−1.911in**(후퇴).
  tan⁻¹(1.911/((1−0.95023)·66.5))=tan⁻¹(1.911/3.31)=**30° 확인**. → **외측 ~5%만 후퇴**. (우리 미모델=한계)
- **Collective 정의 = r/R 0.75 피치**, 블레이드 θc=0 제공 → 우리 `TWIST_REF=0.75` 정확.
- **익형 station**(blend gap): RC4-12 ≤0.65128 / RC4-10 0.70120–0.80075 / RC6-08 0.85053–0.95023 / RC6-08T 1.0.
  우리 경계(0.675/0.825/0.975)는 모두 blend gap 내 → 타당. **RC4-12는 NASA도 "보고서 없음" 명시**(placeholder 불가피).
  RC6-08T = "reduced chord로 nondim TE 두꺼워진" 팁 익형(우리 RC6-08 재사용=근사).
- **★ authentic 좌표 입수 경로**: RC 익형 좌표가 **TM Table 2에 임베드**(scaled×1000), **OML IGES**(`HVAB Blade OML AsDesigned.igs`)도 TM 첨부 → FileShare 외에 **TM에서 직접 RC(4)-12/RC(6)-08T 좌표 추출 가능**.
- 검증 데이터: `rotorcraft.arc.nasa.gov/HVAB/` (Pressure_Airloads_Data + README, 2023-10 release).

출처: NASA/TM-2020-5002153 Table 1/2 (PDF 직접), AIAA HPW HVAB.
