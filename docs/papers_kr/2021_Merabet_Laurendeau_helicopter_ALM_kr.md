# Actuator Line Method for Helicopter Rotors Computations in Various Flight Conditions

- **저자**: Reda Merabet, Eric Laurendeau (Polytechnique Montréal, Montréal, Québec, Canada)
- **발표처**: CASI AERO 2021 — Canadian Aeronautics and Space Institute (2021년 6월 14–18일, 온라인 개최)
- **연도**: 2021
- **분류**: 헬리콥터 로터 공력 / Actuator Line Method (ALM) / blade-resolved U-RANS 검증

## 4줄 한국어 요약

1. 풍력터빈용으로 개발된 ALM(Actuator Line Method)을 헬리콥터 로터에 적용하여, hover·축류비행(climb/descent)·지면효과/장애물이 있는 협소공간이라는 세 가지 비행조건에서 fully blade-resolved(B-R) U-RANS 및 실험과 비교 검증한다.
2. ALM은 Blade Element Theory(BET) + 2D 익형 polar로 절점 양력/항력을 구하고, 이를 ε≈0.25c 크기의 Gaussian kernel로 유동장에 분포시켜 블레이드 메시 없이 로터 효과를 재현한다. 속도 샘플링은 integral(g-가중) sampling을 쓴다.
3. **별도의 tip-loss correction(Prandtl 등)이나 smearing correction을 적용하지 않고도** hover thrust를 실험·B-R 수준으로 잘 예측한다. 그 비결은 (a) ε≈0.25c로 Gaussian 폭을 익형 chord에 맞춘 점, (b) 점-속도가 아닌 적분(g-가중) 속도 샘플링으로 자기유도(self-induction)를 자연스럽게 반영한 점, (c) B-R과 동일한 수준(10–20M cells)의 배경격자로 tip vortex를 해상한 점이다.
4. ALM은 thrust·torque·FoM·tip vortex 위치·지면효과 추력증가·기체 표면압력까지 B-R과 거의 동등하게 재현하되, 수직 BVI(blade-vortex interaction)에서 블레이드 끝단 하중을 과대/과소 예측하여 torque를 약간 높이는 단점이 있다. 메시 절감비(1.8–2.8배)에 비례하는, 실제로는 더 큰(2.0–3.1배) 계산속도 향상과 우수한 병렬 확장성을 얻는다.

---

> **정리본 작성 방침**: 본 문서는 원문을 그대로 옮긴 것이 아니라 한국어로 재서술(paraphrase)한 기술 정리본입니다. 식·수치·결과는 원문에 충실하게 복원했습니다. 핵심 용어는 처음 등장 시 영어를 병기합니다.

---

## 기호 정리 (Nomenclature)

원문 Nomenclature를 정리하면 다음과 같다.

| 기호 | 정의 |
|---|---|
| $C_d$ | 블레이드 단면(2D) 항력계수 (drag coefficient) |
| $C_l$ | 블레이드 단면(2D) 양력계수 (lift coefficient) |
| $C_P$ | 평균 압력계수 (average pressure coefficient), $C_P = \dfrac{p - p_\infty}{0.5\,\rho\, V_\infty^2}$ |
| $C_Q$ | 로터 토크계수 (torque coefficient), $C_Q = \dfrac{Q}{\rho\, V_{tip}^2\, A_{disk}\, R}$ |
| $C_T$ | 로터 추력계수 (thrust coefficient), $C_T = \dfrac{T}{\rho\, V_{tip}^2\, A_{disk}}$ |
| $c$ | 국부 코드 길이 (local chord), m |
| FoM | Figure of Merit, $\mathrm{FoM} = \dfrac{C_T^{1.5}}{\sqrt{2}\,C_Q}$ |
| $M$ | 마하수 (Mach number) |
| $r$ | 스팬 방향 반경 좌표 (radial coordinate) |
| Re | 레이놀즈수 (Reynolds number) |
| $V_c$ | 상승속도 (climb velocity), m/s |
| $V_{ind}^{hover}$ | hover 유도속도 (hover induced velocity), $V_{ind}^{hover} = V_{tip}\sqrt{C_T/2}$ |
| $V_{tip}$ | 블레이드 끝단 속도 (blade tip velocity), m/s |
| $z$ | 로터 축방향 좌표 |
| $\sigma$ | 로터 솔리디티 (solidity), $\sigma = \dfrac{N_b\, c}{\pi\, R_{ref}}$ |
| $\Theta_{75}$ | 75% 스팬 위치의 collective pitch (집합피치), degree |

> **메모(연구자용)**: $C_T,\,C_Q$ 모두 $\rho V_{tip}^2 A_{disk}$로 무차원화한 형태이며($C_Q$는 추가로 $R$로 나눔), 결과 그래프들은 대부분 $C_T/\sigma,\ C_Q/\sigma$ (블레이드 하중계수, blade loading coefficient)로 표시된다. FoM 분모의 $\sqrt2$는 위 $V_{ind}^{hover}=V_{tip}\sqrt{C_T/2}$의 $\sqrt{1/2}$에서 온 것으로, momentum-theory 정의와 일관된다.

---

## I. 서론 (Introduction) — 연구 목적 / 배경

헬리콥터는 수직비행 능력 덕에 접근이 어려운 지형이나 선박 착함 플랫폼처럼 까다로운 유동장 환경에서 운용된다. 이때 로터가 지면·벽·장애물 근처(그리고 바람이 부는 상황)에서 작동하는 경우가 많아, 이러한 조건의 수치해석 수요가 크다. 저자들은 가용한 해석기법을 정확도·계산비용 스펙트럼으로 다음과 같이 분류한다.

- **저비용 모델**: BEMT(Blade Element Momentum Theory)나 특이점 기반 방법(vortex-lattice, free-wake 등). hover 상태의 단순 고립 로터에는 잘 맞지만, 복잡한 사례에서는 경험적 보정(calibration)이나 정교한 모델링에 의존해야 하는 제약이 있다.
- **고충실도(high-fidelity) 방법**: U-RANS(Unsteady RANS) 및 기타 메시 기반 CFD. 지배방정식이 더 완전하여 물리를 잘 포착하지만, 로터 형상과 경계층을 격자로 정확히 표현하려면 셀 수가 매우 많아져 계산시간이 수 자릿수(orders of magnitude) 더 든다.
- **로터 대체 기법(rotor replacement techniques)**: 로터 형상을 명시적으로 메시화하지 않고, 그 효과를 운동량 소스항(momentum source term)으로 대체하면서 배경격자는 유지하는 방법군. 로터-후류 간섭(rotor-wake interaction) 연구처럼 "정밀한 로터 하중"보다 "주변(기체·지면·장애물)에 미치는 후류 효과"가 더 중요한 경우에 특히 적합하다.

로터 대체 기법의 복잡도 단계는 다음과 같이 정리된다.

1. **Actuator Disk (AD)**: 로터 하중을 디스크 위에서 방위각 평균(azimuthal average)한 뒤 압력 불연속 또는 운동량 소스항으로 투영. 비정상(unsteady) 버전도 존재.
2. **Actuator Line Method (ALM)**: 개별 블레이드를 회전하는 운동량 소스항의 "선(line)"으로 취급. 보통 BET로 블레이드 하중을 평가하며, 결과 유동장은 헬리컬 tip vortex 후류 같은 핵심 로터 후류 요소를 재현한다. **본 논문이 채택하는 방법.**
3. **Actuator Surface Method (ASM) / Actuator Blade (AB)**: 같은 개념 위에 코드 방향(chordwise) 하중 분포까지 부여.

본 연구는 Sørensen & Shen [20]이 풍력터빈용으로 처음 제안한 ALM을 헬리콥터 로터에 적용한다. 사전에 지정한 하중 분포나 과도한 모델링에 의존하지 않고, 완전 예측형(fully predictive) 방식으로 블레이드 효과를 소스항으로 재현하는 것이 특징이다. 저자들은 ALM이 B-R U-RANS와 비교 가능한 정확도로 (i) 로터 성능계수, (ii) 블레이드 하중, (iii) tip vortex 위치, (iv) 일반 유동장 특성, (v) 기체 표면압력계수까지 예측할 수 있다고 주장한다. 검증은 잘 알려진 S-76 로터(hover)로 수행하고 실험·B-R과 비교하며, 축류비행으로 확장한다. 이어 지면효과, 전진비행, 그리고 박스형 장애물이 있는 협소공간에서의 로터(축소 모델 헬리콥터)를 다룬다.

---

## II. 시험 케이스 및 로터 형상 (Test Cases & Rotor Geometries)

세 가지 비행조건에 대응하는 세 가지 로터 형상(각각 실험 데이터셋이 존재)을 사용한다. 모두 실제적인 블레이드 형상/종횡비(aspect ratio)를 가지나, 실제 헬리콥터의 관절형(articulated) 블레이드·주기피치(cyclic pitch)와 달리 **강체(rigid)·비관절·고정 collective(주기피치 없음)** 로 단순화되어 있다. 또한 축소 모델이므로 실제 풀스케일보다 Re·Ma가 낮다. 그래도 핵심 물리현상은 잘 포착되며, 순수 공력 관점에서의 비교(=CFD 코드 검증)에 적합하다.

### Table 1 — Hover용 로터: S-76 (AIAA Hover Prediction Workshop 대표 케이스)

Balch & Lombardi [21,22]의 S-76 실험 모델이며, AIAA Hover Prediction Workshop의 중심 시험 케이스 중 하나다. 본 연구는 **직사각형(rectangular) tip, $M_{tip}=0.6$** 조건에 집중한다.

| 항목 | 값 |
|---|---|
| 블레이드 수 $N_b$ | 4 |
| 로터 반경 $R$ | 56.04 in |
| 기준 코드 $c_{ref}$ | 3.1 in |
| 종횡비 AR | 18.077 |
| 솔리디티 $\sigma$ | 0.07043 |
| 선형 트위스트 | $-10^\circ$ |

블레이드 단면 정의 (r/R, 트위스트, 두께비 t/c, 익형):

| $r/R$ | Twist (deg) | $t/c$ (%) | Airfoil |
|---|---|---|---|
| 0.189 | 4.01 | 13 | SC1013-R8 |
| 0.285 | 4.5 | 10.09 | Blend |
| 0.400 | 3.5 | 9.4 | SC1094-R8 |
| 0.750 | 0 | 9.4 | SC1094-R8 |
| 0.800 | -0.5 | 9.4 | SC1094-R8 |
| 0.840 | -0.9 | 9.5 | SC1095 |
| 0.950 | -2 | 9.5 | SC1095 |
| 1.000 | -2.5 | 9.5 | SC1095 |

### Table 2 — 축류비행용 로터: Felker & McKillip [23] (Princeton Long Track, Froude scaled)

Froude 스케일 로터로 Ma·Re가 비교적 낮다. Princeton Long Track 장치에서 로터를 썰매(sled)에 장착해 정지 공기 중을 직선 이동시켜 climb/descent를 모사했다.

| 항목 | 값 |
|---|---|
| 블레이드 수 $N_b$ | 4 |
| 로터 반경 $R$ | 1.2192 m |
| Collective(고정) $\Theta_{75}$ | $9.3^\circ,\ 10.9^\circ$ |
| 회전속도 $\Omega$ | 430.78 RPM |
| 솔리디티 $\sigma$ | 0.0633 |
| 종횡비 AR | 19.2 |
| 익형(일정) | NACA0015 |
| 기준 코드 $c_{ref}$ | 0.0635 m |
| 선형 트위스트 | $-8^\circ$ |
| Tip 속도 $V_{tip}$ | 55 m/s |
| Tip 마하수 $M_{tip}$ | 0.16 |
| Tip 레이놀즈수 $Re_{tip}$ | 212,103 |

### Table 3 — 협소공간용 로터: Zagaglia et al. [24] (축소 MD-500 기체)

축소 MD-500 헬리콥터 기체에 4매 블레이드(단순 익형) 로터를 결합. Politecnico di Milano 풍동에서 시험했으며, 크기 1 m × 0.45 m × 0.8 m (W×H×L)의 직육면체(cuboid) 장애물을 배치했다. 시험 캠페인에서는 로터 이동 sweep, 장애물 유무/위치 등 여러 구성을 다뤘다. 모든 시험은 **무풍** 및 **5.07 m/s 정상풍**(전진비 $\mu = 0.05$에 해당) 두 조건에서 수행했다.

| 항목 | 값 |
|---|---|
| 블레이드 수 $N_b$ | 4 |
| 로터 반경 $R$ | 0.375 m |
| Collective(고정) $\Theta_{75}$ | $10^\circ$ |
| 종횡비 AR | 11.72 |
| 솔리디티 $\sigma$ | 0.10865 |
| 회전속도 $\Omega$ | 2580 RPM |
| 익형(일정) | NACA0012 |
| 기준 코드 $c_{ref}$ | 0.032 m |
| 선형 트위스트 | $0^\circ$ |
| Tip 마하수 $M_{tip}$ | 0.3 |
| Tip 레이놀즈수 $Re_{tip}$ | 220,000 |

---

## III. 수치 모델링 (Numerical Modelling)

### 솔버 및 격자 설정

- **CFD 솔버**: **Star-CCM+ 12.06** [25]. 범용 유한체적(finite volume), 비정렬(unstructured), 셀 중심(cell-centred), 다물리(multi-physics) 유동 솔버. (※ 본 논문의 솔버는 NSCODE가 아니라 Star-CCM+ 임에 유의.)
- **이산화**: 시간·공간 모두 2차 정확도(2nd order). 블레이드 tip 마하수에 따라 **SIMPLE** 알고리즘 [26] 또는 **밀도기반(density-based) 결합형 Weiss–Smith 예조건화 Roe FDS(flux-difference-splitting)** 기법 [27] 중 선택.
- **시간진행**: dual time stepping. 모든 해석에서 시간스텝은 방위각 회전 **$1^\circ$** 에 해당하도록 동일하게 설정.
- **난류모델**: U-RANS를 $k$–$\omega$ **SST** 모델 [28]로 닫음.
- **격자(공정 비교)**: ALM과 B-R의 **배경격자(background mesh)는 크기·해상도가 유사**하며 **10–20 M cells** 범위. B-R은 여기에 overset 방식으로 개별 블레이드 메시를 추가하는데, 각 블레이드가 종횡비에 따라 **5–10 M cells** 수준.
- **구현**: ALM은 Star-CCM+의 User Coding 프레임워크로 로드되는 외부 라이브러리. **C 언어**로 작성, 기반 솔버와 동일한 MPI로 병렬 통신 수행.

### 지배방정식과 ALM 소스항

ALM은 블레이드 형상을 제거하고 그 효과를 운동량 소스항 $\mathbf{f}_{alm}$으로 대체한다. 보존형 유한체적 형태로 쓴 연속·운동량·에너지 방정식은 다음과 같다.

**(식 1) 연속(질량 보존)**

$$
\frac{\partial}{\partial t}\int_V \rho\, dV + \oint_A \rho\,\mathbf{v}\cdot d\mathbf{a} = 0
$$

> 검사체적 내 질량의 시간변화 + 표면을 통한 질량유속 = 0 (질량 보존).

**(식 2) 운동량 보존**

$$
\frac{\partial}{\partial t}\int_V \rho\,\mathbf{v}\, dV + \oint_A \rho\,\mathbf{v}\otimes\mathbf{v}\cdot d\mathbf{a} + \oint_A p\,\mathbf{I}\cdot d\mathbf{a} = \oint_A \mathbf{T}\cdot d\mathbf{a} - \int_V \mathbf{f}_{alm}\, dV
$$

> 운동량의 시간변화 + 대류 운동량유속 + 압력항 = 점성응력항 − ALM 운동량 소스항. 우변 마지막 항이 로터가 유동에 가하는 힘을 부호 반대(equal magnitude, opposite direction)로 부과한다.

**(식 3) 에너지 보존**

$$
\frac{\partial}{\partial t}\int_V \rho E\, dV + \oint_A \rho H\,\mathbf{v}\cdot d\mathbf{a} = \oint_A \mathbf{T}\cdot\mathbf{v}\, d\mathbf{a} - \oint_A \mathbf{q}\cdot d\mathbf{a} - \int_V \mathbf{f}_{alm}\cdot\mathbf{v}\, dV
$$

> 전에너지 시간변화 + 엔탈피 유속 = 점성응력의 일 − 열유속 − ALM 소스항이 하는 일($\mathbf{f}_{alm}\cdot\mathbf{v}$). ALM 힘에 대응하는 일이 에너지식에도 반영됨.

여기서 $\rho$(밀도), $\mathbf{v}$(속도벡터), $p$(압력), $\mathbf{T}$(점성응력텐서), $\mathbf{q}$(열유속), $E,H$(전에너지·엔탈피)는 통상의 유동변수이고, $\mathbf{f}_{alm}$만이 운동량·에너지 식 우변에 추가되는 소스항이다. 음의 부호는 로터 힘과 크기는 같고 방향은 반대로 작용함을 의미한다(즉 로터가 유동에 힘을 가함).

### 절점 하중 — BET 기반

$\mathbf{f}_{alm}$ 계산에는 BET를 사용한다. 블레이드를 **1/4 코드(quarter-chord) 스팬 축**을 따라 블레이드 세그먼트(blade segments)로 이산화하고, 각 절점에서 **integral velocity sampling** [16,29,30]으로 국부 속도를 평가한 뒤 회전속도와 합성하여 유효 상대 자유류 속도 $\mathbf{U}_{rel}$와 받음각 $\alpha_{rel}$을 얻는다. 이 두 값으로 사전계산된 2D 익형 polar에서 국부 양력·항력을 구한다.

**(식 4) 절점 양력**

$$
\Delta L = \frac{1}{2}\,\rho\,\lVert \mathbf{U}_{rel}\rVert^2\, C_l(\alpha_{rel}, Re, M)\, c\, \Delta R
$$

> 세그먼트 양력 = 동압 × 2D 양력계수 × 코드 × 세그먼트 폭. $C_l$은 받음각·Re·Ma의 함수(polar 보간).

**(식 5) 절점 항력**

$$
\Delta D = \frac{1}{2}\,\rho\,\lVert \mathbf{U}_{rel}\rVert^2\, C_d(\alpha_{rel}, Re, M)\, c\, \Delta R
$$

> 세그먼트 항력. $C_d$ 역시 $(\alpha_{rel}, Re, M)$ 함수.

### 힘 투영 — Gaussian kernel

각 ALM 제어점에서 구한 양력·항력 벡터를 Gaussian kernel과 곱해 더함으로써 전체 로터 해를 재구성한다.

**(식 6) 3D Gaussian smearing kernel**

$$
g(x,y,z) = \frac{1}{\epsilon^3\,\pi^{3/2}}\,
\exp\!\left(-\frac{(x-x_0)^2 + (y-y_0)^2 + (z-z_0)^2}{\epsilon^2}\right)
$$

> 점 힘을 여러 격자셀에 걸쳐 분포(smear)시키는 3차원 정규화 Gaussian. $\epsilon$이 분포 폭(smearing parameter)이며, $1/(\epsilon^3\pi^{3/2})$ 정규화로 전체 적분이 1이 되도록 함. $(x_0,y_0,z_0)$는 ALM 제어점 위치.

**(식 7) ALM 운동량 소스항(투영된 체적력)**

$$
\mathbf{f}_{alm} = \sum_i g_i\,\bigl(\Delta L_i\,\mathbf{e}_{L,i} + \Delta D_i\,\mathbf{e}_{D,i}\bigr)
$$

> 각 제어점 $i$의 (양력×양력방향 단위벡터 $\mathbf{e}_{L,i}$ + 항력×항력방향 단위벡터 $\mathbf{e}_{D,i}$)에 Gaussian 가중 $g_i$를 곱해 모든 절점에 대해 합산. 이렇게 분포된 체적력이 식 (2),(3)의 소스항으로 들어감.

### ★ 핵심 설정 (충실히) — ε·sampling·correction·솔버

연구자에게 가장 중요한 ALM 모델링 선택을 명시적으로 정리한다.

- **Gaussian smearing 폭 $\epsilon$**: $\epsilon$은 **국부 코드에 비례(scaled with local chord)** 하도록 두며, 본 연구는 $\boxed{\epsilon \sim 0.25\,c}$ 값이 최선의 결과를 준다고 명시한다 [30–33]. 또한 소스항은 블레이드의 기하학적 끝(루트/팁) 한계를 지키도록 **반경 방향으로 절단(radially truncated)** 되고, 절단 후에도 힘이 유동장에 완전히 투영되도록 **재정규화(re-normalized)** 한다.
- **속도 샘플링 방식**: **integral velocity sampling**(= Gaussian 가중 적분 속도 샘플링, g-weighted)을 사용한다([16,29,30]; 특히 저자 본인의 2D 파라메트릭 연구 [30]에 근거). 즉 단일 점에서 속도를 읽는 point sampling이 아니라, **Gaussian kernel $g$로 가중한 적분 속도**를 국부 속도로 취한다.
  - *왜 point가 아니라 integral인가?* Gaussian으로 분포된 힘이 만드는 자기유도(self-induction)는 분포 폭 $\epsilon$에 의존한다. 점-속도를 그대로 쓰면 이 자기유도가 받음각에 비물리적으로 섞여 들어가, $\epsilon$에 민감한 하중 오차가 생긴다. **g-가중 적분 샘플링은 힘 투영과 동일한 커널로 속도를 평균**하므로, 분포로 인한 자기유도를 일관되게 상쇄/정규화하여 $\epsilon$ 의존성을 줄이고 폭넓은 $\epsilon$에서 안정적인 하중을 준다. 이 점이 본 논문이 별도 보정 없이도 좋은 결과를 얻는 핵심 메커니즘 중 하나다.
- **Tip-loss / smearing correction**: 본 논문은 **Prandtl tip-loss correction이나 별도의 smearing(라인) correction을 명시적으로 적용한다는 언급이 없다.** 즉 tip correction 없이도 hover thrust를 잘 맞춘다. 이는 (i) $\epsilon\approx0.25c$로 분포 폭을 익형 chord 스케일에 맞춰 유지하고, (ii) integral(g-가중) 샘플링으로 자기유도를 일관되게 처리하며, (iii) **B-R과 동급(10–20 M cells)의 충분히 조밀한 배경격자**로 tip vortex 자체를 직접 해상하여 팁 근처 하강류(downwash)/유도효과를 유동해가 스스로 만들어내기 때문이다. 결과적으로 BET가 잡아내야 할 "유도효과"의 상당 부분을 유동장 해석이 직접 담당하므로, 별도 tip correction이 불필요해진다.
- **CFD 솔버**: 위에서 기술한 대로 **Star-CCM+ 12.06**(유한체적, 비정렬, 셀중심, $k$-$\omega$ SST, dual-time, $1^\circ$/step). (NSCODE가 아님.)

> **결론적으로 "ALM이 tip correction 없이 hover thrust를 잘 맞추는 비결"**:
> ① $\epsilon\approx0.25c$ — chord 스케일의 적절한 분포 폭으로 과도한 smearing 오차를 피함.
> ② **integral / g-weighted velocity sampling** — 힘 투영과 같은 커널로 속도를 평균해 self-induction을 일관 처리, $\epsilon$ 민감도와 받음각 오차를 줄임.
> ③ **B-R급 조밀 배경격자**가 tip vortex를 직접 해상 — 팁 유도효과를 유동해가 담당 → Prandtl tip-loss 불필요.
> 단, 이 세팅의 한계로 인접 블레이드 tip vortex와의 **수직(perpendicular) BVI** 영역(≈92% 스팬 이후)에서는 하중 과대/과소가 남는다(아래 결과 참조).

---

## IV. 결과 (Results)

### A. Hover (S-76)

먼저 collective sweep으로 hover 성능계수를 구한다(**Fig. 1**).

- **추력(thrust)**: ALM과 B-R 모두 우수한 예측력을 보인다. 다만 **ALM은 높은 collective에서 thrust를 약간 과소예측(slightly underestimate)** 한다.
- **토크(torque)**: 동일 thrust에서 **ALM의 $C_Q$가 약간 과대예측(slightly overestimated)** 되며, 그 결과 **동일 thrust에서 FoM이 약간 낮게** 나온다.

**원인 분석 (Fig. 2, $C_T/\sigma = 0.09$ trim 상태에서의 단면 하중)**:

- **스팬 첫 80%까지는** 두 방법(ALM, B-R)의 thrust·torque 하중이 **거의 완벽히 일치**.
- **92% 스팬 이후**부터 ALM이 thrust 하중을 한쪽은 과소, 다른 쪽은 과대예측한다. 이는 **앞선 블레이드의 tip vortex와의 조우** 때문으로, 과도한 up-wash/down-wash를 유발한다. 이 **수직 BVI(blade-vortex interaction)** 가 torque 분포를 교란하여 블레이드 팁 부근에서 과도한 항력을 만들고, 이것이 위에서 본 전체 torque 약간 과대예측의 직접 원인이다.

**Fig. 1 — hover 통합 성능계수** ([34]에서 발췌/각색):
- (a) Thrust: $C_T/\sigma$ vs $\Theta_{75}$ — Exp / B-R / ALM이 거의 겹침(ALM은 고 collective에서 약간 낮음).
- (b) Torque: $C_Q/\sigma$ vs $C_T/\sigma$ — ALM이 주어진 thrust에서 약간 높음.
- (c) FoM vs $C_T/\sigma$ — ALM이 약간 낮음(b의 결과).
- *핵심 결론*: **별도 tip correction 없이도 hover thrust는 실험·B-R 수준으로 예측**되며, 차이는 BVI에서 비롯된 약간의 torque 과대예측(→ FoM 약간 저하)에 국한된다.

**Fig. 2 — $C_T/\sigma = 0.09$ 단면 하중** ([34]에서 발췌):
- (a) $d(C_T/\sigma)/dr$ vs $r$, (b) $d(C_Q/\sigma)/dr$ vs $r$.
- *핵심 결론*: 0–80% 스팬은 B-R과 ALM이 사실상 동일, **92% 이후에서 BVI로 인한 thrust 과대/과소 + torque(항력) 과대** 가 발생.

### B. 축류비행 — Climb / Descent (Felker & McKillip 로터)

hover에서 climb로 가면 자유류 속도가 더해져 로터 디스크 위 유입속도가 증가하고, 동일 collective에서 유효 받음각이 줄어 thrust가 감소한다. 운동량 이론상 climb의 소요동력은 "hover 유도동력 + climb 유도동력"의 합이므로, climb 속도에 대한 **FoM은 선형 추세**를 보일 것으로 기대된다 [35].

**Fig. 3 — 축류비행 통합 성능계수**: hover 값으로 정규화한 thrust계수와 FoM을, $V_{ind}^{hover}=V_{tip}\sqrt{C_T/2}$로 정규화한 climb 속도 $V_c/V_{ind}^{hover}$의 함수로 표시.
- (a) $C_T/C_T^{hover}$ vs $V_c/V_{ind}^{hover}$: **thrust 감소를 두 방법 모두 잘 포착**하고 서로 잘 일치하며 실험과도 **양호(fair)** 한 일치.
- (b) $\mathrm{FoM}/\mathrm{FoM}^{hover}$ vs $V_c/V_{ind}^{hover}$ ($\Theta_{75}=10.9^\circ$): **FoM의 선형 추세는 매우 우수(excellent)**.
- **하강(descent) 한계**: 두 방법 모두 descent 거동을 정확히 잡지 못한다. 실험 [23]은 매우 비정상 하중과 vortex ring state(VRS)를 보고하나, **두 해석은 $V_c/V_{ind}^{hover}=-0.75$에서도 거의 정상(near-steady) 수렴 thrust 신호**를 보이며 VRS의 뚜렷한 징후가 없다. 다만 far-wake의 비대칭성이 VRS 발생 조짐일 수 있다.

**Fig. 4 — tip vortex 위치 (축류비행, $\Theta_{75}=10.9^\circ$)**: wake age(후류 나이, deg)에 대한 $z/R$(축방향)과 $r/R$(반경방향) 좌표.
- **hover**: 두 방법이 서로 그리고 Kocurek & Tangler [36]의 경험적 prescribed-wake 모델과 **매우 잘 일치**. 단, 2번째~5번째 블레이드 통과 사이 축방향 속도에 약간의 lag.
- **climb ($V_c/V_{ind}^{hover}=0.75$)**: 추가 하강류로 tip vortex가 hover보다 빠르게 이류(advect)되고, **반경 수축(radial contraction)이 상당히 감소**. 두 수치해는 거의 구별 불가.
- **descent ($V_c/V_{ind}^{hover}=-0.25$)**: 후류가 덜 안정해 4번 통과 후 tip vortex 추적이 중단. climb과 반대로 hover보다 **더 수축**하고 축방향 이류는 더 느림.

### C. 협소공간 (지면효과 / 장애물 / 전진비행) — Zagaglia 로터

협소공간(로터 움직임이 적어도 한 방향으로 형상에 의해 제약되는 환경)을 평가하기 위해 Zagaglia et al. [24] 데이터셋의 두 구성을 사용한다.
1. **지면효과(ground effect)만**: 지면판 위 다양한 수직 높이.
2. **지면 + 장애물**: 로터에서 $2R$ 떨어진 곳에 직육면체 장애물 추가, 역시 다양한 높이.

**Fig. 5 — 협소공간 thrust계수 비** (OGE, $Z/R=4.0$ 추력으로 정규화; [38]에서 각색):
- (a) **지면만**: Cheeseman & Bennett [37] 경험식을 비교용으로 추가. **ALM·B-R이 서로 매우 잘 일치**하며, 고정 collective에서 **지면에 가까워질수록 thrust 증대(thrust augmentation)** 를 정확히 예측($Z/R$ 작아질수록 $C_T/C_T^{oge}$ 증가, 최대 ~1.2 수준).
- (b) **지면 + 장애물**: **장애물이 thrust 상승을 상당히 감소**시킨다. 두 수치해는 서로 잘 일치하나, 실험과의 일치는 수직 sweep의 **양 끝단에서만** 명확. **$Z/R=2.7$에서는 두 방법 모두 thrust를 과대예측**하는데, 이 케이스는 후류 발달에 시간이 오래 걸려(re-circulation 패턴 형성이 느림) 충분한 모사시간을 확보할 계산자원이 과도하기 때문으로 설명된다.

**물리 메커니즘**: 장애물이 있으면 후류가 로터 아래로 흘러내려 지면을 치고 다시 위로 올라가 장애물 수직벽을 만나며 **로터-장애물 사이에 재순환영역(re-circulation zone)** 을 만든다. 낮은 높이에서는 로터가 자신의 재순환 후류를 재섭취(re-ingest)하여 유입속도가 커지고, 이로 인해 블레이드 유효 받음각이 줄어 thrust가 감소한다. 높이가 높으면 재순환 영향이 작고, 중간 높이는 재순환 패턴 정착에 오랜 시간이 걸려 모사시간이 과다해진다.

**Fig. 6 — ALM 지면효과 순간 축방향 속도 + Q-criterion** ($Z/R=4$ vs $Z/R=1$; [38]에서 각색): $Z/R=1$에서 후류가 hover의 전형적 수축에서 **지면효과에 의한 팽창(expansion)** 으로 바뀌며, 블레이드 루트 너머 로터 중앙에서 더 큰 up-wash가 관찰됨.

**Fig. 7 — B-R 지면+장애물 순간 축방향 속도 + Q-criterion** ($Z/R=1.00,\ 2.70,\ 4.20$; [38]): 재순환 패턴을 시각적으로 보여줌.

**Fig. 8 — 전진비행 시 기체 표면 평균 압력계수 + Q-criterion** ($\mu=0.05,\ Z/R=4$; ALM vs B-R; [38]에서 각색): 추가 역량 시연으로 로터+기체를 전진비행에 배치. **두 방법의 평균 압력분포가 거의 동일**하며 tip vortex 후류 패턴도 유사 → ALM이 기체 표면 하중까지 B-R 수준으로 재현 가능함을 입증.

### D. 계산시간 비교 (Computational Time Comparison)

ALM 개발의 최종 목표는 메시 절감을 통한 계산시간 단축이다. **Table 4**는 세 케이스의 총 메시 수·비율과 1회전당 CPU 시간·속도향상(speed-up)을 보여준다.

**Table 4 — 1회전 평균 계산시간 및 speed-up**

| 메시 크기 [백만 셀] | Hover | Axial Flight | Confined |
|---|---|---|---|
| B-R | 28.7 | 47.1 | 32.9 |
| ALM | 10.1 | 20.4 | 18.0 |
| **Ratio (B-R/ALM)** | **2.84** | **2.31** | **1.83** |

| 1회전당 CPU 시간 [hr] | Hover | Axial Flight | Confined |
|---|---|---|---|
| B-R | 2321.5 | 2504 | 1661.2 |
| ALM | 741.9 | 880.0 | 824.3 |
| **Speed-up** | **3.1** | **2.84** | **2.02** |

핵심 관찰:
- ALM의 speed-up은 메시 절감비에 **비례하지만 실제로는 그보다 더 크다**. 이는 ALM의 **병렬 효율이 우수**한 반면, B-R은 overset 기법에 의존해 병렬 성능이 크게 저하되기 때문이다.
- overset 확장성 한계의 예: B-R Axial Flight는 **480 CPU 코어**로 수행됐는데, 같은 메시를 **1000 코어**로 돌리면 1회전당 3758 hr가 되어 명백한 병렬효율 손실(poor scaling)이 발생. 반대로 ALM은 1000 코어 이상에서 동일 메시로 더 나은 scaling을 보인다.
- 따라서 ALM은 (i) 메시가 작아 빠르고, (ii) 확장성이 좋아 충분한 자원이 있으면 풀 헬리콥터 CFD를 시간/일(hours/days) 단위로 돌릴 수 있다. 반면 overset B-R은 (효율 목표 때문에 낮은 코어 수로 제약되어) 보통 며칠~몇 주가 걸린다. 또한 **ALM은 B-R보다 메시 조밀화/조대화(coarsening)에 훨씬 둔감**하여, 총 속도향상을 더 키울 여지가 있다.

---

## V. 결론 (Conclusions)

- 두 고충실도 CFD 도구(ALM, overset B-R U-RANS)를 다양한 비행조건의 헬리콥터 로터에 적용했다.
- 개발한 로터 대체 기법인 **ALM은 전통적 overset B-R 대비 뛰어난 일관성과 추세 예측력**을 보였고, 두 방법 모두 기준 실험과 **양호한 일치**를 보였다.
- **ALM의 주된 단점**: 수직 BVI(perpendicular BVI) 처리. 이로 인해 블레이드 thrust 하중의 과대/과소 예측이 생기고, 궁극적으로 **torque가 약간 높게** 나온다.
- 그 외 모든 지표에서 ALM은 B-R과 **동등(on par)** 한 결과를 **상당한 계산시간 절감**과 함께 제공했다. 절감은 (i) 더 적은 메시 셀 수와 (ii) 더 높은 병렬화 역량에서 비롯된다.

## VI. 감사의 글 / 자금 (요약)

CAE Inc.와 NSERC(캐나다 자연과학·공학연구위원회) 지원, Calcul Québec / Compute Canada의 계산자원으로 수행됨.

---

## 연구자를 위한 종합 메모

1. **tip correction 없이 hover thrust가 맞는 이유 (재강조)**: 본 논문은 Prandtl tip-loss나 라인 보정을 쓰지 않는다. 대신 ($\epsilon\approx0.25c$) + (integral/g-weighted velocity sampling) + (B-R급 조밀 배경격자로 tip vortex 직접 해상)의 조합으로, BET가 떠맡을 유도효과를 유동해가 직접 생성한다. 이 셋이 핵심 레시피.
2. **남는 오차의 위치**: 0–80% 스팬은 완벽, 92% 스팬 이후 수직 BVI에서 thrust 과대/과소 + torque 약간 과대. FoM이 동일 thrust에서 약간 낮게 나오는 것은 이 torque 과대예측의 직접 결과.
3. **솔버 주의**: 본 논문의 솔버는 **Star-CCM+ 12.06**(요청서에 언급된 NSCODE 아님)이며, 익형 polar는 사전계산하여 $C_l,C_d=f(\alpha,Re,M)$로 보간.
4. **검증 로터**: hover=**S-76 (AIAA Hover Prediction Workshop, rect tip, $M_{tip}=0.6$)**, 축류=Felker–McKillip(Princeton Long Track, NACA0015), 협소공간=Zagaglia(MD-500 축소, NACA0012). (요청서의 Caradonna–Tung은 본문에 없으며, 참고문헌 [35]의 Caradonna는 축류비행 momentum-theory 추세 근거로만 인용됨.)
5. **정량 정확도 요약**: hover thrust는 실험·B-R과 거의 일치(ALM은 고 collective에서 약간 과소), torque는 ALM이 약간 과대 → 동일 thrust FoM 약간 저하; climb thrust 감소·FoM 선형추세는 잘 포착(descent/VRS는 미포착); 지면효과 thrust 증대(최대 ~1.2배) 정확; 장애물 케이스 $Z/R=2.7$만 과대예측; 계산 speed-up 2.0–3.1배(메시비 1.8–2.8배보다 큼).
