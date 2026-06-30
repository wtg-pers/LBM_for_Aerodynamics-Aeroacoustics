# 액추에이터 라인 기법을 위한 비반복(non-iterative) 와류 기반 스미어링 보정 (Non-iterative vortex-based smearing correction for the actuator line method)

- **저자(Authors):** Vitor G. Kleine (KTH FLOW / ITA), Ardeshir Hanifi (KTH FLOW), Dan S. Henningson (KTH FLOW)
- **소속:** 1) KTH Royal Institute of Technology, FLOW, Stockholm, Sweden  2) Instituto Tecnológico de Aeronáutica (ITA), São José dos Campos, Brazil
- **출처(Source):** *Journal of Fluid Mechanics* 게재 예정 (Under consideration). arXiv:2206.05448v3 [physics.flu-dyn], 2023-03-18 (초판 2022-06)
- **연도(Year):** 2022 (arXiv), 2023 (JFM accepted)

## 한국어 요약 (4줄)
ALM의 스미어링 보정(vortex-based smearing correction)은 가우시안 투영(Gaussian projection)으로 약해진 유도속도(induced velocity)를 "누락 속도(missing velocity)"로 되살려 양력선 이론(lifting line)의 결과를 재현하지만, 순환(circulation)과 유도속도가 서로 의존하므로 기존 방법들(Dağ & Sørensen 2020, Meyer Forsting et al. 2019)은 매 시간스텝마다 완화계수(relaxation factor) 기반 **반복(iterative)** 절차를 필요로 했다. 본 논문은 ALM과 수학적으로 동일한 비선형 양력선(non-linear lifting line)을 먼저 구성하고, 이를 **선형화(linearization)**하여 반복 과정을 크기 N(제어점 개수)의 작은 선형 시스템 직접 풀이로 대체하는 비반복(non-iterative) 보정을 제안·검증한다. 추가로 (i) 스미어 와류 세그먼트(smeared vortex segment)의 정확한 유도속도 해석식과 (ii) CFD 속도로 와류를 이류시키는 free-vortex wake 모델을 도입해 일반성과 정확도를 높였다. 평판 날개(translating wing)와 NREL 5MW 풍력터빈(전단 유입) 검증에서 반복법과 비반복법의 차이는 순환의 10⁻⁵ 수준으로 무시할 만하며, ALM과 양력선 사이의 유도속도 차이는 0.01%(10⁻⁴) 수준으로 선행 연구보다 훨씬 우수하다.

**키워드(Key words):** actuator line method(액추에이터 라인 기법), vortex-based smearing correction(와류 기반 스미어링 보정), lifting line(양력선), linearization(선형화), free-vortex wake(자유 와류 후류), NREL 5MW

---

## 0. 본 정리본의 핵심 메시지 (구현 관점 한 줄 정리)

> **"왜 비반복인가?"** — 순환 Γ와 유도속도 u는 서로를 결정하는 암묵적(implicit) 관계다. 기존 방법은 이를 fixed-point 반복(완화계수 r)으로 풀었다. 본 논문은 양력 곡선 Cl(α)를 직전 시간스텝 해 주변에서 **선형화**하여, 이 암묵 관계를 1차 항만 남긴 **선형 방정식 (I − diag(b)A)ΔΓ = Γ† − Γⁿ⁻¹** 로 바꾼다. 이 N×N 선형계를 한 번 직접 풀면(LU 분해 등) 반복이 필요 없다. 우리가 ALM+MLG에 구현한다면, **이 버전이 더 결정론적(deterministic)이고 안정적이며 완화계수 튜닝이 불필요**하므로 채택을 권장한다. (자세한 비교는 §4·§5·아래 "구현 결정 요약" 참조)

---

## 1. 서론 (Introduction)

### 1.1 ALM과 스미어링 문제의 배경
액추에이터 라인 기법(actuator line method, ALM)은 블레이드/날개를 Navier–Stokes 해석에서 양력선(lifting line)으로 표현하기 위해 Sorensen & Shen(2002)이 개발하였다. 블레이드 형상 대신 국소 속도와 익형 데이터(airfoil data)로 계산한 체적력(body force)을 운반하는 선으로 대체함으로써 계산 비용을 크게 줄인다.

힘을 정확히 계산하려면 블레이드 팁 근처의 국소 속도를 보정해야 한다. 이는 과거에 다소 논쟁적이었다. 일부(Martinez et al. 2012; Sørensen 2016)는 ALM이 자체적으로 팁 와류(tip vortex)를 만들어 팁 하중을 낮추므로 보정이 불필요하다고 주장했지만, 실제 수치 결과에서 팁 하중이 과대평가(overestimation)되는 현상이 보정의 필요성을 보여준다.

### 1.2 와류 기반 스미어링 보정의 계보 (핵심)
- **Dağ & Sørensen (2020)** (원래는 Dağ 2017 박사학위): ALM이 만드는 속박 와류(bound vortex)의 와도(vorticity) 분포가 **Lamb–Oseen 와류 모델**과 동일한 가우시안 분포임을 관찰했다. 이를 와류 시트(vortex sheet) 전체로 확장하여, 이산화된 Prandtl 양력선이 예측하는 **특이(singular) 와도 분포**가 유도하는 속도에 ALM 속도를 근사시키는 보정을 제안했다. 로터 시뮬레이션에서 명확한 개선(팁·허브에서 힘이 매끄럽게 0으로 수렴)을 보였고, 평판 날개에서는 양력선 결과에 근접했다.
- **Meyer Forsting et al. (2019a)**: 유사한 보정을, 점성 코어 모델(viscous core model)이 있는/없는 회전 블레이드용 와류 기법과 비교했다. 보정 없는 ALM은 **유한 코어 크기**를 가진 와류 기법과 일치하고, 보정한 ALM은 **이상 와류(ideal vortices)**를 쓴 와류 기법과 일치함을 보였다.
- **이론적 근거**: ALM이 만드는 와류와 Lamb–Oseen 모델의 수학적 연결은 속박 와류에 대해 **Forsythe et al. (2015)** 가, 평판 날개 와류 시트에 대해 **Martínez-Tossas & Meneveau (2019)** (원래 Martínez Tossas 2017)가 증명했다. 둘 다 힘을 분포시키고 수치 불안정을 피하기 위해 필요한 **가우시안 커널과의 컨볼루션(convolution)**의 결과임을 보였다. Martínez-Tossas & Meneveau의 보정은 "subfilter-scale velocity correction" 또는 "filtered actuator line model"로 불리며, 본질적으로 와류 기반 스미어링 보정의 한 변형이다(Stanly et al. 2022가 풍력터빈에 적용).
- **Caprace et al. (2019)**: ALM은 아니지만 가우시안 커널로 와도를 분포시킨 mollified(smeared) 양력선을 연구하여, mollified 양력선과 ALM의 연결을 명확히 했다. 팁 근처에서 평활화 파라미터(smoothing parameter)를 0 또는 매우 작은 값으로 가변시키면 결과가 개선됨을 보였다(ALM 연구들 Shives & Crawford 2013; Jha et al. 2014; Jha & Schmitz 2018에서도 관찰).

### 1.3 ε(스미어링 파라미터)를 줄이지 않고 보정하는 이유
ALM에서 평활화 파라미터 ε를 줄이면(가변 ε 포함) 팁 결과가 좋아지지만, ε가 작으면(보통 최소 2~3Δx, Troldborg 2009) 수치 불안정과 진동이 생긴다. 예: Kleusberg et al.(2019b)은 스펙트럴 요소법에서 ε=2Δx일 때 공간적으로 성장하는 진동을 관찰했고, 3Δx≤ε≤4Δx에서는 진동 진폭이 유계였다. 와류 안정성 연구(Kleine et al. 2022)는 ε=3.5Δx를, Shives & Crawford(2013)는 받음각 오차를 줄이려면 ε≈4Δx를 권장했다. 따라서 **ε를 작게 만드는 대신, 컨볼루션된 힘의 와도와 특이(singular) 힘의 와도 차이를 보정**하는 방식(Dağ, Martínez-Tossas, Meyer Forsting)이 더 비용 효율적이다.

### 1.4 기존 방법의 반복(iterative) 문제 — 본 논문이 해결하는 지점
- **Dağ & Sørensen (2020)과 Meyer Forsting et al. (2019a)**: 매 시간스텝마다 반복 절차를 적용한다. 순환은 국소 유도속도에 의존하고, 유도속도는 다시 순환에 의존하기 때문에, 각 반복에서 완화계수(relaxation factor) r로 속도 또는 순환을 갱신한다. 저자들의 경험상 **r이 1에 가까우면 수치 불안정**, **r이 작으면 반복 횟수 증가 → 런타임 증가**라는 딜레마가 있다.
- **Martínez-Tossas & Meneveau (2019)**: 각 시간스텝의 반복을 피하는 대신 **직전 시간스텝의 순환**으로 유도속도를 계산하고, 직전·현재 보정속도의 가중평균을 보정항으로 쓴다. 즉 직전 두 시간스텝의 순환 값으로 유도속도를 계산하되 **현재 순환은 고려하지 않는다**. 개념적으로 타 연구 반복법의 "첫 번째 반복"에 해당한다. 정상(steady) 문제에서는 수렴하지만, **비정상(unsteady) 문제에서는 매 시간스텝에서 현재 순환과 국소 속도의 양립성(compatibility)이 보장되지 않아 오차가 발생**한다(오차는 시간스텝 간 순환 차이와 가중계수에 의존).

### 1.5 본 논문의 기여
본 논문은 **반복을 피하면서도 현재 순환과 속도의 양립성을 유지**하는 방법을 도입한다. 양력선의 **선형화(linearized)** 버전에 기반해 스미어링 보정을 직접 계산하며, 각 시간스텝의 보정은 **크기 N(제어점 개수)의 선형 방정식계 직접 풀이**로 구해진다. 이를 위해 ALM 기반의 이산화 양력선 두 가지(비선형 / 선형화 버전)를 제시한다. 추가 기여:
1. **스미어 와류 세그먼트의 더 정확한 유도속도 근사** (해석적 적분 식 (3.20), 기존 근사 오차 제거)
2. **free-vortex wake 모델 구현** — 와류 시트를 CFD 속도로 이류시켜 정의 (ad hoc 가정 없음, 일반성 유지)

가정을 최소화(특히 회전 블레이드 관련 가정 회피)함으로써 로터를 넘어 **고정익(fixed-wing) 항공기** 등 다른 응용에도 적용 가능하도록 일반성을 추구한다.

---

## 2. 액추에이터 라인 기법 (The actuator line method)

비압축성 Navier–Stokes 방정식(원시 변수 압력 p, 속도 u):

$$
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u} + \mathbf{f}
\tag{2.1}
$$

> (2.1) 비압축성 NS 방정식. 체적력 항 **f**가 ALM에서 터빈/날개를 모델링한다.

체적력은 스팬 단위길이당 2차원 힘 **F₂D**에 기반한다:

$$
\mathbf{F}_{2D} = (F_l, F_d) = \left(\tfrac{1}{2}\rho\, u_r^2\, c\, C_l,\ \tfrac{1}{2}\rho\, u_r^2\, c\, C_d\right)
\tag{2.2}
$$

> (2.2) 2D 양력/항력. 양력 Fl은 상대속도에 수직, 항력 Fd는 평행. 상대속도 $u_r=\sqrt{u_y^2+u_z^2}$, 코드 c, 양력/항력 계수 Cl·Cd는 국소 Re·받음각에서 익형 데이터로부터 보간(그림 1의 국소 좌표계).

2D 힘은 3D 체적력 f와 차원이 맞지 않으므로, 중간 단계로 **이상화된(idealized) 3D 체적력** fⁱ를 정의한다:

$$
\mathbf{f}^i = \frac{1}{\rho}\,\mathbf{F}_{2D}\,\delta(y)\,\delta(z)
\tag{2.3}
$$

> (2.3) 국소 단면에서 힘을 원점에 집중시킨 이상화 체적력. δ는 디랙 델타.

$$
\delta(y) = \lim_{\varepsilon\to 0}\frac{1}{\pi^{1/2}\varepsilon}\,e^{-y^2/\varepsilon^2}
\tag{2.4}
$$

> (2.4) 디랙 델타를 가우시안의 ε→0 극한으로 해석. fⁱ는 원점에서 특이하지만 2D 평면 적분은 F₂D/ρ.

특이성을 피하기 위해 힘을 **3D 가우시안 커널**로 분포시킨다(폭 ε는 세 방향 동일하다고 가정):

$$
\eta_3(x,y,z) := \frac{1}{\pi^{3/2}\varepsilon^3}\exp\!\left(-\frac{x^2+y^2+z^2}{\varepsilon^2}\right) = \eta(x)\,\eta(y)\,\eta(z)
\tag{2.5}
$$

$$
\eta(x) := \frac{1}{\pi^{1/2}\varepsilon}\exp\!\left(-\frac{x^2}{\varepsilon^2}\right)
\tag{2.6}
$$

> (2.5)-(2.6) 3D 가우시안 커널과 1D 성분. (비균일/비등방 커널은 본 논문 범위 외.)

스미어된 힘:

$$
\mathbf{f} = \mathbf{f}^i * \eta_3
\tag{2.7}
$$

> (2.7) 이상화 힘과 가우시안 커널의 컨볼루션이 실제 스미어된 체적력. 이 디랙 델타 중간단계 표현은 고전 표현(Sorensen & Shen 2002; Mikkelsen 2003)과 수학적으로 동등하며, 2D 힘과 3D 컨볼루션의 연결을 형식화한다(§3에서 중요). 양력에 집중하고 항력 스미어링 효과는 향후 연구로 남긴다.

### 2.1 기본 ALM의 양력
각 스팬 단면 j의 양력(스팬 단위길이당):

$$
F_{lj} = \frac{1}{2}\rho\, u_r^2\, c_j\, C_l(\alpha_j)
\tag{2.8}
$$

> (2.8) 단면 j 양력. uz·uy는 제어점 xj에서 CFD로 샘플링.

유효 받음각:

$$
\alpha_j = \alpha_{gj} + \arctan\!\left(\frac{u_y}{u_z}\right)
\tag{2.9}
$$

> (2.9) 유효 받음각 = 기하학적 받음각 αg(트위스트·입사각; 회전 블레이드는 국소 피치각의 음수) + 국소 유동각.

Kutta–Joukowski 정리로 순환:

$$
\Gamma_j = \frac{F_{lj}}{\rho\, u_r} = \frac{1}{2}\,u_r\, c_j\, C_l(\alpha_j)
\tag{2.10}
$$

> (2.10) 순환 = 양력/(ρ·ur). **이 식이 §5 선형화의 핵심 — 순환을 미지수로 푼다.**

### 2.2 와류 기반 스미어링 보정 (개념)
CFD에서 제어점 xj에서 샘플링한 속도 u_s에 "누락 속도(missing velocity)" u_m을 더해 보정 속도 u_c를 얻는다:

$$
\mathbf{u}_c(\mathbf{x}_j) = \mathbf{u}_s(\mathbf{x}_j) + \mathbf{u}_m(\mathbf{x}_j)
\tag{2.11}
$$

> (2.11) 보정 속도 = 샘플링 속도 + 누락 속도.

누락 속도는 ALM이 만든 와류와 "기준(reference)" 와류가 유도하는 속도의 차이다. 본 논문은 기준 와류를 **무한소 코어(infinitesimal core)를 가진 와류 필라멘트(특이 와류)**로 정의한다(Dağ·Meyer Forsting과 동일; Martínez-Tossas는 최적 ε로 만든 와류를 기준으로 삼음):

$$
\mathbf{u}_m(\mathbf{x}_j) = \mathbf{u}^{vi}(\mathbf{x}_j) - \mathbf{u}^{v}(\mathbf{x}_j)
\tag{2.12}
$$

> (2.12) 누락 속도 = 이상(특이) 와류 유도속도 uᵛⁱ − 유한 코어(ALM 생성) 와류 유도속도 uᵛ.

선형 근사에서 CFD 샘플링 속도는 국소 비교란 속도 U와 ALM 와류 유도속도의 합:

$$
\mathbf{u}_s(\mathbf{x}_j) \approx \mathbf{U}(\mathbf{x}_j) + \mathbf{u}^{v}(\mathbf{x}_j)
\tag{2.13}
$$

> (2.13) 샘플링 속도 ≈ 비교란 속도 + 유한 코어 와류 유도속도.

따라서 (2.11)에 대입하면:

$$
\mathbf{u}_c(\mathbf{x}_j) \approx \mathbf{U}(\mathbf{x}_j) + \mathbf{u}^{vi}(\mathbf{x}_j)
\tag{2.14}
$$

> (2.14) 보정 속도 ≈ 비교란 속도 + 이상 와류 유도속도 = **양력선 이론의 결과를 재현**. 단, U는 ALM 시뮬레이션에서 알 수 없으므로(검증용 단순 케이스 제외), uᵛ를 모델링해야 함(§3).

스미어링 보정 위에 양력선 한계 기반의 추가 보정(예: Dağ 2017이 결합한 decambering 보정)을 얹을 수 있으나, 본 논문은 양력선을 검증 기준으로 삼고 추가 보정은 범위 외로 한다.

### 2.3 반복(iterative) 스미어링 보정 — 기존 방식
누락 속도는 속박·후류 와류의 순환에서 계산되고, 순환은 다시 보정 속도(식 2.10)에서 계산된다. 이 순환 의존성 때문에 Dağ·Meyer Forsting은 반복을 사용했다(Meyer Forsting은 속도에, 본 논문 §5는 순환에 완화계수를 적용). **매 시간스텝마다의 반복 절차**:

1. 순환 분포 초기 추정(보통 직전 시간스텝 값)
2. 와류 시트 형성: 프리스크라이브(나선/말굽 와류 가정) **또는** free-vortex wake(CFD 속도 또는 CFD+자유와류 유도속도로 이류)
3. 각 제어점에서 누락 속도 u_m 계산 → 식(2.11)로 국소 보정 속도
4. 식(2.9)로 유효 받음각 계산
5. 익형 데이터에서 Cl 보간
6. 식(2.10)으로 새 순환 Γⁿᵉʷ 계산
7. 완화계수 r로 순환 갱신:

$$
\Gamma_j = r\,\Gamma_j^{new} + (1-r)\,\Gamma_j^{old}
\tag{2.15}
$$

> (2.15) 완화계수 r 기반 fixed-point 갱신. **이 r 튜닝이 반복법의 골칫거리(r→1 불안정, r 작으면 느림).**

8. 와류 시트가 국소 속도에 의존하면 (ii)부터, 아니면 (iii)부터 재시작. 수렴 기준 만족까지 반복. 힘은 수렴 후 계산(본 논문은 보정 속도로 계산; 양력 계산 속도 선택의 모호성은 2차 오차로 무시 가능).

---

## 3. 체적력이 만드는 와도와 누락 속도

### 3.1 체적력이 생성하는 와도
점성 무시·정상 유동의 와도 방정식:

$$
\mathbf{u}\cdot\nabla\boldsymbol{\omega} = \boldsymbol{\omega}\cdot\nabla\mathbf{u} + \nabla\times\mathbf{f}
\tag{3.1}
$$

> (3.1) 정상 와도 방정식.

균일 기저유동 U=const로 선형화:

$$
\mathbf{U}\cdot\nabla\boldsymbol{\omega} = \nabla\times\mathbf{f}
\tag{3.2}
$$

> (3.2) 선형화된 와도 방정식.

z축 정렬 균일 유동 U=(0,0,Uz), y방향 양력만 f=(0,fy,0)일 때:

$$
\frac{\partial\omega_x}{\partial z} = -\frac{\partial f_y}{\partial z}\ \Longrightarrow\ \omega_x = -\frac{f_y}{U_z}
\tag{3.3}
$$

$$
\frac{\partial\omega_z}{\partial z} = \frac{\partial f_y}{\partial x}\ \Longrightarrow\ \omega_z = \int_{-\infty}^{z}\frac{1}{U_z}\frac{\partial f_y}{\partial x}\,dz
\tag{3.4}
$$

> (3.3) 속박 와도 ωx, (3.4) 흘림 와도 ωz. ωx, ωz ≫ ωy≈0.

이상 집중력 $f_y^i = -F_l(x)/\rho\,\delta(y)\delta(z)$의 속박 와도:

$$
\omega_x^i = -\frac{f_y^i}{U_z} = \frac{1}{U_z\rho}F_l(x)\,\delta(y)\,\delta(z)
\tag{3.5}
$$

> (3.5) 양의 양력은 유체에 음의 힘을 가함에 유의.

스팬 방향 와도 적분 = 순환:

$$
\Gamma(x) = \int_{-\infty}^{+\infty}\!\!\int_{-\infty}^{+\infty}\omega_x^i\,dy\,dz = \frac{1}{U_z\rho}F_l(x)
\tag{3.6}
$$

> (3.6) Kutta–Joukowski와 일치.

### 3.2 불연속 힘 분포가 만드는 와도
ALM은 각 세그먼트 j에서 제어점 xj에서 힘을 계산하고 세그먼트 경계 xj⁻≤x≤xj⁺ 사이에서 일정하다고 본다:

$$
\mathbf{F}_{2D}(x) = \sum_{j=1}^{N}\big(H(x-x_{j-}) - H(x-x_{j+})\big)\,\mathbf{F}_{2D}(x_j)
\tag{3.7}
$$

> (3.7) 헤비사이드 계단함수(절반최대 관례)로 세그먼트별 일정 힘 표현.

$$
\mathbf{f}^i(x,y,z) = \sum_{j=1}^{N}\big(H(x-x_{j-}) - H(x-x_{j+})\big)\,\delta(y)\,\delta(z)\,\frac{\mathbf{F}_{2D}(x_j)}{\rho}
\tag{3.8}
$$

> (3.8) 이상 체적력(세그먼트 경계에서 불연속).

가우시안 컨볼루션 후:

$$
\mathbf{f}(x,y,z) = \sum_{j=1}^{N}\big(H_\varepsilon(x-x_{j-}) - H_\varepsilon(x-x_{j+})\big)\,\eta(y)\,\eta(z)\,\frac{\mathbf{F}_{2D}(x_j)}{\rho}
\tag{3.9}
$$

> (3.9) **본 논문이 시뮬레이션에 직접 사용하는 해석적 컨볼루션 식.** 저자들은 이것이 ALM 코드에서 해석적 컨볼루션을 구현한 최초 사례일 수 있다고 언급(대부분 코드는 수치적으로 컨볼루션 수행).

스미어된 헤비사이드(mollified Heaviside) 함수:

$$
H_\varepsilon(x) = \frac{\mathrm{erf}(x/\varepsilon) + 1}{2}
\tag{3.10}
$$

> (3.10) 오차함수 erf 기반 매끄러운 계단함수(Caprace et al. 2019).

양력만 고려한 속박 와도 (식 3.3에서):

$$
\omega_x^i = \sum_{j=1}^{N}\big(H(x-x_{j-}) - H(x-x_{j+})\big)\,\delta(y)\,\delta(z)\,\Gamma_j
\tag{3.11}
$$

$$
\omega_x = \sum_{j=1}^{N}\big(H_\varepsilon(x-x_{j-}) - H_\varepsilon(x-x_{j+})\big)\,\eta(y)\,\eta(z)\,\Gamma_j
\tag{3.12}
$$

> (3.11) 이상 속박 와도, (3.12) 스미어된 속박 와도. Γj=Fl(xj)/(Uzρ). 와도는 라인 법선 방향으로 가우시안 확산할 뿐 아니라 스팬 방향으로 라인 경계 바깥으로도 확산(Hε 항).

흘림 와도(식 3.4에서):

$$
\omega_z^i = -\sum_{j=1}^{N}\big(\delta(x-x_{j-})\delta(y)H(z) - \delta(x-x_{j+})\delta(y)H(z)\big)\Gamma_j
\tag{3.13}
$$

$$
\omega_z = -\sum_{j=1}^{N}\big(\eta(x-x_{j-})\eta(y)H_\varepsilon(z) - \eta(x-x_{j+})\eta(y)H_\varepsilon(z)\big)\Gamma_j
\tag{3.14}
$$

> (3.13) 이상 흘림 와도 = 세그먼트 경계 xj±에서 순환 불연속이 만드는 특이 반무한 와류 = **이산화 Prandtl 양력선**. (3.14) 스미어 흘림 와도 = xj±에 중심을 둔 반무한 Lamb–Oseen 와류 = **가우시안 코어를 가진 양력선**.

**핵심 동등성(equivalence)**: (3.11)·(3.13)의 와도는 "각 세그먼트 내부에서 Uz가 거의 일정"이라는 가정 하에 양력선 이론의 와도와 **동일**하다. 즉 ALM이 세그먼트 내부에서 순환을 일정하게 취급하면, Dağ & Sørensen(2020)의 보정은 선형 근사에서 양력선 이론과 동등하다. (만약 세그먼트 내부에서 비일정 순환을 쓰면 추가 항이 생겨 정확히 일치하지 않음 — 본 논문 범위 외.) 흥미롭게도 Martínez-Tossas & Meneveau(2019)의 이산화 구현은 와류를 세그먼트 경계가 아닌 **제어점**에 두므로 고전 양력선과 정확히 같지는 않다(매우 유사하지만).

속박 와도도 컨볼루션의 영향을 받지만(식 3.12), 대부분의 직선 날개·블레이드에서는 직선 속박 와류가 자기 자신에 속도를 유도하지 않으므로 힘에 영향이 없다. 다중 블레이드에서도 블레이드 간 거리가 ε의 수 배라 차이 없음(허브는 예외 가능성 있으나 순환 낮고 성능 영향 적음). → 속박 와도 컨볼루션 보정은 정당하게 무시.

### 3.3 스미어 와류 세그먼트의 유도속도 (정확한 해석식 — 본 논문 기여)
z방향으로 정렬되고 (x,y)=(0,0)에 위치한, zj⁻~zj⁺ 구간의 일정 순환 Γj 직선 와류 세그먼트:

$$
\omega_z^i = \delta(x)\delta(y)\big(H(z-z_{j-}) - H(z-z_{j+})\big)\Gamma_j
\tag{3.15}
$$

> (3.15) 직선 와류 필라멘트 세그먼트.

3D 가우시안 컨볼루션한 "스미어 와류 세그먼트"의 와도:

$$
\omega_z = \Gamma_j\int_{z_{j-}}^{z_{j+}}\eta_3(x,y,z-z')\,dz' = \eta(x)\eta(y)\big(H_\varepsilon(z-z_{j-}) - H_\varepsilon(z-z_{j+})\big)\Gamma_j
\tag{3.16}
$$

> (3.16) 스미어 와류 세그먼트의 와도(그림 2). Leonard(1980) 접근 사용.

Leonard(1980)에 의한 유도속도:

$$
\mathbf{u}^v(\mathbf{x}) = \frac{\Gamma_j}{4\pi}\int_{z_{j-}}^{z_{j+}}\frac{\hat{\mathbf{e}}_x\times(\mathbf{x}-\mathbf{x}')}{|\mathbf{x}-\mathbf{x}'|^3}\,g(|\mathbf{x}-\mathbf{x}'|)\,dz'
\tag{3.17}
$$

> (3.17) 스미어 와류 세그먼트 유도속도(Biot–Savart에 컷오프 함수 g 적용).

컷오프 함수:

$$
g(s) = 4\pi\int_0^s \eta_3(s')\,s'^2\,ds' = \mathrm{erf}\!\left(\frac{s}{\varepsilon}\right) - 2s\,\eta(s)
\tag{3.18}
$$

> (3.18) 3D 가우시안을 구면좌표 $s'=\sqrt{x^2+y^2+z^2}$로 적분한 컷오프(desingularization) 함수.

방위각 방향 유도속도(해석적 적분):

$$
u^v_\theta(r,z) = \frac{\Gamma_j}{4\pi}\, r\int_{z_{j-}}^{z_{j+}}\frac{\mathrm{erf}\!\big(\tfrac{\sqrt{r^2+(z-z')^2}}{\varepsilon}\big) - 2\sqrt{r^2+(z-z')^2}\,\eta\!\big(\sqrt{r^2+(z-z')^2}\big)}{(r^2+(z-z')^2)^{3/2}}\,dz'
$$
$$
= \frac{\Gamma_j}{4\pi}\big[\Phi(r,\,z-z_{j+}) - \Phi(r,\,z-z_{j-})\big]
\tag{3.19}
$$

> (3.19) 스미어 와류 세그먼트의 방위각 유도속도 = 두 끝점의 Φ 차이. **해석적 적분으로 근사 오차 제거.**

함수 Φ:

$$
\Phi(r,Z) = \frac{1}{r}\left[-\frac{Z}{\sqrt{r^2+Z^2}}\,\mathrm{erf}\!\left(\frac{\sqrt{r^2+Z^2}}{\varepsilon}\right) + \exp\!\left(-\frac{r^2}{\varepsilon^2}\right)\mathrm{erf}\!\left(\frac{Z}{\varepsilon}\right)\right]
\tag{3.20}
$$

> (3.20) **본 논문의 핵심 해석식.** 유한 길이 스미어 와류 세그먼트의 유도속도 커널(기존 연구의 근사식을 대체).
> **⚠ 정정(2026-06-26)**: 이전 전사는 2번째 항에 잉여 `Z/ε`가 있어 Z→±∞에서 발산했음. 원문 PDF p.11 확인 → 올바른 항은 `exp(-r²/ε²)·erf(Z/ε)` (위 식 반영). 구현 `smearing_correction.py:phi_smeared`가 Eq3.19 수치적분·반무한 극한(3.24)과 기계정밀도 일치 검증됨.

특이(이상) 와류의 Φⁱ (ε→0 극한):

$$
\Phi^i(r,Z) := \lim_{\varepsilon\to 0}\Phi(r,Z) = \frac{1}{r}\left[-\frac{Z}{\sqrt{r^2+Z^2}}\right]
\tag{3.21}
$$

$$
u^{vi}_\theta(r,z) = \frac{\Gamma_j}{4\pi}\big[\Phi^i(r,z-z_{j+}) - \Phi^i(r,z-z_{j-})\big]
\tag{3.22}
$$

> (3.21) 이상 와류 커널 = 고전 Biot–Savart(Katz & Plotkin 1991과 일치). (3.22) 특이 와류 세그먼트 유도속도.

**각 와류 세그먼트의 누락 속도**(본 논문이 실제 사용하는 핵심 보정식):

$$
u^m_\theta(r,z) = u^{vi}_\theta(r,z) - u^v_\theta(r,z)
$$
$$
= \frac{\Gamma_j}{4\pi}\Big\{\big[\Phi^i(r,z-z_{j+}) - \Phi^i(r,z-z_{j-})\big] - \big[\Phi(r,z-z_{j+}) - \Phi(r,z-z_{j-})\big]\Big\}
\tag{3.23}
$$

> (3.23) **누락 속도 = (이상 와류 − 스미어 와류) 유도속도.** 속박 와류와 후류 와류 모두에 (방향만 고려해) 동일하게 적용. **이 식이 Dağ·Meyer Forsting의 근사식과 다른 점이 본 논문의 정확도 향상 핵심.**

반무한 와류 극한(z=0, zj⁻=0, zj⁺→+∞):

$$
u^v_\theta(r,0) = \frac{\Gamma_j}{4\pi r}\left[1 - \exp\!\left(-\frac{r^2}{\varepsilon^2}\right)\right]
\tag{3.24}
$$

> (3.24) Martínez-Tossas & Meneveau(2019)가 쓴 반무한 와류 공식과 일치(Dağ·Meyer Forsting의 관계와도 일관). **이것은 Lamb–Oseen 와류의 방위 속도 형태** — 가우시안 코어가 유도속도를 약화시키는 정도를 정량화.

**기존 근사 대비 차이**: 식(3.23)은 Dağ·Meyer Forsting이 쓴 Φ 근사와 명백히 다르다. Meyer Forsting 공식은 반무한 와류에서 음/양 오차가 상쇄되지만 다른 와류 형상에서는 보장되지 않는다. Martínez-Tossas 공식을 회전 블레이드에 그대로 적용(예: Stanly et al. 2022)하면 나선 와류를 직선 말굽 와류로 모델링하는 오차가 생긴다. 그럼에도 기존 보정들이 잘 작동하는 이유: 누락 속도가 Φⁱ−Φ에 비례하는데, 블레이드 근처 와류(보정을 지배)에서는 Φⁱ≫Φ이므로 Φ의 정확한 값이 덜 중요(크기 차수만 맞으면 됨). 본 논문은 해석적 적분 식(3.20)으로 근사 오차를 아예 피한다.

### 3.4 와류 시트 형성 (free-vortex wake — 본 논문 기여)
와류 시트 형성은 ALM+보정과 양력선 이론 모두의 본질적 요소다. 단, 동일 절차로 시트를 만들어도 두 방법이 수학적으로 완전히 같지는 않다(ALM에서는 유도속도가 CFD 내부 스미어 와류 + 보정 와류 시트의 합으로 형성되므로). 이 효과는 §7.1 결과에서 관측되지만 무시할 만하다.

과거 연구들의 시트 형성:
- **Martínez-Tossas & Meneveau (2019)**: 직선 날개·균일 유동의 해석적 반무한 와류 기반이라 명시적 시트 불필요(단 Stanly et al. 2022가 회전 블레이드에 적용 시 나선을 말굽으로 암묵 모델링 → 오차).
- **Dağ & Sørensen (2020)**: 방출 위치의 국소 상대 유동각으로 각 나선 와류의 피치를 정해 나선 와류 부과.
- **Meyer Forsting et al. (2019a)**: 근후류 모델(Pirrung et al. 2016, 2017) 기반 고정 나선 후류 가정. 방출 와류 위치 기록(bookkeeping)은 한 인자에만 필요. Meyer Forsting et al.(2020)이 기록 필요성을 제거해 비용 절감(힘 영향 무시 가능).

**본 논문의 free-vortex wake** (Dağ & Sørensen이 제안한 전략 구현):
- 와류가 유동 속도로 이류된다고 가정. 추적 입자(tracing particles)로 와류 세그먼트 경계 위치 추종.
- 후류에 남아있는 **과거 순환 값을 직접 사용**(전체 후류를 현재 순환으로 두는 것보다 현실적).
- 단점: 이전 시간스텝 방출 와류의 순환·위치 기록 필요. 장점: ad hoc 가정 불필요 → 일반성 유지.
- 와류 위치 추정에 보정 속도 대신 **비보정 CFD 속도** 사용(차이는 2차; 보정 속도를 시트 전체에 쓰면 이상 와류 특이성으로 오차/불안정 위험). 보정 속도는 제어점에서만 계산.
- **비반복법에 중요**: 와류 시트를 보정 속도 계산 **전에 미리 이류** 가능 → 현재 순환이 시트 기하에 영향을 주지 않음(1차 근사와 일관).
- 1차 오일러 시간적분으로 추적 입자 이류. n−n_w보다 오래된 정보는 저장 안 함.
- 메모리 절약: 최근 n_nw 시간스텝 와류는 항상 유지. 추적 입자 거리가 d_w 미만이면 와류 융합(circulation은 평균). 팁 근처 최소 후류 길이 (n_w − n_nw − 1)·d_w 보장. **본 시뮬레이션: n_w=50, n_nw=10, d_w=ε/2 → 팁 근처 약 20ε 후류 길이.** NREL 5MW 파라미터 연구에서 n_w·n_nw를 2배로 해도 순환 영향 무시 가능.
- 비정상 시뮬레이션에서 순환 변화 시 스팬 방향 흘림 와도(Kelvin 정리의 시작 와류)가 생기지만, **준정상(quasi-steady) 접근**으로 이 항은 무시(가우시안 스미어링의 비정상 효과는 향후 연구).

---

## 4. ALM과 일관된 반복(iterative) 양력선

§2.1·§2.3의 ALM 정식화는 **비선형 양력선(non-linear lifting line)**(Anderson 1991; Phillips & Snyder 2000)과 매우 유사하다. 따라서 ALM+보정과 일관된 비선형 양력선을 만들 수 있다. Anderson(1991)은 비교란 균일 유입 U∞를 쓰지만, 본 논문은 식(2.10) 기반으로 **국소 속도**를 사용(Phillips & Snyder 2000과 유사). 또한 비교란 속도 U=(Ux,Uy,Uz)가 스팬을 따라 변할 수 있게 하여 **회전 블레이드 적용에 필수적**인 특징을 확보.

국소 속도:

$$
\mathbf{u}(\mathbf{x}_j) = \mathbf{U}(\mathbf{x}_j) + \mathbf{u}^{vi}(\mathbf{x}_j)
\tag{4.1}
$$

> (4.1) 국소 속도 = 비교란 속도 + 와류계 유도속도(이상 와류).

**비선형 양력선 단계**:
1. 순환 초기 추정(예: u=U로 식 2.10 적용)
2. 와류 시트 형성(프리스크라이브 또는 free-vortex)
3. 속박 와류+와류 시트가 각 제어점에 유도하는 속도 uᵛⁱ 계산 → 식(4.1)로 국소 속도
4. 식(2.9)로 유효 받음각
5. 익형 데이터에서 Cl 보간
6. 식(2.10)으로 새 순환 Γⁿᵉʷ
7. 완화계수 r로 갱신:

$$
\Gamma_j = r\,\Gamma_j^{new} + (1-r)\,\Gamma_j^{old}
\tag{4.2}
$$

> (4.2) 비선형 양력선의 완화 갱신(식 2.15와 동일 구조).

8. 와류 시트가 국소 속도에 의존하면 (ii)부터, 프리스크라이브면 (iii)부터 재시작. 수렴까지 반복.

이 반복 양력선이 §7.1에서 ALM 비교의 **기준(reference)**이다. 유도속도가 비교란 속도에 비해 작다고 볼 수 없는 경우도 다룰 수 있다(Kleine et al. 2023). 이 식들을 기반으로 **선형화 양력선(Appendix A)**을 유도하며, 이것이 비반복 보정의 토대다.

---

## 5. 선형화 양력선 기반 비반복(non-iterative) 스미어링 보정 ★핵심★

스미어링 보정(§2.3)의 단계는 비선형 양력선(§4)과 거의 같다. 차이: (iii)이 누락 속도 계산이 되고, (ii)가 CFD 데이터로 시트를 형성하도록 허용된다는 점. 따라서 Appendix A의 선형화 양력선 공식 기반으로 **비반복 보정**을 만들 수 있다.

반복 양력선과 스미어링 보정의 한 가지 차이: 보정은 직전 시간스텝에서 시작 가능(더 나은 순환 추정). 이 작은 차이가 **익형 Cl이 받음각에 대해 비선형이어도 선형화 비반복 정식화가 좋은 결과를 내는** 이유다.

### Phase 1 — 선형화 기준점(linearization point) 찾기
§2.3의 첫 세 반복 단계를 **딱 한 번만** 적용한다. 직전 시간스텝 순환 Γⁿ⁻¹로 계산한 보정 속도:

$$
\mathbf{u}_c(\mathbf{x}_j,\boldsymbol{\Gamma}^{n-1}) = \mathbf{u}_s(\mathbf{x}_j) + \mathbf{u}_m(\mathbf{x}_j,\boldsymbol{\Gamma}^{n-1})
\tag{5.1}
$$

> (5.1) 직전 순환으로 계산한 보정 속도(선형화의 기준점).

선형화 대상 변수를 †로 표기:

$$
u_y^\dagger := u_{cy}(\mathbf{x}_j,\boldsymbol{\Gamma}^{n-1}) = u_{sy}(\mathbf{x}_j) + u_y^m(\mathbf{x}_j,\boldsymbol{\Gamma}^{n-1})
\tag{5.2}
$$

$$
u_z^\dagger := u_{cz}(\mathbf{x}_j,\boldsymbol{\Gamma}^{n-1}) = u_{sz}(\mathbf{x}_j) + u_z^m(\mathbf{x}_j,\boldsymbol{\Gamma}^{n-1})
\tag{5.3}
$$

> (5.2)-(5.3) 기준점 속도 성분.

기준점에서의 받음각·상대속도·순환:

$$
\alpha_j^\dagger := \alpha_{gj} + \arctan\!\left(\frac{u_y^\dagger}{u_z^\dagger}\right),\qquad u_r^\dagger := \sqrt{u_z^{\dagger 2} + u_y^{\dagger 2}},\qquad \Gamma_j^\dagger := \frac{1}{2}u_r^\dagger c_j\, C_l(\alpha_j^\dagger)
\tag{5.4}
$$

> (5.4) 기준점에서의 받음각/상대속도/순환. **Cl을 이 기준점 주변에서 선형화한다.**

### Phase 2 — 누락 속도를 두 성분으로 분리 (그림 3)
free-vortex wake에서 시트 기하는 CFD 속도로 정해지고 현재 순환이 시트 기하에 영향을 주지 않는다(§3.4, 1차 근사와 일관). 현재 시간스텝 순환 Γⁿ에 대한 누락 속도를 둘로 나눈다:

$$
\mathbf{u}_m(\mathbf{x}_j,\boldsymbol{\Gamma}^n) = \mathbf{u}_{mp}(\mathbf{x}_j,\boldsymbol{\Gamma}^n) + \mathbf{u}_{mc}(\mathbf{x}_j,\boldsymbol{\Gamma}^n)
\tag{5.5}
$$

> (5.5) u_mp = 이전 시간스텝들이 방출한 와류 시트의 유도속도(현재 순환과 무관 → 이미 계산됨). u_mc = 현재 시간스텝 방출 시트 + 속박 와류의 유도속도.

속도-순환 선형성으로 재작성:

$$
\mathbf{u}_m(\mathbf{x}_j,\boldsymbol{\Gamma}^n) = \mathbf{u}_{mp}(\mathbf{x}_j) + \mathbf{u}_{mc}(\mathbf{x}_j,\boldsymbol{\Gamma}^{n-1}) + \Delta\mathbf{u}_{mc}(\mathbf{x}_j,\Delta\boldsymbol{\Gamma}) = \mathbf{u}_m(\mathbf{x}_j,\boldsymbol{\Gamma}^{n-1}) + \Delta\mathbf{u}_{mc}(\mathbf{x}_j,\Delta\boldsymbol{\Gamma})
\tag{5.6}
$$

> (5.6) **핵심 분해(그림 3).** Δu_mc = u_mc(Γⁿ)−u_mc(Γⁿ⁻¹), ΔΓ = Γⁿ−Γⁿ⁻¹. 즉 누락 속도 = (직전 순환의 누락 속도, 이미 알고 있음) + (현재 시간스텝 와류계가 만드는 증분).

### 영향계수(influence coefficient) 행렬
제어점 j에서 세그먼트 k의 와류계(속박 와류 + 현재 시간스텝 시트)가 유도하는 속도:

$$
u_y^{mc}(\mathbf{x}_j,\Gamma_k^n) = a_{y,jk}^{mc}\,\Gamma_k^n,\qquad u_z^{mc}(\mathbf{x}_j,\Gamma_k^n) = a_{z,jk}^{mc}\,\Gamma_k^n
\tag{5.7, 5.8}
$$

> (5.7)-(5.8) 영향계수 a_mc — **반복법의 step(iii)에서 이미 알아야 하는 값**(즉 추가 계산 부담이 거의 없음).

행렬 형태:

$$
\mathbf{u}_y^{mc}(\boldsymbol{\Gamma}^n) = \mathbf{A}_y^{mc}\boldsymbol{\Gamma}^n,\qquad \mathbf{u}_z^{mc}(\boldsymbol{\Gamma}^n) = \mathbf{A}_z^{mc}\boldsymbol{\Gamma}^n
\tag{5.9, 5.10}
$$

> (5.9)-(5.10) 영향계수 행렬 A_mc(현재 시간스텝 와류계만).

누락 속도(증분형):

$$
\mathbf{u}_y^m(\boldsymbol{\Gamma}^n) = \mathbf{u}_y^m(\boldsymbol{\Gamma}^{n-1}) + \mathbf{A}_y^{mc}\Delta\boldsymbol{\Gamma}
\tag{5.11}
$$

> (5.11) 누락 속도 = 직전 값 + 영향계수×ΔΓ.

보정 속도:

$$
\mathbf{u}_y^c(\boldsymbol{\Gamma}^n) = \mathbf{u}_y^\dagger + \mathbf{A}_y^{mc}\Delta\boldsymbol{\Gamma}
\tag{5.12}
$$

$$
\mathbf{u}_z^c(\boldsymbol{\Gamma}^n) = \mathbf{u}_z^\dagger + \mathbf{A}_z^{mc}\Delta\boldsymbol{\Gamma}
\tag{5.13}
$$

> (5.12)-(5.13) 보정 속도 = 기준점 속도 + 영향계수×ΔΓ.

### 선형 방정식 (식 2.10의 선형화)
Appendix A의 단계를 따라 식(2.10)을 기준점 † 주변에서 선형화:

$$
\Gamma_j^n = \Gamma_j^\dagger + \frac{1}{2}c_j\left[\left(C_l(\alpha_j^\dagger)\frac{u_y^\dagger}{u_r^\dagger} + \frac{\partial C_l}{\partial\alpha_j}(\alpha_j^\dagger)\frac{u_z^\dagger}{u_r^\dagger}\right)\Delta u_y^{mc} + \left(C_l(\alpha_j^\dagger)\frac{u_z^\dagger}{u_r^\dagger} - \frac{\partial C_l}{\partial\alpha_j}(\alpha_j^\dagger)\frac{u_y^\dagger}{u_r^\dagger}\right)\Delta u_z^{mc}\right]
\tag{5.14}
$$

> (5.14) 순환의 1차 테일러 전개. **∂Cl/∂α(양력 곡선 기울기)가 등장 → 비반복법은 양력 곡선 기울기 값을 필요로 함.**

행렬 형태:

$$
\big[\mathbf{I} - \mathrm{diag}(\mathbf{b}_y^\dagger)\mathbf{A}_y^{mc} - \mathrm{diag}(\mathbf{b}_z^\dagger)\mathbf{A}_z^{mc}\big]\,\Delta\boldsymbol{\Gamma} = \boldsymbol{\Gamma}^\dagger - \boldsymbol{\Gamma}^{n-1}
\tag{5.15}
$$

> (5.15) **★최종 핵심 선형 시스템★.** by†·bz†는 (식 A5·A6의 † 버전). 이 N×N 선형계를 **한 번 직접 풀어 ΔΓ를 구하고**, 식(5.12)-(5.13)으로 보정 속도를 얻는다. **완화계수도, 반복도 없다.**

### 비반복법의 성질 (구현 관점 핵심)
- Cl이 받음각에 선형이면, 순환 차이가 커도 좋은 결과(고전 양력선이 완전히 틀린 초기값에서도 잘 수렴하는 것과 동일).
- 강점은 **직전 시간스텝의 첫 보정 반복에서 계산한 속도 주변에서 선형화**한다는 점. 대부분 시뮬레이션에서 속도·순환 차이가 작으므로 **Cl이 비선형이어도** 선형화가 정당화됨.
- 성능이 나쁠 수 있는 경우: **양력 곡선 기울기가 받음각에 따라 변하고(비선형) 시간스텝 간 유동 조건이 급변**할 때. 이 극단적 경우엔 본 선형법으로 반복법의 수렴을 **가속**하는 용도로 쓸 수 있음.
- 선형계 크기는 N으로, **CFD 솔버보다 수 차수 작아 계산 비용 무시 가능**.
- 다중 터빈: 거리가 ε보다 훨씬 크면 각 터빈을 독립적으로 풀 수 있음. 블레이드 간 거리가 ε보다 훨씬 크거나 허브 힘이 중요치 않으면 각 블레이드를 독립적으로 풀 수 있음(본 §7 결과엔 미적용).
- **범용성**: 시트 형성·유도속도 계산의 모든 측면이 A_y^mc·A_z^mc 행렬에 담긴다. 따라서 기존 연구들(Dağ, Martínez-Tossas, Meyer Forsting, Stanly)도 자기 방법에 맞는 A 행렬만 구성하면 본 절의 비반복 절차로 혜택을 볼 수 있다. Meyer Forsting et al.(2020)의 속도 향상 기법도 A 행렬 계수에만 영향을 주므로 쉽게 결합 가능.
- 프리스크라이브 시트 방법에서는 (5.5) 분리를 생략할 수도 있으나, 선형화 오차를 줄이려면 구현하는 게 좋음. 즉 비반복 전략은 **이전 순환 기록(bookkeeping) 없이도** 작동 가능.

---

## 6. 수치 방법 (Numerical method)

### 6.1 NS 솔버 + ALM
- 비압축성 3D NS를 **스펙트럴 요소 코드 Nek5000**(Fischer et al. 2008)으로 해석. 약형(weak form), 각 요소에서 기저·시험 함수 전개. 속도 공간은 **7차 라그랑주 다항식**(GLL 구적점), P_N−P_{N−2} 정식화(Maday & Patera 1989).
- 시간 이산화: 3차 implicit/explicit (BDF3/EXT3, Fischer 2003). 고차 모드 필터링으로 안정화(Fischer & Mullen 2001).
- **적응 격자 세분화(AMR)** + 스펙트럴 오차 지표 사용해 비용 절감. 단, 액추에이터 라인 주변은 오차 지표와 무관하게 **최대 분해능 강제**(선택한 ε에 대한 적절한 분해능 보장).
- ALM은 이전에 Nek5000에 Prandtl 팁 보정과 함께 구현·검증됨(Kleusberg et al.). 스펙트럴 요소법의 낮은 소산/분산 덕에 와류 안정성 연구에 적합.
- 반복 보정의 수렴 기준: $\|\boldsymbol{\Gamma}^{new} - \boldsymbol{\Gamma}^{old}\| / \|\boldsymbol{\Gamma}^{new}\| < 10^{-5}$ (유클리드 노름).
- 식(3.9)의 해석적 컨볼루션(세그먼트별 일정 힘) 구현.
- NREL 5MW: 원본은 일부 스팬 위치에만 익형 정의. 본 구현은 단면 간 계수를 **보간**(EllipSys3D 방식, Martinez-Tossas et al. 2018) → 힘 곡선 매끄럽게, 불연속 회피.
- **비반복법은 양력 곡선 기울기 값이 필요** → Cl을 3차 다항식 보간으로 기울기 불연속 회피. 표 익형 Cl은 형상 보존 piecewise cubic(shape-preserving PCHIP)으로 모델링.
- 경계조건: 유입·상·하·측면은 디리클레(Dirichlet), 출구는 자연 유출(natural outflow). (그림 4 도메인 개략.)

### 6.2 양력선 방법(검증 기준)
- §4의 비선형 반복 양력선을 z방향 정렬 **말굽 와류(horseshoe vortices)**로 구현. ALM과 동일 이산화.
- 양력선은 훨씬 빠르므로 더 엄격한 수렴 기준:

$$
\frac{\max_{1\le j\le N}|\Gamma_j^{new} - \Gamma_j^{old}|}{\frac{1}{N}\sum_{j=1}^{N}|\Gamma_j^{new}|} < 10^{-8}
\tag{6.1}
$$

> (6.1) 양력선 수렴 기준(제어점별 순환 차이가 평균 절대 순환의 10⁻⁸ 미만).

---

## 7. 결과 (Results)

### 7.1 양력선 방법과의 비교 — 평판 직선 날개(translating wing)
**일정 코드 평판 직선 날개, 균일 유동, 정상 상태**. 정상이므로 반복·비반복 차이 없음을 기대(결과로 확인). 완전성을 위해 둘 다 제시.

**표 1 (Table 1) — 날개 시뮬레이션 파라미터**:

| 항목 | 값 |
|---|---|
| 종횡비 R/c (Aspect ratio) | 10 |
| 기하 받음각 αg | 1/(2π) rad = 9.189° |
| 도메인 (Lx/R, Ly/R, Lz/R) | (12, 12, 12) |
| 유입 거리 Lzin/R | 6 |

> 표 1: 무차원화는 스팬 R, 무한 속도 U_ref=Uz, 밀도 ρ로. **이상 익형**(항력 없음, Cl=2πα). αg=1/(2π)는 2D에서 Cl(αg)=1, Γ0/(RUz)=0.05를 줌(유도속도 0인 경우). ALM Re_c=cUz/ν=10⁴(양력선엔 Re 비적용). 액추에이터 라인 영역 격자 간격 Δx=R/56. 비교 ε 두 값: **ε=3.5Δx=R/16**, **ε=7Δx=R/8**. (낮은 값은 통상 최소(2~3Δx)보다 크지만 와류 코어 분해능이 좋아 검증 기준으로 적합; 7Δx는 매우 큰 커널 평가용 참조.)

**그림 5 (Figure 5) — 평판 날개 결과 비교** (대칭이라 절반만 표시):
- 범례: LL=비선형 반복 양력선(§4), ALM_d=비반복 보정(§5), ALM_i=반복 보정(§2.3).
- (a) y방향 보정 속도 u_cy/Uz, (b) ALM과 LL의 y속도 차이, (c) z방향 보정 속도 u_cz/Uz, (d) z속도 차이, (e) 순환 Γ/(RUz), (f) 순환 상대 차이.
- **핵심 결론**: 모든 경우에서 일치 우수. **u_y 차이는 10⁻⁴(비교란 속도의 0.01%) 수준** — Dağ & Sørensen(2020), Meyer Forsting et al.(2019a)보다 **훨씬 우수**. 차이는 u_y의 제곱 차수(1차 방법과 일관 → 더 좋은 일치는 기대 안 함). **반복(ALM_i)과 비반복(ALM_d)은 실질적으로 동일.**
- 우수한 일치의 세 가지 이유: ① 식(3.9)로 힘-가우시안 컨볼루션을 해석적으로(세그먼트 일정 힘 = 양력선과 수학적 동등 케이스) 수행, ② 양력선을 ALM과 일관되게 신중히 구성, ③ §3.3의 스미어 와류 세그먼트 유도속도가 다른 이산화 전략보다 오차 작음.
- **두 ε 값(3.5Δx vs 7Δx) 차이 무시 가능** — 이것이 스미어링 보정의 목표(힘을 ε 변화에 둔감하게 만들고 양력선(ε→0 극한)에 근접). 흥미롭게도 큰 ε에서 순환이 양력선과 **더** 잘 맞음: (i) 와류 코어가 커서 와도 표현이 좋아 수치 오차 감소(Appendix B), (ii) 큰 ε일수록 결과가 시뮬레이션 데이터보다 보정에 더 의존(ε→∞이면 시뮬레이션 정보 없이 순수 보정 = 양력선과 동일). 단 ε 증가가 항상 이로운 건 아님(ε는 후류 등 다른 효과도 가짐).
- u_cz: 보정의 영향이 최소(누락 속도 u_z^m이 그림 5(d) 차이보다 최소 한 차수 작음). 차이는 주로 CFD에서 만들어진 와류 시트가 양력선의 처방 말굽 와류와 다르기 때문(자유 와류는 다운워시로 시트가 기울어 음의 z방향 속도 유도; 처방 말굽 와류는 z속도 유도 안 함). 즉 u_z(및 Γ) 차이는 부분적으로 양력선 구현의 한계 탓.
- 순환 상대 차이는 10⁻³ 수준(Γ0의 0.1%)으로 실용상 무시 가능.

**의의**: 스미어링 보정 개발 전엔 ALM이 거의 회전 블레이드 전용이었다(유도속도 부정확 → 평판 날개 양력 영향). 여기서의 우수한 일치는 **고정익/항공기 양력면**에도 ALM을 쓸 동기를 준다. ALM은 양력선을 CFD와 통합해 양력선보다 복잡한 구성·유동을 다루며, 벽 경계조건 CFD보다 격자 요구가 낮다. free-vortex 방법은 일반성을 유지하고 전통 와류 필라멘트 방법의 불안정 문제(Leishman 2000)도 없다.

### 7.2 비정상 로터 시뮬레이션 — NREL 5MW (전단 유입)
**NREL 5MW 풍력터빈에 수직 전단(shear) 유입**: 비교란 풍속이 수직(y)으로 변해 **블레이드 순환이 매 시간스텝 변하는** 조건. 즉 매 시간스텝에서 초기 추정 순환 ≠ 최종 순환(반복·비반복 모두). 출구 BC 안정성을 위해 도메인 축소(Lx/R=Ly/R=8), 유입속도 Uz=y/5+1을 항상 양수로 유지(Lzin/R=4, Lzout/R=6). 비교 연구이므로 도메인 축소가 분석에 영향 없음.

**표 2 (Table 2) — NREL 5MW 시뮬레이션 파라미터**:

| 항목 | 값 |
|---|---|
| 비교란 속도 U(x,y,z)/U_ref | (0, 0, y/(5R)+1) |
| 팁속도비 ΩR/U_ref (tip speed ratio) | 7.55 |
| 도메인 (Lx/R, Ly/R, Lz/R) | (8, 8, 10) |
| 유입 거리 Lzin/R | 4 |

> 표 2: 무차원화는 반경 R, 터빈 중심 속도 U_ref=U(0,0,0), 밀도 ρ로. 격자 간격 Δx=R/56(평판 날개와 동일). Re_R=RU_ref/ν=5×10⁴. 시간스텝 Δt=T/400 (T=회전 주기).

**그림 6 (Figure 6) — t=12T, 방위각 0°(x축 정렬) 블레이드의 반경 방향 분포**:
- (a) 양력 fl, (b) 항력 fd, (c) 순환 Γ/(RU_ref), (d) 순환 차이(기준 ALM_d ε=3.5Δx, 최대 순환으로 정규화).
- **핵심 결론**: 모든 경우 일치 매우 양호. **반복·직접법 실질 동일** — 같은 ε에서 순환 차이 < 10⁻⁵. **비반복 절차가 비정상 유동에서도 오차를 도입하지 않음을 확인.** 두 ε 값(3.5Δx vs 7Δx) 간 순환 차이는 ~1%(힘도 동일 차수) — Meyer Forsting et al.(2019b) 등 선행 연구에서도 관측된, ALM 근사 오차 차수의 차이(Martinez-Tossas et al. 2018 코드 비교에선 더 큰 차이 관측). 실용상 무시 가능.

**그림 7 (Figure 7) — 팁에 가장 가까운 제어점의 순환 시간 이력**:
- 데이터는 T/20 주기로 저장(실제 시간스텝보다 김). 처음 2점은 보정 끔.
- **핵심 결론**: free-wake 기록(bookkeeping)은 첫 시간스텝부터 시작하되 보정 적용은 t=T/20 이후 켬. 보정을 켠 직후 점에서도 반복·비반복 차이는 동일 차수(무시 가능).

**잔여 ε 효과**: ε 효과는 완전히 제거 불가. 보정은 블레이드 힘을 보정하는 게 목표지, 후류는 여전히 ε에 의존(특히 근후류, Meyer Forsting et al. 2019b). ALM 이산화가 매우 작은 코어 와류를 지원 못하므로 흘림 와도는 여전히 스미어됨 → ε에 따라 후류(블로키지 프로파일)가 달라지고, 속도 오차는 순환·힘에 1차 효과를 가짐.

---

## Appendix A. 선형화 양력선 (Linearized lifting line)

식(2.10)을 **비교란 속도(undisturbed velocity) 주변**에서 테일러 전개:

$$
\Gamma_j = \Gamma_{0j} + \frac{1}{2}c_j\left[\left(C_l(\alpha_{0j})\frac{U_y}{U_0} + \frac{\partial C_l}{\partial\alpha}(\alpha_{0j})\frac{U_z}{U_0}\right)u_y^{vi} + \left(C_l(\alpha_{0j})\frac{U_z}{U_0} - \frac{\partial C_l}{\partial\alpha}(\alpha_{0j})\frac{U_y}{U_0}\right)u_z^{vi}\right]
\tag{A1}
$$

> (A1) 순환의 비교란 속도 주변 테일러 전개. 식(5.14)의 모태(여기선 비교란 U, §5에선 기준점 † 주변).

$$
U_0 := \sqrt{U_z^2 + U_y^2}
\tag{A2}
$$

$$
\alpha_{0j} := \alpha_{gj} + \arctan\!\left(\frac{U_y}{U_z}\right)
\tag{A3}
$$

$$
\Gamma_{0j} := \frac{1}{2}U_0\, c_j\, C_l(\alpha_{0j})
\tag{A4}
$$

> (A2)-(A4) 기준 속도 크기/받음각/순환. ∂Cl/∂α(α0j)는 α0j에서의 양력 곡선 기울기.

민감도 계수(sensitivity coefficient):

$$
b_{yj} := \frac{1}{2}c_j\left(C_l(\alpha_{0j})\frac{U_z}{U_0} + \frac{\partial C_l}{\partial\alpha}(\alpha_{0j})\frac{U_y}{U_0}\right)
\tag{A5}
$$

$$
b_{zj} := \frac{1}{2}c_j\left(C_l(\alpha_{0j})\frac{U_y}{U_0} - \frac{\partial C_l}{\partial\alpha}(\alpha_{0j})\frac{U_z}{U_0}\right)
\tag{A6}
$$

> (A5)-(A6) y·z 속도 변화에 대한 순환 민감도(식 5.15의 by†·bz†의 비교란 버전).

$$
\Gamma_j = \Gamma_{0j} + b_{yj}\,u_y^{vi} + b_{zj}\,u_z^{vi}
\tag{A7}
$$

> (A7) 선형화 순환 = 기준 순환 + 민감도×유도속도.

영향계수(influence coefficient) — Biot–Savart로 와류 형상·상대 위치에만 의존:

$$
u_y^{vik}(\mathbf{x}_j) = a_{y,jk}\,\Gamma_k,\qquad u_z^{vik}(\mathbf{x}_j) = a_{z,jk}\,\Gamma_k
\tag{A8, A9}
$$

> (A8)-(A9) 세그먼트 k가 제어점 j에 유도하는 속도(공통 와류 형상 영향계수는 Katz & Plotkin 1991 참조).

전체 N개 와류계 합 → 행렬:

$$
\mathbf{u}_y^{vi} = \mathbf{A}_y\boldsymbol{\Gamma}
\tag{A10, A11}
$$

$$
\mathbf{u}_z^{vi} = \mathbf{A}_z\boldsymbol{\Gamma}
\tag{A12}
$$

> (A10)-(A12) 유도속도의 행렬 표현.

(A7)을 행렬로(◦는 원소별 곱, diag(b)는 b의 대각행렬):

$$
\boldsymbol{\Gamma} = \boldsymbol{\Gamma}_0 + \mathbf{b}_y\circ\mathbf{u}_y^{vi} + \mathbf{b}_z\circ\mathbf{u}_z^{vi} = \boldsymbol{\Gamma}_0 + (\mathrm{diag}(\mathbf{b}_y)\mathbf{A}_y)\boldsymbol{\Gamma} + (\mathrm{diag}(\mathbf{b}_z)\mathbf{A}_z)\boldsymbol{\Gamma}
\tag{A13}
$$

> (A13) 행렬 형태 순환 관계.

**최종 선형 시스템**:

$$
\big[\mathbf{I} - \mathrm{diag}(\mathbf{b}_y)\mathbf{A}_y - \mathrm{diag}(\mathbf{b}_z)\mathbf{A}_z\big]\,\boldsymbol{\Gamma} = \boldsymbol{\Gamma}_0
\tag{A14}
$$

> (A14) **선형화 양력선의 순환 직접 풀이(반복 불필요).** 식(5.15)와 동일 구조 — §5의 비반복 보정은 이것을 ΔΓ·기준점 † 버전으로 만든 것.

**중요한 일반성**: 영향계수만 절대 좌표계에서 정의 후 국소 좌표계로 변환하면 되며, **속도 크기·방향에 제약 없음**(고전 양력선 선형화가 요구하는 Uz=const, Uz≫Uy 가정 불필요). 따라서 **회전 블레이드(고정/회전 좌표계 모두)에 직접 적용 가능**. Uy=0, Uz=U∞=const, 이상 익형·말굽 와류 시 고전 양력선으로 환원:

$$
\frac{1}{2\pi}(\mathrm{diag}(\mathbf{c}/2))^{-1}\boldsymbol{\Gamma} + \mathbf{A}_y\boldsymbol{\Gamma} = -U_\infty\boldsymbol{\alpha_g}
\tag{A15}
$$

> (A15) 고전 양력선으로 환원(Katz & Plotkin 1991 식 8.11과 각 항 대응).

$$
\frac{\Gamma(x)}{2\pi\,\tfrac{c(x)}{2}} - \frac{1}{4\pi}\int_{x_{min}}^{x_{max}}\frac{-\partial\Gamma(x')/\partial x'}{x - x'}\,dx' = -U_\infty\,\alpha_g
\tag{A16}
$$

> (A16) Katz & Plotkin 고전 양력선 적분 방정식(본 논문 표기로 변환).

---

## Appendix B. 가우시안 와류 코어의 분해능 (Resolution of the Gaussian vortex core)

최소 ε는 보통 ε≈2Δx(Troldborg 2009; Martínez-Tossas et al. 2015). 그러나 **ε=2Δx에서는 가우시안 와류 코어가 격자에서 잘 표현되지 않아** 오차 발생(§3 이론은 와도 가우시안 분포의 완벽한 표현 가정).

§7.1 케이스를 ε=2Δx=R/28로 두 격자에서 비교:
- **표준 격자(Δx=R/56=ε/2)**: u_y 차이 ~10⁻³(비교란 속도의 0.1%) — 여전히 문헌보다 우수하고 ALM에 일반적으로 허용 가능하나, §7.1의 큰 ε보다 명확히 악화.
- **세밀 격자(Δx=R/112=ε/4)**: 차이가 §7.1 수준(10⁻⁴)으로 복귀.

**그림 8 (Figure 8) — 동일 ε에서 격자 분해능 영향**:
- (a) y속도 차이, (b) z속도 차이, (c) 순환 상대 차이. 범례: ALM_d ε=R/28=2Δx vs ε=R/28=4Δx.
- **결론**: ε=2Δx의 u_y 오차는 주로 **격자 분해능**(ε 자체가 아님). u_z 차이는 이산화 오차 성분 + ε 관련 성분(특히 팁 근처, ALM·LL 정식화 차이 §7.1).

**그림 9 (Figure 9) — 대칭면(x=0) 속박 와류 영역**:
- u_y 의사색(pseudocolor) + 총 와도 등고선. (a) Δx=R/56=ε/2, (b) Δx=R/112=ε/4.
- **결론**: 거친 격자는 가우시안 와류 코어를 충분히 표현 못함 → 거친 격자의 큰 u_y 오차의 주원인. 이 오차는 Shives & Crawford(2013)가 이미 식별(받음각 오차 줄이려면 ε=4Δx 권장), Meyer Forsting & Troldborg(2020)·Meyer Forsting et al.(2019a)도 권장 최소보다 큰 ε 사용.

---

## 8. 결론 (Conclusions)

1. **선행 연구 평가**: Dağ & Sørensen(2020, 원 Dağ 2017)·Martínez-Tossas & Meneveau(2019, 원 Martínez Tossas 2017)의 스미어링 보정 착안, Martínez-Tossas & Meneveau의 형식적 분석, Meyer Forsting et al.(2019a, 2020)의 점성 양력선 유사성·개선은 ALM의 큰 진전이었다. 그러나 이들은 **완화계수 기반 반복**(느리고·덜 안정·덜 결정론적)에 의존했고, 비반복이었던 Martínez-Tossas & Meneveau는 매 시간스텝 순환-유도속도 양립성을 포기(비정상에서 오차 가능)했다.

2. **본 논문의 핵심 기여**: 양력선 선형화 기반 **비반복 와류 스미어링 보정**을 제안·검증. 반복 절차를 **작은 선형계 직접 풀이**로 대체. 시험 케이스에서 반복·비반복 결과 차이는 ALM 정확도보다 수 차수 작아 **유의미한 차이 없음**.

3. ε는 보정으로 힘 영향이 크게 줄지만 완전히 제거되지 않음(ALM 근사 오차 차수의 힘·순환 차이 잔존, 근후류에 영향). 실용상 무시 가능.

4. **추가 기여**: ① 스미어 와류 세그먼트 유도속도 기반 보정 함수(근사 불필요), ② CFD 속도 기반 free-vortex wake(일반성 유지, ad hoc 가정 불필요). 비용 우선 시 근사 함수·처방 후류·Meyer Forsting et al.(2020) 가속 기법 적용 가능. 본 연구는 기법 제시·검증에 집중하며, 비용 최적 파라미터 선택은 향후 과제.

5. **향후 과제**: blade-resolved 시뮬레이션과의 비교, 3D 항력 보정, 비정상 효과 이해(Kleine 2022).

6. **이론적 의의**: ALM과 일관된 비선형 양력선·ALM을 신중히 구성해 문헌보다 훨씬 작은 유도속도 차이 달성. 이는 선형화의 사후 정당화이며, **세그먼트 내부 순환이 일정하다고 가정하면 반복 양력선과 ALM+스미어링 보정이 수학적으로 동일**함을 증명한 것에 더해진다.

7. **응용 확장**: 일반화된 스미어링 보정은 ALM의 비회전 날개 한계를 제거해 양력선 결과를 재현. **항공 분야** 등 다른 커뮤니티가 양력선보다 복잡한 유동 조건에서 낮은 격자 요구로 날개를 해석하도록 동기 부여.

---

## ★ 구현 결정 요약 — "비반복(non-iterative) vs Dağ 반복(iterative)" ★

우리(ALM+MLG) 구현 시 어느 버전을 쓸지에 대한 직접 비교:

| 항목 | Dağ/Meyer Forsting 반복 (iterative) | 본 논문 비반복 (non-iterative, §5) |
|---|---|---|
| 풀이 구조 | 매 시간스텝 fixed-point 반복 + 완화계수 r | **N×N 선형계 한 번 직접 풀이** (식 5.15) |
| 완화계수 튜닝 | 필요 (r→1 불안정, r↓ 느림) | **불필요** |
| 결정성(determinism) | 수렴 횟수가 케이스 의존 | **결정론적**(고정 비용) |
| 안정성 | 완화계수 따라 불안정 가능 | 더 안정 |
| 비정상 양립성 | 유지(반복으로) | **유지**(Martínez-Tossas 가중평균과 달리 현재 Γ 반영) |
| 추가 입력 | Cl 테이블 | Cl 테이블 + **∂Cl/∂α(양력 곡선 기울기)** (식 5.14) |
| Cl 비선형 대응 | 자연스러움 | 직전 스텝 기준점 선형화로 양호(급변 시만 약점) |
| 추가 계산 부담 | — | **영향계수 A_mc는 반복법 step(iii)에서 어차피 계산하는 값** → 거의 무료 |
| 비용 | 반복 횟수×(유도속도 계산) | 선형계 풀이(CFD보다 수 차수 작아 무시 가능) |

**구현 난이도가 낮은 이유**:
1. 핵심 추가물은 식(5.15)의 **단일 선형계** 뿐 — `numpy.linalg.solve`로 N×N 풀면 끝(반복 루프·수렴 판정·완화계수 스케줄링 코드 불필요).
2. 행렬 A_y^mc·A_z^mc는 **기존 반복법이 이미 매 반복에서 계산하던 영향계수**(식 5.7~5.10)와 동일 → 새로 만들 코드가 거의 없음. 즉 "반복 루프를 직접 풀이로 교체"하는 수준의 변경.
3. by†·bz† 계수(식 A5/A6의 † 버전)는 국소 변수(Cl, ∂Cl/∂α, u_y†, u_z†, u_r†)만으로 구성 — 제어점별로 독립 계산.
4. **유일한 신규 요구사항은 ∂Cl/∂α** — Cl 테이블을 3차 다항/PCHIP로 보간해 기울기를 매끄럽게 얻으면 됨(우리 c81/csv 로더에 이미 보간 인프라 있음).
5. free-vortex wake 도입 시 와류 시트를 **보정 속도 계산 전에 미리 이류**할 수 있어(§3.4) 현재 순환이 시트 기하에 영향 안 줌 → 선형화 가정과 깔끔히 양립.

**권고**: 결정성·안정성·완화계수 제거·비정상 양립성·낮은 추가 구현 부담을 종합하면, 우리 ALM+MLG에는 **본 논문의 비반복 버전(식 5.15) 채택을 권장**한다. 단 ∂Cl/∂α를 매끄럽게 제공하는 익형 보간이 전제이며, 양력 곡선 기울기가 받음각에 따라 급변하고 시간스텝 간 유동이 급변하는 극단 케이스에서는 비반복법으로 초기화 후 반복법으로 마무리하는 하이브리드도 가능하다.
