# ALM-LBM 구현 vs. Watanabe(2026) / Natelson(2026) 엄밀 대조 감사

작성 근거 논문:
- `to_claude/ref_papers/alm/2026_watanabe_Real-time actuator line simulations of wind farm flows enabled by the lattice Boltzmann method and GPUs.pdf`
- `to_claude/ref_papers/alm/2026_Natelson_Advances_in_LB_Modeling_of_Proprotor-wing_aerodynamics.pdf`

대상 코드: `src/actuator/*`, `src/solver/simulation.py`, `configs/hvab/*`

---

## 0. 전제 — 두 논문의 ALM은 서로 다르다

| 항목 | Watanabe 2026 (HAWT 풍력단지) | Natelson 2026 (GT proprotor) |
|---|---|---|
| 격자/충돌 | D3Q27 **cumulant**, implicit LES (고차 cumulant ω=1) | D3Q19 **MRT**, explicit Smagorinsky (Cs=0.042) |
| 유속 샘플링 | actuator점 **trilinear (8점)** | 스테이션당 **20점 링 평균** |
| Gaussian | **등방** −F/(π^{3/2}ε³)·exp(−(d/ε)²), cutoff 3.5ε | **비등방** ε_c, ε_t, ε_r (Churchfield 2017) |
| C_L/C_D | C_L(α)만 (Re·Mach 무관) | 고정밀 CFD 테이블 (BET, quarter-chord) |
| α 부호 | α = φ − γ (**풍력터빈**) | BET |
| F_n/F_θ | F_n=Lcosφ+Dsinφ, F_θ=Lsinφ−Dcosφ (Eq.15-16) | BET |
| 팁/트림 | 없음 / 없음 | 없음 / collective 트림 |

우리 코드 대상은 HVAB **호버 로터(추력 생성)** → 항목마다 "어느 논문과 다른가"가 갈린다.

Watanabe 핵심 수식 (본문 대조 기준):
- Eq.8-9: u_n=u_z, u_θ=u_x cosθ+u_y sinθ
- Eq.10-12: u_rel=√(u_n²+(rω−u_θ)²), φ=arctan(u_n/(rω−u_θ)), α=φ−γ
- Eq.13-14: F_L=½ρu_rel²c_a C_L Δr, F_D=½ρu_rel²c_a C_D Δr (이산력, ×Δr)
- Eq.15-16: F_n=F_L cosφ+F_D sinφ, F_θ=F_L sinφ−F_D cosφ
- Eq.17-19: F_x=F_θ cosθ, F_y=−F_θ sinθ, F_z=F_n
- Eq.20-21: G(x)=Σ_j −F_j/(π^{3/2}ε³)·exp(−(d_j/ε)²), cutoff 3.5ε
- Eq.7: ρu=Σξf+G·Δt/2 (half-force)

---

## A. 논문과 일치하는 핵심 (기본/pure-ALM 경로)

- **F_L, F_D** — `actuator_line.py:1334-1336` `q=½ρu²`, `F_L=q·c·dr·C_L` = Watanabe Eq.13-14 (이산력 ×Δr, per-unit-length 아님) ✓
- **등방 Gaussian** — `spreading.py:171`, `interpolation.py:80` `1/(π^{3/2}ε³)·exp(−d²/ε²)` = Watanabe Eq.20 ✓ (Natelson 비등방과는 다름)
- **Newton 3법칙 부호** — `spreading.py:217-223` `F_grid += −F^AL·η` = Watanabe Eq.20의 −F_j ✓
- **half-force 커플링** — `simulation.py:460` `u += F/(2ρ)` = Watanabe Eq.7 (Guo/속도-shift, Δt=1) ✓
- **격자/충돌** — `_hvab_hover_base.py:280` `D3Q27+cumulant` = Watanabe 계열 ✓

---

## B. 의도적·타당한 차이 (HVAB에 맞춘 것 — 버그 아님)

### B-1. 프로펠러(Leishman) 부호 규약 vs Watanabe 풍력터빈 규약 ★핵심
- α: 우리 `α = twist − φ` (`rotor.py:500`) ↔ Watanabe `α = φ − γ` (Eq.12)
- F_n/F_θ: 우리 `F_n=L cosφ − D sinφ`, `F_θ=L sinφ + D cosφ` (`actuator_line.py:1349-1350`)
  ↔ Watanabe `F_n=L cosφ + D sinφ`, `F_θ=L sinφ − D cosφ` (Eq.15-16)

드래그 항 부호가 반대. 추력-생성(프로펠러/헬기) vs 에너지-추출(풍력터빈)의 물리적 방향 차이에서 나오는 규약. HVAB 호버 로터에는 **우리 규약이 옳다** (`coeff_mode:"rotorcraft"`, CT=GT Fluent 일치가 확증). → 차이지만 정상. 단 D-1 결함 있음.

### B-2. Mach 의존 폴라 (C81 Mach-pass)
`_lookup_cl_cd` + `M=u_rel·(dx/dt)/a` (`actuator_line.py:1256`). Watanabe C_L(α)만, Natelson Re/Mach 무관 테이블. 우리는 C_L(α,Re,Mach) — 천음속 HVAB 팁(M~0.6)에 필요. **양 논문 대비 추가.**

### B-3. Gaussian-가중 체적평균 샘플링
기본 `Σu·η/Ση` (±3ε 정규화 Gaussian, `interpolation.py:99`) — Watanabe trilinear(8점), Natelson 20점 링평균 어느 쪽과도 다른 제3의 방식(Sørensen-Shen 계열). `sampling:"point"`로 trilinear 전환 가능(`interpolation.py:930`). 팁 인플로우에 영향.

---

## C. 확인/주의가 필요한 설정 차이

- **C-1. Prandtl 팁/루트 손실이 HVAB base 기본 ON** (`_hvab_hover_base.py:208,324` `prandtl_loss=True`).
  두 논문 모두 팁보정 없음. ALM은 팁와류를 이미 해상 → Prandtl 손실은 **이중계산 위험**(알려진 함정).
  `watanabe`/`prtipR`/`epscorr` 변형은 False로 꺼둠 — 이 방향이 논문 정합. ★가장 중요.
- **C-2. 명시적 dynamic Smagorinsky SGS ON** (`_hvab_hover_base.py:337`).
  Watanabe는 implicit LES(명시 SGS 없음) 명시 선언. cumulant 위 dyn_smag는 Watanabe와 다름(Natelson Smagorinsky에 가까우나 우리는 dynamic).
- **C-3. ε 정의**: `ε=max(chord/4, 2Δx)` (chord 비례, `blade.py:426`) ↔ Watanabe 균일 ε=2.5Δx. 둘 다 권장범위나 정의 다름(주석의 "Watanabe Eq.13" 귀속 부정확).
- **C-4. cutoff n_cut=3.0** (`create_..._from_config:1875`) ↔ Watanabe 3.5ε.
- **C-5. 두 논문에 없는 추가 옵션** (전부 기본 OFF, C-1 예외): Merabet 반경절단+재정규화(`spreading.py:74`), 스윕보정(cosΛ), Dağ/Kleine smearing, free-wake.

---

## D. 실제 결함 (수정 권장)

- **D-1. `_compute_bem_forces` 상단 docstring 부호 오기** (`actuator_line.py:1176-1177`):
  Watanabe 터빈 부호(`F_n=FLcosφ+FDsinφ`)를 적어놨으나 실제 코드는 Leishman 프로펠러 부호(1349행; 1343행 주석은 올바름). 같은 함수 안 두 주석이 모순 → 물리 정상, **문서 버그**.
- **D-2. `decompose_velocity` 규약 주석** (`coordinates.py:581-594`):
  "Watanabe는 radial 투영"이라 적었으나 Watanabe u_θ는 표기상 tangential이 맞고 우리 tangent 투영이 물리적으로 정확. 순수 축류 동일, swirl에서만 미세차. 물리 정상, 주석 표현만 오해 소지.

---

## 결론

- **pure-ALM 경로**의 힘 계산·spreading·LBM 커플링은 Watanabe와 수식적으로 일치. `hvab_hover_c10_fine_watanabe_nasa.py`가 충실 재현판.
- **의도적 차이**(B): 프로펠러 부호규약, Mach 폴라, Gaussian 샘플링 — HVAB에 타당하나 논문과 다름.
- **프로덕션에서 실제 갈라지는 지점**(C): ①Prandtl 팁손실 기본 ON, ②명시 dyn-Smagorinsky, ③ε=chord/4, ④cutoff 3.0 — 특히 ①②는 두 논문 방법론과 정면으로 다름.
- **수정 대상**(D): 부호 docstring 오기 1건.

가장 신경 쓸 것 하나: **C-1 (Prandtl 팁손실 이중계산)** — 팁 과부하 조사와 직결.
