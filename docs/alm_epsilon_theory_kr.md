# ALM과 ε(epsilon) — 이론 학습 노트

> 우리 LBM 솔버의 Actuator Line Method(ALM)에서 ε이 정확히 무엇인지, 왜 그렇게 정해지는지,
> 그리고 LBM과 어떻게 맞물리는지를 정리한 노트. Stage B(팁 테이퍼) 작업의 배경 이론이기도 함.
> 코드 위치는 본문 끝 "코드 맵" 참고.

---

## 0. 한 문장 요약

> **ε은 "선(line)으로 추상화한 블레이드의 공기력을 격자에 퍼뜨리는 Gaussian 구름의 폭"이고,
> 물리 제약(≈0.25c)과 수치 제약(≥2Δx) 사이에서 정해지는 정규화 길이 스케일(regularization
> length scale)이다.**

---

## 1. 왜 ALM인가 — 블레이드를 "안 그린다"

날개/블레이드를 다루는 두 가지 방식:

```
(A) Body-fitted (직접 해상)            (B) Actuator Line (ALM)
   ┌─────────────┐                        · · · · · · · ·  ← marker(점)들이 늘어선 "선"
   │  격자가 표면을 │                       (블레이드 형상은 격자에 없음)
   │  감싸야 함     │                       각 점에서 익형표(C_L,C_D)로 힘 계산
   └─────────────┘                        → 그 힘을 유체에 body force로 주입
   경계층까지 해상 → 셀 폭발              격자는 블레이드를 "느끼기"만 함
```

로터는 얇고(코드 수 cm) 길고(반경 수 m) 빠르게 회전 → body-fitted로 표면+경계층을 해상하면
격자수가 폭발한다. ALM은 블레이드 **형상**을 버리고 블레이드가 유체에 주는 **힘**만 모델링한다.

우리 코드에서 그 "점"이 **marker**이고, 각 marker는 익형 속성을 들고 있다:
`marker_r`(반경 위치), `marker_chord`(코드 길이), `marker_twist`(비틀림각), `marker_epsilon`(ε).

---

## 2. ALM 한 타임스텝의 흐름

```
   ┌─ 격자(LBM)의 속도장 u(x) ──────────────────────────────┐
   │                                                          │
   │  (1) 샘플링: 각 marker 위치의 유속을 Gaussian(ε)로 보간   │
   │         u_marker_j = Σ_x u(x) · η_ε(|x - x_j|) · Δx³      │
   │                                                          ▼
   │                                            (2) BEMT 힘 계산 (marker별)
   │                                              속도삼각형: u_n, u_θ → u_rel, φ
   │                                              받음각 α = twist − φ
   │                                              C_L,C_D = polar(α, Re/Mach)
   │                                              F_L,F_D → F_n(추력), F_θ(토크)
   │                                                          │
   │  (4) Guo forcing으로 LBM에 주입            (3) Spreading: 그 힘을 격자로 퍼뜨림
   │      body_force → collision  ◀───────────   F_grid(x) = −Σ_j F_j · η_ε(|x − x_j|)
   └──────────────────────────────────────────────────────────┘
```

(1)과 (3)이 **같은 Gaussian 커널 η_ε** 을 쓰는 게 핵심 — 뒤에서 설명.

---

## 3. ε의 정의: 정규화 Gaussian 커널

힘을 한 격자점에 그대로 꽂으면 국소 속도 스파이크 → 수치 폭발 + 격자 의존성. 그래서 힘을
**주변 셀로 부드럽게 분산**한다. 분산 모양이 3D 등방 Gaussian이고 그 폭이 ε:

```
            1                ⎛   d²  ⎞
 η_ε(d) = ───────────  · exp ⎜ − ─── ⎟        d = |x − x_j|  (marker→격자점 거리)
          π^{3/2} ε³         ⎝   ε²  ⎠
          └────┬────┘
         정규화 상수: 공간 전체 적분 = 1
         → 퍼뜨려도 "총 힘 보존"
```

유체에 들어가는 body force(반작용, 뉴턴 3법칙이라 −부호):

```
 F_grid(x) = − Σ_j  F_j^AL · η_ε(|x − x_j|)        [lattice force / lu³]
```

**직관 (ε의 의미):**

```
 작은 ε                     큰 ε
  ▲ 힘밀도                   ▲ 힘밀도
  │  █                       │   ▄▄▄
  │  █                       │ ▄█████▄
  │ ███                      │▄███████▄
  └──┴──► x                  └────┴────► x
 좁고 뾰족 (몇 셀에 집중)    넓고 완만 (많은 셀에 옅게)
```

같은 **총 힘**이라도 분포 모양이 달라진다. ε = "얼마나 넓게 번지는가".

차원 확인: `1/(π^{3/2} ε³)` = [1/lu³], `× F_j` [lattice force] = **[lattice force / lu³]** =
힘 **밀도**. LBM body force가 정확히 이 단위라 바로 들어간다.

---

## 4. 샘플링 ↔ Spreading 대칭성 (왜 같은 ε인가)

- **Spreading**: 점의 힘 → 격자 (위 식).
- **Sampling**: 격자의 속도 → 점. `u_marker_j = Σ_x u(x) · η_ε(|x−x_j|) · Δx³`.

둘이 **같은 커널·같은 ε**이어야 운동량 교환이 일관된다. "내가 힘을 준 만큼의 영역에서 속도를
읽는다." 서로 다르면 인위적 운동량 생성/소실이 생긴다. 그래서 우리 코드는 sampling·spreading이
**하나의 `marker_epsilon` 배열**을 공유한다 (`get_all_marker_epsilon`이 양쪽에 같은 배열 공급).

---

## 5. ε은 얼마? — 양쪽에서 조이는 trade-off

| ε이 너무 **작으면** | ε이 너무 **크면** |
|---|---|
| 힘이 너무 적은 셀에 집중 | 과확산(over-smear) |
| 격자가 표현 못 함(셀보다 작은 구조) | 속도 구배가 뭉개짐 |
| aliasing·국소 u 스파이크 → **불안정** | **유도속도(다운워시) 과소** → 추력 과대 |

그래서 우리 `set_lattice_spacing`의 공식은 두 기준의 `max`:

```
 ε = max( chord/4 ,  2·Δx )
        └───┬───┘   └──┬──┘
       물리 기준      수치 하한(floor)
       (≈0.25c)      (격자 표현 한계)
```

- **chord/4 = 0.25c (물리):** Martínez-Tossas 등이 보인 "최적 ε ≈ 0.2~0.25c". 이 폭일 때
  퍼뜨린 양력 분포가 **실제 익형의 bound circulation**과 가장 잘 맞는다. 익형 코드(chord)에
  비례하는 게 물리적으로 자연스럽다 (큰 코드 = 두꺼운 양력 분포).
- **2·Δx (수치 하한):** Δx의 약 2배보다 좁으면 격자가 Gaussian을 못 그린다. 그 아래로는 못
  내려가게 막는다.

→ "물리적으로 원하는 폭, 단 격자가 표현 가능한 최소 이상."

**이게 우리가 본 현상의 정확한 이유다:**
- **D=16 smoke**: `chord/4 = 0.667 < 2.0` → `2Δx` floor가 이김 → ε 균일 2.0 → 팁 테이퍼 no-op.
- **D32 production**: `chord/4 = 5.3 > 2.0` → 물리 기준이 이김 → ε = 5.3 → 테이퍼 작동.

---

## 6. 왜 "유한 ε"이 유도속도를 못 살리나 (추력 과대의 근원, Stage B/C의 핵심)

이상적 lifting line 이론에선 블레이드 양력 분포 Γ(r)가 **trailed vortex(흘러나가는 와류)**를
만들고, 그 와류가 **다운워시(induced downwash)** = 아래로 부는 유도속도를 만든다. 이 다운워시가
블레이드가 보는 **유입각 φ**를 키워 **받음각 α = twist − φ**를 줄인다 → 팁에서 양력이 자연스레
roll-off(감소).

```
 실제(이상적):  팁에서 trailed vortex 강함 → 다운워시 큼 → φ 큼 → α 작음 → 팁 양력 ↓ (roll-off)

 유한 ε ALM:    와류가 ε만큼 뭉개짐(over-smear) → 다운워시 결손 → φ→0 → α≈기하각 → 팁 과하중 ↑
                                                              └──────────┬──────────┘
                                                            우리가 진단한 추력 +8~17% 과대의 원인
```

ε이 클수록(팁 와류 대비) 유도속도 누락이 커지고, **팁에서 최악**이다(φ→0). 이건 **버그가 아니라
smeared-ALM의 모델 한계**다 (우리 진단 + 독립 BEMT 비교에서 확인).

**고치는 두 갈래:**
- **Stage B (지금 작업)**: 팁 쪽 ε을 floor까지 좁혀 와류를 덜 뭉갠다. 빠르고 부분적.
- **Stage C (보류)**: 누락된 유도속도를 해석적으로 되더한다 (JFM-2019, filtered lifting-line).
  정석이지만 hover 타당성 검토 필요.

---

## 7. LBM과의 관계 (핵심)

### 7.1 LBM 한 줄 복습

LBM은 N-S를 직접 풀지 않고, 격자 위 **분포함수** `f_i(x,t)` (속도 방향 `c_i`, D3Q27이면 27방향)를
진화시킨다. 거시량은 모멘트로 얻는다:

```
 ρ   = Σ_i f_i           (밀도)
 ρu  = Σ_i c_i f_i       (운동량)
 p   = c_s² ρ            (상태방정식 — 약압축성)
```

매 스텝 두 과정:
1. **Collision(충돌)**: `f_i`를 평형 `f_i^eq(ρ,u)`로 완화(relax). 우리는 **cumulant** 충돌
   (BGK보다 고차 모멘트 안정성↑, 저점성에 유리 → 고-Re·와류 보존에 좋음).
2. **Streaming(이류)**: `f_i`를 `c_i` 방향 이웃 셀로 이동.

**약압축성 제약(중요):** LBM은 음속이 격자로 고정(`c_s = 1/√3`)된 weak-compressibility 솔버.
격자 속도는 작아야 한다 (`|u_lattice| ≲ 0.1`). 넘으면 비물리적 압축성·불안정.

### 7.2 ALM 힘이 LBM에 들어가는 길 — **Guo forcing**

body force `F`를 LBM에 넣는 표준 = **Guo forcing scheme** (2차 정확). 우리 cumulant 커널이 이걸 한다:

```
 (a) 속도 보정:   u ← u + F/(2ρ)          ← 평형 f_i^eq를 이 보정된 u로 계산
 (b) source 항:   collision에 S_i 추가     ← F를 f_i 분포에 second-order로 주입
```

전체 연결:

```
 ALM markers ─spread(ε)→ F_grid [lattice force/lu³]
        → simulation.body_force
        → cumulant collision 안의 Guo forcing { u+=F/2ρ ,  +S_i }
        → 다음 스텝 u(x)가 블레이드 힘을 반영
        → 그 u(x)를 다시 marker에서 sampling … (루프)
```

즉 **ε이 만든 force 밀도장이 곧 LBM의 body_force**다. ε이 그 밀도장의 "공간 분포 모양"을
결정하므로, ε은 LBM 입장에서 "운동량 소스를 얼마나 넓게 뿌릴지"를 정하는 값.

> 주의: 코드의 `mem_force_*` 커널은 **벽면(HWBB/IBB)의 momentum-exchange 힘 측정**용으로 ALM과
> 무관하다. ALM 주입 경로는 위의 `_compute_body_force` → Guo forcing.

### 7.3 왜 LBM에서 특히 ε ≥ 2Δx floor가 중요한가

세 가지 LBM 고유 이유:

1. **격자 표현성**: LBM 격자는 균일 직교, lattice units에서 Δx=1. ε<2면 Gaussian이 1셀에
   몰려 격자점으로 못 그린다 → 운동량이 불연속적으로 주입.
2. **약압축성 = 음향 노이즈**: 힘을 너무 좁게(급하게) 주면 LBM이 그걸 **압력파(음파)**로
   방사한다 (약압축성 솔버라 예민). 넓게 펴면 매끄럽게 흡수.
3. **속도 상한**: 좁은 ε → 국소 `u` 스파이크 → `|u|>0.1` 초과 → 불안정/NaN. 펴면 국소 u가
   유계로 유지. (그래서 ramp_steps로 초기 힘을 서서히 키우는 보완책도 있음.)

### 7.4 lattice units에서의 ε, 그리고 MLG

ε은 물리 길이지만 코드는 **lattice units**로 환산해 쓴다: `ε_lu = ε_phys / Δx`.

- `set_lattice_spacing(dx=1.0)`이 권위 있는 계산 — 이미 lattice units라 chord도 lu, floor도
  `2·1 = 2.0 lu`.
- **MLG(multi-level grid) fine-level ALM**: ALM은 hub 근처 **가장 fine한 레벨**에 붙는다.
  레벨 k는 `Δx_fine = Δx_L0 / 2^k`로 더 촘촘 → 같은 물리 chord가 **더 많은 셀**에 걸침 →
  `chord_lu` 큼 → `chord/4`가 floor를 이김 → 테이퍼가 작동.
  - 레벨별 재계산: `ε_fine = max(c_fine/4, 2·Δx_fine)` (각 레벨 lattice units).
  - 그래서 D=16 2-level smoke는 floor에 걸리고(테이퍼 no-op), D32 4-level production은
    안 걸린다(테이퍼 작동). **§5·§6에서 본 차이가 바로 이 레벨 해상도 문제.**

---

## 8. 우리 코드 맵 (어디서 무엇을 보나)

| 개념 | 파일 / 함수 |
|---|---|
| ε 계산 (권위) | `src/actuator/blade.py::set_lattice_spacing` (`max(c/4, 2dx)` + Stage B 분기) |
| Gaussian spreading | `src/actuator/spreading.py` (CPU 루프 + GPU `_SPREAD_KERNEL_SRC`, `epsilons[m]`) |
| Gaussian sampling | `src/actuator/interpolation.py::gaussian_kernel_3d` |
| marker별 ε 집계 | `src/actuator/rotor.py::get_all_marker_epsilon` (blade[0] 타일링) |
| BEMT 힘 / 속도삼각형 | `src/actuator/actuator_line.py::_compute_bem_forces`, `rotor.py::compute_relative_velocity` |
| LBM Guo forcing | `src/solver/simulation.py::_compute_body_force` → `src/kernels/cumulant_d3q27.py` (`u+=F/2ρ`, `S_i`) |
| fine-level ALM(MLG) | `src/solver/setup.py` (`ε_fine = max(c_fine/4, 2Δx_fine)`) |
| 진단 출력 | `blade_diagnostics/*.csv`의 `eps_lu`, `blade_geometry.csv`의 `epsilon_lu` |

---

## 9. Stage B(팁 테이퍼)는 이 이론에서 어디?

- §6의 "유한 ε → 팁 다운워시 결손"을 **팁 쪽 ε만 floor로 좁혀** 부분 완화.
- 공식: `r/R ≥ epsilon_taper_start`부터 선형으로 `ε = (1−t)·max(c/4,2dx) + t·max(factor·2dx, 2dx)`.
- 안쪽 스팬은 `0.25c` 유지(거긴 단면 유도가 이미 잘 맞음 = Martínez-Tossas 최적), 팁만 손봄.
- **검증 가능 신호**: 팁 ε↓ → 팁 와류 또렷 → 다운워시 회복 → 팁 φ 상승 → 팁 과하중↓ → C_T↓
  (실험 0.00473 쪽으로). `spanwise_post.py`로 `eps_lu`·`phi`·`M2CL`를 r/R 대비 확인.

---

## 10. 더 읽을거리

- **Sørensen & Shen (2002)** — ALM 원전 (Gaussian projection 도입).
- **Martínez-Tossas et al.** — 최적 ε ≈ 0.2~0.25c, filtered actuator line.
- **Martínez-Tossas & Meneveau, JFM 2019** — filtered lifting-line 보정(Stage C 후보, 1D
  spanwise, wake 모델 불필요). `to_claude/ref_papers/`.
- **Diaz (2023) §2.1.4** — 가변 ε 팁 테이퍼(Stage B 근거).
- **Dağ & Sørensen (2020)** — helical wake 기반 보정(BVI 포착, but 무겁고 hover 2-rev 절단 부정확).
- **Geier et al.** — cumulant LBM 충돌 모델(우리 collision).
- **Guo, Zheng & Shi (2002)** — LBM forcing scheme(우리 body force 주입).

---

### 부록 A — 자주 헷갈리는 점

- **ε ≠ 격자 해상도.** ε은 "힘을 퍼뜨리는 물리적 폭", Δx는 "셀 크기". 단 ε은 Δx보다 충분히
  커야(≥2) 격자가 표현 가능. 둘은 `chord_lu = chord/Δx`를 통해 연결.
- **ε ≠ 코드(chord).** 보통 ε = 0.25·chord로 비례시키지만 같은 양은 아님. floor에 걸리면
  chord와 무관하게 2Δx.
- **테이퍼가 thrust를 "직접" 줄이는 게 아니다.** ε을 바꿔 **유도속도장**을 바꾸고, 그게
  α를 바꿔, 그 결과 양력·추력이 바뀐다. 항상 유동을 거쳐 간접적으로 작용.

---

### 부록 B — marker 위치와 ε의 전달 반경 (Q&A)

**Q1. marker가 격자점 위에 있어야 하나? → 아니다.**

`spreading.py`에서 `marker_pos`는 **연속(float) 좌표** `[lu]`이고, 커널은 **정수 격자점에서
실수 marker까지의 거리**로 평가한다:

```python
dx = gx - marker_pos[0]   # gx = 정수 격자좌표, marker_pos[0] = 실수 (예: 37.42)
d_sq = dx*dx + dy*dy + dz*dz
eta[mask] = norm * np.exp(-d_sq / eps_sq)
```

```
 격자점:   ──┼─────┼─────┼─────┼──     marker(×)는 셀 사이 임의 위치에 있을 수 있음
            34    35    36 ×  37        d = |격자점 − 37.42| 의 분수 거리로 Gaussian 평가
```

블레이드는 회전하며 **연속 이동**한다. marker를 격자점에 스냅하면 위치가 계단처럼 튀어
비물리적 노이즈가 생긴다 → Gaussian이 분수 오프셋을 매끄럽게 처리한다(spreading·sampling
동일). `hub_center`·marker 간격(`marker_dr`)도 격자와 무관.

**Q2. ε은 "정보 전달 거리"처럼 작동하나? → 두 개념을 구분.**

- **ε = Gaussian의 폭** (힘 대부분이 몰리는 크기).
- **r_cut = n_cut·ε = 실제 잘림 반경** (`gaussian_cutoff`, 기본 **3**). 이 구(sphere) 밖은
  정확히 0: `mask = d_sq ≤ r_cut²`.

3D 질량 누적(반경 m·ε 안에 들어오는 힘 비율):

| 반경 | 누적 질량 | 그 거리 커널값(peak 대비) |
|---|---|---|
| 1·ε | 42.8% | 37% |
| 2·ε | **95.4%** | 1.8% |
| 3·ε (잘림) | 99.96% | 0.01% |

→ 힘의 ~95%가 **2ε 안**, 3ε에서 잘라도 손실 0.04% (그래서 n_cut=3이 표준).

칸 수(lattice units, Δx=1이라 ε 값 = 칸 수):

| ε | r_cut=3ε (영향 반경) | 스캔 박스 | 95%가 드는 반경 2ε |
|---|---|---|---|
| 2.0 (floor) | 6칸 | ~15³ | 4칸 |
| 5.3 (D32 production) | 15.9칸 | ~35³ | 10.6칸 |

거리별 커널값(ε=5.3, peak 대비): 1칸 96% · 3칸 73% · 5칸 41% · 8칸 10% · 15칸 0.03% ·
16칸~ = 0(잘림).

**비용 노트:** `gaussian_cutoff`(n_cut)↓ → 박스 작아져 빠름(특히 GPU stencil이
`(2·⌈3ε⌉+3)³`로 큼). 단 너무 작으면 Gaussian 꼬리를 잘라 총 힘 보존이 약화. 3이
정확도-비용 균형점이라 모든 config가 3.0.
