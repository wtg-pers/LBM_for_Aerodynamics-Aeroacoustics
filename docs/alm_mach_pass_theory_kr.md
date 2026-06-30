# ALM Mach-pass — 이론 · 배경 · 구현 의도 · 효과

> ALM에서 단면 공력(C_L/C_D)을 "올바른 국소 Mach"로 조회하기 위한 Mach-pass 확장의
> 배경과 구현 정리. 등현(HART2/CT) → 테이퍼(HVAB)로 넘어가며 왜 필요해졌는가.
> 관련: `docs/alm_epsilon_theory_kr.md`, `docs/hvab_geometry_kr.md`,
> `patch_notes/alm_mach_pass/`.

---

## 0. 한 줄 요약

> **ALM은 격자에서 단면 상대속도 `u_rel`만 얻는다. 단면 공력 C_L/C_D는 외부 폴라표에서
> 조회하는데, 그 표가 Mach에 의존(압축성)한다. 등현 블레이드에선 Mach를 Reynolds수로부터
> 우회 복원할 수 있었지만(트릭), 테이퍼 블레이드에선 그 우회가 깨진다. Mach-pass는 단면
> Mach `M = u_rel/a`를 직접 계산해 폴라에 넘겨, 우회를 없애고 정확하게 만든다.**

---

## 1. 배경: ALM은 공력을 "표에서 조회"한다

ALM은 블레이드 형상을 격자로 풀지 않는다(→ `alm_epsilon_theory_kr.md`). 대신 각 마커에서:

1. 격자 속도장에서 단면 상대속도 `u_rel`, 받음각 `α`를 구하고,
2. **익형 폴라표**에서 `C_L(α, ...)`, `C_D(α, ...)`를 **조회(lookup)** 한 뒤,
3. 양력/항력 → body force로 환산해 격자에 되돌린다.

즉 "공력의 질"은 **폴라표 조회가 얼마나 올바른 조건에서 이뤄지는가**에 달려 있다.

## 2. 폴라는 무엇에 의존하나 — Reynolds vs Mach

익형 C_L/C_D는 두 무차원수에 의존한다:

- **Reynolds수** `Re = u_rel·c/ν` — 점성 효과(경계층, 박리). 시위 `c`에 의존.
- **Mach수** `M = u_rel/a` — **압축성** 효과 (`a` = 음속). **시위와 무관.**

로터 팁은 `M`이 0.6~0.9로 커서 압축성이 중요하다:
- Prandtl-Glauert: `C_L`이 `1/√(1−M²)`로 상승,
- 천음속(M≳0.7~0.8): 충격파로 `C_L` 급강하·`C_D` 급증.

그래서 로터용 폴라는 **C81 덱 = (α, Mach) 2차원 표**로 주어진다. 우리 `C81PolarSet.get_CL(α, M)`이
이 표를 (α, Mach)로 보간한다. **이 Mach 보간 기능은 HART2 때 이미 구현·검증됨.**

## 3. 문제: ALM은 격자에서 "속도"를 얻지, Re도 Mach도 직접 주지 않는다

ALM이 격자에서 직접 얻는 1차량은 **`u_rel`(상대속도, lattice 단위)** 이다. 여기서:

```
Re = u_rel · c / ν          (시위 c에 의존)
M  = u_rel · (dx/dt) / a    (= u_rel_physical / a,  시위와 무관)
```

`u_rel·(dx/dt)`는 lattice 속도를 물리 속도[m/s]로 환산한 것(`dx/dt` = 속도 스케일).
**핵심: Mach는 시위가 아니라 속도만의 함수다.**

## 4. HART2(등현)의 우회 트릭 — 왜 그렇게 했나

역사적으로 ALM의 폴라 인터페이스는 `polar_query(α, Re)` 였다 (Re-인덱스 XFOIL/CSV 폴라용).
Mach-인덱스 C81을 이 인터페이스에 **ALM 코드 변경 없이** 끼워넣으려고, HART2 구현은 closure에서
Re로부터 Mach를 **역산**했다:

```
M = k · Re,   k = ν / (c · a)        (make_c81_polar_query, c81_loader.py)
  = [ν/(c·a)] · [u_rel·c/ν] = u_rel/a   ✓  (단, c가 일정할 때만 c가 소거됨)
```

HART2는 **등현(c = 0.121 m 일정)** 이라 `k`가 고정 상수 → 이 트릭이 **정확**하다. 그래서
HART2엔 새 코드가 필요 없었다. 영리한 최소-변경 hack.

## 5. 테이퍼에서 트릭이 깨지는 이유 (HVAB)

테이퍼면 시위 `c`가 스팬마다 다르다. closure가 **단일 기준시위 `c_ref`** 로 `k`를 고정하면:

```
M_조회 = k·Re_local = [ν/(c_ref·a)]·[u_rel·c_local/ν] = (u_rel/a) · (c_local / c_ref)
                                                        └ M_true ┘   └ 오차 인자 ┘
```

- 등현: `c_local = c_ref` → 오차 인자 1 → 정확.
- **HVAB 팁**: `c_local/c_ref = 3.27/5.45 = 0.60` → **M_조회 = 0.60·M_true**.
  - M_true 0.65 → M_조회 0.39. 0.39는 아음속(압축성 거의 없음), 0.65는 PG+압축성 시작.
  - 결과: **팁 C_L이 ~20% 틀어짐** (하필 가장 중요한 팁에서).

> 만약 closure에 **국소 시위**를 넣어 `k_local = ν/(c_local·a)`로 했다면 `M = u_rel/a`가 되어
> 정확해진다 — 하지만 그건 결국 "Mach를 속도로부터 직접 계산"하는 것과 동일하다. 즉 우회를
> 버리고 **Mach를 직접 계산·전달**하는 게 정공법.

## 6. Mach-pass — 무엇을 하나

**BEM 루프에서 단면 Mach를 직접 계산해 폴라에 전달한다:**

```
M = u_rel · (dx_phys/dt_phys) / a_phys      # 시위 무관, 물리적으로 정확
polar_query(α, Re, mach=M)                  # 폴라는 이 M으로 (α,M) 표 조회
```

- `dx_phys/dt_phys`는 그 레벨의 속도 스케일 (MLG fine-level도 올바름 — `dx/dt` 비율 레벨 불변).
- `a_phys = c_s_phys` (음속, 물리량, 레벨 무관).
- Re는 여전히 점성용으로 계산·전달(폴라가 쓰면 사용). Mach만 우회 없이 직접.

**기본 inert(중요):** 폴라가 `mach` 인자를 **선언했을 때만** Mach를 계산·전달한다. 등현
HART2/CT 폴라는 `mach` 인자가 없으므로(2-arg `(α,Re)`) Mach-pass가 **전혀 작동하지 않음** →
기존 결과 **bit-identical** (CT 스모크 T_lu=0.080959로 검증).

## 7. 구현 의도 & 설계 원칙

| 원칙 | 실현 |
|---|---|
| 기존 자산 재사용 | C81 Mach 보간(`get_CL(α,M)`) **그대로** 재사용. 새 폐쇄는 ~5줄 wrapper. |
| 기존 동작 불변 | `mach` 미선언 폴라엔 inert → HART2/CT bit-identical. |
| 시그니처 자동 탐지 | `inspect.signature`로 `'mach' in params` 판별 → 명시 플래그 불필요. |
| 레벨 정확성 | per-level `dx_phys/dt_phys` 사용 (fine-level ALM도 정확). |

## 8. 효과

- **테이퍼 팁 공력 정확**: 팁 Mach를 시위와 분리해 물리적으로 정확히 조회 → 팁 C_L/C_D,
  나아가 팁 하중·추력이 올바르게.
- **등현 불변**: HART2/CT는 영향 0.
- **HVAB enabling**: 테이퍼+다익형 로터를 제대로 돌리는 전제 조건.

## 9. 코드 맵

| 개념 | 위치 |
|---|---|
| C81 Mach 보간 (재사용) | `c81_loader.py::C81PolarSet.get_CL/get_CD(α, M)` |
| 등현 Re→M 트릭 (HART2) | `c81_loader.py::make_c81_polar_query` (불변) |
| **Mach-native 폐쇄 (신규)** | `c81_loader.py::make_c81_polar_query_mach` |
| Mach 계산·전달 | `actuator_line.py` BEM 루프 (`M=u_rel·(dx/dt)/a`), 시그니처 탐지 `_polar_wants_mach` |
| 음속 주입 | `setup.py` → `create_actuator_line_from_config(sound_speed=self.c_s_phys)` |

## 10. 다음 (R3) — multi-airfoil + Mach 결합

HVAB는 **다종 RC 익형**(RC(4)-12/RC(4)-10/RC(6)-08/RC(6)-08T)이라, 폴라 조회가
`(α, Re, airfoil_name, mach)` 4입력이어야 한다. 현재 multi-airfoil 매니저는
`(α, Re, airfoil_name)`까지만 → **`mach`를 thread하도록 확장 + per-RC Mach-native 폐쇄**가 필요.
이게 R3. (단일 익형 Mach-pass는 §6에서 완료.)

---

### 부록 — 자주 헷갈리는 점
- **Mach-pass는 형상이 아니라 공력 조회를 고친다.** 테이퍼 chord(r) 분포는 config/geometry에서
  정의되고, Mach-pass는 그 위에서 "각 단면을 올바른 Mach로 조회"만 담당.
- **Re는 사라지지 않는다.** 점성(박리/Re-stall)용으로 계속 계산·전달. Mach만 우회 제거.
- **등현이면 Mach-pass 불필요.** Re→M 트릭이 이미 정확. HVAB(테이퍼)에서만 의미.
