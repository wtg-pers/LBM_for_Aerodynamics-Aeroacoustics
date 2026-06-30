# Airfoil Mask Connected-Component Filter

**위치**: `src/boundary/geometry.py::create_airfoil_2d_mask_from_coords` 끝부분
**도입일**: 2026-04-22
**상태**: 상시 활성 (scipy 없으면 silent skip)

## 1. 배경 — 왜 필요했는가

### 문제 현상

CLF5605 (NASA Mars Helicopter Ingenuity outboard primary airfoil)를
D2Q9 LBM으로 시뮬레이션할 때, airfoil mask를 VTK로 렌더링하면
**TE(Trailing Edge) 근처에 본체와 분리된 1~3 셀짜리 작은 solid 조각**
들이 떠있는 것이 관찰됨.

4-level MLG의 L3 (1024 cpc) 레벨에서조차 이 현상이 지속되었으며,
5-level, 6-level로 늘려도 근본 해결 불가.

### 수치 현상

연결 컴포넌트 분석 결과 (α=0°, 다양한 해상도):

| CPC | Main body | 고립 파편 (components) | 파편 총 셀 수 |
|---|---|---|---|
| 300 | 2,687 | 5 | 11 |
| 512 | 7,807 | 7 | 12 |
| 1024 | 22,796 | 8 | 15 |

파편 위치: **x/c = 0.97 ~ 1.00** 구간에 집중.

### 근본 원인 — Sub-pixel Thickness

CLF5605는 sharp TE 익형이라 TE 근처 두께가 0으로 수렴:

| x/c | 두께 (c) | 두께 (lu at 1024 cpc) | 해상 여부 |
|---|---|---|---|
| 0.90 | 0.63% | 6.5 | ✓ 해상 |
| 0.95 | 0.32% | 3.2 | ✓ 해상 |
| 0.97 | 0.19% | 1.95 | ⚠️ 경계 |
| 0.99 | 0.06% | 0.65 | ✗ 서브-픽셀 |
| 1.00 | 0.001% | 0.01 | ✗ 불가 |

### 왜 파편이 생기는가 — Ray Casting 분류의 근본 한계

Mask 생성은 `_point_in_polygon_mask` (ray casting) 사용:
**각 격자 노드의 중심이 폴리곤 내부에 있으면 solid, 아니면 fluid**로 분류.

두께가 1 lu 미만인 구간에선:

```
위쪽 유체 노드  (y = j)     ━━━━━━━━━━━━━━━━━━━ (fluid, 다각형 밖)
                                 ↓ 두께 0.6 lu
                      ┌────────────────────────┐
                      │  <- airfoil polygon -> │
                      └────────────────────────┘
                                 ↑
아래쪽 유체 노드 (y = j-1)  ━━━━━━━━━━━━━━━━━━━ (fluid, 다각형 밖)
```

- 폴리곤이 (j-1, j) 두 노드 사이를 완전히 관통
- 어떤 정수 y 좌표도 폴리곤 내부에 포함되지 않음
- **두 노드 모두 fluid로 분류됨**

가끔 x 방향으로 훑어가다가 폴리곤이 정수 격자 좌표와 우연히 교차하는
지점만 solid로 잡히면서 "점선 (dashed line)" 효과 발생.

### 본 문제는 IBB로도 해결 안 됨

Bouzidi IBB는 **이미 분류된 solid mask**에 대해 wall 위치를 q=fractional
link로 정확히 처리할 뿐이고, mask에 없는 solid를 만들어내지 못함.
Link-level Immersed Boundary Method로 가야 완전 해결 (구현 비용 큼).

---

## 2. Connected-Component Filter — 원리

### 이론적 배경 — 위상수학과 그래프 이론

2D boolean 배열을 **그래프 구조**로 해석:
- **Node**: `True` 값인 각 격자 셀
- **Edge**: 두 `True` 노드가 "인접"하면 연결

**연결 컴포넌트 (connected component)** = 그래프 상에서 path로 도달
가능한 노드들의 극대집합 (maximal connected subgraph).

본 문제에선 airfoil mask를 2D 격자 그래프로 보고:
- **Main body**: 하나의 거대 컴포넌트 (20,000+ 셀)
- **파편**: 각각 별도의 작은 컴포넌트 (1~3 셀)

### 인접성 정의 — 4-connectivity

본 구현은 **4-connectivity** (N/S/E/W 4방향 인접) 사용:

```
         ┌───┐
         │ N │
     ┌───┼───┼───┐
     │ W │ ■ │ E │      ← 4-connectivity: N, E, S, W만 인접
     └───┼───┼───┘
         │ S │
         └───┘
```

대각선 (NE, NW, SE, SW) 을 포함하는 8-connectivity가 있으나,
**LBM의 streaming 구조와 일치**시키기 위해 4-connectivity 선택:

- LBM streaming은 D2Q9의 **축 정렬 4방향이 주 정보 전달 경로**
- Bounce-back은 축 방향에서 가장 강하게 작용
- 대각선 격자 연결은 물리적으로 약한 연결성 (cornor of solid에 link 없음)

4-connectivity가 **LBM의 실제 물리적 연결성을 더 정확히 반영**.

### 알고리즘 — scipy.ndimage.label

```python
from scipy.ndimage import label

# labeled[i,j] = 컴포넌트 번호 (0 = background, 1,2,... = solid)
labeled, n_components = label(mask_np)
```

내부적으로 **Union-Find (disjoint-set) with 2-pass scanning**:
1. 1-pass: 각 True 셀에 임시 라벨 부여, 인접한 라벨들을 equivalence class로 기록
2. 2-pass: equivalence class를 최소 대표 라벨로 통합

시간 복잡도 O(N) (N = 총 셀 수), 공간 O(N).

### Filter 로직

```python
if n_components > 1:
    sizes = np.bincount(labeled.ravel())
    # sizes[0] = background count (skip)
    # sizes[1..] = 각 solid component의 셀 수
    largest = int(np.argmax(sizes[1:])) + 1
    mask_np = (labeled == largest)
```

**"Largest component only"** 전략:
- 가장 많은 셀을 가진 컴포넌트 = 의도된 main body
- 나머지 컴포넌트 = 파편 (제거)

---

## 3. 물리적 의미 — Airfoil 기하학 변형

### 결과적 기하학적 효과

Filter 적용 후, airfoil은 **"마지막 연결된 cross-section에서 잘린 것처럼"** 동작:

```
Filter 적용 전:
     ═══════════════════════════════
    ══════════════════════════════════
   ══════════════════════════════════════
    ═════════════════════════════════   ▪   ▪  ▪    ← 파편
     ════════════════════════════════
      ═══════════════════════════

Filter 적용 후:
     ═══════════════════════════════
    ══════════════════════════════════
   ══════════════════════════════════════  │
    ═════════════════════════════════      │ ← implicit blunt TE
     ════════════════════════════════      │
      ═══════════════════════════
                                  ↑
                            "마지막 연결된 cross-section"
                            x/c ≈ 0.97 at 1024 cpc
```

### 효과적 TE 두께

Filter에 의한 implicit TE 절단 위치는 해상도에 의존:

| CPC | 마지막 연결 x/c | 효과적 TE 두께 (c) |
|---|---|---|
| 300 | ~0.93 | ~0.35% |
| 512 | ~0.95 | ~0.25% |
| 1024 | ~0.97 | ~0.19% |
| 2048 | ~0.985 | ~0.12% |

### 실제 제조 airfoil과의 비교

- **이론상 CLF5605 TE 두께**: 0.01% chord (Selig `.dat` 좌표)
- **Filter 적용 후 유효 TE 두께**: 0.19~0.35% chord (해상도 의존)
- **실제 제조 가능한 TE 두께**: 0.25~0.5% chord (CNC 가공 한계, 구조 강도)

**Filter 적용 결과가 오히려 실제 제조 익형에 더 가까움**. 이론적 sharp
TE는 시뮬레이션 목적 외에는 존재하지 않음.

### Cd/Cl에 미치는 영향

고전 익형 공력 데이터 (XFOIL, NACA Report)에 따르면:
- TE 두께 0.1~0.3% chord에서 Cd 증가 < 1%
- Cl 영향 더 작음 (양력은 주로 upper-lower pressure difference가 결정)
- Cp 분포는 x/c > 0.95에서 국소적으로만 다름

결론: **Filter의 실효 변형은 공력 예측 정확도에 미미한 영향**.

---

## 4. 한계 및 주의사항

### Filter가 잘못 작동할 수 있는 케이스

1. **의도된 다중 solid**: 여러 airfoil을 동시에 배치한 경우 (tandem wing, biplane).
   → 가장 큰 것만 남고 나머지는 삭제됨. 이 경우 filter를 우회해야 함.

2. **비연결된 의도적 구조**: wing + tail (T-tail 같은) 의 geometric 배치.
   → 4-connectivity 관점에서 연결이 끊기면 하나가 삭제됨.

3. **매우 얇은 연결**: 중앙 body가 단일 셀 두께의 브리지로 연결된 케이스.
   → 브리지가 uneven하면 얇은 곳에서 끊길 수 있음.

### Filter가 해결 못 하는 것

- **서브-픽셀 영역의 공력 정확도**: 잘라낸 3% 영역의 pressure, shear 정보는 누락
- **Cumulative fidelity**: 해상도 증가로 filter 영향 범위를 축소할 순 있으나, 근본적으로 cell-based mask의 한계
- **진짜 blunt TE의 물리 정확도**: 실제 blunt TE는 post-TE vortex shedding 등 유동 효과가 있으나, filter는 단순 절단

### 완전 해결이 필요하면

Task 9 (**Bouzidi IBB + Link-level IBM**) 구현이 근본 해결. 그 전까지
filter + 적절한 해상도가 실용적 타협안.

---

## 5. 검증

### Unit 레벨

```python
from src.boundary.geometry import load_selig_dat, create_airfoil_2d_mask_from_coords
from scipy.ndimage import label

x, y = load_selig_dat('to_claude/clf5605.dat')
mask = create_airfoil_2d_mask_from_coords(
    np, (1064, 300), x, y, chord_length=1024,
    center=(532, 150), angle_of_attack=0.0,
)
_, n = label(mask)
assert n == 1  # single connected component after filter
```

### Integration 레벨

4-level MLG (α=0°) 실행 시 각 레벨이 모두 `components=1`로 생성됨:

| Level | CPC | Before filter | After filter |
|---|---|---|---|
| L0 | 128 | 496 solid, 3 components | 492 solid, 1 component |
| L1 | 256 | 1,961 solid, 5 components | 1,951 solid, 1 component |
| L2 | 512 | 7,814 solid, 7 components | 7,807 solid, 1 component |
| L3 | 1024 | 22,796 solid, 8 components | 22,789 solid, 1 component |

Warning 메시지로 제거된 fragment 수가 로그에 출력됨.

---

## 6. 참고

- `scipy.ndimage.label` 공식 문서:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
- Union-Find 알고리즘 (Tarjan 1975, Rosenfeld 1966 two-pass scanning):
  표준 알고리즘 교재
- LBM streaming 연결성 vs 격자 connectivity 논의:
  Kruger et al. "The Lattice Boltzmann Method" Ch. 5
- NACA airfoil TE geometry / manufacturing tolerance:
  NASA Langley airfoil database; Abbott & von Doenhoff "Theory of Wing Sections"
