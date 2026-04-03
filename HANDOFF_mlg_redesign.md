# HANDOFF: Multi-Level Grid (MLG) 재구현

## 작성일: 2026-04-04
## 상태: Phase 1~3 폐기, 처음부터 재설계 필요

---

## 1. 이전 작업 요약 및 실패 원인

### 수행한 작업
- Phase 1: `src/grid/unit_converter.py`, `grid_level.py` 작성
- Phase 2: `src/grid/interpolation.py`, `coarse_to_fine.py`, `fine_to_coarse.py` 작성
- Phase 3: `src/solver/multi_level_simulation.py` 작성, `setup.py`/`output_manager.py` 패치

### 근본적 설계 오류
**도메인 배치를 완전히 잘못 이해했다.**

잘못된 구현 (multi-domain을 "나란히 배치"로 오해):
```
|── Level 0 (coarse) ──|── Level 1 (fine) ──|
```

올바른 구현 (Geier/Palabos/Lagrava 방식):
```
|────────────── Level 0 (coarse, 전체 도메인) ──────────────|
              |──── Level 1 (fine, 부분 영역, 겹침) ────|
                    |── Level 2 (finer, 더 작은 영역) ──|
```

**Level 0은 전체 도메인을 coarse 해상도로 커버한다.**
**Level 1은 Level 0 안의 관심 영역을 2배 해상도로 중첩 커버한다.**
**fine 영역에서는 coarse 결과를 fine 결과로 덮어쓴다 (two-way coupling).**

이 오류로 인해 커플링 로직, config 구조, VTK 출력이 모두 잘못되었다.

### 폐기 대상 코드
- `src/solver/multi_level_simulation.py` — 도메인 배치와 커플링 인터페이스가 잘못됨
- `src/grid/coarse_to_fine.py` — 인터페이스 정의가 잘못됨 (나란히 배치 기준)
- `src/grid/fine_to_coarse.py` — 동일
- `configs/mlg_poiseuille_2d.py` — 도메인 구조가 잘못됨
- `setup.py`의 MLG 패치 — `_setup_mlg()`, `_build_level_simulation()`, `build_simulation()` MLG 경로
- `output_manager.py`의 MLG 패치 — `_write_vtk_mlg()`

### 재사용 가능한 코드
- `src/grid/unit_converter.py` — τ, ν 레벨 간 변환은 올바름 (τ_{k+1} = 2τ_k - 0.5 검증 완료)
- `src/grid/grid_level.py` — BoundingBox, 좌표 변환 등 유틸리티는 재사용 가능
- `src/grid/interpolation.py` — Cubic/CompactSecondOrder 보간 알고리즘 자체는 올바름 (적용 위치가 잘못됐을 뿐)

---

## 2. 올바른 MLG 구조

### 2.1 물리적 구조 (Geier Fig.10, Lagrava Ch.5)

```
전체 도메인 (예: 100 x 50 x 50)
┌─────────────────────────────────────────────────┐
│  Level 0: coarse, dx=1.0, 전체 도메인 커버       │
│                                                  │
│      ┌─────────────────────┐                     │
│      │  Level 1: dx=0.5    │                     │
│      │  Level 0과 겹침!     │                     │
│      │   ┌──────────┐      │                     │
│      │   │ Level 2   │      │                     │
│      │   │ dx=0.25   │      │                     │
│      │   │  ● 물체   │      │                     │
│      │   └──────────┘      │                     │
│      └─────────────────────┘                     │
│                                                  │
└─────────────────────────────────────────────────┘
```

- Level 0: 전체 도메인을 coarse 해상도로 시뮬레이션
- Level 1: Level 0의 **부분 영역**을 2배 해상도로 시뮬레이션 (Level 0과 공간적으로 겹침)
- Level 2: Level 1의 부분 영역을 4배 해상도로 시뮬레이션
- **겹치는 영역에서는 fine 레벨의 결과가 coarse를 덮어쓴다**

### 2.2 Nested Time-Stepping (올바른 알고리즘)

Lagrava Sec. 4.4.3의 complete algorithm:

```
advance_coarse_step():
    1. coarse 전체 도메인에서 collide-and-stream (t → t+δt_c)
    
    2. 겹치는 영역(overlap region)에서:
       - coarse의 ρ, u, f^neq를 fine 경계로 전달 (C→F)
       - 시간 보간: t와 t+δt_c 사이의 t+δt_c/2 값 생성
       - 공간 보간: coarse 노드 사이의 fine 노드 값 생성
    
    3. fine 영역에서 첫 번째 step (t → t+δt_f = t+δt_c/2)
       - fine 경계에 C→F 보간된 값 적용
    
    4. 겹치는 영역에서 C→F (full step, 시간 보간 불필요)
    
    5. fine 영역에서 두 번째 step (t+δt_f → t+δt_c)
       - fine 경계에 C→F 값 적용
    
    6. 겹치는 영역에서 F→C:
       - fine의 결과를 coarse의 겹치는 노드에 덮어쓰기
       - f^neq 필터링 (high-frequency 제거) + rescaling
       - ρ, u는 필터링하지 않음
```

### 2.3 커플링 인터페이스 (겹치는 영역)

**커플링은 "경계면"이 아니라 "겹치는 영역(overlap region)"에서 수행된다.**

```
Coarse grid (dx_c):    ○───○───○───○───○───○───○───○───○
                                    ↕ overlap ↕
Fine grid (dx_f):                ●─●─●─●─●─●─●─●─●
                                 ↑               ↑
                           fine 시작         fine 끝
```

- 오버랩 영역: 최소 1 coarse cell 폭 (= 2 fine cells)
- C→F: 오버랩 영역의 coarse 데이터를 fine 경계(모든 면)에 전달
- F→C: 오버랩 영역의 fine 데이터를 coarse 노드에 덮어쓰기
- fine 경계의 **모든 면**(상하좌우)에서 커플링 수행 (나란히가 아니라 감싸는 구조)

### 2.4 f^neq Rescaling (이 부분은 올바르게 구현됨)

Convective scaling 하에서:
- f^eq는 ρ, u에만 의존 → 격자 간 연속 → rescaling 불필요
- f^neq ∝ ∇u → 격자 해상도에 따라 rescaling 필요

```
C→F: f_{fine} = f^eq(ρ,u) + (τ_f / 2τ_c) · f^neq_{coarse}
F→C: f_{coarse} = f^eq(ρ,u) + (2τ_c / τ_f) · f^neq_{fine,filtered}
```

τ 점화식: τ_{k+1} = 2·τ_k - 0.5 (Cheylan et al. Eq.12, 검증 완료)

### 2.5 VTK 출력 (올바른 방식)

ParaView의 **vtkOverlappingAMR** 또는 **vtkNonOverlappingAMR** 포맷 사용:
- 하나의 파일에 모든 레벨이 포함됨
- ParaView에서 레벨별 ON/OFF 전환 가능
- 격자 해상도 차이가 자동으로 표시됨

대안: 각 레벨을 별도 VTI로 쓰되, 같은 물리 좌표계에서 origin/spacing을 정확히 설정하고, VTM(MultiBlock)으로 묶기. 이 경우 겹치는 영역에서 두 레벨이 모두 보임.

---

## 3. Config 구조 (올바른 형태)

```python
# Level 0 = simulation.domain = 전체 도메인
simulation = {
    "domain": {"Nx": 200, "Ny": 100, "Nz": 100},
    "physics": {"tau": 0.56, ...},
    ...
}

# MLG: 각 fine 레벨의 bounding box를 Level 0 좌표계에서 지정
mlg = {
    "num_levels": 3,
    "interpolation": "cubic",
    "filter_level": 1,
    "levels": [
        {},  # Level 0 = simulation.domain 전체
        {    # Level 1: Level 0 안의 부분 영역 (Level 0 좌표계)
            "region": {"x_min": 40, "x_max": 160, "y_min": 20, "y_max": 80, "z_min": 20, "z_max": 80},
            # Nx, Ny, Nz는 region 크기 × 2 (refinement ratio)로 자동 계산
        },
        {    # Level 2: Level 1 안의 부분 영역 (Level 0 좌표계)
            "region": {"x_min": 70, "x_max": 130, "y_min": 35, "y_max": 65, "z_min": 35, "z_max": 65},
        },
    ],
}
```

Level k의 격자 크기:
- dx_k = dx_0 / 2^k
- 영역 크기 = (x_max - x_min) / dx_k 노드

---

## 4. 구현 계획 (재설계)

### Phase 1 (재사용): Unit Converter + Grid Level
- `unit_converter.py`: 그대로 사용 (검증 완료)
- `grid_level.py`: BoundingBox를 "Level 0 좌표계 내의 부분 영역"으로 재정의

### Phase 2 (재작성): 겹침 기반 커플링
- **OverlapRegion 클래스**: 두 레벨 간 겹치는 영역의 인덱스 매핑
  - coarse 인덱스 ↔ fine 인덱스 대응표 사전 계산
  - fine 경계면(6면 in 3D, 4면 in 2D) 식별
- **CoarseToFine**: 겹치는 영역의 coarse → fine 경계 전체면에 전달
  - 공간 보간: coarse 노드 사이의 fine 노드 보간
  - 시간 보간: half-step에서 linear interpolation
- **FineToCoarse**: 겹치는 영역의 fine → coarse 노드 덮어쓰기
  - 필터링 + rescaling
- `interpolation.py`: 알고리즘은 재사용, 적용 대상을 "경계면"에서 "겹치는 영역 전체"로 수정

### Phase 3 (재작성): MultiLevelSimulation
- 각 레벨이 **자체 Simulation 객체**를 소유 (B안 유지)
- Level 0의 Simulation은 전체 도메인
- Level k의 Simulation은 부분 도메인 (자체 BC = 커플링 경계)
- advance()에서 nested time-stepping 수행
- **F→C 덮어쓰기**: fine 결과를 coarse의 대응 노드에 직접 복사

### Phase 4: VTK 출력
- vtkOverlappingAMR 또는 VTM+VTI (겹치는 영역 포함)
- ParaView에서 레벨별 가시화

### Phase 5: Validation
- 2D Poiseuille flow (전체 도메인 coarse + 중앙 영역 fine)
- 3D sphere (Geier 논문 Fig.10 재현)

---

## 5. 참고 논문 핵심 (재확인)

### Lagrava Sandoval (학위 논문)
- Ch.4: Grid refinement 이론 — 모든 수식 정확
- Ch.5: Palabos 구현 — **multi-domain 방식, 레벨별 multi-block**
- Fig.5.1: "A multi-domain of three levels is actually three multi-blocks"
  → 각 레벨이 자기 영역만 소유하지만, **겹치는 영역이 존재**
- Sec.4.3.4, Fig.4.5: 커플링 위치 정의 — **겹치는 영역의 경계**에서 수행
- Sec.4.4.3: Complete algorithm — 시간 순서 정확

### Geier et al. (2015)
- Fig.10: 5-level sphere — **중첩된 사각형 영역들**
- Sec.6: "compact second order interpolation and nested time stepping"
- 각 레벨의 노드 수가 전체 도메인보다 훨씬 작음 → 부분 영역임을 확인

### Cheylan et al. (Shape Optimization)
- Eq.12: τ^f = 2τ^c - 1/2 (검증 완료)
- Fig.3: Grid refinement splitting — edge를 2개로 분할
- "five resolution domains (five edge lengths)" — 중첩 구조

---

## 6. 현재 Git 상태

### 삭제할 파일
- `src/solver/multi_level_simulation.py`
- `src/solver/PATCH_setup_mlg.py`
- `src/solver/PATCH_phase3_complete.py`

### 되돌릴 파일 (패치 제거)
- `src/solver/setup.py` — `_setup_mlg()`, `_build_level_simulation()`, `_build_level_bc_manager()`, `build_simulation()` MLG 경로 제거
- `src/solver/output_manager.py` — `_write_vtk_mlg()` 제거, `_write_vtk()` 원복

### 유지할 파일 (검증 완료)
- `src/grid/__init__.py` — export 목록은 수정 필요
- `src/grid/unit_converter.py` — 올바름
- `src/grid/grid_level.py` — BoundingBox 재정의 필요하지만 기본 구조는 유지
- `src/grid/interpolation.py` — 알고리즘 올바름
- `src/grid/coarse_to_fine.py` — 폐기 (인터페이스 정의 잘못됨)
- `src/grid/fine_to_coarse.py` — 폐기 (인터페이스 정의 잘못됨)

### Config
- `configs/mlg_poiseuille_2d.py` — 폐기, 재작성 필요

---

## 7. 다음 컨텍스트에서 해야 할 일

1. 이 handoff.md를 읽고 올바른 MLG 구조를 이해했는지 확인
2. Lagrava Fig.4.5, Fig.5.1을 기준으로 OverlapRegion 설계
3. Phase 2부터 재작성 (겹침 기반 커플링)
4. **반드시 코드 작성 전에 설계를 사용자와 논의**
5. **프로젝트 구조를 임의로 변경하지 않음**

---

## 8. 중요 교훈

1. MLG의 핵심은 **겹침(overlap)**이다. 나란히 배치가 아니다.
2. Level 0은 항상 전체 도메인을 커버한다.
3. 코드 작성 전에 물리적 구조가 올바른지 반드시 확인한다.
4. 프로젝트 구조를 임의로 변경하지 않는다.
5. 기존 솔버의 실행 흐름(main.py → setup → sim.advance → output)을 존중한다.
