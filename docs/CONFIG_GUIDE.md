# LBM Solver Configuration Guide

## 설계 원칙

**물리/격자/수치를 분리한다.** 사용자는 SI 물리값과 해상도만 지정하고,
`UnitConverter`가 나머지 lattice 값(tau, nu_lu, dt, ...)을 자동 유도한다.

- **physics**: SI 단위 물리값 (rho [kg/m³], U_inf [m/s], Re, L_char [m])
- **grid**: 격자 크기와 해상도 (Nx, Ny, [Nz], resolution [cells/L_char])
- **numerics**: 수치 선택 (u_max [lattice], collision, ...)

Lattice 단위를 직접 지정하는 **legacy 포맷은 지원하지 않는다**.
`simulation` 블록 안에 `physics`나 `domain`을 중첩하면 즉시 실패한다.

---

## Config 구조 개요

```python
config = {
    "simulation":        { ... },  # [필수] 장치, 격자 모델, 충돌 모델
    "physics":           { ... },  # [필수] 물리 파라미터 (SI)
    "grid":              { ... },  # [필수] 격자 크기와 해상도
    "numerics":          { ... },  # [필수] u_max 등 수치 파라미터
    "boundaries":        { ... },  # [필수] 경계 조건
    "internal_geometry": { ... },  # [선택] 내부 장애물
    "mlg":               { ... },  # [선택] Multi-Level Grid
    "interval":          { ... },  # [선택] 출력/로그 주기
    "time":              { ... },  # [선택] max_steps
    "airfoil_polar":     { ... },  # [ALM 필수] 익형 공력 데이터
    "actuator_line":     { ... },  # [선택] Actuator Line Model
    "conservation":      { ... },  # [선택] 질량 보존 모니터
    "convergence":       { ... },  # [선택] 수렴 감지
    "force_calculation": { ... },  # [선택] MEM force
    "output":            { ... },  # [필수] 출력 경로
}
```

---

## 1. simulation

충돌 모델/격자 모델/장치 선택. **물리값은 여기에 넣지 않는다**.

| Key | Type | Default | 필수 | 설명 |
|-----|------|---------|:----:|------|
| `device_mode` | str | — | O | `"gpu"` 또는 `"cpu"` |
| `device_id` | int | 0 | | GPU 번호 (CLI `--gpu`로 덮어씀) |
| `precision` | str | `"float64"` | | `"float32"` 또는 `"float64"` |
| `dimension` | int | — | O | `2` 또는 `3` |
| `lattice_model` | str | `"D3Q27"` | | `"D2Q9"` / `"D3Q15"` / `"D3Q27"` |
| `collision_model` | str | `"bgk"` | | `"bgk"` 또는 `"cumulant"` |
| `validate_lattice` | bool | False | | 격자 직교성 검증 (디버그) |

### Cumulant 전용 파라미터 (simulation 루트)

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `omega_bulk` | float | None | Bulk viscosity rate. 2D에서 None이면 ω_shear 사용 |
| `omega_3` | float | 1.0 | 3차 모멘트 relaxation (2D) |
| `omega_4` | float | 1.0 | 4차 모멘트 relaxation (2D) |
| `omega_high` | float | 1.0 | 고차 모멘트 relaxation (3D) |

> **Re > ~8,000은 `"cumulant"` 권장.** BGK는 고Re에서 불안정.

---

## 2. physics — SI 단위 물리 파라미터

**모든 값은 SI 단위**(kg/m³, m/s, m)로 지정. Lattice 단위로 변환은 `UnitConverter`가 자동 처리.

| Key | Type | 필수 | 단위 | 설명 |
|-----|------|:----:|------|------|
| `rho` | float | | kg/m³ | 참조 유체 밀도 |
| `U_inf` | float | | m/s | 자유류 속도 |
| `Re` | float | * | — | Reynolds number. `Re` 또는 `nu` 중 하나 필수 |
| `nu` | float | * | m²/s | 동점성. `Re` 대신 직접 지정 가능 |
| `L_char` | float | O | m | 특성 길이 (예: chord, 직경). **필수** |
| `flow_direction` | list | | — | 자유류 방향 단위벡터 (기본 `[1,0,0]`) |

> - `Re` 지정 시 `nu_phys = U_ref * L_ref / Re` 자동 계산.
> - 회전기(ALM)가 있으면 `Re_U_ref = tip_speed`, `Re_L_ref = chord`로 자동 전환.
> - 필요하면 `Re_U_ref`, `Re_L_ref`로 무차원화 기준을 직접 지정 가능.

---

## 3. grid — 격자 크기 + 해상도

| Key | Type | 필수 | 설명 |
|-----|------|:----:|------|
| `Nx` | int | O | x방향 격자 수 [L0 lattice units] |
| `Ny` | int | O | y방향 격자 수 |
| `Nz` | int | | z방향 격자 수. 2D에서는 생략 |
| `resolution` | int | O | 특성 길이당 격자 수 [cells/L_char] |

> `dx_phys = L_char / resolution`. 예: 익형 chord=0.1 m, resolution=64 → dx=1.56 mm.

---

## 4. numerics — 수치 파라미터

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `u_max` | float | 0.1 | Lattice 단위 최대 속도. `Ma = u_max / (1/√3)` |
| `collision` | str | — | (선택) simulation.collision_model 미러 |
| `resolution` | int | — | grid.resolution 대신 여기에 써도 됨 |

> - `u_max < 0.17`이면 Ma < 0.3 유지. `u_max > 0.17`은 `UnitConverter`가 raise.
> - `dt_phys = u_max * dx_phys / U_max_phys` (U_max_phys = max(U_inf, tip_speed))

---

## 5. boundaries

Named dict. 각 면은 아래 키를 가진다. **속도는 SI 단위 [m/s]**로 지정 —
setup.py가 `UnitConverter.phys_to_lu_velocity()`로 자동 변환한다.

### 공통 키

| Key | Type | 필수 | 설명 |
|-----|------|:----:|------|
| `location` | str | O | `"xmin"`, `"xmax"`, `"ymin"`, `"ymax"`, `"zmin"`, `"zmax"` |
| `method` | str | O | 경계 조건 방식 (아래 참조) |
| `velocity` | float/list | | [m/s]. scalar=법선 방향, list=[ux,uy,uz] |
| `rho` | float | | 경계 밀도 (기본 1.0) |

### 사용 가능한 method

| Method | 설명 | 필수 키 |
|--------|------|---------|
| `regularized_inlet` / `reg_inlet` | Regularized 속도 입구 | `velocity` |
| `equilibrium` / `eq` | f = f_eq(ρ, u) | `velocity` |
| `regularized_outlet` / `reg_outlet` | 압력(ρ) 출구 | `density` |
| `regularized_wall` / `reg_wall` | 벽 (no-slip, regularized) | — |
| `bounce_back` / `hwbb` | Half-way bounce back | — |
| `neumann` / `zero_gradient` | Zero-gradient 출구 | — |
| `sponge` / `sponge_layer` | 흡수 스펀지층 | `velocity`, `density`, `thickness`, `strength` |
| `periodic` / `none` | 주기 (BC 없음) | — |

### sponge 전용 키

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `thickness` | int | 20 | 스펀지 두께 [lu] |
| `strength` | float | 0.5 | 최대 감쇠 강도 (sigma_max의 alias) |
| `velocity` | list | — | 감쇠 목표 속도 [m/s] (freestream 권장) |
| `density` | float | 1.0 | 감쇠 목표 밀도 |

### 예시 (2D 익형, M=0.6)

```python
U_INF_PHYS = 144.0  # m/s

boundaries = {
    "inlet":  {"location": "xmin", "method": "equilibrium",
               "velocity": [U_INF_PHYS, 0.0]},
    "outlet": {"location": "xmax", "method": "sponge",
               "velocity": [U_INF_PHYS, 0.0], "density": 1.0,
               "thickness": 40, "strength": 0.5},
    "ymin":   {"location": "ymin", "method": "equilibrium",
               "velocity": [U_INF_PHYS, 0.0]},
    "ymax":   {"location": "ymax", "method": "equilibrium",
               "velocity": [U_INF_PHYS, 0.0]},
}
```

> **주의**: lateral 면에서 scalar `velocity`를 주면 법선 방향 유입이 된다.
> Axial flow에서는 반드시 vector `[U, 0, 0]`로 지정.

---

## 6. internal_geometry

내부 장애물. 사용하지 않으면 dict를 비우거나 모든 entry의 `enabled=False`.
좌표/반경은 **L0 lattice units**.

### sphere / cylinder / box

```python
internal_geometry = {
    "sphere":   {"enabled": True, "center": [50, 50, 50], "radius": 10.0},
    "cylinder": {"enabled": True, "center": [50, 50], "radius": 10.0, "axis": "z"},
    "box":      {"enabled": True, "corner_min": [20,20,20], "corner_max": [40,40,40]},
}
```

### airfoil (2D, Selig .dat)

```python
internal_geometry = {
    "airfoil": {
        "enabled": True,
        "selig_file": "/path/to/clf5605.dat",
        "chord": 64,                     # L0 lattice units (= resolution)
        "center": [LE_X_L0, Ny // 2],    # LE 기준 위치 [lu]
        "angle_of_attack": 2.0,          # [deg]
    }
}
```

> TE가 subpixel인 익형(예: CLF5605)에서는 mask 생성 후 connected-component
> filter가 자동으로 고립된 셀을 제거한다 (`docs/MASK_COMPONENT_FILTER.md`).

---

## 7. mlg — Multi-Level Grid

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `enabled` | bool | False | MLG 활성화 |
| `num_levels` | int | 1 | 격자 레벨 수 (L0 + fine) |
| `overlap_width` | int | 2 | 겹침 영역 폭 [coarse cells] |
| `interpolation` | str | `"cubic"` | `"cubic"` / `"compact_second_order"` |
| `filter_level` | int | 1 | 필터 적용 레벨 |

### levels (list)

Level 0은 빈 dict `{}`. Level k (k ≥ 1)에 refined region 지정:

```python
"levels": [
    {},                                          # L0: 전체 도메인
    {"region": {"x_min": 80, "x_max": 280,      # L1
                "y_min": 40, "y_max": 160}},
    {"region": {"x_min": 100, "x_max": 200,     # L2
                "y_min": 75, "y_max": 125}},
]
```

> - region 좌표는 **L0 lattice units**.
> - 각 fine level은 refine ratio 2 (dx_k = dx_0 / 2^k, tau_k = 2·tau_{k-1} - 0.5).
> - **ALM+MLG**: rotor hub가 finest level region 내부에 있어야 함.

---

## 8. interval

출력/로그/체크포인트/질량보존 체크 주기.

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `output` | int | 500 | VTK/marker 출력 주기 [steps] |
| `log` | int | `output` | 성능/로터 CSV 로그 주기 |
| `checkpoint` | int | 2000 | 체크포인트 저장 주기 |
| `conservation` | int | `output` | 질량 보존 체크 주기 |

---

## 9. time

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `max_steps` | int | 10000 | 최대 시뮬레이션 스텝 수 |

---

## 10. airfoil_polar (ALM 필수)

익형 공력 데이터 로딩 방식.

### method: "neuralfoil" (단일 익형)

```python
airfoil_polar = {
    "method": "neuralfoil",
    "airfoil_name": "naca0012",
    "Re_target": 1e5,
    "Re_min": 1e4, "Re_max": 1e5, "Re_steps": 10,
    "mode": "asb",        # "asb" (aerosandbox) | "user" (좌표 입력)
    "ncrit": 9.0,
}
```

### method: "multi" (복수 익형)

```python
airfoil_polar = {
    "method": "multi",
    "default": "e63",
    "airfoils": {
        "e63":      {"method": "neuralfoil", "airfoil_name": "e63", ...},
        "naca4412": {"method": "neuralfoil", "airfoil_name": "naca4412", ...},
    },
}
```

> 각 blade section의 `airfoil` 키와 이 dict의 키가 일치해야 함.

### method: "csv"

| Key | 설명 |
|-----|------|
| `csv_path` | CSV 파일 경로 |
| `alpha_col` | AoA 컬럼명 (기본 `"AoA(deg)"`) |
| `Re_col` / `CL_col` / `CD_col` | 각 컬럼명 |

### method: "flat_plate"

디버깅용 고정 CL/CD.

---

## 11. actuator_line

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `enabled` | bool | False | ALM 활성화 |
| `gaussian_cutoff` | float | 3.0 | Gaussian spreading 절단 반경 [ε] |
| `rho_ref` | float | 1.0 | BEM force 참조 밀도 |
| `coeff_mode` | str | `"auto"` | `"auto"` / `"rotorcraft"` / `"wind_turbine"` |
| `ramp_steps` | int | 0 | Force ramp-up 스텝 (0=즉시) |
| `prandtl_loss` | bool | False | Prandtl tip/root loss 보정 |

### rotor (단일 로터)

| Key | Type | 필수 | 단위 | 설명 |
|-----|------|:----:|------|------|
| `n_blades` | int | O | — | 블레이드 수 |
| `hub_center` | list(3) | O | **L0 lu** | 허브 위치 (UnitConverter가 [m]로 변환) |
| `radius` | float | O | m | 블레이드 반경 |
| `rpm` | float | * | rpm | 회전수. `rpm` 또는 `omega` 중 하나 필수 |
| `omega` | float | * | rad/s | 각속도 |
| `rotation_axis` | list(3) | | — | 회전축 단위벡터 (기본 `[1,0,0]`) |
| `thrust_direction` | list(3) | | — | 추력 방향 (기본: -rotation_axis) |
| `theta_0` | float | | rad | 초기 방위각 (기본 0) |

### blade (rotor 하위)

```python
"blade": {
    "sections": [
        {"r": 0.028, "chord": 0.021, "twist": 34.26,
         "airfoil": "e63", "active": True},
        # ...
    ],
},
"grid": {"n_radial": 40},
```

| Section key | Type | 단위 | 설명 |
|-------------|------|------|------|
| `r` | float | m | 반경 방향 위치 |
| `chord` | float | m | 현지 코드 길이 |
| `twist` | float | deg | 비틀림/피치각 (rotor plane 기준) |
| `airfoil` | str | — | 익형 이름 (airfoil_polar의 키와 일치) |
| `active` | bool | — | True=공력 하중 계산, False=hub 비활성 구간 |

### rotors (복수 로터 — rotor 대체)

```python
"rotors": [
    {"name": "upper", "rotor": { ... }},
    {"name": "lower", "rotor": { ... }},
]
```

### coeff_mode

| 모드 | C_T 정의 | 자동 선택 조건 |
|------|---------|---------------|
| `rotorcraft` | T / (ρ·A·(ωR)²) | U_inf < 0.01·ωR |
| `wind_turbine` | T / (0.5·ρ·U²·A) | U_inf ≥ 0.01·ωR |
| `auto` | 위 조건으로 자동 판별 | — |

> 터미널 summary와 CSV에는 propeller/rotorcraft 양쪽 convention 모두 출력.

---

## 12. conservation

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `enabled` | bool | True | 질량 보존 모니터 활성화 |
| `verbose` | int | 0 | 0=quiet, 1=summary, 2=detailed |
| `log_to_csv` | bool | True | CSV 파일 기록 |

---

## 13. convergence

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `enabled` | bool | False | 수렴 감지 활성화 |
| `on_converged` | str | `"checkpoint_and_stop"` | 수렴 시 동작 |
| `on_diverged` | str | `"stop_with_checkpoint"` | 발산 시 동작 |

### cauchy (수렴 기준)

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `epsilon` | float | 1e-5 | 운동에너지 변화 임계값 |
| `Cd_epsilon` | float | 1e-3 | Cd 변화 임계값 |
| `window_size` | int/str | `"auto"` | 이동 평균 윈도우 |
| `n_required` | int | 3 | 수렴 판정 연속 횟수 |

---

## 14. force_calculation (MEM)

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `enabled` | bool | False | MEM force 측정 활성화 |
| `interval` | int | 10 | 측정 주기 [steps] |
| `start_step` | int | 0 | 과도 상태 건너뛰기 |

### reference (무차원화)

| Key | 단위 | 설명 |
|-----|------|------|
| `rho` | — | 참조 밀도 (기본 1.0) |
| `velocity` | Δx/Δt | 참조 속도 (기본 numerics.u_max) |
| `char_length` | lu | 특성 길이 (기본 grid.resolution) |
| `span_length` | lu | Span 길이 (2D=1, 3D=Nz 등) |

> **internal_geometry에 장애물이 있어야 측정 가능.**

---

## 15. output

| Key | Type | Default | 설명 |
|-----|------|---------|------|
| `output_dir` | str | `"./results/vtk"` | VTK 출력 경로 |
| `checkpoint_dir` | str | `"./checkpoints"` | 체크포인트 경로 |
| `csv_dir` | str | `"./results/csv"` | CSV 로그 경로 |
| `clear_previous` | bool | False | 이전 출력 삭제 |

### vtk

| Key | Default | 설명 |
|-----|---------|------|
| `enabled` | True | VTK 출력 |
| `precision` | `"float32"` | 데이터 정밀도 |
| `variables` | — | `["density","velocity","pressure","solid_mask",...]` |

### checkpoint

| Key | Default | 설명 |
|-----|---------|------|
| `enabled` | True | 체크포인트 활성화 |
| `keep_last_n` | 3 | 최근 N개만 유지 |

---

## UnitConverter가 자동 유도하는 값

```
dx_phys = L_char / resolution
U_max_phys = max(U_inf, tip_speed)
dt_phys = u_max * dx_phys / U_max_phys
nu_phys = Re_U_ref * Re_L_ref / Re   (or 직접 'nu')
nu_lu   = nu_phys * dt_phys / dx_phys²
tau     = 0.5 + 3 * nu_lu
U_inf_lu = U_inf * dt_phys / dx_phys
```

### 안정성 자동 검증

- `tau <= 0.5` → ValueError (증가 해상도 or 감소 Re)
- `Ma = u_max / (1/√3) > 0.3` → ValueError (감소 u_max)
- `Ma_tip > 0.3` → ValueError (ALM)

---

## 의존성 및 호환성

### 함께 사용해야 하는 옵션

| 기능 | 필요 조건 |
|------|----------|
| ALM (`actuator_line.enabled`) | `airfoil_polar` 필수 |
| ALM + multi airfoil | `airfoil_polar.method = "multi"` + blade section 키 일치 |
| ALM + MLG | hub_center가 finest level region 내부 |
| MEM force (`force_calculation.enabled`) | `internal_geometry`에 장애물 |
| Sponge BC | `thickness`, `strength`, `velocity`, `density` 지정 |
| Cumulant 전용 파라미터 | `collision_model = "cumulant"` |

### 함께 사용할 수 없는 옵션

| 조합 | 이유 |
|------|------|
| `bgk` + Re > ~8,000 | 수치 불안정 |
| `force_calculation` + 장애물 없음 | 측정 대상 없음 |
| `rotor.omega` + `rotor.rpm` | 하나만 지정 (omega 우선) |
| `dimension = 2` + `D3Q27` | 차원 불일치 |
| `dimension = 3` + `D2Q9` | 차원 불일치 |
| `simulation.physics` 또는 `simulation.domain` 중첩 | **Legacy 포맷 — 제거됨** |

### 권장 설정

| 조건 | 권장 |
|------|------|
| Re < 1,000 | `collision_model = "bgk"`, tau > 0.55 |
| Re > 10,000 | `collision_model = "cumulant"` |
| Hover (U_inf=0) | `coeff_mode = "rotorcraft"` 또는 `"auto"` |
| Axial flight (J>0) | `coeff_mode = "auto"`, lateral BC를 vector로 |
| Sponge outlet | `velocity`를 freestream과 동일하게 |
| MLG + ALM | `overlap_width ≥ 4`, gaussian_cutoff = 3.0 |

---

## 최소 예시 (2D 익형)

```python
config = {
    "simulation": {
        "device_mode": "gpu", "precision": "float32",
        "dimension": 2, "lattice_model": "D2Q9",
        "collision_model": "cumulant",
    },
    "physics": {
        "rho": 0.015, "U_inf": 144.0, "Re": 20000.0,
        "L_char": 0.1046,                 # [m] chord
        "flow_direction": [1.0, 0.0],
    },
    "grid":     {"Nx": 1024, "Ny": 640, "resolution": 64},
    "numerics": {"u_max": 0.05, "collision": "cumulant"},
    "boundaries": {
        "inlet":  {"location": "xmin", "method": "equilibrium",
                   "velocity": [144.0, 0.0]},
        "outlet": {"location": "xmax", "method": "sponge",
                   "velocity": [144.0, 0.0], "density": 1.0,
                   "thickness": 40, "strength": 0.5},
        "ymin":   {"location": "ymin", "method": "equilibrium",
                   "velocity": [144.0, 0.0]},
        "ymax":   {"location": "ymax", "method": "equilibrium",
                   "velocity": [144.0, 0.0]},
    },
    "internal_geometry": {
        "airfoil": {
            "enabled": True, "selig_file": "/path/clf5605.dat",
            "chord": 64, "center": [192, 320], "angle_of_attack": 2.0,
        },
    },
    "interval": {"output": 200, "log": 50, "checkpoint": 1000},
    "time":     {"max_steps": 10000},
    "output":   {"output_dir": "./result/vtk",
                 "checkpoint_dir": "./result/ckpt",
                 "csv_dir": "./result/csv", "clear_previous": True,
                 "vtk": {"enabled": True, "precision": "float32",
                         "variables": ["density", "velocity", "solid_mask"]}},
}
```
