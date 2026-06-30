# 4단계 — `eps_lu` 진단 컬럼 (4개 동기 편집)

**내용:** per-marker ε을 blade 진단 CSV에 노출해 테이퍼를 A/B로 점검하고 r/R 대비
플롯할 수 있게 함. ε은 blade에 있으므로(`BEMResult` 아님) `BEMResult` 변경 없음.

`chord_lu` 바로 뒤에 배치(기하량 그룹핑). `spanwise_post.py`는 헤더 이름으로 읽으므로
(`pd.read_csv`) 컬럼 위치 자체는 무관 — 단 **writer 순서 == header 순서**는 반드시 유지.

## 편집 4.1 — `actuator_line.py::get_blade_diagnostics`

```python
'chord': blade.marker_chord,        # [lu]
'epsilon': blade.marker_epsilon,    # [lu] Gaussian projection width
'twist': blade.marker_twist,        # [degrees]
```
`'chord'`처럼 full per-marker 배열 (모든 blade는 deep copy → ε 동일).

## 편집 4.2 — `output_manager.py::_log_blade_diagnostics` (writer)

```python
f"{diag['chord'][j]:.4f},"
f"{diag['epsilon'][j]:.4f},"
f"{diag['twist'][j]:.3f},"
```

## 편집 4.3 — `setup.py` `_blade_csv_header`

```python
'step,revolutions,blade,r_R,r_lu,chord_lu,eps_lu,twist,'
'u_n,u_theta,u_rel,phi,alpha,Re,CL,CD,'
'F_n,F_theta,F_L,F_D\n'
```
헤더 = **20**개 컬럼; writer = **20**개 필드 — 동기화 확인 완료.

## 편집 4.4 — `spanwise_post.py` 컬럼 선택 (선택적, 계획서 포함)

```python
cols = [c for c in ["r_R", "alpha", "phi", "Re", "CL", "CD",
                    "mach", "M2CL", "F_n", "F_theta", "eps_lu"] if c in df.columns]
```
`if c in df.columns` 가드 → `eps_lu` 없는 구버전 CSV와 하위 호환.

## 검증

- 4개 파일 모두 `python -m py_compile` → **OK**.
- 컬럼 개수 일치(header 20 == writer 20) 수동 확인. 5단계 스모크 런이 실제
  `blade_diagnostics/*.csv`를 쓰고 `eps_lu` 컬럼 존재·채워짐을 확인.
