# 04 — 폴라조회 벡터화 (`_lookup_cl_cd` batch path)

날짜: 2026-07-04 · 대상: `src/actuator/actuator_line.py`, `src/actuator/airfoil_data.py`,
`src/actuator/c81_loader.py` · 검증: `test_polar_batch.py` (ALL PASSED)

## 배경 — GPU util 병목 확정

`diag_pure_alm` + `ALM_PROFILE=1` (RTX 3090, light preset):

```
sample=1.8  bem=21.0  spread=0.4  ms/call  (fine sub-step당, GPU-synced)
```

bem 21ms의 정체 = `_lookup_cl_cd`의 **마커별 Python 루프**: fine sub-step당
4블레이드 × 48마커 = **192회 `polar_query` 호출**, 호출마다
unified_query → manager.query → `get_query()` **선형 이름탐색** → C81 보간
(scipy RegularGridInterpolator를 점 1개씩). 8 sub-step × 23ms ≈ 186ms
= 코어스텝(709ms)의 26% → util 67~78%와 정합. 보정 활성 시 재조회로 2×.

## 수정 (3파일)

1. **`c81_loader.py`**: 두 closure(`make_c81_polar_query{,_mach}`)에
   `supports_batch=True` 마킹. `_C81Table.__call__`은 원래 배열 broadcast +
   단일 보간 호출을 지원 — 스칼라와 **동일 보간자·동일 클램프** 경로.
2. **`airfoil_data.py`**:
   - `MultiAirfoilPolarManager.get_query()`에 이름 해석 캐시(선형탐색·fallback
     경고를 이름당 1회로).
   - `to_unified_query()`가 closure에 `manager` 핸들 부착 → ALM이 per-name
     query 함수를 직접 해석 가능.
3. **`actuator_line.py` `_lookup_cl_cd`**:
   - 마커→익형 그룹(정적)을 blade당 1회 계산·캐시(`_polar_groups`,
     identity 체크로 무효화), query 함수도 이름당 1회 해석(`_polar_qf_cache`).
   - 그룹별 **한 번의 배열 호출**(`supports_batch` C81). mach 전달 규칙은
     원본과 동일(멀티=wants_mach일 때만, 싱글=Mach 비-None이면).
   - 스칼라 전용 폴라(CSV/NeuralFoil)는 기존 per-marker 루프 유지(무회귀).
   - **`ALM_POLAR_BATCH=0`** 환경변수 → 구 경로 강제(A/B·회귀 검증용).

## 검증

- **bit-identical**: HVAB nasa_overflow 4덱(멀티+Mach-pass) 48마커,
  inactive/α·Mach 클램프/u_rel<1e-10 포함 — 구루프 대비 `array_equal` 통과.
  no-Mach 경로도 동일. `ALM_POLAR_BATCH=0` 경로도 동일.
- **속도(조회만)**: 4블레이드 lookup 8.0ms → 0.7ms (**11×**).
- **end-to-end**: diag_pure_alm 100step A/B — 본문 하단 결과 참조.

## 주의

- 동작 변화: unknown-airfoil fallback 경고가 호출마다→이름당 1회.
- Kleine `cl_eval`(polar_slope)도 같은 함수 경유라 자동 수혜.
