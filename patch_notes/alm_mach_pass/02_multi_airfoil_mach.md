# ALM Mach-pass — R3: multi-airfoil + Mach 결합 (HVAB 다종 RC)

**날짜:** 2026-06-24
**목적:** HVAB는 다종 RC 익형(RC(4)-12/RC(4)-10/RC(6)-08/RC(6)-08T) + 테이퍼라, 폴라 조회가
`(α, Re, airfoil_name, mach)` 4입력이어야 한다. 01단계(단일 익형 Mach-pass)에 이어,
multi-airfoil 매니저에 Mach를 thread.

## 변경
1. `c81_loader.py::make_c81_polar_query_mach`: 폐쇄에 `.wants_mach = True` 태그 추가
   (매니저가 per-airfoil로 mach 전달 여부 판별).
2. `airfoil_data.py::MultiAirfoilPolarManager`:
   - **`add_airfoil_from_c81(name, c81_path, set_default)`** 신규 — Mach-native c81 폐쇄를
     `_queries[name]`에 등록.
   - `query(airfoil_name, alpha, Re, mach=None)` — `mach is not None and
     getattr(qf,'wants_mach',False)`일 때만 `qf(alpha, Re, mach=mach)`; 아니면 2-arg
     (CSV/neuralfoil/flat_plate 안전).
   - `to_unified_query()` → `unified_query(alpha_deg, Re, airfoil_name=None, mach=None)`
     — **`mach` 파라미터가 있어야 ALM이 Mach-pass 탐지**. 혼합 덱(Re-only + Mach) 안전.
   - `airfoil_names` → `_queries` 기반(c81은 _databases 없이 _queries만 채움).
3. `airfoil_data.py::create_polar_from_config` `method=="multi"`: `af_method=="c81"` 분기 추가
   → `manager.add_airfoil_from_c81(path=af_cfg["path"])`.

## 검증
- compile OK (airfoil_data / c81_loader / actuator_line).
- **기능(다종 c81)**: 기존 C81 덱 2종을 multi로 묶음 →
  `unified_query` params = `['alpha_deg','Re','airfoil_name','mach']` →
  **ALM 탐지: _multi_airfoil=True, _polar_wants_mach=True (둘 다)**. 익형별·Mach별 CL 구분 확인
  (대칭형 vs 캠버형, 둘 다 M↑ PG 상승). `mach` 미지정 호출도 2-arg 폴백으로 안전.
- **회귀(단일 c81 CT 스모크)**: T_lu=0.080959 **bit-identical** (multi 경로 추가가 단일 경로 불변).

## config 사용 예 (HVAB)
```python
airfoil_polar = {
  "method": "multi",
  "default": "RC4-12",
  "airfoils": {
    "RC4-12": {"method":"c81", "path":"data/airfoils/RC4-12.C81"},
    "RC4-10": {"method":"c81", "path":"data/airfoils/RC4-10.C81"},
    "RC6-08": {"method":"c81", "path":"data/airfoils/RC6-08.C81"},
    "RC6-08T":{"method":"c81", "path":"data/airfoils/RC6-08T.C81"},
  },
}
# + 블레이드 섹션의 airfoil 이름이 위 키와 일치 + grid에 sound_speed(=c_s_phys) 자동 주입
```

## Mach-pass 전체 상태 (01+02)
- ✅ 단일 익형 Mach-pass (01): 등현 inert, CT bit-identical, Mach-native 폐쇄 작동.
- ✅ 다종 익형 + Mach (02): 매니저 thread, ALM 둘 다 탐지, 익형·Mach별 조회.
- **남은 HVAB 셋업**: RC polar 생성(`c81_from_neuralfoil.py`로 RC(4)-12/RC(4)-10/RC(6)-08[T]
  Mach C81 덱; RC 좌표 출처 확인) → HVAB config/grid(tip chord ≥16 cell, 공력 0.25~1.0,
  sweep 직선 근사) → smoke → 클러스터 production(M0.65 sweep) → sectional M²Cn vs NASA 비교.
