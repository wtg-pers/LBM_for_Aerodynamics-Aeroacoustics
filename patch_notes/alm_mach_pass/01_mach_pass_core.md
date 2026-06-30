# ALM Mach-pass — core 구현 (single-airfoil)

**날짜:** 2026-06-23
**목적:** 테이퍼/가변시위 로터(HVAB)에서 등현 Re→M 트릭이 깨지므로, BEM 루프에서
per-element 국소 Mach를 직접 계산해 폴라에 전달. **기본 inert**(등현 폴라는 `mach` 인자 없음
→ 기존과 bit-identical).

## R1~R4 사전 확인 (코드 근거)
- **R1** 시그니처 휴리스틱: 3-arg 폐쇄는 multi-airfoil `unified_query(...,airfoil_name)` 하나뿐
  → 이름기반 탐지로 전환 안전.
- **R2** Mach 단위/fine-level: model이 dx_phys/dt_phys 저장(actuator_line.py:201-203); fine-level은
  `dx_fine=dx_L0/2^k, dt_fine=dt_L0/2^k`(setup.py:1737-8, 1775-6). `dx/dt` 비율 레벨 불변 +
  u_rel_lu convective 보존 → `M=u_rel·(dx/dt)/a` 모든 레벨 일관. polar_query L0/fine 공유(1774).
  a_phys = `self.c_s_phys`(setup.py:514).
- **R3** multi+mach: 매니저 per-airfoil 폐쇄 + unified_query에 mach thread (다음 단계).
- **R4** sweep: 마커 직선 radial → 30° swept tip(외측 5%) 표현 불가 → 직선 근사·한계 명시.

## 변경 (구현됨)
1. `actuator_line.py` `__init__`: ctor `sound_speed=None` 추가 → `self.a_phys`. 시그니처 탐지
   이름기반: `_multi_airfoil='airfoil_name' in params`, **신규** `_polar_wants_mach='mach' in params`
   (len≥3 휴리스틱 제거).
2. `actuator_line.py` BEM 루프: Re 다음 `Mach=u_rel·(dx_phys/dt_phys)/a_phys` (단 `_polar_wants_mach
   and a_phys`일 때만, 아니면 None). 호출은 4조합(single/multi × mach/no-mach); **no-mach 경로는
   기존 2/3-arg 호출과 동일**, mach 경로만 `mach=...` kwarg 추가.
3. `c81_loader.py`: `make_c81_polar_query_mach(polar)` → `polar_query(alpha,Re,mach)` (mach 직접
   사용). 기존 `make_c81_polar_query`(등현) 불변.
4. `actuator_line.py` 팩토리 2종(single/multi): `sound_speed` 파라미터 + 전달.
5. `setup.py`: ALM 생성 3곳(L0 single/multi, fine-level)에 `sound_speed=self.c_s_phys`.

## 검증
- **회귀(등현 CT 스모크)**: `ct_hover_smoke.py --max-steps 6` → **T_lu=0.080959 = baseline과
  bit-identical**. 등현 c81은 `mach` 미선언 → Mach-pass inert. 기존 동작 완전 보존.
- **기능(Mach-native 폐쇄)**: 시그니처 탐지 정확(`['alpha_deg','Re','mach']`→wants_mach True).
  Mach 의존 CL 물리적 정확: α=4°에서 M0.2→CL0.446, M0.4→0.475(PG), M0.6→0.529, M0.7→0.292
  (천음속 CL 급강하). 
- compile OK: actuator_line / c81_loader / setup / airfoil_data.

## 남은 단계 (HVAB용)
- **R3 multi-airfoil + mach**: `airfoil_data.py` MultiAirfoilPolarManager.to_unified_query를
  `unified_query(alpha, Re, airfoil_name=None, mach=None)`로 확장 + per-RC Mach-native 폐쇄.
  create_polar_from_config에 multi-RC c81 경로.
- RC polar 생성(`c81_from_neuralfoil.py`), HVAB config/grid(tip chord ≥16 cell), smoke.
