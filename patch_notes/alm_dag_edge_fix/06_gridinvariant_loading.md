# 06 — 격자 불변 하중 정규화 + 모니터 C_T 표시 (2026-07-06)

대상: `src/utilities/compare_spanwise.py`, `src/solver/output_manager.py`
계기: slab5(5-level) vs light(4-level) 비교 시 F_n·F_θ·thrust_lu가 4× 차이 →
"스케일 오류인가?" (사용자). 원인 규명 + 수정.

## 근본 원인 (스케일 아티팩트, 버그 아님)

ALM은 **finest 레벨**에서 작동(R_lu=256 slab5 / 128 light). 마커 절대 힘:
  F_n[lu] = ρ_lu·u_rel²·c_lu·dr_lu
u_rel(=0.1)는 레벨 불변이나 **c_lu, dr_lu는 finest 셀 길이**라 해상도에 비례.
→ F ∝ (해상도)² → 5-level이 4-level의 **정확히 4×**. 검증(같은 r/R 마커):
r_lu·ε_lu·dr_lu ratio 2.000, u_rel·CL·CD ratio ~1.00, **F_n·F_θ ratio 4.03~4.09**.
thrust_lu(21.82 vs 5.19)·area_lu(205887 vs 51471)도 동일 (해상도)² 스케일.

## 정규화 코드는 이미 정확 (확인 완료)

- **dx_phys = finest 레벨 dx** (uc.dx_phys; R_lu=R_phys/dx_phys=128/256로 확인).
  Mach = u_lu·(dx/dt)/a 는 dx/dt 비(레벨 불변)라 정확.
- **C_T = T/(ρ·A·(ωR)²)**: T∝res², A∝res², (ωR)=0.1 불변 → **res² 상쇄, 격자 불변**.
  실측 검증: light 0.01045 vs slab5 0.01065 (차이 1.9%=실제 물리+도메인, 스케일 무).
- **to_physical_units** (F_phys=F_lu·ρ·dx⁴/dt², dx=finest): force_scale ratio
  0.5⁴/0.5²=0.25 → T_phys ratio = 4.2×0.25 = 1.05 ≈ 1 (격자 불변 확인).
- CSV(rotor_performance)는 dim force + norm params 저장 → post서 CT 도출, 정확.

**결론: 물리 정규화는 코드에 올바르게 반영돼 있음. 문제는 "raw lattice 값을
그대로 노출"한 두 곳뿐.**

## 수정 1 — compare_spanwise: 격자 불변 하중 패널

- load_spanwise: span 자체의 r_lu/r_R로 R_lu 복원 → `dCT_n = F_n/(ρ·A·u_tip²)`,
  `dCT_t = F_θ/(ρ·A·u_tip²)` 추가(마커별 무차원 하중 기여; Σ=C_T/blade).
- 플롯 F_n·F_θ 패널 → **dCT_n·dCT_t로 교체**(격자 간 곡선 겹침; 그림
  `260706_5level_test/testB_260706_norm_spanwise_compare.png` 하단 좌2 확인).
- tip_metrics: `CT_n=ΣdCT_n`(격자 불변) 추가, Fn_sum은 "grid-scale" 명시 유지.
- 표·판정 출력: "적분 C_T/blade(grid-invariant)"로 교체(구 "T=sumFn +307%"=
  스케일 아티팩트였음). per-blade 플롯도 dCT로.

## 수정 2 — output_manager 모니터: T_lu → C_T 우선 표시

`_update_progress`가 raw `T_lu`(finest lattice, ∝res²) 노출 → 4·5-level 런서
동일 물리인데 4× 점프 = 오독. **C_T(격자 불변, perf['C_T']/`_last_ct`)를 주
지표로**, T_lu는 "lattice-scale" 명시하고 보조 유지(단·다중로터 both).
※ 물리 thrust[N]은 rho_phys가 output_manager에 미배선 → C_T로 충분(호버 표준
지표). 필요 시 rho_phys 배선해 T_phys[N] 추가 가능.

## 검증
py_compile OK. testB 3-run 재현: dCT 패널 3격자 겹침, CT/blade 0.00261/0.00266/
0.00260(×4=rotor CT 0.0104~0.0107, rotor_performance와 일치), 모니터 postfix
{C_T:0.01065, T_lu:21.820} 정상.
