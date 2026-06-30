# Kleine 2022 비반복 vortex smearing 보정 — 구현 설계 계획

작성: 2026-06-26 / 근거: `docs/papers_kr/2022_noniterative_vortex_smearing_correction_kr.md` (Kleine, Hanifi,
Henningson, arXiv 2206.05448, JFM). 관련: [[docs/almlbm_paper_analysis_kr.md]] §0.5,
현행 Dağ 구현 `src/actuator/actuator_line.py:475` `_viscous_core_correction`.

## 0. 목표·범위·우선순위

**목표**: 우리 ALM의 팁 유도속도 결손을, smeared 와류와 ideal(특이) 와류의 **차이(누락속도)**를
**비반복(선형화)**으로 더해 양력선(LLT) 결과를 재현한다. 사용자가 제안한 "free-wake filament +
Biot-Savart 유도속도 갱신"의 엄밀 정식화 = Kleine §3.4(free-vortex wake) + §5(비반복 선형해).

**우선순위(갱신된 경로)**: ① 해상도(fine/tip5, Merabet 레시피)로 먼저 결손 축소 → ② **잔차**에 본
보정. 즉 Kleine는 "해상도로 못 메운 부분 + DGX를 못 쓰는 케이스용 모델 보정". (ABC가 샘플러는
반증했고 Dağ single-pass는 7%였으므로, 본 보정은 Dağ의 **정확·강화판**.)

**핵심 안전장치(이중계산 회피)**: 누락속도 = `u_m = u^vi(ideal) − u^v(smeared, core ε)` (Kleine 식
2.12, 3.23). LBM이 이미 가진 smeared 유도(u^v)를 빼므로 **full Biot-Savart를 그대로 더하지 않는다.**

## 1. 현행 Dağ 대비 무엇이 바뀌나

| 항목 | 현행 `_viscous_core_correction`(Dağ) | Kleine 2022 (목표) |
|---|---|---|
| 커널 | 근사 `exp(−(d/ε)²)/d` | **정확 해석식 Φ (식 3.20)**, ideal Φ^i (3.21) |
| 후류 기하 | 직선 trailed (2D LL 근사) | **free-vortex wake**(CFD 속도 이류, 팁 rollup/수축 포착) |
| 폐쇄 | single-pass(현재 Γ로 1회) | **선형화 비반복**(현재 Γ-속도 양립성 유지, 식 5.15) |
| 완화계수 | `relax`(기본 1.0) | **불필요**(N×N 직접풀이) |
| 추가 입력 | Cl 테이블 | Cl 테이블 + **∂Cl/∂α** |
| 정확도(ALM↔LLT) | ~7% 회복(실측) | 유도속도 0.01% (논문) |

## 2. config 스키마 (기존 `eps_correction` 확장, 하위호환)

```python
eps_correction = {
    "enabled": True,
    "method": "kleine",          # "dag"(현행 기본) | "kleine"(신규)
    # --- kleine 전용 ---
    "wake": "prescribed",        # Phase1 "prescribed" | Phase2 "free"
    "n_w": 50, "n_nw": 10, "d_w_over_eps": 0.5,   # free-wake bookkeeping(논문값)
    "per_blade": True,           # 블레이드 독립 풀이(간격≫ε이면 유효, 저비용)
    # --- dag 전용(method="dag"일 때만) ---
    "target": "inviscid", "eps_opt_factor": 0.25, "relax": 1.0,
}
```
`method` 미지정/`"dag"` → 현행 코드 그대로(bit-identical). `loader`(actuator_line.py:1206) 확장.

## 3. 모듈 아키텍처

신규 `src/actuator/smearing_correction.py`:
- `class KleineCorrection`: 와류 상태(블레이드별 shed segment 목록) 보유, per-step `solve()` 제공.
  - `__init__(rotor, eps, params)` — 기하·ε·파라미터.
  - `state`: 블레이드별 `[{pos[:,3], Gamma, age}]`(free-wake) 또는 prescribed 기하.
  - `solve(u_markers, Gamma_prev, polar_query, dCl_dalpha) -> u_corrected, Gamma_new` — 식 5.1~5.15.
  - `shed_and_convect(Gamma_new, u_field_sampler)` — Phase2 후류 갱신.
- 커널 함수(모듈 레벨, 순수): `phi_smeared(r,Z,eps)`(식 3.20), `phi_ideal(r,Z)`(식 3.21),
  `seg_induced_velocity(...)`(식 3.17~3.23, 벡터 Biot-Savart + cutoff g 식 3.18).
- CPU 전용(마커 ~192, N×N 풀이는 `numpy.linalg.solve`로 무시 가능 비용). 샘플 속도는 이미 CPU.

`actuator_line.py` 통합: `_compute_bem_forces`의 `if self._eps_corr` 블록(:627)에서
`method=="kleine"`면 `KleineCorrection.solve()` 호출, 아니면 현행 Dağ 경로. 와류 상태는
`self._kleine`(모델 보유)에 persist.

## 4. per-step 알고리즘 (블레이드별, Kleine 식 매핑)

1. **샘플 속도** `u_s` (현행 sampling 그대로; actuator_line.py:317 결과 재사용).
2. **누락속도(직전 Γ)** `u_m(Γⁿ⁻¹)`: 와류(속박+shed)에서 deficit 커널 `(Φ^i−Φ)`로 (식 3.23).
3. **기준점 †**: `u† = u_s + u_m(Γⁿ⁻¹)` (5.1~5.3) → `α†, u_r†, Γ†` (5.4).
4. **영향계수** `A_y^mc, A_z^mc`: 현재스텝 shed + 속박 와류 → 제어점, deficit 커널 Biot-Savart (5.7~5.10).
5. **민감도** `b_y†, b_z†` (A5/A6의 † 버전): `Cl(α†)`, **`∂Cl/∂α(α†)`**, `u_y†,u_z†,u_r†`.
6. **선형해** `[I − diag(b_y†)A_y^mc − diag(b_z†)A_z^mc] ΔΓ = Γ† − Γⁿ⁻¹` (5.15) → `numpy.linalg.solve`.
7. **보정속도·힘** `u_c = u† + A^mc ΔΓ` (5.12/5.13) → `α_c` → `recompute_velocity_triangle` → 폴라 → 힘.
   `Γⁿ` 저장(다음 스텝 기준점).
8. **(Phase2) 후류 갱신**: `Γⁿ`의 trailed(dΓ/dr)로 새 segment shed, 기존 segment를 **샘플 CFD 속도로
   이류**(Euler), `age>n_w` 폐기, 간격<d_w 융합(Γ 평균). 보정속도 계산 **전에 미리 이류**(현재 Γ가
   시트 기하에 영향 안 줌 → 선형화와 양립, 5.5 분해).

## 5. 단계별 구현(Phase) — 리스크 순

- **Phase 0 — ∂Cl/∂α 인프라** (선행 필수, 저위험):
  c81/csv 폴라에 **lift-curve-slope 질의** 추가. `airfoil_data.py`/`c81_loader.py`에 PCHIP(shape-preserving
  cubic) 보간으로 `dCl/dα(α, Re, M)` 매끄럽게(논문 §6.1: 기울기 불연속 회피). 기존 Mach-pass 보간
  인프라 재사용. 단위검증: 평판 2π/rad 근사.
- **Phase 1 — prescribed-wake 비반복** (중위험, Dağ 즉시 능가):
  직선 semi-infinite trailed 와류(현 Dağ 기하 재사용) + **정확 Φ 커널** + **선형해(식 5.15)**.
  bookkeeping 불필요. → Dağ 대비 (정확커널+선형화)만으로 개선. CT/HVAB에서 Dağ·Prandtl과 A/B.
- **Phase 2 — free-vortex wake** (고위험, 사용자 제안 본체):
  shed segment 이류·bookkeeping(n_w=50,n_nw=10,d_w=ε/2 → 팁 ~20ε). 팁와류 rollup/수축 포착 →
  팁 결손의 잔차를 더 잡을 잠재력. 메모리·정렬 주의(블레이드별 list, 최근 n_w만 유지).
- **Phase 3 — 검증·통합**:
  ① translating wing(Kleine §7.1, 가능하면) — LLT 0.01% 재현 확인.
  ② HVAB c10: light+Kleine vs (fine 해상도) vs (Dağ) vs (Prandtl) — 팁 φ·α·CT·FM.
  ③ MLG fine-level ALM에 연결(좌표 fine-lu, 이미 sampler가 쓰는 hub/axis/R 재사용). smoke(CPU) 게이트.

## 6. 통합 지점 (file:line)
- 분기/호출: `actuator_line.py:627` (`if self._eps_corr` 블록) → method 분기.
- 기하: `rotor.hub_center/rotation_axis/radius`(lu), `blade.marker_r/marker_epsilon/marker_dr`,
  `rotor.get_blade_unit_vectors`(ê_n,ê_θ,ê_r).
- 삼각형 재계산: `rotor.recompute_velocity_triangle`(:504) 재사용.
- 폴라+기울기: `self.polar_query` + 신규 `polar_slope_query`(Phase 0).
- config loader: `actuator_line.py:1206` eps_correction 파싱 확장(method/wake/n_w...).

## 7. 리스크·주의
1. **이중계산**: deficit form(ideal−smeared) 필수. 검증: 보정 후 팁 하중이 LLT 위/아래로 **튀지 않음**.
2. **∂Cl/∂α 급변**: 실속 부근 기울기 변동·시간스텝 급변 시 선형화 약화 → 논문 권고대로 그 케이스만
   "비반복 초기화 + 반복 마무리" 하이브리드 fallback(Phase 2+ 옵션).
3. **해상도 의존(잔존)**: Kleine App.B — ε/Δx<4면 코어 미해상으로 보정 정확도↓. 즉 **light(ε/Δx=2)에선
   Kleine도 약할 수 있음** → 해상도(fine/tip5)와 **병행**해야 효과. 우선순위 ①해상도 ②Kleine 유지.
4. **per-blade 독립 가정**: 블레이드 간격≫ε(HVAB OK)일 때만. 허브 근처·근접 블레이드는 결합 풀이.
5. **GPU/MLG**: 보정은 CPU(작은 N). 단 매 스텝 호출이라 free-wake bookkeeping이 병목 안 되게 벡터화.

## 8. 산출물(예정)
- `src/actuator/smearing_correction.py` (KleineCorrection + 커널)
- `airfoil_data.py`/`c81_loader.py` 기울기 질의(Phase 0)
- `actuator_line.py` method 분기 + loader
- 검증: `configs/hvab/*_kleine_c10.py` + CPU smoke + spanwise A/B(`compare_sampler_abc` 확장 or 신규)
- 단계별 패치노트 `patch_notes/kleine_smearing_correction/01..` ([[feedback_stepwise_patch_notes]])

## 9. 열린 결정(사용자 확인 필요)
- **a)** Phase 1(prescribed)부터 단계적으로 vs 바로 Phase 2(free-wake)? → 권장: Phase 1 먼저(Dağ 즉시
  능가·저위험, free-wake 추가이득은 잔차 측정 후).
- **b)** ABC/해상도 결과로 잔차가 작으면(해상도로 대부분 해결) Kleine는 "저해상 케이스용"으로만 둘지.
- **c)** 검증 기준: translating wing(LLT 0.01% 재현, 구현부담 큼) 포함 vs HVAB만(실용).
