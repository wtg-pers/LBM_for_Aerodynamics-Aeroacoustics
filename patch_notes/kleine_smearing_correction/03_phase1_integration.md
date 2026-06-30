# 03 — Phase 1 통합 (actuator_line) + smoke

**파일:** `src/actuator/actuator_line.py` (+ `smearing_correction.py` active param).

## 변경
1. **필드**(__init__): `_eps_corr_method="dag"|"kleine"`, `_kleine_A`(per-blade 영향행렬 캐시),
   `_kleine_gamma_prev`(per-blade Γⁿ⁻¹ warm-start persist).
2. **`_kleine_w_corr(k,blade,u_n,u_theta,cos_sweep,chord,dr,CL,active,n)`** 신규 헬퍼:
   - 영향행렬 캐시 → `correct_noniterative` 호출. `u_tan=|ω|r−sign(ω)u_θ`(rotor 일치).
   - **Re/Mach freeze @ †**(=u_n+AΓⁿ⁻¹), `cl_eval`=`_lookup_cl_cd`, `dcl`=`lift_curve_slope_batch`.
   - cold-start Γⁿ⁻¹=무보정 Γ. inactive 마커는 `active`로 b=0 (root 안정).
   - **안전망**: 비유한 or `max|w|>0.5·max|u_tan|`(불안정 fixed point) → **Dağ single-pass fallback**
     (나쁜 Γ persist 안 함). LinAlgError도 fallback.
3. **eps_corr 블록 분기**(:632): `method=="kleine"`→헬퍼, else Dağ. 다운스트림(삼각형 재계산·폴라
   재조회) 공유. **method 미지정/"dag"→bit-identical 회귀.**
4. **loader**(:1212): `eps_correction.method`.

## smoke (CPU, end-to-end, PASS)
`ActuatorLineModel`(from_simple) + 합성 다운워시장 + `step()`:
- **dag 회귀**: 정상(팁 α9.18°, w+2.0e-3).
- **kleine, constant-chord**(Γ rolloff 없음=불안정 fixed point): 안전망이 매 스텝 **Dağ fallback**
  → 안정(w=dag값). (frozen-field 합성이라 보수적; 실제 LBM은 유동응답이 damp.)
- **kleine, taper(Γ rolloff)**: **Kleine 활성·안정 수렴** (Γ persist, `max|w_dag−w_kleine|=1.4e-2≠0`).
- 오프라인 코어검증(02): warm-start 잔차 cold 3.7e-3→1e-10.

## ★핵심 한계(정직)
**팁에서 Phase 1 Kleine ≈ Dağ** (taper smoke: 팁 Δα=−0.01°, w 3.03e-3 vs 3.16e-3). 같은
semi-infinite 직선-wake deficit 커널이라 팁 크기는 Dağ와 동급. 차이는 주로 **inboard 분포**
(Σ|Δα|=24.5°, 비반복이 수렴해를 잡음). **드라마틱한 팁 회복은 Phase 2(free-wake가 팁
rollup/수축 포착 → deficit↑) + exact-Φ.** Phase 1은 (i)비반복/안정 인프라(Phase 2 토대),
(ii)수렴 inboard 분포가 가치.

## 미해결 / 다음
- 실제 HVAB CPU smoke / 클러스터 A/B(`hvab_hover_c10_kleine.py`): dag vs kleine 팁 φ·CT +
  **fallback 빈도** 관찰. fallback 잦으면 across-step under-relax(0.3~0.5) 옵션 추가 검토.
- **Phase 2**: free-vortex wake(이류·bookkeeping) + exact-Φ 유한세그먼트 커널 → 진짜 팁 개선.
