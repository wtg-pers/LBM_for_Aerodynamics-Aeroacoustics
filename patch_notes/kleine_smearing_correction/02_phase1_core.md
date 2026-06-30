# 02 — Phase 1 core: 비반복 선형해 (완료·검증)

**파일:** `src/actuator/smearing_correction.py` (신규) / 설계 `00_design.md` §5.

## 범위 정정 (중요)
우리 Dağ(`_viscous_core_correction`)는 이미 **semi-infinite 직선 trailed 와류의 deficit 커널**
(Lamb-Oseen `Γ/4πr·exp(-r²/ε²)`)을 쓴다. 그래서 **Phase 1의 개선 = 커널이 아니라 "비반복
선형해"**(완화계수 single-pass → 직접 1회 풀이). exact-Φ 유한세그먼트 커널(식 3.20) + 이류
free-wake는 **Phase 2**.

## 구현 (smearing_correction.py)
- `influence_matrix(r, eps, dr) -> A[i,k] = ∂w_i/∂Γ_k`
  - `A = -(1/4π)·Δr·(Kmat @ G)`, `Kmat_im=exp(-((r_i-r_m)/ε_m)²)/(r_i-r_m)`(self=0), `G`=gradient matrix.
  - `w = A@Γ`는 기존 Dağ `_viscous_core_correction(Γ)`를 **선형 행렬로 정확 재현**(np.gradient 일치).
- `correct_noniterative(...)` — Kleine 식 5.4/5.14/5.15:
  - 기준점 † : `u_n† = u_n + A@Γⁿ⁻¹`, `α†,u_rel†` → `CL†, dCL†(Phase0), Γ†`.
  - 민감도 `b†_j = ½c(u_n†·CL† − u_tan·dCL†)/u_rel†`  (= ∂Γ/∂u_n at †).
  - 풀이 `[I − diag(b†)A]ΔΓ = Γ† − Γⁿ⁻¹` (`numpy.linalg.solve`), `Γⁿ=Γⁿ⁻¹+ΔΓ`, `w=A@Γⁿ`.
- `correct_iterative_reference(...)` — 검증용 fixed-point 반복.

## 검증 (PASS)
물리적 합성 블레이드(chord taper→Γ rolloff):
- **fixed-point 잔차: cold 3.7e-3 → warm1 7.9e-6 → warm2 1.1e-10** — 타임스텝 루프(warm-start)로
  비선형 fixed point에 수렴(=Kleine의 "현재 Γ-속도 양립성").
- **부호 물리적**: 팁 w>0(downwash 추가) → α 2.82°→2.58°(de-load). Dağ 부호와 일치.
- `cond(I−diag(b)A)≈90` (가역). **완화반복(r=0.2,0.3)은 발산** → 비반복의 필요성 실증.

## 다음 (Phase 1 통합)
`actuator_line._compute_bem_forces`의 eps_corr 블록(:627)에 `method=="kleine"` 분기:
- `cl_eval`/`dcl_dalpha`를 `self.polar_query`+`polar_slope.lift_curve_slope_batch`로 바인딩(Re/Mach/airfoil).
- per-blade `u_tan = |ω|r − sign(ω)u_θ`, `eps=marker_epsilon`, `Γⁿ⁻¹`는 모델에 persist.
- config `eps_correction={"enabled":True,"method":"kleine"}` loader(:1206) 확장.
- CPU smoke(4-level light)로 end-to-end. → `03_phase1_integration.md`.
