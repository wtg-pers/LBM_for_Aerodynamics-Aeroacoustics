# 06 — Phase 2 통합 (free-wake → 솔버) 완료·검증

**파일:** `src/actuator/actuator_line.py`, config `hvab_hover_c10_kleine_free.py`.

## 변경
1. **필드**: `_kleine_wake_mode`("straight"|"free"), `_kleine_wake_nw`(=50), `_kleine_wake`(per-blade FreeWake).
2. **`_convect_and_shed_wake(u_field,positions,dt,xp)`**: wake 점에서 `_sample_trilinear`로 **비보정
   CFD 속도** 샘플 → Euler 이류 → 현재 마커 위치에 새 ring shed (per-blade).
3. **`step()`**: kleine+free면 BEM 전에 convect+shed 호출.
4. **`_kleine_w_corr`**: free + 링≥2면 `A = Δr·(freewake_influence(ctrl3d, wake.rings, eps, axis) @ G)`
   매 스텝 재빌드. 링<2면 straight A(Phase1) fallback.
   - ★**축방향 부호**: `axis = -sign(ω)·rotation_axis`. (u_n은 +axis로 측정되나 free-wake 유도속도의
     deficit 투영이 회전 sense에 의존 → Phase1/Dağ의 회전-불변 de-induction과 부호 맞춤. 검증함.)
5. **loader**: `eps_correction.wake`, `n_w`.

## 검증 (smoke + loader)
- **부호**: free w_tip 양수(de-load), straight와 동일 부호. (수정 전엔 −부호로 뒤집힘이었음.)
- **Phase1 환원**: 링 1개(cold)면 free=straight. 짧은 wake에서 free≈straight(부호+크기 근접).
- **end-to-end**: dag/straight/free 모두 유한 force, Γ persist, wake 빌드업(이류·shed). loader 3종 정상.

## ★정직한 관찰
**synthetic smoke(순수 -z 다운워시, 수축 없음)에선 free≈straight**(Δw~-5%, helix만·수축無).
**실제 팁 개선(수축으로 팁와류 inboard 감김 → 추가 유도)은 LBM hover에서만** → HVAB 런이 진짜 테스트.
통합 자체는 검증 완료.

## 성능
free: per-step 영향행렬 재빌드(24.7ms×blade≈100ms) + wake점 이류 샘플(~9600점/step).
18rev≈30–45min CPU. 정상hover면 K스텝마다 재빌드 최적화 여지(후속).

## 다음 (사용자, 클러스터)
HVAB A/B: `hvab_hover_c10_{,kleine,kleine_free}.py` (baseline / Dağ / Phase1 / Phase2) +
기존 dag epscorr. 팁 φ·α·CT·FM + fallback 빈도. free가 수축으로 straight 능가하는지 확인.
