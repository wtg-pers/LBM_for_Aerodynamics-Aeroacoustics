# 05 — Phase 2 step 2-3: free-wake 계산 코어 (완료·검증)

**파일:** `src/actuator/smearing_correction.py`.

## 구현 (전부 검증)
| 함수/클래스 | 역할 | 검증 |
|---|---|---|
| `phi_smeared/phi_ideal/segment_missing_theta` | exact-Φ 커널 (Eq 3.20-3.23) | Eq3.19 수치적분 + 반무한 Lamb-Oseen 1e-16 |
| `segment_induced_velocity` | 일반기하 3D 세그먼트 deficit 속도 (벡터) | z-정렬=Φ, 직선반무한=Lamb-Oseen 1e-16 |
| `FreeWake` (class) | per-blade wake 기하: shed(n_w cap)/convect | 단위 OK |
| `_build_segments`, `_seg_vz_batch` | 세그먼트 평탄화 + 벡터 Biot-Savart | — |
| `freewake_influence` (벡터화) | B[i,m] = 축방향 유도속도, A=Δr·(B@G) | **vs loop 5.7e-16**, **직선극한=Phase1 3.4e-16** |
| `_freewake_influence_loop` | 검증용 reference (느림) | — |

## 핵심 anchor
**free-wake 영향행렬이 직선극한에서 Phase1 `influence_matrix`(=Dağ)를 기계정밀도 재현.**
→ wake가 곡선(helix/수축)이 되면 그 차이가 추가 팁 유도. 물리 검증된 토대.

## 성능
N=48, L=50: **24.7 ms/call** (loop 2050ms 대비 83×). per-step 4 blades ≈ 100ms.
18rev×1005step ≈ 30min CPU. 허용 가능하나, 정상 hover면 wake가 천천히 변하므로
**K스텝마다 A 재빌드** 최적화 여지(후속).

## 남은 것 — 통합 (Phase 2 마지막)
`_kleine_w_corr` + `step()`에 wake 연결:
1. config `eps_correction.wake = "straight"(Phase1 기본) | "free"(Phase2)`.
2. 모델 state `_kleine_wake[k]=FreeWake(n_w)`.
3. `step()`에서 (markers 샘플 후, BEM 전): per-blade **convect**(wake 점에서 u_field
   trilinear 샘플 → Euler 이류) + **shed**(현재 마커 3D 위치). u_field 접근 필요.
4. `_kleine_w_corr`: wake=="free"면 `A = Δr·(freewake_influence(ctrl3d, wake.rings,
   marker_eps, axis)@G)` 빌드 → `correct_noniterative` 투입. (straight면 Phase1 A.)
5. 4-level light CPU smoke → HVAB A/B(straight vs free).
