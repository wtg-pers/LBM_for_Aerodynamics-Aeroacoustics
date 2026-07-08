# Q2 — free-wake tip-only shedding (Task 3 item 2) 구현·검증 (2026-06-30)

free-wake를 **지정 spanwise 마커에서만 shed**하도록. 목적=free-wake 애초 의도(팁 rollup)에 맞춘
최소 팁-와류 모델(단일 팁 필라멘트).

## 변경
- `__init__`: `_kleine_wake_markers="all"`(기본). config `eps_correction.wake_markers`:
  `"all"`(전스팬, byte-identical) | `"tip"`(최외곽 활성 1) | int N(외곽 N) | float f(r/R≥f) | list(idx).
- `_shed_idx(blade)`: spec → 마커 인덱스(또는 None=all). 블레이드 공통.
- `_convect_and_shed_wake`: `shed(pos_k)` → `shed(pos_k[shed_idx])` (subset).
- `_kleine_w_corr` free 분기: 링이 len(shed_idx)점 → `eps_src=eps[shed_idx]`,
  `G_used=G[shed_idx,:]`(src×n), `A=B@(dr_sub[:,None]·G_used)`.
- **★잠재 버그 수정**: 기존 `A=dr*(B@G)`는 **array dr서 잘못된 축(Γ index) 스케일**. 정합 형태로
  교정(`B@(dr[:,None]·G)`). all+scalar는 기존식 유지 → **byte-identical 보존**.

## 검증 (`test_q2_tipwake.py`)
- (d) shed_idx: all→None, tip→[19], int3→[17,18,19], 0.9→[17,18,19] ✓
- (a/b) end-to-end free-wake: **tip 모드 ring_points=1**(단일 필라멘트), A(n,n), 力·w_corr finite ✓
- (c) **all+scalar A == legacy `dr*(B@G)` maxabs=0.0** (byte-identical) ✓

## config
- `hvab_hover_c10_kleine_free_tip.py` (kleine free + `wake_markers="tip"`). 짝=기존 `_kleine_free`(전스팬).
  A/B: 팁 φ/α 회복 + fallback 빈도 + walltime(필라멘트 1점→↓).

## 주의 (n_w "1 level")
n_w=1은 free-wake 비활성(필라멘트엔 ≥2 링 필요 → straight fallback). "tip 1 level"=spanwise 1점(tip)
=wake_markers="tip"로 충족. 시간깊이 n_w는 별도(기본 50).
