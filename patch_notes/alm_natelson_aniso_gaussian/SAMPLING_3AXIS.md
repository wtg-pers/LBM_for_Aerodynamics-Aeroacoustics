# 비등방 샘플링 3축화 + MPI (2026-07-21)

## 문제 (사용자 지적)

기존 `_sample_aniso`(sampling mode "aniso")는 **radial-vs-perpendicular
2방향**(ε_r, ε_⊥)뿐이라 스프레딩의 **3축**(chord/thickness/span, Natelson
Eq.7)과 대칭이 안 맞았고, **MPI(분산) 미지원**(fail-fast)이었다. 사용자
요구: G = exp(−ξ²/ε_c² − η²/ε_t² − ζ²/ε_s²) 3축 + MPI.

## 구현

**커널** (`interpolation.py`): `_sample_aniso` 전면 재작성 —
- 입력 = blade-local 프레임 ec/et/er + 축별 폭 ε_c/ε_t/ε_r (스프레딩과
  **동일 프레임·규약**, `get_all_marker_aero_frame`).
- arg = (d_c/ε_c)² + (d_t/ε_t)² + (d_r/ε_r)², 타원체 cutoff arg≤n_cut².
- `return_sums` + `clip_bounds` 지원 → 분산 ALM 파샬섬(소유 셀만).
- `_alt_stencil`에 `clip_bounds` 소유권 마스크(`owned`) 추가.

**배선** (`actuator_line.py`): `_build_sampling_aniso(ε_iso)` = 프레임 +
ε_{c,t,r}={c,t,r}·ε_iso. 매 스텝 재구성(azimuth 회전). 단일-rank는
sample_velocity_alt, MPI는 `_velocity_sampler(..., aniso=spec)`로 전달.
MPI fail-fast를 gaussian/aniso 허용으로 수정.

**분산** (`alm_dist.py`): sampler closure가 `aniso` 받으면 `_sample_aniso`
파샬섬(clip_bounds=소유 slab) → allreduce(기존 (N,4) 구조 그대로).

**config**: `sampling={"mode":"aniso","c":1.0,"t":0.5,"r":1.0}`.
기본 c=1,t=0.5,r=1 (Churchfield 두께 sub-grid 타협, 스프레딩과 동일).
c=t=r=1 → 등방(gaussian) 정확 환원.

## 검증

- **①등방 환원**: aniso(c=t=r=1) vs gaussian 적분 = **2.2e-16** (fp).
- **②실제 비등방**: t=0.5 → 0.024 차이 (>0).
- **③MPI 파샬섬**: 2-slab 분할 합 vs 전체 = **1.1e-16** (CPU, 소유권 정확).
- **④단일-rank GPU e2e**: bench5 pure+aniso 완주 (CT 0.018912).
- **⑤2-rank MPI GPU e2e**: CT = 단일-rank **정확 일치**(rel 0.0e+00).
- caveat: cuda-aware=1 + 2랭크 1GPU는 segfault(**gaussian도 동일** =
  환경 아티팩트, aniso 무관). 클러스터 랭크당 별도 GPU는 정상.

## 다음

pure ALM 2×2 factorial (sampling {iso,aniso} × spreading {iso,aniso}):
- 단계1(iso,iso)·3(iso samp+aniso spread) = MPI 다중-GPU.
- 단계2·4(aniso samp) = **이제 MPI 지원** → 다중-GPU 가능(이전엔 단일-rank).
config·factorial 확정 후 진행 (Prandtl은 승자에 나중에).

## 물리적 폭 + spacing ε_r (2026-07-21 후속, 사용자 지적)

기본 c=1,t=0.5,r=1은 chord·radial이 ε_iso로 등방이라 사실상 thickness만
좁힌 near-등방(과거 aniso "약함"의 원인). 사용자 물리 spec 채택:
- **ε_c=0.25c (c=1.0), ε_t=0.1c (t=0.4), ε_r=0.5·δr (r=0.5, r_ref="spacing")**.
- ε_r을 δr 기반으로 쓰려 `_build_sampling_aniso`에 r_ref 추가(스프레딩 미러).
- ★버그 수정: 스프레딩 aniso가 `get_marker_spacing()`(미정의) 호출 →
  `get_all_marker_dr()`로 수정(aniso 스프레딩 spacing 경로는 실행 전무였음).
- 검증: D40 팁서 ε_c=22.0mm(0.25c)·ε_t=8.8mm(0.1c)·ε_r=9.87mm(0.5δr) 정확.
  bench5 single=2-rank MPI CT 0.01892036 정확일치.
- ★caveat: ε_t=0.1c=1.67Δx (<2Δx) sub-grid → 두께방향 샘플링 격자제약
  (Churchfield도 물리 ε_t는 sub-grid라 지적). 효과는 factorial이 판정.
- ★factorial 주의: 스프레딩 aniso의 ε_r은 0.5δr 쓰면 안 됨(δr<δr →
  라인 불연속/blob). 스프레딩은 ε_r≥δr 필요(continuity). 샘플링(국소)과
  스프레딩(연속)의 ε_r은 달라야 함 — 사용자 확정 필요.
