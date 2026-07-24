# LINEAVG — coupled LineAverage 샘플러 (Melani 2024) (2026-07-22)

## 결정 (사용자 합의)
공식화 개정: **spreading = iso 가변 ε(0.25c, 불변) + sampling = coupled
LineAverage constant r_s=0.25c** (Melani TORQUE 2024 벤치마크 승자;
Schrenk/elliptic은 저자들이 §4.2에서 탈락 — Schrenk는 마지막 10% α 대폭
과소 = 가짜 이득 함정). N_s=80(논문값; ≥30 둔감). DELA(출력전용) 아닌
coupled(유동 반영) — CT 교정이 목적. 기준선 = mlg4+K17L+SGSoff(ILES).

## 구현
- `interpolation._sample_ring` 확장: 기존 ring(Natelson) = LineAverage와
  동일 기하(마커 중심 section-평면 원둘레 균일평균) → ①per-마커 반경
  배열(radii) ②분산 파샬섬(return_sums+clip_bounds: 센서 '위치' 소유권
  [lo,hi) 반개구간 → (sum,count), count 총합=N_s 정확) ③**전 센서 단일
  보간호출 벡터화**(파이썬 루프는 80센서×16substep에서 수 s/step —
  타임아웃으로 적발) + bincount 집계.
- `actuator_line`: mode "lineavg" — config sampling={mode, profile
  (constant|elliptic|schrenk), n_points=80, rs_chord_factor=0.25,
  time_filter_deg=6}. 반경 프로파일 캐시(_build_lineavg_radii: rs=f·c,
  floor max(0.1c, 2Δx)), **EMA 시간필터**(방위각 τ_ψ, coupled 루프의
  통과와류 노이즈 가드 — Melani는 정상 RANS였음; α=dψ/τ, 재시작 시
  상태 재시드). rs를 eps_samp_c/t(eps_samp_r=0)로 마커 기록.
- `alm_dist.make_distributed_sampler`: ring kwarg 분기(파샬섬→allreduce).
  dist fail-fast에 lineavg 허용 추가.

## 검증 (bench5+K17L+SGSoff, 로컬)
- 300스텝 안정, 0.248 s/step(gaussian 0.209 대비 +19%).
- **2-rank ≡ 1-rank**: CT 인쇄정밀 동일(1.884340e-02) = 파샬섬 정확.
- α 프로파일 vs gaussian 쌍둥이: **midspan(0.4–0.8R) |Δ| 0.085°**(일치),
  팁(0.99R) **−0.38°**(방향 ✓; D16 조악+300스텝이라 소폭 — mlg4 예측 −1.2°).
- CT@300: −0.26% (예측 −0.5~−1%의 초기 단계).

## 실행 계획 (사용자 순서)
1. **mlg4_k17l_la (30 rev)**: 새 공식화 무보정 앵커 + 수렴 rev 확정
   (k17l이 rev20에 dCT/rev +8.5e-5 drift). configs/.../eso_mlg4_k17l_la.py.
2. 수렴 위치 확인 → Prandtl(shen g=1.0) → Shen g-스윕 (전부 재앵커).
