# ② 링 평균 인플로우 샘플링 (Natelson 2026 / ROAM·Helios)

현행: 유속을 actuator 점 1곳서 샘플(Gaussian ±3ε 가중평균). 그 점은 자기 body
force 결손 한가운데 → 자기유도 오염.

목표: **각 스테이션 주위 원 위 N개 센서**에서 trilinear 샘플 후 평균.
- 링 평면 = **익형 단면 평면** (ê_r에 수직, ê_c[chord]·ê_t[thickness] span) — 사용자
  확인(Figure 2 노란 sphere). 프레임은 ①의 `get_all_marker_aero_frame()` 재사용.
- N = 20 (논문). R_ring = **1.0·ε** (논문 미제시 → 추천값; 자기유도 결손이 ε 스케일).
- 각 센서 = trilinear(8노드) — 논문 "linearly interpolated from LBM fluid nodes".
- 전부 config 조절: `sampling={"mode":"ring","ring_n":20,"ring_r_factor":1.0}`.

수식: sensor_i = x_m + R·(cos φ_i·ê_c + sin φ_i·ê_t), φ_i=2πi/N.
u_m = (1/N) Σ_i trilinear(u, sensor_i).

## 구현
- `interpolation._sample_ring(u_field, positions, ec, et, ε, xp, n, r_factor)`
  (`_sample_trilinear` N회 재사용).
- `sample_velocity_alt`에 "ring" 분기 + ec/et/ring_n/ring_r_factor 인자.
- `actuator_line`: __init__ 기본값(_sampling_ring_n=20, _sampling_ring_r_factor=1.0),
  step()이 mode=="ring"일 때 프레임 계산·전달, config 파싱.
- 기본 mode="gaussian" 불변 → 기존 run byte-identical.

## 검증
- (1) 균일 유동 u=const → 링 평균 = const (정확). (2) 선형 유동 u=a·x → 링 평균 =
  중심값(원 대칭 상쇄). (3) 센서 위치가 단면평면(ê_r 성분 0), 반경 R=ε. (4) step()
  end-to-end (gaussian vs ring 상이, NaN 없음). (5) GPU/CPU 동일(trilinear 재사용).

### 상태: 완료 (2026-07-06)
- `interpolation._sample_ring`(단면평면 원, trilinear N회 평균), `sample_velocity_alt`
  "ring" 분기 + ec/et/ring_n/ring_r_factor 인자.
- `actuator_line`: __init__ 기본값(ring_n=20, ring_r_factor=1.0), step()이 mode=="ring"
  일 때 `get_all_marker_aero_frame()`로 ê_c/ê_t 전달, config 파싱.
- config: `sampling={"mode":"ring","ring_n":20,"ring_r_factor":1.0}`.
  A/B: `hvab_hover_c10_ring.py` (대조군=`hvab_hover_c10_aniso_iso.py` 순수 ALM gaussian).
- 검증(test_ring.py): (1)균일→상수 2e-16, (2)**선형→링평균=중심 2e-16**(대칭·trilinear),
  (3)센서 단면평면(⊥span 1e-16)·반경 ε 정확, (4)step() gaussian vs ring 상이·NaN無.
  wiring: config→model 정확. 기본 gaussian → 기존 run byte-identical.
- 미결: GPU=CPU(trilinear는 xp 공용이라 동일 기대), sweep 프레임 회전 미반영(①과 동일).
- 실행(클러스터): ring run + compare_M2Cn로 팁 하중 비교.

## 원칙
단계 검증·체크인. 자동 진행 금지. 기본 OFF(gaussian)로 기존 run byte-identical.
