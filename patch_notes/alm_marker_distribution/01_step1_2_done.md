# Step 1·2 완료 — per-marker dr + 분포 모드 (2026-06-30)

## Step 1 — per-marker dr 안전화 (보정 2함수)
- `smearing_correction.influence_matrix`: scalar 분기(기존식 EXACT) + array 분기
  `A = -(1/4π)·Kmat @ (dr[:,None]·G)` (dr=source 세그먼트폭, 축 검증됨).
- `actuator_line._viscous_core_correction`: scalar `Σ(gradG·kernel)·dr` (EXACT) /
  array `Σ(gradG·kernel·dr)` 분기.
- **검증**(`scratchpad/test_step1_dr.py`): (a) scalar **byte-identical**(np.array_equal True),
  (b) array=brute-force ref 2.3e-16, (e) 불변식 `influence@Γ==viscous_core` scalar·array 둘 다 1e-17.

## Step 2 — `blade.generate_markers(distribution=, cosine_side=)`
- 모듈 헬퍼 `_cosine_map`(both/tip/root, f(0)=0 f(1)=1 단조) + `_trapezoid_widths`(Σ=span).
- 3 분포:
  - `"uniform"`(기본): `r_start+(j+0.5)dr`, **marker_dr=스칼라** → 기존 byte-identical.
  - `"cosine"`: edge-map 셀중심 클러스터, marker_dr=배열, **Σdr=span**.
  - `"endpoint"`: `linspace(root_cut,tip,N)` 끝점포함 + 사다리꼴 marker_dr, **Σdr=span**.
- 출력 포맷(blade/rotor `:.5f`) `np.mean` 래핑(배열 대응). `get_all_marker_dr` 호출처 無.
- config 연결: `rotor.grid.{marker_distribution,cosine_side}` → `Rotor.from_config`(rotor.py:1061).
  HVAB `build_config(marker_distribution=, cosine_side=)` 인자 추가.
- **검증**:
  - `test_step2_dist.py`: uniform byte-identical, cosine 클러스터 방향 정확, endpoint 끝점·사다리꼴.
  - `test_step2_e2e.py`(HVAB 전체 config): uniform byte-identical, **endpoint 팁마커 시위=3.27in(진짜 팁)**.
  - `test_step12_forcepath.py`(model.step + Dağ): 전 분포 finite·Σ|F|>0·Σdr=span, 보정 array-dr 정상.

## 비교 config (사용자 클러스터 실행)
- `hvab_hover_c10_markerdist_cosine.py` (cosine both), `_endpoint.py` — 순수ALM·NASA·25rev,
  Task2 pureALM_nasa와 동일셋업, **marker_distribution만 변경**. 팁/루트 φ·α·CT·FM 비교.

## 백워드 호환
기본 marker_distribution="uniform" → marker_dr 스칼라 → **기존 69 config 전부 byte-identical**(검증).

## 남음 — Step 3 (옵션 3): 보정 endpoint Γ=0 closure
`eps_correction.endpoint_closure`. 보정의 dΓ/dr에 root/tip Γ=0 가상노드. base projection 이중계산 주의.
