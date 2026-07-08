# ALM 마커 분포 비교 기능 — 설계 (2026-06-30)

사용자 요청: 마커 배치를 3가지로 비교 가능하게.
1. **cosine 조밀화** — 팁(+루트) 쪽 가중 배치 (가파른 gradient 해상).
2. **endpoint + 사다리꼴 적분** — root-cut·tip 끝점에 마커 배치, midpoint(사각) 대신 trapezoid dr 가중.
3. **endpoint Γ=0 closure 노드** — 보정(trailed vorticity)에 root/tip Γ=0 노드 도입 (force 마커 아님).

## 전제: `marker_dr` 스칼라 → per-marker 지원 (bit-identical 보존)
현 `blade.marker_dr`은 스칼라. 사용처 3곳:
- 힘: `F_L = q·chord·dr·CL` (actuator_line.py:843) — dr 스칼라/배열 둘 다 element-wise OK.
- 보정 A: `influence_matrix(r,eps,dr)` → `A = -(1/4π)·dr·(Kmat@G)`. dr=**source 마커 세그먼트폭**.
  비균일 정식: `A = -(1/4π)·Kmat @ (dr[:,None]·G)`. (스칼라면 기존식 그대로 = bit-identical.)
- 보정 B: `_viscous_core_correction` `w[i]=-inv4pi·Σ(gradG·kernel)·dr`. 비균일: `Σ(gradG·kernel·dr_arr)`.

**bit-identical 전략**: `marker_dr`을 **uniform이면 스칼라 그대로**, cosine/endpoint면 배열. 두 보정함수는
`np.ndim(dr)==0`이면 기존 스칼라 경로(EXACT), 배열이면 새 경로 분기. → **기존 69개 config 전부 불변.**

## 구현 (stepwise)
### Step 1 — per-marker dr 안전화 (보정 2함수 분기)
`influence_matrix`·`_viscous_core_correction`에 scalar/array 분기 추가. **검증**: uniform 스칼라 = 기존 bit-identical (단위테스트).

### Step 2 — `blade.generate_markers(distribution=...)`
config `actuator_line.marker_distribution`:
- `"uniform"` (기본): `r_j = r_start + (j+0.5)·dr`, marker_dr=스칼라. **현행 bit-identical.**
- `"cosine"`: 팁 조밀. `r_j = r_tip - (r_tip-r_start)·(1-sin(π/2·(j+0.5)/N))` 류(팁 클러스터). marker_dr=배열(midpoint diff).
- `"endpoint"`: `r_j = linspace(r_start, r_tip, N)` (끝점 포함). marker_dr=**사다리꼴**(끝점 ½, 내부 full).
- 출력 검증: Σ(marker_dr)=effective_span (추력 보존), r 단조, 활성범위 일치.

### Step 3 — 보정 endpoint Γ=0 closure (옵션 3, 직교)
config `eps_correction.endpoint_closure: bool`(기본 False). True면 `_gradient_matrix`/`influence_matrix`/
free-wake에서 **r 양끝에 Γ=0 가상노드**(r_start, r_tip) 추가해 dΓ/dr 계산 후 실제 마커로 환원.
force·sampling엔 영향 없음(보정 전용). base projection과 이중계산 주의 → A/B로 효과 분리.

## 검증 (Claude=smoke, 사용자=클러스터 HVAB 비교)
- 단위: uniform bit-identical(스칼라경로), Σdr=span, 보정 array-dr=균일시 스칼라와 일치.
- CPU smoke: 3 분포 × (보정 on/off) 무오류·유한 force.
- config: `hvab_hover_c10_markerdist_{cosine,endpoint,closure}.py` 신설(비교용).

## 백워드 호환
모든 신규 = config opt-in. 기본 marker_distribution="uniform" + endpoint_closure=False → 기존 전부 bit-identical.
