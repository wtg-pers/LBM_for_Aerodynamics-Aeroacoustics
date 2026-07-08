# 0703_dag_edge_fix — Dağ 보정 edge-fix 재실행

`_viscous_core_correction`을 Dağ&Sørensen(2020) Eq.17-18대로 **edge 기반**으로
재구현(팁·루트와류 포함, +1/4π) 후 재실행. 배경/검증: `patch_notes/alm_dag_edge_fix/`.

## 공통 셋업
collective 10°, M0.65, light 4-level MLG, NASA OVERFLOW 덱, 25rev,
**gauss 샘플러**, prandtl_loss OFF, **eps_correction=dag(inviscid, relax 0.5)**.

## 케이스
### Dağ 단일패스 (edge-fixed, relax=0.5)
| config | 분포 | 용도 |
|---|---|---|
| `dagfix_uniform_gauss.py` | uniform | **대조군** — 구 260630/dag_csv(깨진 버전, uniform+gauss)와 직접 A/B → 코드수정 효과만 격리 |
| `dagfix_endpoint_gauss.py` | endpoint | 끝점 마커(팁 r/R=1.0) |
| `dagfix_cosine_gauss.py` | cosine(both) | 양끝 클러스터 |

### Kleine 비반복 (edge-fixed, free-wake, relax 불필요=self-consistent)
| config | 셋업 | 용도 |
|---|---|---|
| `kleine_free_fast_endpoint.py` | free wake, rebuild_every=**5**, n_w=50, all edges | Kleine free-wake "fast" |
| `kleine_free_tip_nw2_endpoint.py` | free wake, **tip edge만**, **n_w=2**, rebuild_every=1 | 팁와류만 convected 모델 |

Kleine는 endpoint 분포 사용(Dağ endpoint와 method A/B). 필요시 cosine/uniform로 변경 가능.
**Dağ vs Kleine 기대**: Kleine가 self-consistent Γ 감소 반영 → 팁 w_corr 더 작고(≈0.005 vs 단일패스 0.019) relax 불필요, 더 깨끗한 팁 de-load 예상.

## 실행
```
python main.py --config configs/0703_dag_edge_fix/dagfix_uniform_gauss.py
python main.py --config configs/0703_dag_edge_fix/dagfix_endpoint_gauss.py
python main.py --config configs/0703_dag_edge_fix/dagfix_cosine_gauss.py
```

## 판정 기준 (재실행 후)
- **성공**: 팁 M²Cn이 0.39(현행) → 실측 ~0.15 쪽으로 하락, CT/σ가 0.10 → 0.076 쪽으로.
- 대조군(uniform+dagfix)이 구 dag(CT/σ 0.1017)와 확연히 다르면 = 코드수정이 실효.
- 주의: relax=0.5는 부분보정. 팁이 내려가되 부족하면 relax↑(→1.0), thrust CV 튀면 relax↓.
- 후처리: `aeromechanics_workshop/HVAB/case1_analysis/rerun_compare.py` 류로 실측 M²Cn 오버레이.

## 2026-07-04 갱신 — helix pitch guard + 폴라조회 배치
- **`eps_correction.helix_pitch_floor`** (신규, 기본 `"auto"`): 호버에서 팁 자기유도
  업워시/root 재순환으로 국소 φ≤0인 edge의 prescribed-helix 필라멘트가 **상류로
  상승**하던 비물리 거동을 차단(디스크 평균 유입류로 하강). `"off"`=논문 literal
  (기존 15rev helix 런 재현), float=명시적 w_floor [Δx/Δt].
  상세: `patch_notes/alm_dag_edge_fix/03_helix_pitch_floor.md`.
  **기존 helix 결과와 비교 시 변인**: 260703 폴더의 helix 런들은 floor 없던 코드.
- **폴라조회 배치화**(코드 전역, config 무관): bem 21ms→~1ms대, GPU util 개선.
  `ALM_POLAR_BATCH=0`으로 구경로 강제(A/B용). 수치 bit-identical
  (`patch_notes/alm_dag_edge_fix/04_polar_batch_lookup.md`).
- **민감도 config 2종 추가**: `sens_straight_smooth{1,3}.py` — 비문헌 성분(Γ 평활화) 민감도
  매트릭스용(smooth 0=발산/1/2=기존/3). 편차 총괄표 = docs/alm_fundamental_issues_audit_kr.md.

## 2026-07-05 — TEST B: slab-L4 격자 (격자 축)
preset **light_slab5** — 도메인=구 L1 박스(3.25D×2.5D², hub 상류 0.75D), L1=도메인
거의 전체, **L4 로터 슬랩**(dx=L0/16, lat ±1.0625R, 두께 48 fine cells=±3ε 지지폭).
45.3M셀/~18.6GB(24GB OK), wall ~3.3× light. 팁 ε/Δx 2.0→3.4, 팁 chord 13.6셀
(light_tip5급을 1/3 메모리로). 스모크 통과(2026-07-05, `testB_slab5_smoke.py`).
⚠ 도메인 축소로 절대 CT/FM은 기존 light와 직비교 불가 — spanwise/w_corr/팁α로 판정.

**모델 결정(2026-07-05, testA 종결 후)**: 물리성 서열 kleine_free ≈ pure ALM > Dağ
(Dağ=진동·root 스파이크·floor 불연속 아티팩트) → 본런 기준 모델 변경.
| config | 모델 | 역할 |
|---|---|---|
| `testB_slab5_pure_alm.py` | **pure ALM (보정 無)** | ★본런 — 격자 효과 무혼입 격리. 비교쌍=light pureALM_nasa 25rev |
| `testB_slab5_kleine_free.py` | Kleine free (rebuild1) | 2차 런 — 해상된 수축에서 free-wake 응답 측정 (pure 후, 사전 smoke 권장) |
| `testB_slab5_dag_straight.py` | Dağ straight | (구버전 기준안 — 미사용 보관, testA 연속성용) |
