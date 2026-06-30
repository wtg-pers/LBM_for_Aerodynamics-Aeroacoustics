# ALM 속도 샘플러 A/B — 패치 노트 (overview)

**날짜:** 2026-06-26
**근거 분석:** `docs/almlbm_paper_analysis_kr.md` §2 차원① / `to_claude/tip_sampling_diag.py`
**관련 메모리:** [[next-session-pickup-alm-lbm-5]] (P1), [[feedback_stepwise_patch_notes]]

## 목표

우리 ALM 속도 샘플러는 현재 **등방 Gaussian(±3ε) 가중 평균**(`interpolation.py`,
`u(x_j)=Σu·exp(-d²/ε²)/Σexp(-d²/ε²)`, d≤3ε)이다. 참고논문 5편은 전부 point/trilinear
또는 ring-linear 샘플을 쓴다 — **우리만 ±3ε 적분 샘플**. 진단(`tip_sampling_diag.py`)
결과 **팁 마커가 자기 샘플의 ~30%를 디스크 밖(cyl radius>R, 유도≈0)에서 가져온다**
(DEFAULT light preset 29.8%, 마커 47 단 하나에 국소). 이는 팁 φ→0(유입 결손)을 증폭하는
"우리 고유" 요인의 유력 후보.

이 패치는 **샘플링 방식을 config로 선택 가능**하게 만들어 그 기여(=분석문서의 (b))를
정량 분리하는 A/B 실험을 가능케 한다. 적용 범위 = **전 스팬 일괄**(사용자 결정 2026-06-26;
깨끗한 (b) 분리 목적). 팁-국소 매끄러운 taper는 분해 결과 확인 후 후속.

## 절대 제약 (재현성)

`sampling.mode="gaussian"` (**기본값**)은 기존 `interpolate_velocity_batch_gpu` 경로를
**그대로 호출** → 기존 HVAB/HART2/CT 결과 **bit-identical** 보존. 새 모드는 config opt-in.

## 신규 config 키 (`actuator_line` 하위)

```python
"sampling": {
    "mode": "gaussian",     # A(기본) | "point"(B-i) | "aniso"(B-ii) | "mask_disk"(B-iii)
    "eps_r_factor": 0.5,    # B-ii 전용: ε_r = factor·ε (반경), ε_perp = ε
}
# 문자열 단축형도 허용: "sampling": "point"
```

## 네 모드 수식 (lattice units, 마커별 ε, d = 노드−마커 오프셋)

| 모드 | 가중치 w(node) | 컷오프 | 제거하는 것 |
|------|----------------|--------|-------------|
| **A** gaussian | `exp(−|d|²/ε²)` | \|d\|≤3ε | (현행, 불변) |
| **B-i** point | trilinear(8 코너), ε 무관 | 1셀 | b1+b2 + 코드·축 필터 전부 |
| **B-ii** aniso | `exp(−d_r²/ε_r² − d_⊥²/ε²)`, ε_r=factor·ε | \|d\|≤3ε | 반경 필터(b1+b2)만, 코드·축 유지 |
| **B-iii** mask_disk | `exp(−|d|²/ε²)`·[cyl(node)≤R] | \|d\|≤3ε | off-disk(b1)만, 반경 smoothing(b2) 유지 |

- d_r = d·ê_r (반경 성분), d_⊥² = |d|² − d_r² (반경수직 성분)
- ê_r(마커) = normalize( (p−hub) − ((p−hub)·n̂)·n̂ ),  p=마커, n̂=축
- cyl(node) = |(g−hub) − ((g−hub)·n̂)·n̂|,  g=노드 전역위치
- 기하: hub=`rotor.hub_center`[lu], n̂=`rotor.rotation_axis`, R=`rotor.radius`[lu]
- 모든 모드 분모 Σw로 정규화 (정규화상수 약분), 반환은 numpy(BEM=CPU)

## A/B/C 실험 설계 — (b) 분해

진단상 (b)는 두 성분: **b1=off-disk 혼입**(측정 ~30%), **b2=반경 smoothing**(유동장 필요).

| 런 | b1 | b2 | 차분 의미 |
|----|----|----|-----------|
| **A** gaussian | 有 | 有 | baseline |
| **B-iii** mask_disk | **제거** | 有 | A − B-iii = **b1 기여** |
| **B-i** point | 제거 | 제거 | B-iii − B-i = **b2 기여** |
| **B-ii** aniso | 감소 | 감소 | production fix 후보(매끄러움) |

지표: 팁 φ·α·u_n 스팬분포, C_T, FM. **A·B-iii·B-i 세 점**이면 b1/b2 정량 분해.
프로덕션 무거운 런은 사용자 클러스터 실행([[feedback_simulation_execution]]); Claude는
config + CPU smoke까지.

## 구현 단계

1. `interpolation.py` §7 — xp-generic 샘플러 3종 + dispatch (`01_interpolation.md`)
2. `actuator_line.py` — `_sampling_mode` 필드 + `step()` dispatch + `from_config` (`02_actuator_line.md`)
3. 검증 — bit-identical(gaussian) + 3 모드 동작 + mask_disk≠gaussian (`03_verification.md`)
