# 0630 marker/sampler 비교 실험

ALM 마커 분포 × 샘플러 비교. **분석 초점 = 마커별 spanwise 분포(φ·α·u_n·CL·CD·F_n vs r/R),
특히 팁/루트**. 적분 CT/CP/FM은 부차(상쇄로 둔감, 스팬 앵커 없음).
공통: HVAB c10, M_tip0.65, light, **NASA OVERFLOW 덱**, 25rev, 순수 ALM(보정·Prandtl OFF; Case2/3 제외).
분석: `src/utilities/compare_spanwise.py --run label=<결과폴더> ...` (r/R 축 오버레이).

## Case 1 — 순수 ALM, marker 분포 효과 격리 (×2 sampler)
| 분포 \ sampler | gauss | point(trilinear, 5편 LBM-ALM 표준) |
|---|---|---|
| uniform(셀중심, 현행) | `md_uniform_gauss` = 기존 pureALM_nasa(결과 보유) | `md_uniform_point` |
| cosine(양끝 조밀) | `md_cosine_gauss` | `md_cosine_point` |
| endpoint(끝점포함+사다리꼴) | `md_endpoint_gauss` | `md_endpoint_point` |

## Case 2 — virtual Γ=0 end-node 보정 (옵션 3) — 끝점에 마커 없는 분포만
uniform/cosine × {gauss,point} = 4 (endpoint는 실제 끝점 마커가 있어 제외).
→ 옵션 3 구현 후 추가.

## Case 3 — Dağ 기준 + (Case1·2서 물리적으로 타당했던 방법) 추가
Case1·2 결과 본 뒤 결정.
