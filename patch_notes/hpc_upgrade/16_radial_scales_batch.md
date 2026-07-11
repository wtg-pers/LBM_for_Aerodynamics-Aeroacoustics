# 16 — archB(radial truncation) 스프레딩 감속 수정: scales GPU 배치화

> 2026-07-12. 배경: D40 4-케이스 클러스터 실런에서 archB(case3/4)가 pure(case1/2) 대비 ~2× 감속.
> case5/6(SGS-off A/B) 발사 전 수정 요청(사용자).

## 진단 (ALM_PROFILE_BEM, 로컬 3090, D40 case4 vs case1; ms/fine-substep ×16/coarse)
| 단계 | case1 | case4(前) | case4(後) |
|---|---|---|---|
| sample | 26.3 | 26.4 | 26.8 |
| bem(solve+polar) | 1.3 | 7.3 | 7.4 |
| **spread** | 5.1 | **37.4** | **8.6** |
| coarse step | 2.66s | 3.70s | **3.29s** |

범인 = kleine solve(4.8ms, patch04로 이미 최적)가 아니라 **`compute_radial_scales`**: 재정규화 스케일을
매 substep **호스트 numpy 루프**(near-limit ~25-30마커 × meshgrid 박스합, root ε=6.56→~41³노드)로 재계산.

## 수정
`compute_radial_scales_batch(xp, ...)` (spreading.py): near-limit 마커 전체를 **padded (M,K,K,K) 브로드캐스트
1회**로 평가, xp=cupy면 GPU에서 계산·소비(스프레딩 커널이 어차피 scales를 디바이스로 받음 → H2D 왕복 소멸).
box floor/ceil·도메인 클립·sphere cut = 참조 구현과 **동일 노드 선택**; 합산 순서만 달라 fp64 라운드오프 차.
CPU 경로(§2)는 원본 참조 구현 유지.

## 검증
- `gates/alm_radial_scales_gate.py`: D40급 기하 3-seed — batch(np/cupy) vs 참조 **max rel ≤ 4.6e-16**,
  truncated 20-30/256 마커, scale 최대 ~2.03. 단독 8×(13.9→1.7ms).
- end-to-end: case4 D40 2-step |F_grid|합 9.547397e-02 = 변경 전과 전 자릿수 일치.

## 잔여(비병목, 기록)
- sample 26-27ms/call(×16=0.43s/coarse)는 **전 케이스 공통** 최대 ALM 비용 — 추후 후보.
- archB 잔여 오버헤드 = kleine solve+polar ~6ms/call = 구조적(보정의 본질 비용).
