# 05 — compare_spanwise per-blade 분석 (`--per-blade`)

날짜: 2026-07-05 · 대상: `src/utilities/compare_spanwise.py`

## 배경

블레이드 위치별 힘 분포가 달라(사용자 관찰; testA 결과 step 15050에서 팁 F_n이
블레이드별 0.030~0.040) 전-블레이드 평균만으로는 blade-locked 비대칭이 가려짐.
시간평균(정수 rev tail)에도 남는 블레이드별 차이는 회전좌표계에 고정된
비대칭(예: per-blade floor, 독립 필라멘트 발달)의 신호.

## 변경

- `load_spanwise(path, avg_revs, avg_steps, blade=None)`: `blade=int`면 해당
  블레이드 행만 필터해 프로파일 재구성. `None`(기본)=기존 동작(전 블레이드
  평균) — 기존 경로 byte-불변.
- `list_blades(path)`: diagnostics의 blade 인덱스 목록.
- `--per-blade` 플래그: 런별 per-blade 표(sumFn·dT/mean·tip Fn/α/φ) +
  비대칭 요약(블레이드 추력 스프레드 %, 최대 스팬 F_n scatter %와 위치) +
  `<prefix>_spanwise_perblade.png`(행=런, 열=u_n/α/F_n/F_θ, 얇은선=블레이드,
  검정=평균).

## 검증 (testA 4런, 15rev tail 3rev)

- mean 경로 회귀: `--per-blade` 없이 기존 출력과 동일.
- per-blade 결과가 물리적으로 정합: 적분 스프레드 ≤0.4%(축대칭), 국소
  scatter는 기전 있는 위치에만(helix floor 팁 16.4% — per-blade "auto" floor;
  kleine free inboard 11.7% — 독립 필라멘트; upwash helix root 7.7% — 상류
  정체 필라멘트). 상세: `aeromechanics_workshop/HVAB/260703_dag_edge_fix/
  testA_260705_verdict_kr.md`.
