# ALM Stage B — 가변 ε 팁 테이퍼 · 패치 노트

**날짜:** 2026-06-22
**계획서:** `~/.claude/plans/taper-hashed-rocket.md` (이전 설계: `reactive-snacking-donut.md`)

## 목표

ALM+LBM hover 솔버가 로터 추력을 과대예측한다 (Caradonna-Tung 기준 아음속 +8% /
천음속 +17% vs 실험). 원인은 **smeared-ALM 팁 유도 다운워시 결손**: 유한한 ε Gaussian
힘 투영이 진짜 lifting line의 유도속도를 재현하지 못하며, 그 오차가 팁에서 가장 크다
(팁이 거의 기하학적 받음각을 보고, 유입각 φ→0, 팁 과하중). Stage B (Diaz 2023 §2.1.4)는
투영 Gaussian 폭 ε을 팁 쪽으로 좁혀 팁 와류가 덜 과확산되도록 하는 **빠르고 부분적인**
완화책이다. Stage C (filtered lifting-line, JFM-2019)는 설계만 되어 있고 **보류**.

## 절대 제약

`epsilon_mode="default"` (기본값)은 기존
`marker_epsilon = max(chord/4, 2·dx)`를 **bit-identical**하게 재현해야 한다 — 기존
HART2/CT 결과의 재현성 보존. 테이퍼는 config로 opt-in.

## 신규 config 키 (`actuator_line` 하위)

```python
"epsilon_mode": "default",   # "default" | "tip_taper"
"epsilon_tip_factor": 1.0,   # 팁 ε 목표 = max(factor·2.0, 2.0) lu
"epsilon_taper_start": 0.7,  # 테이퍼 시작 r/R
```

## 테이퍼 공식 (lattice units, floor 2.0)

```
eps_base = max(marker_chord/4, 2.0)
r_norm   = marker_r / r_tip
t        = clip((r_norm - taper_start)/(1 - taper_start), 0, 1)
eps_tip  = max(epsilon_tip_factor*2.0, 2.0)
eps      = (1-t)*eps_base + t*eps_tip
```

## 단계 체크리스트

| # | 영역 | 파일 | 노트 | 상태 |
|---|------|---------|------|--------|
| 1 | 테이퍼 본체 | `src/actuator/blade.py` | `01_blade.md` | ✅ |
| 2 | config → blade | `src/actuator/rotor.py` | `02_rotor.md` | ✅ |
| 3 | 팩토리 주입 | `src/actuator/actuator_line.py` | `03_actuator_line.md` | ✅ |
| 4 | 진단 컬럼 | `actuator_line.py` + `output_manager.py` + `setup.py` + `spanwise_post.py` | `04_diagnostics.md` | ✅ |
| 5 | 스모크 검증 | `configs/caradonna_tung/ct_hover_smoke.py` | `05_verification.md` | ✅ |
| 6 | Production A/B config | `configs/caradonna_tung/ct_hover_t08_m088_taper.py` | `06_production_config.md` | ✅ |

## 결과 (전부 통과)

- **Default-OFF bit-identical** (rotor_performance.csv + blade_geometry.csv) ✓
- **테이퍼 수학 정확** — production 해상도에서 확인 (inboard 불변, 선형 테이퍼, 팁 floor 수렴) ✓
- **`eps_lu`** 진단 컬럼 연결 ✓ · **커널 무수정** ✓
- 수정 파일: `src/actuator/{blade,rotor,actuator_line}.py`,
  `src/solver/{output_manager,setup}.py`, `src/utilities/spanwise_post.py`.
- 검증용 임시 config/결과 폴더는 디스크에 남아 있음 (정리 거부됨). 정리하려면:
  `rm -f configs/caradonna_tung/_tmp_eps_*.py && rm -rf result_ct_smoke_eps{A,B,DEF} result_ct_smoke_log{A,B}`
- A/B 비교 스크립트: `src/utilities/compare_taper_ab.py` (실데이터 end-to-end 검증 완료).

**다음 (사용자, 클러스터):** production CT M0.877 `light`에 `epsilon_mode:"tip_taper"`로
실행 → C_T vs exp 0.00473 + 팁 φ 회복 (`spanwise_post.py`). 이후 B 충분 vs Stage C 결정.
