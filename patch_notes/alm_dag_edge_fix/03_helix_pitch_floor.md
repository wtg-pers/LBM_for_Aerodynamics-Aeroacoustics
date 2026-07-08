# 03 — Prescribed-helix 호버 pitch guard (`eps_correction.helix_pitch_floor`)

날짜: 2026-07-04 · 대상: `src/actuator/actuator_line.py` (`_dag_prescribed_helix_wcorr`)
검증: `test_helix_pitch_floor.py` (ALL PASSED)

## 증상 (260703_dag_edge_fix 15rev helix 런, 검증 세션서 발견)

wake VTP에서 일부 필라멘트가 로터 디스크 **상류(−x, 추력측)** 로 올라감:

| edge | r_edge (L0) | 축방향 이동 (2rev) |
|---|---|---|
| 0 (root closure) | 4.03 | **−0.26 lu (상승)** |
| 47 | 15.75 | **−0.41 lu (상승)** |
| 48 (tip closure) | 16.00 | **−5.61 lu (상승)** — wake x_min 90.4 < hub 96의 정체 |
| 그 외 46개 | — | +2.5 ~ +26.0 lu (정상 하강) |

## 원인 — Dağ 규정의 전제 위반 (구현 버그 아님)

Dağ §3.2: helix pitch = "trailing vortex가 방출되는 위치의 국소 유동각 φ".
풍력터빈(논문 대상)은 through-flow가 항상 양(φ>0)이라 문제없음. 호버에서는:

- **팁 마커**: Gaussian 샘플 구가 팁 바깥(자기 팁와류 **업워시** 영역)에 걸쳐
  u_n<0 → tanφ<0 → 팁 closure edge 필라멘트가 **위로** 감김. field 검증에서도
  최외곽 마커 u_x = −0.003~−0.004 확인.
- **root 마커**: 허브 재순환 상승류 → 동일 기전.

물리적으로 방출된 filament는 주위 웨이크 평균류에 실려 **하류로만** 이송된다
(자유류가 0인 호버에서 디스크 위로 기어오르는 tip vortex는 없음). 상승 나선은
팁/root 마커 바로 옆에 filament를 수 회전 머물게 해 root α 스파이크
(helix −27° vs straight −13°)를 악화시키는 부작용도 확인됨.

## 수정

`tanphi_e ≤ 0`인 edge만 floor pitch로 치환 (유효한 양의 pitch는 논문 그대로,
바이트 동일):

```
tanφ_floor(e) = w_floor / (|ω|·r_e)      # 축방향 하강속도 w_floor로 환산한 pitch
```

- config: `eps_correction.helix_pitch_floor`
  - `"auto"` (기본): w_floor = 해당 블레이드 active 마커의 **양의 u_n 평균**
    (= 디스크 평균 유입류로 이송)
  - float: 명시적 w_floor [Δx/Δt]
  - `"off"`: 논문 literal (기존 동작 재현용)
- 적용 위치: subset(`wake_markers="tip"` 등) 절단 **이전** → 모든 모드 공통.
- `helix_fast`(rebuild_every>1) 캐시는 기하만 바뀌므로 그대로 유효.

## 검증 (합성 HVAB 규약: axis=+x, thrust=−x, u_n 팁/root 음수)

1. `off`: root/tip filament 상승 재현 (dx −46.8/−64.9 lu)
2. `auto`: 49개 filament 전부 단조 하강 (dx +24.8~+112.7 lu)
3. 유효 pitch edge 45개 기하 **byte-identical** (논문 경로 불변)
4. 팁 edge 하강 = 4π·w_floor/|ω| 정확 일치
5. float floor 동작 확인

## sawtooth와의 관계 (사용자 질문에 대한 답)

**상류 필라멘트는 sawtooth의 원인이 아님.** 근거: (a) sawtooth는 filament가
아예 없는 straight(해석적 반경 커널)에서 발생·관측됐고, (b) 잔여 중간파장
진동(외측 w_corr ±0.007)도 straight와 helix에서 사실상 동일. sawtooth의
뿌리는 edge 보정의 w∝−Γ″ 피드백(02 참조, smooth로 억제)이다. 상류 필라멘트의
실제 피해는 **root α 스파이크 악화**와 팁 근접장 오염(helix가 straight 대비
추가 de-load를 못 만든 요인 후보)이며, 효과 크기는 재실행으로 판정 필요.

## 영향 범위

- Dağ `wake="prescribed_helix"` 경로만. straight·Kleine-free는 불변.
  (Kleine free-wake는 field 속도로 이류하므로 상승류 구간에서 유사 거동이
  가능하나, 그쪽은 "CFD-consistent 이송"이라는 모델 정의 자체 — 별도 관찰 항목.)
- 기존 helix 런과 비교 시 `helix_pitch_floor` 차이를 변인으로 기록할 것.
