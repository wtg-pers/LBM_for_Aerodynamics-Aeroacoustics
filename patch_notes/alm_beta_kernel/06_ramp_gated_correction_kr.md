# 06 — 스미어링 보정 램프 게이팅 (2026-07-19)

## 1. 문제 (05 §5c에서 확정한 근본원인 후보)

램프 팩터는 스프레딩 후 `_F_grid`에만 적용된다(`actuator_line.py`
step 말미). BEM/kleine·Dağ 보정은 램프를 모른 채 **풀 Γ의 w_corr를
램프 중에도 전액 가산** — 초기 램프에서 보정이 실제 침적력 대비
1/ramp 배 과대. 이것이 D40 case-4′(Wendland archB)의 팁 심저(α≈−16°)
진입 창이다: R2′에서 다이브=램프 중(step 629), 회복=램프 종료 직후
(step 1258). gaussian은 같은 창에서 아임계로 통과할 뿐 구조는 동일
(커널 무관 결함; 커널 연산자 자체는 전수 무죄 — 05 §5c 표).

## 2. 수정

`src/actuator/actuator_line.py` — 보정 적용 직전 (kleine/Dağ 공통
지점, `u_n = u_n + w_corr` 위):

```python
if self.ramp_steps > 0 and self._step_count < self.ramp_steps:
    w_corr = w_corr * ((self._step_count + 1) / self.ramp_steps)
```

- 팩터는 **이번 스텝의 `_F_grid` 램프와 정확히 동일 규약**
  (`_step_count`는 스프레딩 후 증가하므로 보정 시점엔 +1).
- 게이팅은 **적용(과 VTP `w_corr` 진단)에만**. 솔버 워밍스타트 상태
  (Γⁿ⁻¹, `_kleine_w_prev`, `_dag_w_prev`)는 무램프 유지 — 풀 하중
  선형화의 자기일관성을 보존.
- 램프 종료 후·ramp_steps=0에서는 분기 자체가 실행되지 않음 →
  **bit-identical** (게이트 리스크 없음).

`src/solver/setup.py` — 부수(05 §6): ALM 섹션에 `Kernel family:` echo
추가 (D40 디버그에서 런 로그에 커널 기록이 없던 문제).

## 3. 검증

- **G-β0 kernel abstraction gate: PASS** (S/G/U/E 전 항목 — 연산자
  수준 무영향 확인).
- **bench5 archB wendland 1-rank 300스텝 A/B** (base vs fix, VTP
  step 251 = mid-ramp, 램프팩터 0.499): 적용 w_corr 비율(fix/base)
  팁 4마커 0.37~0.47 (스팬 전체 0.37~0.82 — 게이팅→α↑→CL↑→무램프
  w↑ 피드백으로 0.5 주변 분포). 팁 α 3.03°→4.07° (진입 창 폐쇄 방향).
- 수정 diff: actuator_line.py +14 / setup.py +6 (미커밋).

## 4. 클러스터 검증 — R2′ 재실행 **PASS** (2026-07-19)

mk63 α (수정 전 → 수정 후, per-blade):

| step(rev) | 수정 전 | 수정 후 |
|---|---|---|
| 629 (0.5) | 전 블레이드 **−16.4°** 심저 | −0.8/+18.0/+17.9/+17.9 (미발달 유동 고α 과도) |
| 1258 (1.0) | −13° 2블레이드 고착 중 | −4.5~−9.5° 일시 스윙 |
| 1887 (1.5) | b1 −15.7 고착 | 전원 +2.0~+3.2 회복 |
| 2516/3145 | b1 −16.5 **영구 고착** | 스프레드 0.13°/**0.02°** 완전 대칭 |

- 진입 창 폐쇄 + 고착 소멸 + 최종 팁 프로파일 R4′ gaussian과 ≤0.04°
  일치(mk63 2.24 vs 2.25). 수정 전 코드는 동일 설정에서 결정론적으로
  −16.4를 재현했으므로 이 결과 자체가 수정판 배포 지문.
- 데이터: `0718_beta_kernel/result_..._case4w_.../vtk/markers/`(17:24
  마커만 신규 복사 — csv/setup_log.txt는 수정 전 런 잔재, 주의).
- 잔여 관찰: 램프 종료 직후(1258) −9.5° 일시 스윙은 보정이 전강도로
  들어오는 과도 — 반 rev 내 자체 감쇠. 필요시 램프 후 짧은 보정
  soft-start(게이팅 연장)로 다듬을 수 있으나 현재로선 불요 판단.

## 5. 다음 (당시 계획)

case4′ 25rev 재런 + case4 gaussian 동일코드 기준선(2-rank×2, GPU
0,1/2,3) → §5.4 판정. --vtk-fields-last 3 (32052d8).

## 6. 25rev 본런 결과 (2026-07-20) — 잔여 결함 1건 + 물리 답

**(a) gaussian 기준선 회귀 PASS**: CT 0.009541 (0711 아카이브 대비
−0.07%), 스팬 RMS 1.1e-3, 팁 dip 0.2159, 4블레이드 대칭 — 게이팅은
gaussian 수렴 상태 불변 입증. --vtk-fields-last 3 작동(필드 3개).

**(b) case4′ 잔여 결함**: 블레이드 2가 **램프 종료 직후 과도(step
1257: +2.6/−9.5/−4.5/−7.3 — R2′fix 3rev와 동일한 결정론적 스윙)에서
포획**, mk63 α≈−5.6°로 25rev 전체 고착(mk60-63 톱니 +10/+11/+4.7/
−5.6, w_corr 부호반전 — 05의 attractor 축소판). 게이팅 성과: 진입
강도 −16.5→−5.6°, 파손 2→1블레이드, 램프 중 진입 창은 폐쇄. 잔여
진입 경로 = **보정이 램프 종료에서 전강도 도달할 때 유동이 아직
풀-포스 평형에 못 미침** (06 §4에서 '불요'로 봤던 그 과도 — 오판).
후속 수정 후보: 보정 게이트를 k·ramp_steps(k≈2)까지 연장하는
소프트스타트(램프 후에도 유동이 따라올 시간 부여).

**(c) §5.4 물리 판정 (healthy b013, ×4/3 스케일 — 서브셋 방법오차는
gaussian 대조로 +0.13% 확인)**:

| | CT | vs GT | 피크영역 vs KSAS rigid | 팁 dip |
|---|---|---|---|---|
| gaussian (4bl) | 0.009541 | +14.7% | +12.4% | 0.216 |
| **wendland b013** | 0.009647 | **+15.9%** | **+13.4%** | 0.214 |

스팬 RMS(wend−gauss, b013) = 2.3e-3. **게이트 FAIL — 그리고 이번엔
오염 때문이 아니라 물리 답**: ε-등가 Wendland + 유도된 자기일관 보정
= gaussian + Kleine와 ~1% 이내 동일. case1′(pure 무영향)과 합치면
**"Gaussian 커널 형태 자체의 잔여 편향" 가설(00_handoff §2)은 기각
방향** — compact 커널로 형태를 바꿔도, 그 커널의 닫힌형 보정을 결합
해도 피크영역 +13~15%는 그대로다. 잔여 편향의 원인은 커널 형태가
아닌 다른 곳(유한-ε 스케일 자체, 또는 ALM 모델 외적 요인).

Caveat: b2 파손 웨이크가 b013 하중에 간접 영향 가능 → 확정 도장은
소프트스타트 후 clean 4-blade 1회 재런으로. 단, Δ(wend−gauss)=+1.1%
가 부호까지 뒤집힐 가능성은 낮음.

산출물: analyze_beta25.py, summary_beta25.csv,
spanwise_M2Cn_beta25.png, convergence_CT_beta25.png.

**(d) 아카이브 완결(밀집 CSV·로그 회수 후)**: setup_log kernel echo =
wendland/gaussian 각각 확인(배포 지문 완결). 두 런 모두 2-rank axis=y
cuda_aware=1, 0.99 s/step ≈ 8.6h. 밀집 CT(last 2rev, 64-step 샘플):
case4g 0.009546(drift +0.03%) / case4w ALL 0.009420(drift +0.24%) —
rev-locked VTP 추정치와 각각 −0.05%/+0.06% 일치(방법 교차검증).
b013 healthy 수치는 VTP 기반(밀집 CSV는 블레이드 적분값만 기록).
