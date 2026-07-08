# ALM 모델 근본 문제 점검 (audit) — 하나씩

**시작 2026-06-30.** ALM 파이프라인을 `step()` 순서대로 따라가며 **구조적/물리적 근본 문제**를 카탈로그화하고 하나씩 검증·수정. Task 2(NASA baseline)에서 드러난 익형 경계 아티팩트가 출발점.
범례: 🔴 확인된 결함 · 🟡 의심/미검증 · 🟢 양호(강점) · ⬜ 미착수.

파이프라인 (`src/actuator/actuator_line.py:step()` 293–402):
`① 속도샘플링 → ② 속도삼각형/마커이산화 → ③ 폴라조회 → ④ 팁/유도보정 → ⑤ force spreading → ⑥ LBM 커플링`

---

## ① 속도 샘플링 (actuator_line.py:345 `interpolate_velocity_batch_gpu`) — 🟢 검증완료: 결함 아님 (2026-06-30)
**공식** `u(x_j)=Σ u·exp(-d²/ε²)/Σ exp(-d²/ε²)`, 3ε 절단 (interpolation.py:400). **정규화 Gaussian 평균** =
표준 ALM(Troldborg/Sørensen/Martínez-Tossas). 균일장 보존·3성분·**최정밀 레벨(L3)서 샘플**·sampling ε≡spreading ε(대칭). **공식 자체 결함 없음.**
- 🟡 **off-disk 오염(팁 1~2마커 국한)**: ε=**0.25c** 외측 전반(floor 2Δx는 최외곽 1마커뿐), 3ε/R≈5~6%.

### ★sampler A/B — 실데이터 분석 (2026-06-30, `260629_results_sampler` samA_gauss/samB1_point/samB3_mask, 18rev)
**smoke(합성장) 결과는 신뢰 안 함**(사용자 확정) — 아래는 실런:
| sampler | CT | vs GT 0.00832 | FM | 루트 거동 |
|---|---|---|---|---|
| **gaussian** | 0.00924 | +11% | **0.622 최고** | 매끈 ✓ |
| point | 0.00957 | +15% | 0.611 | 매끈 |
| mask_disk | 0.00896 | +7.7% | 0.604 최저 | **루트 CD 스파이크 0.011→0.028 + α붕괴 3.9→2.3°** ✗ |
- **★핵심: 샘플러의 지배적 영향은 팁이 아니라 루트**(r/R 0.26~0.4; 컷아웃 가장자리서 disk/off-disk 기하차 최대). 팁 α 차 ~0.6°(7.24→6.66)뿐.
- mask_disk CT가 GT에 근접한 건 팁 교정이 아니라 **전스팬(루트) 하중을 비물리적으로 깎은 부작용** = "옳은 답, 틀린 이유."
- **결정: 기본값 gaussian 유지** — 근거는 **실데이터**(최고 FM·유일 루트 매끈·spanwise 앵커 없어 point/mask 우월 근거 無)지 "표준"이 아님.
  ★문헌은 갈림: **5편 LBM-ALM 전부 point/trilinear(우리만 Gaussian)**, Gaussian은 비-LBM 로터 ALM(Merabet/Sørensen-Shen)서 정석. "Merabet 동일=표준" 단순화는 정정.
  즉 "지금 데이터로 바꿀 이유 없음"이지 "gaussian이 정답" 아님. (Merabet 레시피 ε↓+point+조밀격자를 fine서 함께 가면 결과 달라질 수 있음=⑤/해상도 묶음 별도 실험.)
- **결론**: ① sampling 공식 = 표준·정상. "샘플링 문제" 실체 = **ε-scale/해상도**(ε/Δx_fine≈2.6) → **②·⑤의 증상**. 레버는 샘플러 아님.
- 플롯: `aeromechanics_workshop/HVAB/hvab_sampler_real_spanwise_compare.png`.

## ② 속도삼각형 / 마커 이산화 (compute_relative_velocity; blade.py) — 🟡 (2026-06-30, ★초기 🔴 과잉규정 정정)
NASA pureALM 분석:
- 🟢 **마커 간격 = 균일 Δr≈2.0 fine lu** (48 markers): 팁/루트 조밀화 없음. 표준 이산화 — 그 자체 결함 아님.
- **관찰**: 바운드 순환 Γ가 최외곽 마커(r/R 0.992)서 **최대값의 73%**(0.31 vs 0.43@r/R0.945) — 팁서 0으로 안 닫힘.
- 🟡 **★정정(초기 🔴→🟡)**: 이 Γ-비closure는 **대부분 F1 과부하의 *증상*** 이다 (보정 없는 pureALM이라 팁 과부하→CL↑→Γ 유지). **독립 결함 아님.**
  - "virtual Γ=0 tip/root 노드" 보정은: (a) base force-projection이 팁 力을 ε로 스미어해 LBM에 약한 팁와류를 이미 생성 → **이중계산 위험**,
    (b) 사용자가 과거 제안한 "팁/루트 force마커 강제배치"(비물리로 기각됨)와 **본질 함정 공유**. → 깨끗한 fix 아님, **보정의 2차 정제 후보로 보류.**
  - (용어구분: virtual Γ=0 노드=力·샘플 없이 dΓ/dr 계산용 순환 BC뿐 / force마커=力 spread+속도샘플하는 실제 점. 다르지만 closure 함정 공유.)
  - **★구현+검증 완료(옵션3, 2026-06-30)**: `eps_correction.endpoint_closure`. 기계검증 통과(byte-identical OFF, net순환→0, 노드정확)이나
    **물리검증서 결함**: 팁 w_corr 부호 역전(+→−, 업워시=과부하 악화). 원인=①base/smeared 불일치(이중계산) ②Γ→0 over dr/2 인공급경사(격자의존).
    → **옵션3 비권장**, 정합 대안=옵션2(endpoint, 실제 팁마커). 상세 `patch_notes/alm_marker_distribution/03_step3_closure.md`.
- 🟡 **팁 chord ~6 cells**(미해상, ε/Δx≈2.6): 진짜 레버는 이 **해상도**(⑤와 결합, Merabet ε/Δx≈4)지 마커 BC 아님. [[next-session-pickup]]

## ③ 폴라 조회 (`_lookup_cl_cd:470`, multi-airfoil)
- 🔴 **익형 경계 블렌딩 부재** (Task 2 F2 확정): `blade._get_airfoil_at_r` → `_hvab_hover_base._airfoil:97` piecewise-constant 배정. r/R 0.825(RC4-10→RC6-08)서 **CL −16% 계단**(C81 직접조회로 Mach아티팩트 반증). 실제 lofted 블레이드/NASA sectional은 매끈. **fix=경계 ±Δr 밴드 인접 덱 CL/CD 선형 블렌드.**
- 🟡 **C81 덱 품질**: RC4-12=placeholder(내측 40%), RC6-08T=tab근사(팁). authentic은 RC4-10·RC6-08만. (`docs/hvab_geometry_kr.md`)
- 🟢 **Mach-pass + sweep cosΛ** (799, 770–796): 각 단면 정확 Mach 조회 — 5편 참고논문보다 앞선 강점.

## ④ 팁/유도 보정 (`_compute_prandtl_factor:408` / `_viscous_core_correction:500` / `_kleine_w_corr:545`)
- 🔴 **유한-ε smearing 팁 유도결손** (Task 2 F1, 코어 문제): 팁 u_n −89% 붕괴 → 과부하 → CT +26% 과대. Dağ/Kleine/Prandtl 보정을 위에 적층.
- 🟡 **Prandtl tip-loss 이중계산 위험** (Diaz 경고): ALM이 induction 일부 해상하는데 Prandtl로 또 깎음 → variable-ε 권장. task3 production이 Prandtl ON.
- 🟢 **Kleine 비반복 + free-wake**(Phase 2) 구현·검증 완료. 단 Task 3 정리 필요(아래).

## ⑤ Force spreading (actuator_line.py:382 `spread_forces_to_grid_gpu`; ε=`max(0.25c,2Δx)` blade.py:359)
- 🟡 **ε/Δx 비**: 팁서 2Δx floor → 과스미어 + 코어 미해상. 해상도와 결합된 문제(②).
- ⬜ **등방 vs 비등방 커널**: 마커당 단일 등방 ε. Natelson은 chord/span 비등방. fundamental 커널형상 선택.

## ⑥ LBM 커플링 / 시간적분 (step body force; MLG sub-step; ramp; Γ warm-start)
- 🟢 **tight coupling 작동 확인** (Task 3 item3, 2026-06-30): `u_n += w_corr`(819) → 속도삼각형 재계산(821) → 폴라 재조회(826) → 같은 스텝 F→body force→LBM. free-wake 유도속도가 독자결과 아님. (상세 아래)
- 🟡 **Γ warm-start 1-step lag** (607 `_kleine_gamma_prev`): Kleine 비반복 선형화의 의도된 lag. 검증됨.
- ⬜ **MLG sub-step간 ALM force 처리**: fine level(L3) ALM, steps_per_coarse=8. 매 sub-step 재계산 vs hold? 확인 필요.

---

## Task 3 — kleine free-wake 정리 (상태)
- **item 1 (n_w 노출)**: ✅ 이미 config 노출됨(`actuator_line.py:1414` `ec.get('n_w',50)`). ⚠️ 단 **n_w=1은 free-wake 비활성**(`:568` `len(wake)>=2` 필요 → straight Phase1 fallback). "1 패널" 의미 재확인 필요.
- **item 2 (팁-마커 only shedding)**: ⬜ 미구현. 현재 `:693` 전체 마커 shed. 구현경로 확정: shed 서브셋 + `_kleine_w_corr`서 `G_used = G[shed_idx,:]`(m×n) → `A=dr·(B@G_used)` (n×n 유지). config `wake_markers`(all|tip|N|r/R) 추가, 기본 all=bit-identical.
- **item 3 (tight coupling)**: ✅ 검증 완료(⑥). free-wake convect=CFD속도(`:653` Kleine §3.4), 보정 w_corr이 같은 스텝 body force에 반영.

---

## 문헌 대비 편차 총괄표 (2026-07-04 기준 — 발표/보고 시 명시 항목)

현재 활성 ALM+보정 파이프라인의 성분을 "논문 그대로 / 우리 임의 / 수치 중립" 3급으로 분류.
(질문 배경: 적용 방법들이 문헌 방법인지 임의 보정인지 — 세션 2026-07-04)

### A. 논문 그대로 (paper-faithful)
| 성분 | 출처 | 비고 |
|---|---|---|
| edge 기반 trailed-vorticity 보정 (N+1 edge, Γw=ΔΓ, +1/4π, K=exp(−(d/ε)²)) | Dağ & Sørensen 2020 Eq.17-18 | 2026-07-03 수정 = 구코드(마커 dΓ/dr, 팁·루트와류 누락)를 논문대로 **복원** |
| prescribed helix (pitch=국소 φ, 2rev, 2° 스텝) | Dağ §3.2 | 기하·커널 faithful |
| relaxation w=relax·w_new+(1−relax)·w_prev | Martínez-Tossas & Meneveau 2019 (Kleine 2022 §1 method B) | Dağ 원문은 method A(스텝내 반복); B는 문헌 표준 대안, steady fixed-point 동일 |
| Kleine 비반복 선형해 + 3차다항 Cl slope + 2° 셰딩 + CFD-advect free-wake | Kleine 2022 | 잔여 편차는 patch_notes/alm_dag_edge_fix/02 (rebuild_every>1=근사 등) |
| Gaussian integral 샘플링, ε=max(0.25c,2Δx) | Merabet 2021 / Martínez-Tossas 2017 / Asmuth | 표준 관행 |
| descent·projection=−thrust_axis | — | 논문 의도(다운워시 방향)의 올바른 구현 = 버그수정, 편차 아님 |

### B. 우리 임의 추가 (비문헌 — 공개 시 명시 + 민감도 부록 필요)
| 성분 | config | 동기(튜닝 아님) | 논문 literal 재현 |
|---|---|---|---|
| **Γ 스팬 평활화** ([0.25,0.5,0.25]ⁿ) | `eps_correction.smooth` (기본 0) | w∝−Γ″ 피드백 Nyquist 이득 7.0>1 폭주(sawtooth) → smooth=2서 0.73<1. 안정성 해석 근거. 논문 환경(매끈한 풍력 하중·성긴 마커)선 미발현 | `smooth: 0` |
| **helix pitch guard** (φ≤0 edge만 양의-u_n 평균으로 하강) | `eps_correction.helix_pitch_floor` (기본 "auto") | 호버서 팁 자기유도 업워시/루트 재순환이 Dağ의 φ>0 전제 위반 → 필라멘트 상류 상승(비물리). 로터 prescribed-wake 전통(Landgrebe 등)은 하강률을 양수 규정 — 개념은 로터 문헌 표준, 구현은 우리 것 | `"off"` |
| sweep cosΛ + Mach-pass | (rotor 기하) | 표준 공기역학(단순 후퇴각 이론)이나 LBM-ALM 5편엔 없음 | Λ=0 → byte-identical |
| `wake_markers`/`n_w` 서브셋 | `eps_correction.*` | 진단 계측용 | `"all"` = 논문형 |

**권장 보고 문구**: "numerical regularization of the trailed-vorticity kernel (2-pass binomial
spanwise Γ filter)" + "wake-transport guard for locally reversed sampled inflow at the blade
ends". 부록 = smooth {0(발산),1,2,3} × floor {off,auto} CT·spanwise 민감도.

### C. 수치 중립 (모델링 내용 없음 — 편차 아님)
freewake pruning(prune_factor=6, bit-동일 2e-16) · 폴라조회 배치화(bit-동일, `ALM_POLAR_BATCH=0`
A/B 토글) · helix/kleine rebuild 캐싱(=1이면 exact; >1만 근사로 명시) · get_query 이름캐시.

### 함정 메모
- 임의 성분 ①②는 실측 근접용 노브가 아님: ①은 안정화(방치=발산), ②는 적정성 가드(φ≤0서
  모델 정의 자체가 붕괴). 둘 다 최소개입(①opt-in 패스수, ②무효 edge만 치환·유효 edge
  byte-동일)이며 literal 재현 스위치 보존.
- 단 smooth는 Γ″를 깎으므로 **보정 크기 자체를 약간 줄이는 방향** — smooth 민감도 없이
  "Dağ가 약하다" 단정 금지(민감도 런으로 분리할 것).
