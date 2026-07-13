# 작업 보고서: LBM 솔버 HPC/병렬화 트랙 완결 (2026-07-08 ~ 07-13)

목적: 외부 연구자 검토용 — 수행한 작업의 전모와 결과를 제시하고,
**개선 가능 지점에 대한 평가를 요청**한다. 모든 수치는 실측이며,
주장-증명의 상세 대응은 `docs/MULTIGPU_DESIGN_PHILOSOPHY_kr.md`(1·2차
검토 승인 완료), 시간순 전체 로그는 `patch_notes/hpc_upgrade/17_*.md`.

## 1. 배경과 목표

- 솔버: Python/CuPy D3Q27 cumulant LBM + 5-레벨 MLG(cell-vertex) + ALM.
- 검증 케이스: HVAB 로터 호버, D40 = 91.6M 셀, RTX 4090 24GB.
- 장기 목표: 음향(수억 셀급) — 이번 트랙의 요구는 ①D40이 단일 24GB에
  들어갈 것 ②멀티 GPU로 속도·용량 확장 ③전 과정의 정확도 증명.

## 2. 단일 GPU 트랙 결과

| 항목 | 방법 | 결과 |
|---|---|---|
| f 메모리 반감 | Esoteric Pull(단일버퍼 in-place) 부활·cumulant 이식 | D40 19.8→9.9GB, **24GB 단일 실행 성립** |
| 커플링 병목 | cubic-z coalescing + C2F rescale 융합 커널 | 4090 pure-LBM 391→264ms (1.48×) |
| region 변환 | 27회 fancy-index → 단일 RawKernel | 커플링 5.19→1.20 s/step (@NR=1) |
| ALM 샘플링 | (N,S³) 청크 체인 → 마커당 1블록 커널 | ALM 구간 0.605→0.261 s/step |
| 초기화/체크포인트 | x-슬랩 청킹, slab-stream 저장 | 피크 52.6→19.2GB, ckpt OOM 해소 |

단일 GPU 최종: D40 **1.084 s/step** (트랙 시작 시점 3.1 → 2.9×).

## 3. 멀티 GPU 트랙 결과 (M1–M5 + 백로그 5종)

구조: 축-일반 1D 슬랩 분해, v1 물리밴드 halo(ghost=3 w/ SGS), 파생 컷
기반 rank-로컬 MLG 커플링(레벨 간 통신 0), ALM 부분합 Allreduce(16KB)
+ 복제 solve, 부하균형 컷(최내곽 박스 내 배치), SPMD 러너 + CUDA-aware
MPI. 상세 논증은 철학 문서 §2–5.

| 지표 | 값 |
|---|---|
| 정확도(결정성) | 분해-vs-단일 **bit-identical** — 5레벨 필드·마커·추력, 2/3/4-rank, 2회전 수용시험(40,224 substep) |
| 정확도(장기 통계) | 25-rev 풀런 CT **+0.027%**(rev-내 σ ±0.6%의 1/20), 팁 max\|ω\| +3.5%(카오스 폭 내). ※bit와 모순 아님 — ALM 경계마커 재결합의 f32-cast 방화벽(플립 확률 ~1e-9/값)이 두 결과를 정량 정합하게 예측: 2-rev 기대 플립 ~0.03회 → bit, 25-rev ~10⁹회 반올림 → 수 회 플립 → 카오스 증폭 → 통계 동일성만 성립 |
| 성능 (D40 4×4090) | **0.442 s/step** — 동코드 분해이득 2.45×, 병렬구간 분할효율 ~100% (잔여 = BEM 복제 0.08 + lockstep 스큐 0.1) |
| 용량 | 분산 초기화(dist-init)로 4-GPU ~110M → **~450-500M 셀** (빌드 피크 19.2→0.91GB/rank) |
| 기능 | 재시작(bit roundtrip), VTK/checkpoint/마커 production 포맷 rank0 조립, 3-tier 진행로그(ALM: CT/CP/FM, body: CD/CL/CS, flow: rho/u_max) |
| 물리 케이스 커버리지 | ALM·고체경계(HWBB MEM force)·순수유동 전부 bit 게이트; 레거시 2D 익형/실린더 56개 config nu-마이그레이션으로 복원 |

## 4. 검증 체계와 그 성과

bit-parity 3등급 사다리(bit / fp-lastbit / CV-band) + 단계별 게이트
12종. **연산마다 등급을 사전 논증하고 위반은 등급 내라도 규명**하는
원칙이 잡아낸 실결함 9건: esoteric 초기화 롤 누락, SGS ghost 깊이,
x-체인 퇴화, 균등분할 불능, shape-의존 라이브러리 리덕션, 커플링 연산자
미공유, verify 진입점 파손, **재시작 로터 위상 리셋(production 버그)**,
NR=1 메모리 이중보유. 이 중 3건은 tolerance-only 검증이었다면 물리를
조용히 오염시켰을 것이다.

최종 수용시험: 단일 GPU vs 3-rank, 2회전(40,224 substep) — 전 필드·
마커·추력 **완전 bit 동일**.

## 5. 물리 트랙 현황 (참고 — 다음 작업의 배경)

HVAB rigid CT 과대예측 +19~27%를 분해: 팁(r/R≥0.9) roll-off 실패가
지배 → archB(반경 절단+Kleine 보정)로 팁은 해결(0.217 vs KSAS 0.215),
**잔여 +15%는 피크영역(0.85–0.97) finite-ε 커널 편향**으로 수렴. 격자
(D20–80 외삽), 마커 수, 덱, SGS 소산(3중 기각) 전부 배제됨. 다음 작업
= β kernel(compact 커널 + 보정 재유도) — 별도 handoff 문서 참조.

## 6. 알려진 한계 (전부 fail-fast 또는 문서로 명시)

- dist-init: 균일 IC 전용, 재시작 병용 불가
- 분산 ALM: kleine free-wake·비gaussian 샘플러 미지원(production은 straight)
- v2 slot halo(6× 트래픽↓): 프로토콜 bit 증명 완료, production 결합은
  음향급 강스케일링 실측 대기(차단 분석 문서화)
- 1D 분해: 4–8 GPU 규모 적정; 그 이상은 재평가
- IBB는 표준 경로(main.py) 전용(esoteric 커널은 HWBB 암시적)
- hover-ALM 레거시 config 9개: 수동 nu 마이그레이션 필요 상태로 잔류

## 7. 검토 요청: 개선점 평가를 바라는 지점

1. **BEM 복제 (0.08 s/step)** — 마커 분할+Allgather로 병렬화 가능하나
   복제의 단순성을 깨는 가치가 있는가? (rank 수가 커지면 재평가?)
2. **lockstep 스큐 (0.1 s/step)** — 현재 fresh-skip+early-post/Irecv
   프리포스트까지 적용. 남은 스큐를 줄일 표준 기법 중 우리 구조(재귀적
   MLG 스케줄)에 맞는 것이 있는가? (interior/edge 분할 런치의 비용 대비?)
3. **에너지 효율/occupancy** — 우리는 대역폭·런치 병목만 다뤘고 커널
   내부 occupancy 튜닝(블록 크기, 레지스터)은 미착수. 의미 있는 여지가
   있다고 보는가?
4. **2D/3D 분해 확장** — 음향급(8+ GPU)에서 1D 슬랩의 한계 예상 지점과
   전환 기준에 대한 의견.
5. **checkpoint 확장성** — rank0 host 조립(450M이면 npz당 ~50GB)의
   대안(병렬 I/O, 레벨 분할 파일)이 필요해지는 규모는?
6. **검증 체계 일반화** — bit-사다리 방법론을 다른 그룹 코드에 이식할
   때의 최소 요건(치환-순수 프리미티브의 격리)에 대한 견해.

## 8. 재현 방법

- 실행: `docs/SIMULATION_RUN_GUIDE_kr.md` (tier 스모크 3종 = 각 ~1분)
- 게이트 전수: 같은 문서 §5 (로컬 1-GPU에서 전부 재현 가능)
- 학습용 해설: `docs/LEARNING_hpc_parallelization_kr.md`

---

## 부록 A. 3차 검토 반영 (2026-07-14)

검토 원문: `docs/MULTIGPU_REVIEW_kr.md` 3차 절(트랙 마감 승인 + 要폐쇄 2건).
코드 변경 상세: `patch_notes/hpc_upgrade/19_review3_response.md`.

| 지적 | 판정 | 처리 |
|---|---|---|
| R3-1 prepost Irecv race (要폐쇄) | 수용 | prepost를 commit() 뒤로 이동(스트림 sync가 직전 scatter 완료를 구조적 보장) + ORDERING CONTRACT docstring. G-verify 재게이트 PASS. cuda-aware 실경로 확인은 다음 클러스터 세션 1회 |
| R3-2 eso_mem_force 쌍둥이 게이트 (要폐쇄) | 수용 | `eso_mem_force_twin_gate` 신설 → **실결함 #10 즉검출**: "암시적 HWBB"가 실제로는 2-스텝 지연 bounce(정상상태 수렴이라 rank-불변·CD 그럴듯 — "잘못된 규약도 rank-불변일 수 있다"의 실증). LOAD parity-swap 수정 + IC 시딩(`eso_seed_solid_bounce_ic`, restore 제외) + v2 메일박스 NODE_TRANSIT 분리. 추출 규약은 동일 상태 위 f64 완전 일치(diff=0)로 증명 |
| R3-3 2D end-to-end 게이트 | 수용 | `cyl2d_re100_gate`(~100s, main.py 전 체인) + 기록 Cd 밴드 1.181±0.05. 회귀 스위트 등재 |
| R3-4.1 strict-bit 레그 ALM config | 수용 | pure_lbm config로 교체(사다리 등급과 게이트 단언의 형식 모순 해소) |
| R3-4.2 §3 bit/통계 혼합 서술 | 수용 | 본 보고서 §3을 결정성/장기 통계 2행으로 분리, 방화벽 모델의 정량 정합(2-rev bit ↔ 25-rev +0.027%) 명시 |
| R3-4.3 config 재임포트 | 수용 | setup 해제 전 `setup.config`에서 추출(이중 파싱 제거). 스모크 CD 동일 |
| R3-4.4 Gaussian 4번째 지점 | 확인 | β kernel 핸드오프 목록 포함 확인 — β 첫 수술 대상 |
| R3-4.5 하우스키핑 | 수용 | canonical-axis 단독 커밋(45caea9), eps_r/collector/stale-handoff 정리 커밋(b5e5c94), 검토 보고서 track |
| R3-5 §7 답변 6건 | 백로그 재조정 | 즉시 수정 아닌 로드맵 반영(patch 19 §5): ckpt 슬랩-npz **450M 런 전 필수(최상위)**, CUDA Graph 재도입 승격, 스큐 조성분해 선행·우선순위 하향, 2D 분해 설계 보류, BEM 복제 유지(트리거 명시), 방법론 요건 = 논문 §2 초안 |

게이트 회귀: 기존 14종 + 신규 2종 **전부 PASS** (로컬 3090; sphere 스모크
replicated·dist-init --verify 전 레벨 bit). HWBB 수정은 순수유동·ALM 케이스에
bit-중립(분기 미활성; 게이트 무변화로 실증). sphere 스모크 CD 트레이스 기록은
지연-bounce 값에서 교정판(32-step 종점 +0.3296)으로 대체 — R3-0의 대조 앵커
(−0.4465)는 구 스킴 기준이었음.
