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
| 정확도 | 분해-vs-단일 **bit-identical** (5레벨 필드·마커·추력; 2/3/4-rank; 25-rev 풀런 CT +0.027%) |
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
