# Patch 19 — 3차 검토(R3) 대응 (2026-07-14)

검토 원문: `docs/MULTIGPU_REVIEW_kr.md` "3차 검토" 절. 대응 요약표는 작업 보고서
(`docs/WORK_REPORT_2026-07_kr.md`) 부록 G. 이 문서는 코드 변경의 상세 기록.

## 1. R3-1 (要폐쇄) — prepost Irecv를 commit 뒤로 [수용·수정]

지적 그대로: `HaloBandExchangerV1.post`의 side 루프 안에서 `prepost`가
`commit()`의 stream sync **이전**에 Irecv를 게시 → 직전 라운드 scatter 커널
(동일 persistent rbuf를 읽음)이 미실행인 시점에 스큐 앞선 이웃의 Isend가
매칭되면 UCX cuda_ipc가 읽는 중 버퍼에 device-write하는 race.

- 수정: `halo.py` — prepost 루프를 `commit()` **뒤** 별도 루프로 이동.
  commit의 stream sync가 "직전 scatter 완료 후 Irecv 게시"를 구조적으로 보장.
  오버랩 이득(post~complete 구간 전송 은닉)은 유지.
- `MPITransport.prepost` docstring에 ORDERING CONTRACT 명문화 + "오버랩
  작업에서 가장 먼저 깨질 가정"(commit sync 제거 시 이중버퍼/event 대체) 명시.
- 재게이트: G-verify(mpirun 2-rank, host-staged) PASS. cuda-aware 실경로는
  로컬 재현 불가(검토 지적 그대로) — **다음 클러스터 세션 1회 확인 항목**.

## 2. R3-2 (要폐쇄) — eso_mem_force 쌍둥이 게이트 [수용·게이트 신설 → 실결함 #10 발견·수정]

신설: `gates/eso_mem_force_twin_gate.py` — 동일 sphere 케이스를 표준 경로
(two-buffer + HWBBKernel + `mem_force_d3q27`)와 esoteric 경로(암시적 HWBB +
`eso_mem_force`)로 구동, 4-claim 등급 사전 논증(P/A1/A2/B — 게이트 docstring).

**게이트가 즉시 잡은 실결함(트랙 결함 #10): esoteric "암시적 HWBB"가 실제로는
2-스텝 지연 bounce였다.**

- 기제: solid 이웃은 STORE하지 않으므로, 통상 슬롯을 읽는 LOAD는 t−1이 아닌
  **t−2의 예치**를 읽는다(짝 t 예치=slot i, 홀 t+1 읽기=slot i+1 — 어긋남).
  정상상태에선 HWBB와 수렴해 CD·rank-불변성 모두 그럴듯했고(수용시험 bit도
  분해-불변성만 검증하므로 통과), 과도/비정상 응답은 틀린 스킴 — 검토자의
  "잘못된 규약도 rank-불변일 수 있다"의 실증.
- 수정 1 (`esoteric_d3q27.py`/`esoteric_cumulant_d3q27.py` LOAD): solid 이웃
  방향의 읽기를 **parity-swap된 슬롯**으로 전환 = 자기 자신의 t−1 outgoing
  = 교과서 halfway BB. 순수유동 셀은 분기 미활성(bit-중립; eso/mgpu 게이트
  전수 무변화 PASS로 실증).
- 수정 2 (IC 시딩, `eso_seed_solid_bounce_ic`): swap-슬롯의 "step −1 예치"가
  fresh IC에는 없으므로 f0의 incoming으로 시딩(std 경로와 step 0부터 동일
  규약). **restore에는 호출 금지**(gather∘scatter 전단사가 예치를 bit 보존).
  시딩은 유체-파트너가 있는 링크만 기록 — solid-solid 링크까지 쓰면 슬랩
  재구축(gather 기반 f0) 시 직접-IC 빌드와 어긋남(sphere --verify가 검출).
- 수정 3 (NODE_TRANSIT=5): v2 halo의 ghost 메일박스가 SOLID 표기를 재사용
  중이었음 → HWBB 수정 후 벽으로 오동작(G-M4 즉검출). "skip하되 bounce 의미
  없음"의 TRANSIT 타입 신설, v2 게이트/문서 전환. G-M4 3축 bit + 6.00× 복원.
- 결과: twin gate PASS — [P] std↔eso 필드 fp-lastbit(8.3e-7/24step; 두 collision
  codegen이라 bit는 원리적 비대상), [A1] **동일 상태 위 f64 완전 일치(diff=0)**
  = 추출 규약 등가의 강한 증명, [A2]/[B] 밴드 내.
- sphere tier 스모크 재기록: 32-step CD 트레이스 −1.0120(8) → −0.1635(16) →
  −5.6990(24) → **+0.3296(32)** (구 지연-bounce 기록 −0.4465→0.5133 대체).
  replicated·dist-init 양쪽 --verify 전 레벨 bit 유지.

## 3. R3-3 — 2D end-to-end 게이트 [수용·게이트 신설]

신설: `gates/cyl2d_re100_gate.py` + `configs/hpc_bench/cyl_re100_2d_gate.py`.
main.py 실제 진입점으로 D2Q9 cumulant+IBB+MLG3+MEM force 전 체인 구동(~100s,
32 MLUPS). 수용 기준: 기록 밴드 tail-mean Cd 1.181±0.05(37 CT 발달 중 정상파;
문헌 케이스는 full-size config가 담당) + |Cl|<0.05 + force CSV 생성.
range(3)류 2D 파손은 이제 게이트가 즉검출.

## 4. R3-4 사소/일관성 4건

1. **strict-bit 레그 → pure_lbm** [수용]: G-verify의 strict-bit 레그를
   `bench5_pure_lbm.py`로 교체(ALM=fp-lastbit tier와 게이트 단언의 형식 모순
   해소). PASS.
2. **보고서 §3 bit/통계 분리** [수용]: 정확도 행을 결정성(2-rev bit)과 장기
   통계(25-rev CT +0.027%)로 분리, 방화벽 확률 모델이 둘을 정량 정합하게
   예측함을 명시(2-rev 기대 플립 ~0.03회 ↔ 25-rev 수 회 → 카오스 증폭).
3. **body-tier config 재임포트 제거** [수용]: main_mpi가 이미 파싱된
   `setup.config`에서 `force_calculation.reference`를 setup 해제 전에 추출
   (importlib 이중 실행 우회 제거). sphere 스모크 CD 동일 확인.
4. **Gaussian 하드코딩 4번째 지점(샘플링 RawKernel)** [확인]: β kernel
   핸드오프 추상화 목록에 포함 확인 — β 작업 첫 수술 대상으로 재확인만.

## 5. R3-5 — §7 답변 6건: 백로그 우선순위 재조정 (즉시 수정 아님)

검토자 답변을 로드맵에 반영. 코드 변경 없음:

| # | 항목 | 재조정 |
|---|---|---|
| 1 | BEM 복제 | **유지 확정**. 재평가 트리거 = R≥8-16 또는 β 보정의 iterative화(후자가 개연적 — β 트랙과 연동 감시) |
| 2 | lockstep 스큐 | 기법 추가 전 **조성 분해 선행**(Iprobe로 "데이터 미도착" vs "정적 불균형 +6.4%" 판별). 정적이면 컷 가중을 실측 per-rank ms로. 완전 제거보다 β 트랙 시간이 더 가치 있음 — 우선순위 하향 |
| 3 | occupancy | 루프라인 1점 측정(실측 GLUPS vs 대역폭/216B) 선행 — LBM은 대역폭 포화 근처라 튜닝 여지 한 자릿수 %. 실질 레버 = **CUDA Graph 재도입**(레벨-버스트 부분 capture; SU2 방향의 per-step 투자와 일치) — 백로그 승격 |
| 4 | 2D/3D 분해 | 두 천장(기하: 최내곽 span/ghost; 경제: halo+스큐>20%)을 케이스 정의에서 선계산. SU2 커플링 확정 시 8+ GPU 시나리오 소멸 가능 → **케이스가 강제할 때까지 설계 금지** |
| 5 | checkpoint | **임계 이미 도달**(450M ≈ 48GB/ckpt). rank별 슬랩 npz + 매니페스트(bounds), same-R 재시작은 자기 파일만, 조립은 lazy. **450M production 런 전 필수 — 백로그 최상위 삽입** |
| 6 | 방법론 이식 | 요건 ①~⑦ + "동일 디바이스 세대 한정" 명시 = 방법론 논문 §2 초안으로 채택 (2차 검토 Proposal A와 연결) |

## 6. 게이트 회귀 총결산 (로컬 3090, 전부 현재 트리)

eso 7종(gather_scatter/cumulant_equiv/bgk_equiv/sim_integration/mlg/
coupling_scoped/sgs_alm) + mgpu 7종(M1/M2a/M2b/M3/M4/G-verify/G-restart)
+ 신규 2종(twin/cyl2d) = **16종 전부 PASS**. sphere 스모크 replicated·
dist-init --verify 전 레벨 bit. 순수유동·ALM 게이트는 HWBB 수정 전후
bit-무변화(분기 미활성 논증의 실증).
