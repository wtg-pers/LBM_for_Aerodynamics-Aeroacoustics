# Multi-GPU LBM 솔버 검토 보고서 (외부 검토자 관점)

검토일: 2026-07-12 · 대상: `docs/MULTIGPU_DESIGN_PHILOSOPHY_kr.md` + `src/parallel/` + `main_mpi.py`
+ production 훅(`src/actuator/actuator_line.py`, `src/grid/coupling.py`) + 게이트
(커밋 `d52b375` 작업트리 기준; 멀티GPU 코드는 `2c40712`와 동일)

---

## 0. 총평

**승인.** 보고서의 주장과 코드가 일치하며, 검토 요청 포인트 5건의 논증은 전부
성립한다(각각 독립적으로 재유도/재실행하여 확인 — §2). 설계 철학(비트-parity
사다리, 프리미티브 재사용, 게이트 사다리)은 문헌 표준(tolerance 기반 검증)보다
강한 기준이고, §6의 1-ulp 사건이 그 실효성을 실증한다. 이 수준의 검증 체계를
갖춘 분산 LBM 구현은 드물다.

유보 1건: **성능 주장(~1% 오버헤드)은 아직 견적**이다. 클러스터 §2(farfield40
실런)가 남아 있고 보고서도 이를 정직하게 표기한다. 아래 F-2/F-3의 두 구현
디테일이 이 견적에 반영되지 않았으므로, 실측이 견적을 상회하면 그 둘부터 볼 것.

지적 사항: 중요도 중 1건(F-1: ALM 부분합에 잔존하는 라이브러리 리덕션 — §6
교훈과의 긴장), 성능 2건(F-2, F-3), 프로세스 1건(F-4), 문서 표현 2건(F-6).
모두 현재 검증 결과를 뒤집지 않는다.

## 1. 검토 방법

1. `src/parallel/` 7개 모듈 + `main_mpi.py` 전수 독해, production 훅
   (`actuator_line.py`의 `_grid_offset`/`_velocity_sampler`/scale 컨텍스트,
   `coupling.py`의 고정순서 리덕션, `interpolation.py`의 `return_sums`) 대조.
2. 핵심 논증 3건을 **코드에서 독립 재유도**: ghost=3 충분성(esoteric
   population-set 의미론으로), MLG 커플링 여백(cubic 4점/필터 7·19점 스텐실
   커버리지), v2 슬롯 대수(gather/scatter 공식에서 STORE 규칙 유도 → A/B 테이블
   검산).
3. 로컬 재실행 가능한 게이트 4종을 현재 HEAD에서 재실행(RTX 3090):

| 게이트 | 재실행 결과 | 보고서 주장과 일치 |
|---|---|---|
| G-M1 | x/y/z 3축 f·ρ·u 전부 bit (max\|df\|=0) | ✓ |
| G-M2b | bench5 5레벨(dyn_smag+eq/sponge) y/z 전 레벨 bit, x축 자동기각 회귀 ✓ | ✓ |
| G-M3 | pure-ALM·archB 필드 bit, F_grid rel 6.65e-17 | ✓ |
| G-M4 | 3축 bit + 트래픽 정확히 6.00× (486→81 KiB/step) | ✓ |

클러스터 결과(§1(a)(b)(c))는 로컬 재현 불가 — 진단 체인(§6)의 논리 정합성으로
간접 평가(아래 2.6).

## 2. 검토 요청 포인트 5건 판정

### 2.1 중복 ghost 계산 논증 (§3.2) — **성립, 단 재검토 규칙을 더 날카롭게**

기제를 재유도한 결과, 논증은 성립하되 보고서/게이트 docstring의 서술보다 실제
구조가 더 깔끔하다. 정확한 그림:

- 교환 후 ghost 셀의 **population set은 완전**하다(scatter가 pair-슬롯을 wrap
  포함해 전부 기입하고, 커널 LOAD의 주기 인덱스 산술과 wrap 의미론이 일치).
  따라서 macro u는 **동기화된 ghost 전 층에서 유효**하다 — "ghost 안쪽으로
  갈수록 u가 한 층씩 오염"이 아니다.
- 오염의 유일한 진입로는 **비국소 스텐실이 로컬 배열의 wrap 경계를 넘는 것**
  이다. dyn_smag는 5×5×5(반경 2) u-스텐실이므로(`dyn_smag_d3q27.py`), 최내곽
  ghost(소유 edge에 STORE하는 층)의 nu_t가 유효하려면 그 셀에서 반경 2까지 u가
  유효해야 하고, 이는 ghost=3에서 **마진 0으로 정확히** 충족된다
  (최내곽 ghost의 스텐실이 ghost 3층 + 소유 2층을 정확히 덮음).
- 소유 셀의 t+1 population 중 다음 교환이 덮지 못하는 것은 최내곽 ghost의
  STORE 산출뿐이고, 그 셀의 입력(자기 set·force·nu_t)이 전부 정확하므로 소유
  궤적은 bit 동일 — G-M2b가 이를 실증.

**권고 (재검토 규칙의 명문화)**: 보고서의 "비국소 입력을 읽는 커널이 끼면 한 층
더"는 과소평가 위험이 있다(그때 +1로 충분했던 것은 dyn_smag가 반경 2이고 pure
ghost=2에 이미 마진 1이 있었기 때문). 올바른 일반 규칙은:

> **ghost ≥ 1 + r**, r = "최내곽 ghost 셀의 collision 출력에 영향을 주는
> 비국소 입력의 총 반경" (dyn_smag: 그래디언트 1 + test filter 1 = 2 → ghost 3).

test filter를 넓히거나(7³), 비국소 입력이 있는 커널을 추가하면 ghost=3은 즉시
깨진다(마진 0). 이 규칙을 `halo.py` docstring 또는 커널 추가 체크리스트에 넣을
것. 아울러 M2b 게이트 docstring의 "u corrupt at ghost-3"는 기제 서술로는
부정확하다(§F-6) — 검토 포인트로 지목된 논증인 만큼 표현을 교정해 두는 것이
좋다.

### 2.2 MLG 커플링 여백 (§3.3) — **성립**

- **C2F**: cubic upsample은 4점(반경 2 coarse행) 스텐실. 소유 fine 스트립의
  부모 coarse행 i ∈ [own_start, own_end)에 대해 스텐실 {i−1..i+2}는 블록
  (owned±2행) 안에 정확히 들어간다 — ±2 여백은 필요·충분. one-sided 스텐실은
  블록 edge 1행 이내에서만 발생하며 이는 소유 스트립 밖이고, 진짜 box 끝에서는
  블록 edge=global edge라 global과 동일한 one-sided ✓ (`mlg_coupling.py`
  docstring 논증 그대로 확인).
- **F2C**: 7/19점 필터는 strided(coarse 간격) 배열에서 반경 1 — 읽기 +1
  coarse행 여백으로 기록 전 행이 centered ✓. box 끝 클리핑은 global의 interior-
  only 필터 edge 처리와 일치 ✓ (`_filter_f_neq`가 interior만 갱신함을 확인).
- 여백 데이터의 신선도: 커플링 직전 해당 레벨 re-sync가 러너 스케줄에 있고
  (ghost=3 ≥ 필요 2), 파생 컷의 겹침/빈틈 없음은 `RankLocalCouplingV1.__init__`
  의 일관성 assert(파생치와 불일치 시 raise)가 구조적으로 강제한다.

### 2.3 balance_cuts 보장 (§2.3) — **성립, 실질 보장은 이중 안전장치**

"컷이 최내곽 박스 span 안에 있으면 전 rank 전 레벨 소유"는 중첩(nesting) 하에서
옳다: 각 rank 구간이 최내곽 span과 교차하고, 최내곽 span ⊂ 모든 상위 박스
span이므로. 다만 **실질 보장은 논증이 아니라 검사에서 온다**는 점이 이 설계의
미덕이다 — `chain_owns`가 러너와 동일한 `fine_range_from_coarse` 산술로 전
(rank, level) own을 시뮬레이션하고, `balance_cuts`와 러너 양쪽에서 min_own <
ghost 시 raise. 논증이 틀려도 조용히 깨질 수 없다.

한계 2건(보고서 §8.3의 자문에 대한 답):
- **단일 중첩 박스 체인 가정은 partition/coupling/runner 전반에 구조적**이다
  (`couplings[k]._region.fine_domain_coarse` 단수 접근, `level_spans_L0`의 체인
  합성). 비중첩 다중 박스 도입 시 이 계층 전체 재설계가 필요 — 보고서 표기
  그대로이며, 현 솔버가 중첩 전용이므로 지금은 리스크 아님.
- rank 수 상한이 최내곽 span에 묶인다: R−1개 컷이 간격 ≥ghost로 span 안에
  들어가야 하므로 대략 R ≤ 1 + span_L0/ghost. farfield40(L4 span이 수십 L0셀)
  에서 4~8 rank는 여유이나, 1D 분해의 스케일 한계(§7)와 같은 지점에서 만난다.

### 2.4 ALM 소유권 정확성 (§4.1) — **성립, 발견 1건(F-1) 첨부**

- 자연 클립 확인: 샘플러가 소유 뷰만 전달(`owned_local()` 슬라이스 + ghost
  오프셋 보정)하고, `interpolation.py`의 valid 마스크가 뷰 밖 스텐실 노드를
  정확히 0 가중 처리 — ghost 이중계상이 코드 경로상 불가능 ✓.
- 내부 마커의 bit 재현 논증도 정확: 타 rank의 기여가 정확한 +0.0(IEEE에서
  x+0.0=x)이고, 리덕션 배열 shape (N, S³)가 rank-불변이라 동일 디바이스에서
  결합순서 동일 ✓. 경계 마커만 연속 분할에 의한 재결합 차이 = fp last-bit ✓.
- 스프레딩: 각 rank가 전 마커를 로컬 전역(ghost 포함)에 뿌리는 구조가 옳다 —
  ghost 셀의 force 정확성은 중복 ghost 계산 논증(2.1)의 전제이기도 하다(최내곽
  ghost의 collision이 force를 씀). radial-trunc 재정규화 scale만 전역 기하
  (`scale_domain_shape`/`scale_positions`/`scale_hub`)로 계산함을 확인 ✓.
- 비대칭 대안(rank0 solve+bcast) 기각은 타당: 결정적 파이프라인 복제가 통신·
  분기 없이 동일 출력을 주고, 7ms 중복은 R에 상수다.

### 2.5 v2 슬롯 대수 (§3.4) — **성립 (독립 검산 완료)**

gather 공식(짝 parity: f_std[i](x)=mem[i+1][x], f_std[i+1](x)=mem[i][x+cᵢ];
홀은 역할 교환)에서 STORE 규칙을 유도하면 — 짝 t: gᵢ(x)→mem[i][x+cᵢ],
g_{i+1}(x)→mem[i+1][x]; 홀 t: 슬롯 스왑 — 이로부터:

- **A그룹**(유출분이 우리 ghost 평면에 예치): +face, cᵢ[axis]=+1 → 슬롯
  (짝 t: i / 홀 t: i+1) = `_slot_tables`의 a_even/a_odd와 일치 ✓
- **B그룹**(edge 평면 상주분을 이웃이 원격 LOAD): cᵢ[axis]=−1 pair → 슬롯
  (짝: i+1 / 홀: i) = b_even/b_odd ✓
- low-face 미러 = "부호 반전 pair + parity 라벨 반전" → A_low=B[1−par],
  B_low=A[1−par] ✓, 수신 슬롯이 수신자 자신의 STORE 슬롯과 pair 내 상보(i vs
  i+1)라 충돌 불가 ✓.

same-slot copy(재라벨링 불필요) 주장까지 대수적으로 확인. G-M4 재실행: 3축
bit + 정확히 6.00× ✓. 결합 보류(실측 후) 판단도 §1.7 원칙과 정합.

### 2.6 §6 사례 연구(1-ulp 사건) — 서사와 코드가 일치

`coupling.py`의 `_compute_macroscopic`/`_compute_f_eq`가 실제로 고정순서 ±직렬
누적으로 교체되어 있고(q승순 serial, c∈{−1,0,+1} 곱셈 제거), `_filter_f_neq`
도 명시적 elementwise 합(라이브러리 리덕션 없음)임을 확인. 진단 체인(loopback
재현→디바이스 동일성→`--devices 0,0,0,0`→diff 위치 국소화)은 각 단계가 정확히
하나의 가설을 기각하는 모범적 포렌식이다. 2⁻²⁷=2/27 가중 population의 f32
1 ulp라는 수치 정합도 옳다.

## 3. 추가 발견 (보고서 밖)

### F-1 (중) — ALM 부분합에 라이브러리 리덕션 잔존: §6 교훈과의 긴장

§6의 교훈은 "분산 경로에 파이썬 수준 라이브러리 리덕션이 하나라도 남으면 shape
의존성이 생긴다"인데, **분산 ALM 샘플링의 부분합 자체가 `xp.sum(weights/
u_local, axis=(1,2,3))`** (CuPy 리덕션, `interpolation.py`)이다.

오늘 이것이 안전한 이유는 명확하다: 리덕션 배열 shape (N_markers, S, S, S)가
rank-불변이고 단일-GPU 레퍼런스 경로와도 동일 함수·동일 shape이므로, **동종
디바이스에서는** 결합순서가 항상 같다(경계 마커의 rank 간 재결합만 last-bit —
이는 프로토콜의 공인된 등급). 게이트/클러스터 bit 결과와도 모순 없다.

그러나 §6에서 잡은 버그 클래스의 잔존 표본이다: 이기종 GPU 혼용(§6의 3090 vs
4090 재현 불가가 정확히 이 기제), CuPy 버전의 리덕션 전략 변경, 향후 chunking
허용 시 재발한다. **권고**: (a) 최소한 §7 한계표에 "ALM 부분합 = shape-불변
조건부 결정적(동종 디바이스 전제)"을 명시하거나, (b) §6 수정과 같은 급으로
고정순서 커널화. (a)만으로도 검토 관점에서는 충분하다.

### F-2 (성능, 소) — 러너의 중복 halo 교환

`runner.py` `_advance_fine`: `if has_finer: self._sync(k); self._save_fprev(k)`
직후 `self._sync(k)`가 다시 실행된다(두 지점). `_save_fprev`는 읽기 전용이므로
두 번째 교환은 동일 데이터의 재전송 — bit에는 무해하나 **finer 자식을 가진
모든 레벨의 halo 트래픽이 2×**다. §2의 ~1% 오버헤드 견적에 이 중복이 반영돼
있는지 불명. 실측(§2) 전 제거하거나, 실측 해석 시 유의할 것.

### F-3 (성능, 각주) — 메시지당 전역 디바이스 동기화

`MPITransport.post`가 cuda_aware에서 메시지마다 `deviceSynchronize()`를 호출
한다(레벨×서브스텝×2면×2메시지). v1의 "증명 단순" 철학과는 정합하나, 강스케일링
실측에서 latency 항이 견적을 상회하면 첫 용의자다(스트림-이벤트 동기화로 대체
가능). §1.7 원칙대로 실측 후 판단하면 됨 — 기록만 남긴다.

### F-4 (프로세스, 소) — verify의 종합판정이 tolerance 기준

`main_mpi.py --verify`의 RESULT는 max|df|<1e-4로 PASS를 찍는다. bit 플래그와
diff 위치 출력은 있으나(1-ulp 사건을 잡은 바로 그 출력), "예측 위반은 등급 안
이라도 규명"(§1.1)의 강제는 사람 규율에 의존한다. 순수-LBM 케이스는 bit이
예측이므로 `--strict-bit`(bit 실패 시 RESULT FAIL) 플래그를 권고 — §1.1을
도구에 내장하는 마무리다.

### F-5 (로드맵) — 복제 solve의 환경 동일성 전제

복제 BEM/보정의 rank 간 동일성은 "동일 입력 + 결정적 파이프라인"에 더해 **rank
간 동일 CPU/numpy/BLAS 바이너리**를 전제한다. 단일 노드 4×4090에서는 자명하나,
멀티노드 확장 시 한계표에 올릴 것(§7의 1D 분해 재평가와 같은 시점).

### F-6 (문서, 미세)

- M2b 게이트 docstring "u corrupt at ghost-3 → nu_t valid at ghost-1": 실제
  기제는 "동기화된 ghost의 u는 전 층 유효, 오염은 스텐실의 wrap-경계 초과에서만"
  (2.1) — 결론은 같으나 서술 교정 권장(검토 포인트 #1로 지목된 논증이므로).
- §3.2 v1 트래픽 "면당 2셀×27슬롯"은 pure(ghost=2) 기준 — SGS production은
  ghost=3라 3×27(v2 대비 9×). 표기 각주 권장.
- `partition.py` `choose_axis`의 `ok` 변수는 항상 True(죽은 코드). 코스메틱.

## 4. 문헌 대비 평가 (waLBerla-wind / Holzer 프레임워크)

- **ALM 통신**: walberla-wind는 마커의 서브도메인 침투를 예측·marking하여
  buffered MPI로 actuator 데이터를 교환한다. 본 구현의 partial-sum allreduce +
  복제 solve는 그와 다른 선택인데, 이 문제 설정(디스크-평행 축 분할에서 로터가
  컷을 상시 횡단, 마커 ~10²개, 16KB 메시지)에서는 **더 우월하다**: 소유권 추적
  상태가 아예 소멸하고 축-불가지론이 공짜로 얻어진다. 트레이드오프(rank당 중복
  solve)는 R에 상수라 4~8 GPU에서 무의미.
- **분해**: waLBerla의 블록구조+SFC 로드밸런싱 대비 1D 슬랩+quantile 컷은
  4~8 GPU PCIe에 맞는 옳은 단순화다(통신 상대 2 고정). 보고서가 한계(얇은 슬랩
  중복계산 비율)와 재평가 시점을 명시한 것도 적절.
- **검증 문화**: 문헌의 분산 LBM 검증은 통상 물리량 tolerance(스케일링 논문은
  성능만)다. bit-parity 사다리 + "등급 내 예측 위반 규명" + LoopbackTransport
  게이트는 그보다 강하며, §6 사건(tolerance면 영원히 숨었을 라이브러리 비결정성
  포착)은 그 자체로 방법론 기여다. 별도 짧은 논문/테크리포트 가치가 있다.
- **Esoteric Pull 분산**: Lehmann의 단일버퍼 스트리밍을 분산한 공개 구현은
  드물다(FluidX3D는 자체 방식). v1(물리밴드 왕복 변환)→v2(parity-slot 부분집합,
  재라벨링 불필요)의 2단 전략과 v2의 대수적 증명은 이 지점의 독자적 기여로,
  patch 17과 함께 기록 보존 가치가 높다.

## 5. 결론 및 권고 우선순위

**판정: 승인.** 검증된 범위(정확성) 내에서 production 사용에 이견 없음. 성능
주장은 클러스터 §2 실측으로 확정할 것.

| 우선순위 | 항목 | 성격 |
|---|---|---|
| 1 | 클러스터 §2 실런/스케일링 실측 (기예정) | 성능 주장 확정 |
| 2 | F-2 중복 `_sync` 제거 — §2 실측 **전에** | 실측 신뢰도 |
| 3 | F-1 ALM 부분합 리덕션: 한계표 명시 or 고정순서화 | §6 교훈의 일관성 |
| 4 | 2.1의 ghost ≥ 1+r 규칙 명문화 + F-6 문서 교정 | 미래 커널 안전 |
| 5 | F-4 `--strict-bit`, F-5 멀티노드 전제 한계표 추가 | 프로세스 |

---
---

# 2차 검토 (2026-07-13) — M5 완결 + 성능 패스 4종 + 출력/재시작 배선 이후

검토 대상: `d52b375..c792ba1`(11 커밋 — perf pass 1~4, M5b/M5c 출력, --restart,
production 재시작 버그 수정) + 갱신된 보고서(부록 C/D/E).

## R2-0. 총평

**승인 유지 + 회귀 버그 1건(必修) 발견.** 1차 검토의 F-1~F-4는 전부 적절히
처리되었고(F-1 문서화, F-2/F-3 수정, F-4 --strict-bit, ghost≥1+r 명문화),
성능 패스 4종은 bit-parity를 유지하면서 4.4× 개선을 달성했으며 그 과정 자체가
방법론의 추가 실증이 되었다(연산자 미공유 3.9e-07을 G-M2b가 즉검출). 부록 E의
성능 재정산("동코드 분해 이득 1.6–1.8×"로 스스로 정정)은 드문 수준의 지적
정직성이다. **다만 `main_mpi.py --verify`가 현재 HEAD에서 깨져 있다**(R2-1).

게이트 재실행(전부 현재 HEAD, 로컬 3090):

| 게이트 | 결과 |
|---|---|
| G-M1 / G-M4 | 3축 bit / 3축 bit + 6.00× — 유지 |
| eso_gather_scatter | region 커널 왕복 bit — PASS |
| G-M2b | 5레벨 bit (boundary-only upsample + 융합 feq 반영 후) — 유지 |
| G-M3 | 필드 bit, F_grid rel 6.4e-17 — 유지 |
| **G-restart** | 2+2 vs 4 연속, **전 5레벨 checkpoint bit** — PASS |

## R2-1. (必修) `--verify` 회귀: build() 튜플 언패킹

M5b에서 `build()`가 `(mlg, setup)`을 반환하도록 바뀌었는데(`main_mpi.py:79`),
`verify()`는 여전히 단일 반환을 가정한다:

- `main_mpi.py:262` — `ref = build(args.config, 0)` → `ref`가 튜플
- `main_mpi.py:264` — `ref.advance()` → AttributeError → `comm.Abort(1)`

**런북 §1의 클러스터 검증 명령 3개가 전부 `--verify`를 사용**하므로 다음
클러스터 세션에서 즉시 터진다(fail-fast라 조용히 틀리진 않음). 수정은 1줄
(`ref, _ = build(args.config, 0)`).

메타 교훈: 재시작 게이트는 `--ckpt-every`/`--restart` 경로만 구동하고
`--verify`는 어떤 게이트도 커버하지 않는다 — **게이트 사다리의 사각지대는
"게이트가 없는 진입점"**이다. mgpu 게이트에 `main_mpi --verify` 스모크(bench5
1~2 step, mpirun 2-rank) 1개를 추가하면 이 클래스가 닫힌다. 참고로 F-4의
`--strict-bit`도 이 깨진 함수 안에 있으므로 현재 실행 불가능한 상태다.

## R2-2. 성능 패스 4종의 정확성 검토 — 전부 성립

- **staged-commit transport** (`halo.py`): post=persistent 버퍼로 async 복사,
  commit=라운드당 1회 stream sync 후 Isend, complete=Recv+flush. 정확성 확인.
  한 가지 **암묵적 의존**: persistent recv 버퍼의 재사용 안전성(이전 라운드의
  scatter 커널이 버퍼를 읽는 중에 다음 Recv가 UCX device-write하는 race)이
  "다음 post의 commit이 하는 stream sync"에 의해 우연히 보장된다. 이 가드를
  docstring에 명시할 것 — **백로그 #5(통신/계산 오버랩)에서 가장 먼저 깨질
  가정**이다(오버랩은 정확히 이 sync를 없애려는 작업이므로).
- **융합 `_feq_fneq` RawKernel** (`coupling.py`): 셀별 직렬 q-루프 = shape/
  디바이스 불변 결정성 + 단일 패스. global `fine_to_coarse`와 rank-로컬 `f2c`가
  **같은 연산자 객체를 공유**함을 확인 — 3.9e-07 사건의 교훈("수식이 아니라
  연산자를 공유하라")이 §1.4의 정밀화로 보고서에 반영된 것도 적절.
- **boundary-only upsample** (`mlg_coupling.py`): production `_bnd_face_specs`
  객체를 그대로 공유하고, 분할축-수직 face는 containment assert(own≥ghost가
  보장), 횡방향 face는 M2 ±2행 여백 논증이 그대로 이전됨. G-M2a/M2b bit 유지로
  실증. 30× 과잉 쓰기 제거는 정당한 수리.
- **region gather/scatter 커널** (`esoteric_d3q27.py`): 파이썬 fancy-index
  경로와 동일한 치환(짝/홀 parity 분기, ±c wrap 산술 일치 확인) — 값 연산이
  없는 순수 복사라 bit-by-construction 주장이 구조적으로 성립. stride(F2C
  0::2) 지원, 비정형 region은 파이썬 폴백. 단일-GPU 경로에도 이득 — 부록 E의
  재기준 필요성을 낳은 바로 그 변경. 사소 1건: scatter fast path가 `values`
  shape을 region 크기와 대조하지 않음(flat 인덱싱이라 mismatch 시 조용한
  OOB 읽기) — assert 1줄 권고.
- **chunked return_sums** (`interpolation.py`): 청크 소속이 마커 자신의 스텐실
  리덕션을 바꾸지 않으므로 투명 — 논증 옳음. chunk_size가 마커 수·ε만의
  함수라 전 rank/단일 동일 분할인 것도 확인.

## R2-3. 출력/재시작 배선 검토

- **Rank0OutputBridge**: production writer 재사용 전략 옳음(체크포인트가 단일
  GPU 경로와 상호 재시작 가능 — G-restart가 실증). 이중 슬라이스 버그를 npz
  shape 체크로 잡았다는 기록도 게이트 문화의 연장. tag 네임스페이스(300/400대)
  는 halo(≤65)와 분리, flush 후 호출이라 안전.
- **재시작**: 표준 f를 복원 parity로 scatter(t0) + **rotor.advance() replay**
  (fp 누적까지 연속런과 동일 — theta=ω·t 직접 세팅보다 옳은 선택) + ramp
  민감 구간을 게이트 창으로 고른 것 모두 좋다. **production 버그(로터 상태
  미복원 — 단일 GPU 재시작에도 있던 실버그)를 분산 bit 게이트가 발견**한 것은
  방법론 가치의 세 번째 실증 사례다.
  주의 1건: replay 비용이 O(완료 substep × 마커) — D40 25-rev 중반 재시작이면
  ~40만 회 advance. 현재는 수 초 수준으로 추정되나, 길어지면 체크포인트에
  rotor(theta/time/step) f64 상태를 직접 저장하는 것도 동등하게 exact하다
  (f64 저장·복원은 bit 무손실). 실측 후 판단하면 됨.
- `--csv` append, 절대 step 루프, ETA, `--profile`의 rank별 gather 출력: 전부
  정상. `--verify`+`--restart` 조합은 (튜플 수정 후) 의미상 올바르게 동작한다
  — 레퍼런스가 0부터 절대 step까지 연속 실행되므로 "재시작+분산 vs 연속+단일"
  이라는 더 강한 대조가 됨. docstring 한 줄 가치.

## R2-4. 진행 중인 백로그에 대한 선제 권고

1. **ALM 로컬화(#3)는 F-1과 정면 충돌한다 — 순서를 지킬 것.** 로컬화의 자연스러운
   구현(rank별 마커 부분집합)은 리덕션 배열 shape을 rank-의존으로 만든다. 그
   순간 KNOWN RESIDUAL(F-1)이 잠재에서 **활성**으로 바뀐다(rank마다 CuPy 리덕션
   전략이 달라질 수 있음 — §6과 동일 기제). 권고 순서:
   (a) 마커별 고정순서 부분합 RawKernel(셀 순회 직렬 — §6/feq 수정과 동형)로
   F-1을 먼저 해소 → shape 무관 결정성 확보,
   (b) 그 다음 부분집합 로컬화(스텐실 박스 ∩ owned 슬랩 = ∅ 인 마커 스킵 —
   기여가 정확히 0이므로 bit-중립),
   (c) 스프레딩도 동일(∩ 로컬 배열(ghost 포함) 기준, 마커 인덱스 오름차순 유지).
2. **로컬화는 새 스큐를 만든다.** 현재 복제 ALM은 전 rank가 같은 시간을 쓰는
   "균형 잡힌 낭비"다. 로컬화하면 로터 근접 rank만 느려진다 — `balance_cuts`의
   업데이트 밀도 프로파일에 **ALM 행-비용 항**(마커 스텐실의 L0-행 분포 가중)을
   추가해 컷이 이를 흡수하게 할 것. quantile 기계는 이미 있으므로 밀도 함수에
   항 하나 더하는 일이다.
3. **분산 초기화(#4)**: 이 솔버의 초기화 필드(기하 마스크, BC 배열, 평형 분포)는
   전부 전역 좌표의 순수 함수라 슬랩-로컬 평가가 구조적으로 가능하다. 검증은
   "seam 게이트"를 권고 — 인접 rank가 여유 겹침으로 빌드해 겹침 영역을 bit
   대조(전역 레퍼런스가 존재할 수 없는 크기에서의 유일한 bit 검증 수단) +
   축소 쌍둥이(bench5급)에서 복제-빌드와 전역 bit 대조.
4. **v2 결합(#5)의 우선순위 재고**: 현 halo 비용 0.20 중 0.18이 스큐 대기다.
   v2의 6× 트래픽 절감은 스큐를 줄이지 못한다 — 오버랩(interior/edge 분할)이
   맞는 공격이고, 그 전에 2번(스큐의 근원 제거)이 먼저다. 순서: ALM 로컬화+컷
   재가중 → 오버랩 → (halo_post가 그때도 지배적이면) v2. v2는 이미 증명돼
   있으므로 서랍에서 기다리는 비용이 없다.

## R2-5. 문헌 대비 contribution 재평가 (1차 검토의 정정 포함)

**정정**: 1차 검토에서 "Esoteric Pull의 분산 공개 구현은 드물다(FluidX3D는
자체 방식)"라고 썼는데, 이는 부정확했다. FluidX3D(Lehmann, 2022~) 자체가
멀티-GPU esoteric pull이며, 경계 횡단 population만 전송하는 통신 방식도 v2와
같은 정신이다. 따라서 **"최초의 분산 esoteric pull" 류의 주장은 성립하지
않는다.** v2의 차별점은 (a) same-slot 무재라벨 대수의 명시적 증명, (b) bit-
parity 보장, (c) MLG refinement와의 결합이지만, 헤드라인 기여로는 부족하다.

같은 기준으로 다른 축들도 재평가하면:

- **분산 grid refinement 자체**도 신규 아님 — waLBerla(Schornbaum & Rüde)가
  블록 구조로 대규모 수행. 단 그들은 tolerance 검증 + 2-버퍼 스트리밍이다.
- **유도 컷(derived cuts)에 의한 커플링 통신 소멸**은 1D+중첩박스라는 제약을
  받아들인 대가로 얻은 구조적 결과로, 일반 블록 그래프(waLBerla)에서는 불가능한
  성질이다. "제약 수용 → 증명 가능성 획득"이라는 트레이드오프가 설계 기여.
- **ALM partial-sum allreduce**는 walberla-wind의 침투예측+buffered MPI 대비
  이 문제 설정에서 명백히 단순·우월하지만, 단독 논문감은 아니다.

**그래서 진짜 기여는 무엇인가**: 개별 기술이 아니라 **"bit-재현 가능한 분해"
라는 검증 방법론과 그것이 실제 버그를 잡는다는 증거**다. 이제 실증 사례가
3건으로 늘었다:

| 사례 | 잡은 것 | tolerance 검증이었다면 |
|---|---|---|
| §6 1-ulp | CuPy/cuBLAS shape-의존 리덕션 | PASS로 영원히 은폐 |
| 3.9e-07 (pass 2) | 연산자 미공유(kernel vs elementwise 라운딩 분기) | 1e-4 게이트 통과, 은폐 |
| G-restart | **production 재시작 로터 상태 버그(단일 GPU에도 실재)** | ramp 미세 오차로 장기간 잠복 |

여기에 "결정성의 비용" 반증까지 갖췄다: 고정순서를 커널 융합과 결합하면 성능
불이익이 없다(오히려 4.4× 개선과 동행) — **"determinism is free after
fusion"**은 그 자체로 강한 메시지다. 부록 E의 재기준 규율("비교 기준 없는
speedup은 검증 불가능한 수치")도 같은 방법론의 성능 버전이다.

### 권고 방향

**Proposal A (추천) — 방법론 논문**: *"Bit-reproducible multi-GPU domain
decomposition of a multilevel lattice Boltzmann solver with LES and actuator
lines"* (Computer Physics Communications / J. Computational Science 급).

구성 제안:
1. **정식화**: LBM 분해가 bit-동일할 수 있는 충분조건의 정리화 — per-cell
   커널 + 순수 치환 프리미티브 + ghost ≥ 1+r + 유도 컷 + 연산자 객체 공유 +
   고정순서(또는 커널 내 직렬) 리덕션. 본 검토와 보고서 §1.1이 이미 초안이다.
2. **3건 포렌식 사례** (위 표) — 방법론이 tolerance 검증 대비 무엇을 더 잡는지.
3. **비용 정량화**: 결정성 유지가 성능과 충돌하지 않음(4-pass 과정, 0.671
   s/step, 동코드 1.6–1.8×와 그 Amdahl 분해 — 부록 E 그대로).
4. **아티팩트**: 게이트 사다리 + LoopbackTransport("전선 없는 MPI") — 클러스터
   없이 재현 가능한 검증 하네스.

필요한 추가 실험은 적다: 1/2/4-rank 강스케일링 커브 1개, bit-게이트 vs
tolerance-게이트 검출력 ablation(3건 사례 재현 스크립트), 끝. 지금이 적기인
이유: M5로 코드가 "완결" 상태로 얼어 있고, ALM 로컬화 등 후속이 들어가면
스냅샷이 다시 움직인다.

**Proposal B**: HVAB 응용논문에 "분산 검증" 절로 +0.027% CT 물리 동일성과
ALM 프로토콜을 수록(방법론 논문과 상호 인용).

**Proposal C (보류 권고)**: v2+오버랩+수십-GPU 강스케일링의 HPC 벤치마크
경쟁 — 하드웨어(4×4090, PCIe)로는 경쟁력이 없고, 장기 방향(SU2 근접장 커플링)
과도 어긋난다.

**SU2 방향과의 정합**: Transport/region-커널 계층은 향후 SU2 경계장 교환의
기반 그대로다(설계 시 이 인터페이스를 안정 API로 취급할 것). 단, bit-parity
앵커는 SU2 커플링으로 이전되지 않는다(이종 솔버 간에는 "단일 실행 레퍼런스"가
없음) — 그쪽 검증은 CV-band + 겹침영역 MMS/보존량 감사로 갈 수밖에 없다는
점을 미리 인정하고 설계하는 편이, bit 문화가 만든 기대치를 잘못 이식하는 것보다
낫다. 방법론 논문(Proposal A)은 bit가 성립하는 마지막이자 가장 순수한 스냅샷을
기록으로 남기는 일이기도 하다.

## R2-6. 판정 요약

| # | 항목 | 등급 |
|---|---|---|
| R2-1 | `--verify` 튜플 회귀(런북 §1 전멸) + verify 스모크 게이트 부재 | **必修 (1줄 + 게이트 1개)** |
| R2-4.1 | ALM 로컬화 전 F-1 고정순서 커널화 선행 | 순서 제약 (WIP 직결) |
| R2-4.2 | 로컬화 후 balance_cuts에 ALM 비용 항 | WIP 설계 반영 |
| R2-2 | recv 버퍼 재사용 가드의 암묵성 문서화 | 오버랩 착수 전 |
| R2-2 | scatter fast path values-shape assert | 사소 |
| R2-3 | 장기 재시작의 replay 비용 실측(필요시 rotor f64 상태 저장) | 관찰 |

---
---

# 3차 검토 (2026-07-13, 트랙 마감 시점) — 슈퍼바이저 최종 리뷰

검토 대상: `c792ba1..a08ea91`(16 커밋 — 백로그 #3/#4/#5, NR=1 ghost-free,
3-tier, MEM force, nu-마이그레이션, 수용시험) + `docs/WORK_REPORT_2026-07_kr.md`.

## R3-0. 총평

**트랙 마감 승인. 단, 구조적 폐쇄가 필요한 잠재 위험 1건(R3-2)과 교차검증
공백 1건(R3-3)을 남긴 채의 마감이다** — 둘 다 현재 결과를 뒤집을 증거는
없으나(수용시험 40,224 substep bit), "경험적으로 안 터졌다"와 "구조적으로
못 터진다"를 구분하는 것이 이 프로젝트가 스스로 세운 기준이다. 2차 검토
권고는 전부 이행되었고(G-verify 게이트, F-1 커널화 선행, v2 이연 분석),
특히 ALM 로컬화 대신 clip-bounds 커널로 스큐 문제 자체를 회피한 것은
권고보다 나은 해법이다.

게이트 재검증(전부 현재 HEAD `a08ea91`, 로컬 3090):
eso_gather_scatter / G-M1 / G-M4 / G-M2b / G-M3 / **G-verify(plain+strict-bit)**
/ G-restart — **7종 전부 PASS**. 추가로 sphere HWBB tier 스모크
(NR=1 ghost-free + dist-init obstacle + MEM force + verify) 직접 실행 —
전 레벨 bit, CD 트레이스가 문서 기록값(−0.4465)과 일치.

## R3-1. (要폐쇄) prepost Irecv와 이전 라운드 scatter의 디바이스 순서

백로그 #5의 `MPITransport.prepost`는 Irecv를 **persistent recv 버퍼에
조기 게시**한다. 문제: 게시 시점이 `HaloBandExchangerV1.post`의 side 루프
안 — 즉 `commit()`의 stream sync **이전**이다. 이 시점에 우리 스트림에는
직전 라운드 `complete()`의 scatter 커널(같은 rbuf를 **읽는** 커널)이 아직
미실행 상태로 남아 있을 수 있다. 상대 rank가 앞서 있으면(스큐 ~0.1s로
충분) 상대의 Isend가 이미 발사된 상태에서 우리의 Irecv 호출이 즉시 매칭
→ UCX cuda_ipc가 **자체 non-blocking 스트림으로 디바이스 쓰기**를 개시
→ 읽기 중 버퍼에 쓰기(race) 가능성.

- **왜 로컬 게이트가 못 잡는가**: host-staged 경로는 `cp.asarray`(pageable
  H2D)가 동기라 race 창이 없다. cuda-aware 클러스터에서만 존재하는 위험
  클래스 — §6 1-ulp 사건과 같은 "로컬 재현 불가" 부류다.
- **왜 아직 안 터졌나(추정)**: UCX/CuPy의 실제 스트림 의미론(legacy default
  stream 직렬화 등)이 우연히 순서를 보장하고 있을 수 있다. 수용시험
  40,224 substep bit는 강한 경험적 증거지만, 보장의 근거가 **문서화되지
  않은 라이브러리 내부 동작**이라는 점이 문제다. 터지면 ghost 오염 →
  production 중간부터 **조용히** 틀린다(verify는 런 끝에만 돈다).
- **폐쇄 방법(권고, 2줄)**: `prepost` 호출을 `commit()` **뒤로** 이동 —
  commit의 stream sync가 "이전 scatter 완료 후 Irecv 게시"를 구조적으로
  보장하고, 오버랩 이득(post~complete 구간의 전송 은닉)은 그대로 유지된다.
  대안: 라운드 parity별 rbuf 이중버퍼, 또는 scatter 후 CUDA event를
  prepost 전에 wait. 어느 쪽이든 재게이트는 mpirun 2-rank verify + 클러스터
  1회면 충분하다.

## R3-2. (要폐쇄) eso_mem_force의 교차 경로 검증 공백

MEM force 커널의 슬롯 유도(완료-parity에서 toward-solid 예치 슬롯)는 내가
STORE 규칙으로부터 독립 검산했고 **정확하다**. owned-배타 clip + Allreduce의
rank-불변성도 1/2-rank 일치로 실증됐다. 그러나 이 커널은 이번 트랙에서
**유일하게 물리량을 새로 산출하면서 등가 게이트가 없는 코드**다: 검증된
것은 "분해 불변"이지 "힘 추출이 옳다"가 아니다(둘은 독립 — 잘못된 규약도
rank-불변일 수 있다).

- **폐쇄 방법(권고)**: 표준 경로 쌍둥이 게이트 — 동일 sphere config를
  standard 경로(main.py, `mem_force_d3q27`)로 수 스텝 돌려 힘 트레이스를
  esoteric 커널 출력과 대조(필드는 이미 eso 등가 게이트로 bit 동급이므로,
  이 대조는 순수하게 두 힘-추출 구현의 등가 시험이 된다). atomicAdd
  진단-tier이므로 기준은 f64 lastbit면 충분.
- 참고: bench_sphere(D=2 L0셀)는 계단형 blob이라 문헌 CD 앵커는 무의미 —
  교차-구현 게이트가 유일하게 올바른 검증이다. 진행 라인이 "진단 tier"로
  명시된 점은 적절하나, CSV로 저장되는 순간 사용자는 물리로 읽는다.

## R3-3. 신규 코드 정밀 검토 — 통과 항목

- **fresh-skip + early-post의 스케줄 무결성**: `_touch`의 "in-flight 중
  변이 assert"가 스케줄 오류를 fail-fast로 바꾸는 설계가 옳다. 재귀 전
  구간을 수동 추적했다 — parity 안전성(fresh는 t 불변을 함의: t 변경은
  advance뿐이고 advance는 touch), 이중 c2f의 동일-sync-상태 읽기(production
  스케줄과 일치), f2c 후 coarse touch의 posted-assert 정합까지 전부 성립.
- **ALM 샘플링 RawKernel**: 마커당 1블록 + 스레드-stride 직렬 f64 + 고정
  트리 리덕션 = shape/디바이스/런 불변 결정성 — F-1의 올바른 해소 방식이고,
  clip-bounds 소유권(뷰/시프트 제거)은 로컬화가 만들었을 스큐 문제를
  구조적으로 회피한다(전 rank 동일 비용 유지). `llrint`=half-even의 구경로
  일치 주석, env kill-switch, 폴백 fail-fast 모두 적절.
- **LocalLevel 청크 구성**: pair-슬롯 쓰기가 소스 셀 기준 분할이라 청크
  합집합 = 전체 scatter(bit-exact) — 논증 옳음. 재사용 버퍼(`cp.take(out=)`)
  + 하드리밋 에뮬레이션 교훈의 명문화도 좋다.
- **dist-init**: 메타데이터 host 빌더 재사용(§1.4 원칙 유지), 균일 IC
  27-벡터 broadcast의 bit 논증(동일 스칼라 입력의 elementwise 결정성) 성립,
  obstacle solid-mask 경로 bench5급 bit 게이트 ✓. 신규 커널 전부 64-bit
  인덱싱 확인(450M 안전).
- **NR=1 ghost=0**: "로컬 배열=도메인, 커널 %N wrap=물리 주기성" — 정확한
  구조적 단순화. 단일 GPU가 mpirun 없이 같은 진입점을 쓰게 된 것은 유지보수
  관점의 실질 이득(경로 분기 소멸).
- **2D fixed-order 폴백 수정**: `range(3)` 하드코딩 → dim-generic. 이 회귀의
  교훈은 R2의 verify와 동일하다 — **2D는 게이트 사다리의 사각지대였다**
  (3D 게이트만 존재). Q=9 등가 체크가 추가됐으나, 2D end-to-end 게이트
  1개(cyl Re100 Cd 밴드)를 회귀 스위트에 넣을 것을 권고.
- **flow_stats SOLID 마스킹 버그**: 스모크가 잡았고 수정 정확(미초기화
  버퍼 노출 — cp.empty 관행의 대가). 통과.

## R3-4. 사소/일관성 지적

1. **G-verify의 strict-bit 레그가 ALM config를 쓴다**: 자체 분석대로 ALM
   케이스의 bit는 f32-cast 방화벽에 기댄 **확률적** 관측이다(플립 기대
   ~1e-9/값). 2-step이라 실제 위험은 무시 가능하나, 사다리의 등급 분류
   (ALM=fp-lastbit tier)와 게이트 단언(bit)이 형식적으로 모순 — strict-bit
   레그는 pure_lbm config로 바꾸는 것이 자기일관적이다.
2. **작업 보고서 §3의 압축 서술**: "bit-identical (…; 25-rev 풀런 CT
   +0.027%)"는 두 개의 다른 런/주장(2-rev 수용시험=bit, 25-rev=통계 동일)을
   한 괄호에 섞었다 — 외부 검토자는 "bit인데 왜 CT가 다른가"를 물을 것이다.
   실은 방화벽 모델이 이 둘을 **정량적으로 정합**하게 설명한다: 25-rev는
   반올림 사건 ~10⁹회 규모 → 기대 플립 수 회 → 카오스 증폭 → CT는 σ/20
   수준에서 갈라짐; 2-rev는 기대 플립 ~0.03회 → bit. 이 문장을 그렇게
   풀어 쓰면 약점이 아니라 모델 검증 사례가 된다.
3. **main_mpi의 body-tier config 재임포트**: `importlib`로 config를 다시
   실행해 reference를 읽는다 — `setup` 삭제 전에 뽑아두면 없어질 우회.
   동작은 하나 이중 파싱(부수효과 있는 config면 위험)이라 리팩터 후보.
4. **Gaussian 하드코딩 4번째 지점**: 신규 샘플링 RawKernel의 `exp(-d2/e2)` —
   β kernel 핸드오프의 추상화 목록에 이미 포함돼 있음을 확인했다. β 작업
   시작 시 이 커널이 첫 수술 대상이라는 점만 재확인해 둔다.
5. **하우스키핑**: canonical-axis 수정(rotor.py/setup.py +50줄)이 이틀째
   미커밋, 검토 보고서(본 문서) untracked, alm_multigpu/HANDOFF.md 삭제
   미반영. β 세션이 rotor.py 위에서 시작된다 — **트랙 마감 커밋과 섞이기
   전에 정리할 것.**

## R3-5. 작업 보고서 §7 질문에 대한 답변

1. **BEM 복제(0.08s)**: 유지하라. 마커분할+Allgather의 상한 이득은 4-rank
   에서 ~0.06s인데, 대가는 복제-상태 불변식(마커 VTP 무통신, rank0 진단
   =production 상태) 전부의 포기다. 재평가 트리거는 R≥8-16 또는 β 보정이
   반복적(iterative)이 되어 solve 비용이 커질 때 — 후자가 더 개연적이다.
2. **lockstep 스큐(0.1s)**: 기법 추가 전에 **조성 분해**부터 — halo_complete
   대기를 "데이터 미도착 대기"(Iprobe로 판별)와 "정적 불균형"(worst-share
   0.266의 +6.4%≈0.03s)으로 쪼개라. 정적이면 컷 가중을 업데이트-수 모델에서
   **실측 per-rank ms**로 교체(quantile 기계는 그대로); 동적 지터면 4-rank
   에서 남는 표준 기법은 사실상 없다(interior/edge 분할은 halo_post 0.009
   에는 과잉). 스큐 0.1s의 완전 제거보다 β 트랙의 시간이 더 가치 있다.
3. **occupancy 튜닝**: 착수 전에 루프라인 숫자 하나만 — 커널 구간의 실측
   GLUPS vs (대역폭/바이트-퍼-업데이트≈216B) 이론치. LBM 커널은 거의
   확실히 대역폭 포화 근처라 블록/레지스터 튜닝은 한 자릿수 %다. 남은
   실질 레버는 오히려 **CUDA Graph 재도입**(러너가 `_graph_enabled=False`로
   끄고 있음 — 서브스텝 런치 오케스트레이션 절감; SU2 방향의 per-step 속도
   투자항목과도 일치)이다. 단 fresh-skip의 조건 분기가 graph capture와
   충돌하므로 레벨-버스트 단위 부분 capture가 현실적.
4. **2D/3D 분해 전환 기준**: 두 개의 독립 천장 — ①기하: 최내곽 span에
   간격≥ghost로 R−1개 컷이 들어가야 함(farfield40 L4 span이 직접 상한)
   ②경제: halo+스큐가 스텝의 ~20%를 넘는 지점. 음향급 스케일-업 전에
   이 두 수를 케이스 정의에서 먼저 계산하라 — 그리고 SU2 커플링이 LBM
   외부장 역할로 확정되면 8+ GPU 시나리오 자체가 소멸할 수 있으니, 2D
   분해는 **케이스가 강제할 때까지 설계도 하지 마라**(정도 원칙 그대로).
5. **checkpoint 확장성**: 임계는 이미 왔다 — 450M이면 조립 f ~48GB/ckpt
   (host RAM+NFS 쓰기 시간). rank별 슬랩 npz + 매니페스트(bounds 기록)로
   가라: 재시작은 파티션이 결정적이라 same-R이면 각 rank가 자기 파일만
   읽으면 되고(조립 자체가 불필요), 글로벌 조립은 VTK/분석 때만 lazy로.
   different-R 재시작도 매니페스트의 global 인덱스로 재슬라이스 가능.
   450M production 런 **전에** 필요하다.
6. **방법론 이식의 최소 요건**: ①모든 rank 간 데이터 이동을 "커널과 wrap
   의미론이 일치하는 순수 치환 프리미티브"로 격리 ②per-cell 커널 순수성
   (라이브러리 리덕션 전면 금지 또는 고정순서화) ③분해·레퍼런스 경로의
   **연산자 객체 공유**(수식 복제 금지) ④결정적 빌드 ⑤재결합이 불가피한
   지점의 등급 사전 선언 ⑥게이트는 재구현이 아니라 production 경로 자체와
   대조 ⑦진입점마다 게이트(R2의 메타 교훈). 경계 조건: 동일 디바이스
   세대 내에서만 bit가 성립(FMA 계약/아키 차이) — 이 한계도 요건의 일부로
   명시해야 이식이 정직해진다. 이 목록이 곧 방법론 논문 §2다.

## R3-6. 최종 판정

| # | 항목 | 등급 |
|---|---|---|
| R3-1 | prepost를 commit 뒤로(또는 이중버퍼/event) + 클러스터 재확인 | **要폐쇄 (2줄, β 착수 전)** |
| R3-2 | eso_mem_force ↔ mem_force_d3q27 쌍둥이 게이트 | **要폐쇄 (게이트 1개)** |
| R3-4.5 | canonical-axis 커밋 분리 + 문서 track | 하우스키핑 (즉시) |
| R3-4.1 | strict-bit 레그를 pure_lbm으로 | 사소 |
| R3-4.2 | 보고서 §3 bit/통계 서술 분리(방화벽 정합 명시) | 보고서 품질 |
| R3-4.3 | body-tier config 재임포트 리팩터 | 사소 |
| R3-3 | 2D end-to-end 회귀 게이트 1개 | 사다리 보강 |

위 R3-1/R3-2를 닫으면 이 트랙은 "경험적으로 검증됨"이 아니라 "구조적으로
닫힘" 상태가 된다 — 방법론 논문(2차 검토 Proposal A)을 쓴다면 그 상태가
논문의 주장 그 자체이므로, 두 항목은 논문 전 필수 선행이기도 하다.
방법론 논문의 소재는 이번 스프린트로 더 강해졌다: 결함 9건 목록, 동코드
2.45×/분할효율 ~100%, 40k-substep bit 수용시험, 그리고 **방화벽 확률
모델의 정량적 자기정합**(2-rev bit ↔ 25-rev 통계 동일이 같은 모델의 두
예측이라는 것)까지 — 마지막 항목은 그 자체로 논문의 하이라이트감이다.
