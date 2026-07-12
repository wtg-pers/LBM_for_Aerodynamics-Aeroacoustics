# 17 — Multi-GPU (Option A) 설계: esoteric layout 위 도메인 분할

> Status: **DESIGN (구현 전)** · 2026-07-12
> 로드맵: [[feedback_long_term_structural]] 정도 순서 = esoteric(완료, patch15) → **멀티GPU(본 노트)**.
> 명분: 음향 대규모(207M+ 격자) + 케이스 처리량(4×4090) + SU2 커플링 투자항목 정합.
> 참조: `docs/MULTI_GPU_DESIGN.md`(2026-04-12 blueprint), `patch_notes/alm_multigpu/task1_cluster_verify.py`,
> walberla-wind 리뷰(`docs/papers_kr/2023_walberla_wind_kr.md` §4 — actuator 통신/로드밸런싱 패턴).

## 0. ★자산 실사 정정 (handoff §3의 오류)
- **`src/parallel/`(Partition/HaloExchange/DomainDecomp)은 git history에 없음** — esoteric_cumulant와 동일한
  "blueprint만 잔존, 코드 미커밋" 유실 패턴. **재구현이다** (부활 아님).
- 생존 자산: ①`MULTI_GPU_DESIGN.md` — x-분할·halo·NCCL 프로토콜 + **§2.3 esoteric 호환 두 방안**(물리 gather/scatter
  교환 vs raw esoteric 버퍼 lockstep 교환) + §3 MLG 레벨별 분할·halo2 ②Task1 클러스터 검증(2026-07-08):
  anode1 4×4090, **CUDA-aware OpenMPI 5.0.5 + UCX 1.18 device-direct PASS**, ★운영필수
  `mpi4py.rc.thread_level='serialized'`(import 전; MPI_THREAD_MULTIPLE이면 UCX 탈락→ob1 저속), NVLink 無
  PCIe → **halo 데이터량 최소화가 성능 지배**.

## 1. 결정 사항 (설계 확정치)
| 항목 | 결정 | 근거 |
|---|---|---|
| 분할 축 | ★**axis-generic 1D 분할 — config 파라미터**(`parallel.axis ∈ {auto,x,y,z}`). 모든 Partition/halo/슬라이스 코드는 축 인덱스를 인자로 받아 동작(하드코딩 금지 — 유입류/회전축이 케이스마다 ±x/±y/±z로 다름, 사용자 지시 2026-07-12). `auto` = 레벨별 해당 축 extent의 최소값을 최대화하는 축(=지배 레벨 일 균등분할 보장; 동률 시 tie-break=**halo 연속성 내림차순 x→y→z**((Q,x,y,z) C-순서라 x-cut 밴드=연속 평면, y-cut=Nz-연속 조각, z-cut=2원소 파편=패킹 최악; 사용자 질의로 원칙화 2026-07-12). farfield40은 x 탈락(57)·y/z 동률→y) | ①x-분할은 farfield40 nested 토폴로지에 불균형 치명(L4=x-두께 57셀 슬랩) — 단 이는 이 케이스의 사정이지 코드 제약이 아님 ②ALM partial-sum 프로토콜은 축·회전축 불가지론적이라 어떤 축에도 동일 적용 ③region gather/scatter 프리미티브도 이미 축별 slice 3-튜플 인자라 generic ④1D가 halo 면 2개=최소 |
| 통신 | **mpi4py CUDA-aware(UCX device-direct)** 우선, NCCL은 옵션 | Task1에서 실검증된 경로. runner=`mpirun -np N python main.py ...` |
| esoteric halo v1 | **ghost=2 + 물리-밴드 교환**(스텝 전: 자기 경계 2셀 밴드 `esoteric_gather_std_region`→MPI→상대 ghost에 `scatter_std_region`; 커널은 ghost 포함 전체 계산=1층 중복계산 후 다음 교환이 덮어씀) | bit-검증된 프리미티브 재사용 → 정확성이 구성적으로 보장. 슬롯-parity 추론 불필요 |
| esoteric halo v2 | **raw 슬롯 교환**(halo=1, 스텝 후 경계면의 '건너간 슬롯 부분집합'만 merge; parity별 13쌍 매핑) | blueprint §2.3 권장. v1 대비 데이터 ~4×↓(2면·27슬롯→1면·~13슬롯) — PCIe에서 중요. **v1을 참조로 게이트** |
| MLG 분할 | **레벨별 독립 분할(선택 축)**(각 레벨 자신의 해당-축 extent를 N등분) + 레벨별 halo(코스=2: C2F cubic stencil, 파인=2: 안전) | blueprint §3.1/3.2. 균형: 각 레벨 updates가 균등 분할(L4=총 58%도 균등) |
| MLG 커플링 | 커플링 연산은 **랭크-로컬 슬라이스**에서 수행: C2F 코스 서브볼륨·F2C strided 읽기가 halo 2셀로 충족되도록 슬라이스 경계 정렬(코스 2셀=파인 4셀 정렬 필수) | 현 coupling.py·region-scoped 브릿지(patch15 e2) 무변경 목표 |
| ALM | ★**partial-sum allreduce 프로토콜**(신규 설계, walberla-wind 방식보다 우리에 적합): ①**샘플링**: 각 랭크가 자기 소유 셀에서 전 마커의 Gaussian 가중 부분합(분자·분모) 계산 → `allreduce` (256마커×8쌍 double = ~16KB!) ②**BEM+kleine solve**: 전 랭크 복제 실행(7ms, 결정적→동일 결과) ③**스프레딩**: 각 랭크가 전 마커를 자기 셀에만 뿌림(통신 0) ④radial scales: 복제 계산(1.7ms) | 마커 ±3ε 샘플링 반경(≤20셀)이 halo로 감당 불가 ↔ 부분합은 **정확**(합산만 재배열)+메시지 초소형. 디스크-평행 축 분할에서 로터가 cut을 반드시 가로지르므로 마커 소유권 추적 자체가 소멸(축 불가지론) |
| 검증 표준 | **N-rank vs 1-rank 등가**: 물리-밴드(v1)는 합산순서까지 동일 연산이므로 결정적 GPU(3090)에서 **bit 목표**, 부분합 allreduce·v2는 fp32 last-bit/CV-band | 기존 3단 게이트 체계 계승 |

## 2. 데이터량 견적 (D40, 4-rank, PCIe ~12GB/s 가정)
- v1 물리-밴드/face: L4 = 2셀×57×681×27×4B ≈ **8.4MB**; 전 레벨 합 ≈ 21MB/face-교환.
  L4는 coarse당 16회 교환 → L4만 16×2면×8.4 ≈ 269MB → ~22ms/coarse (+latency). 전 레벨 합산 ~35ms/coarse
  = 현 3.3s 대비 ~1% — **v1도 성능상 충분**, v2는 음향급(207M+, 스텝수↑)에서 가치.
- ALM allreduce: 16KB×32회/coarse = 무시가능.

## 3. 단계 (stepwise; 단계 경계마다 게이트+패치노트)
- **M1 — 인프라+단일격자**: `src/parallel/`(Partition[axis-generic], HaloBandExchanger[v1], MPI runner 배선),
  esoteric 단일레벨 주기박스 2-rank. **게이트 G-M1**: 2-rank vs 1-rank rho/u bit(3090)·질량보존 — **x/y/z 3축 전부**(axis-generic 증명).
  (mpi4py 로컬 미설치 시: 통신층에 mock-loopback 모드 넣어 로컬 2-"rank" 검증 + 클러스터 실검증 이원화.)
- **M2 — MLG**: 레벨별 분할(선택 축; OverlapRegion/coupling 슬라이스의 랭크-로컬화), 경계정렬 규칙. 
  **G-M2**: bench5 pure-LBM 2-rank vs 1-rank(레벨별 rho/u).
- **M3 — ALM**: partial-sum 샘플링 allreduce + 복제 BEM + 로컬 스프레딩. 
  **G-M3**: bench5_baseline 2-rank vs 1-rank 추력 trace(결정적 3090 bit 목표/카오스면 median 게이트).
- **M4 — v2 raw-슬롯 halo + 통신/계산 오버랩**(CUDA stream). **G-M4**: v2 vs v1 등가 + 성능.
- **M5 — 클러스터**: anode1 4×4090 — D40 4-way(≈4.3GB/GPU→여유로 음향급 예열), weak/strong scaling,
  `serialized` 체크 내장. 산출=사용자 실행 가이드.

## 4. 리스크 / 미결
- (a) mpi4py 로컬(WSL2) 가용성 → M1의 mock-loopback으로 개발-검증 분리(클러스터 최종검증은 사용자).
- (b) MLG 경계정렬: 레벨별 해당-축 extent가 2^k 배수 관계 — 분할점을 코스 격자 정렬로 강제(파인은 자동 정렬).
  farfield40 y-extent: L0 240/L1 193(!)... **홀수 extent 레벨 존재** → 균등분할 대신 '코스-정렬 최근접' 분할 규칙 필요.
- (c) checkpoint/VTK: v1은 physical_f per-rank 슬라이스 → rank0 조립 저장(체크포인트 포맷 유지) — M2에서.
- (d) CUDA graph·MLG_PROFILE 등 부가 경로는 멀티GPU에서 명시 거부(단계적).
- (e) 분할축 면의 BC/sponge: 해당 축 min/max face BC는 끝 랭크만 적용 — bc_manager 슬라이스화(axis-generic).

## 5. 이 설계가 의도적으로 미루는 것
2D 분할(z 추가), 동적 로드밸런싱(SFC), GPUdirect-RDMA 노드간, esoteric 슬롯-halo의 커널內 융합.
음향급에서 병목이 실측되면 그때(정도 원칙: 실측 후 최적화).

## 6. 구현 로그
### ✅ M1 — 인프라 + 단일격자 (2026-07-12, 로컬 3090)
- **axis-generic 재설계 반영**(사용자 지시): 분할축은 config 파라미터(`auto|x|y|z`), 모든 코드가 축 인덱스
  인자로 동작. `choose_axis`(레벨별 최소 extent 최대화) = farfield40 형상에서 y 자동 선택 확인.
- 신규 `src/parallel/`: `partition.py`(`Partition1D` — owned/edge_band/ghost_band 슬라이스 3-튜플 헬퍼,
  periodic 이웃, 비균등 분할), `halo.py`(`HaloBandExchangerV1` 2-phase post/complete + `LoopbackTransport`
  in-process 검증용 + `MPITransport` 골격[M5에서 recv-버퍼 방식으로 완성]).
- **게이트 G-M1 PASS**: `gates/mgpu_m1_gate.py` — esoteric cumulant 주기 TGV 24³, 16스텝,
  2-rank loopback vs 1-rank: **x/y/z 3축 전부 f·rho·u bit-identical(max|df|=0.0)**. 추가로 5-rank
  비균등(5,5,5,5,4)도 3축 bit ✓. v1 물리-밴드 halo의 "구성적 정확성" 실증(ghost-2 중복계산 + 프리미티브
  재사용 전략 그대로 적중).
- 다음 → M2: MLG 레벨별 분할(코스-정렬 cut 규칙, coupling 랭크-로컬화, 홀수 extent 처리).

### ✅ M2a — MLG rank-local 커플링 핵심 수학 (2026-07-12, 로컬 3090)
- `Partition1D.from_range`(명시 범위 — fine 분할은 coarse cut에서 **유도**: fine=2·(coarse−box_lo), 마지막
  랭크가 co-located +1 노드 흡수) + `src/parallel/mlg_coupling.py`(`RankLocalCouplingV1`):
  **global GridCoupling의 수학 프리미티브(fused rescale·cubic upsample·filter·macro/f_eq)를 그대로 재사용**,
  슬라이스 오케스트레이션만 로컬화. margin 논증: C2F=코스 블록 ±2행 확장→owned 파인 스트립은 centered
  stencil(=global bit); F2C=strided 읽기 ±1 코스행→filter centered(=global bit).
- 스케줄: 커플링 읽기 직전 해당 레벨 re-sync(advance가 ghost layer2를 오염시키므로). 파인 레벨도 periodic
  ring 교환 = 1-rank 커널의 global wrap 의미론을 정확 재현(스트립 밖 오염 행이 동일하게 생성·덮임).
- **게이트 G-M2a PASS**: 합성 2-레벨(코스 36×32×32 주기 + 파인 47³ 중첩, 실제 GridCoupling+fused rescale,
  production 스케줄 5 coarse step) — 2-rank vs 1-rank **x/y/z 3축 coarse·fine 전부 bit-identical**;
  3-rank 비대칭 분할도 3축 bit ✓.
- 남은 M2b: bench5 5-레벨 연쇄(레벨쌍 4개 체인) + 솔버 통합(setup/MultiLevelGrid의 분산 구동, BC/sponge
  끝랭크 슬라이스, checkpoint rank0 조립). fprev는 로컬 블록 영역으로 자연 확장(코드 동일).

### ✅ M2b — 실제 bench5 5-레벨 체인 (2026-07-12, 로컬 3090)
- **G-M2b PASS**: bench5_pure_lbm을 1-rank로 빌드해 **production `MultiLevelGrid.advance()` 자체를 레퍼런스**로,
  동일 초기상태에서 분산판(레벨별 wrap-슬라이스된 f/node_type/bc, production advance 복제=dyn_smag pre-pass→
  WALE-branch cumulant, 레벨쌍 4개 RankLocalCoupling 체인, production 재귀 스케줄 미러) 2-rank 구동 —
  **y/z 양축, 5레벨 전부 bit-identical**.
- ★발견 1 — **SGS는 ghost=3 필요**: dyn_smag u-구배 stencil이 ghost2의 오염된 u를 읽으면 ghost1 nu_t→owned로
  전파. ghost=3이면 u 유효층이 한 겹 깊어져 owned 보호(게이트 실증). SGS-off 경로는 ghost=2로 충분(M1/M2a).
- ★발견 2 — **auto-axis 지표 격상(체인 시뮬레이션)**: bench5 x-체인은 pair2 box(x[18,62])가 rank0의 유도된
  L2 범위에 포함돼 **L3/L4가 rank0 독점**(rank1 own=0) — 원시 extent 지표(x=56>y=48)로는 x를 뽑는 함정.
  `choose_axis(shapes, pair_boxes, n_ranks)`가 축별 cut 체인을 시뮬레이션해 worst-rank 업데이트 점유율을
  최소화 → bench5에서 x 기각·y 선택(게이트에 회귀 포함). box 없으면 legacy min-extent 폴백.
- 잔여(M2c→M5로 이관): mpirun runner/per-rank setup 배선(모든 수학 경로는 이미 게이트됨 → 배선 리스크 낮음),
  own_count=0 랭크 허용(현재 fail-fast assert), checkpoint rank0 조립.

### ✅ M3 — 분산 ALM: partial-sum allreduce (2026-07-12, 로컬 3090)
- **production 훅 3종**(actuator_line.py, 미설정 시 완전 no-op — 1-rank ALM smoke 자릿수 보존 확인):
  ①`_grid_offset`(그리드 연산=positions−offset; 로터 물리/wake 기하는 global 유지=이동불변)
  ②`_velocity_sampler`(step() 내부 주입점 — MPI 구현이 이 자리에서 Allreduce)
  ③radial-trunc **전역 scale 컨텍스트**(scale_domain_shape/positions/hub — 재정규화 스케일은 전 랭크 동일 필수,
  커널의 per-node radial cut은 로컬 hub). +`interpolate_velocity_batch_gpu(return_sums=True)`(분자/분모 원시합).
  가드: 분산+비-gaussian 샘플링, 분산+kleine free-wake → fail-fast(NotImplementedError).
- `src/parallel/alm_dist.py`: owned-view 부분합(스텐실 박스가 owned 슬랩 경계에서 자연 클립=소유권 정확,
  ghost 이중계상 불가) + `ThreadAllreduce`(barrier 기반 — MPI.Allreduce와 동형 제어흐름) + 샘플러 팩토리.
- **게이트 G-M3 PASS**(bench5, 2-rank, y축, 랭크별 실제 ActuatorLineModel을 스레드 step으로 구동):
  | pure-ALM | **field max|df| = 0.0 (bit)** | F_grid rel 6.5e-17 |
  | archB-straight(radial trunc+kleine) | **field max|df| = 0.0 (bit)** | F_grid rel 6.6e-17 |
  bit는 이 구성에서의 실측(부분합 재결합이 우연히 재현)이며 일반 보장은 아님 — 게이트 기준은 tolerance
  (field<1e-4, F rel<1e-5) 유지. BEM/kleine solve는 동일 u_markers로 전 랭크 복제=결정적 동일.
- M3 범위 제외(문서화): kleine free-wake(웨이크 점 속도 샘플링 미분산 — production은 straight 사용이라 무영향),
  비-gaussian 연구용 샘플러들.

### ✅ M4 — halo v2: raw parity-slot 부분집합 교환 (2026-07-12, 로컬 3090)
- `src/parallel/halo_v2.py` `SlotHaloExchangerV2`: **ghost=1**, 고스트 평면=SOLID 마킹(커널이 아예 스킵 —
  v1의 중복계산 제거, 순수 transit mailbox). 스텝 **후** 면당 횡단 population 9개/셀만 두 그룹으로 교환:
  A(우리 edge STORE가 고스트에 예치한 유출 슬롯 → 이웃 owned edge), B(이웃의 다음 LOAD가 원격으로 읽는
  우리 edge 상주 슬롯 → 이웃 고스트). **두 그룹 모두 same-slot copy** — esoteric parity swap이 스트리밍을
  이미 인코딩하므로 재라벨링 불필요. 수신 슬롯과 자기 STORE 슬롯은 pair 내 상보 슬롯이라 충돌 불가(레이아웃 성질).
  테이블: A_hi(t)={c_i[a]=+1: i+1 odd/i even}, B_hi(t)={c_i[a]=−1: i odd/i+1 even}, low면은 parity 라벨 반전 미러.
- **G-M4 PASS: 3축 전부 f/rho/u bit-identical, 트래픽 v1 대비 정확히 6.00×↓**(486→81 KiB/step, 2-rank TGV).
- 범위/이연(§5 실측-후-최적화 원칙): production 기본은 v1 유지(D40 4-rank v1 오버헤드 ~1%).
  v2의 SGS 결합(u edge-plane 교환)·MLG 결합(커플링 시점만 v1 물리밴드 하이브리드)·comm/compute 오버랩
  (interior/edge 분할 런치 필요)은 **음향급 강스케일링 실측이 정당화할 때** M5+ 후속으로. v2는 단일레벨
  스트리밍 계층에서 프로토콜 수학이 bit로 증명된 상태.

### ✅ M5 — SPMD 배선 + 로컬 실 mpirun 검증 (2026-07-12)
- **production 승격**: `src/parallel/local_level.py`(게이트 검증 LocalLevel/extract), `runner.py`
  (DistributedMLGRunner — G-M2b/M3 재귀의 SPMD화, ALM 훅 배선 포함), `alm_dist.MPIAllreduce`,
  `MPITransport` 완성(CUDA-aware device-direct / host-staging 폴백, shape-결정적 Recv, flush),
  exchanger tag_base(레벨별 네임스페이스), `main_mpi.py`(serialized guard, node-local rank→GPU,
  rank0 thrust CSV, --verify, comm.Abort fail-fast).
- **★균등분할 함정 발견→해결**: farfield40 NR=4는 균등 L0 분할 시 전 축 infeasible(중첩 박스
  중앙집중 → 바깥 rank L4 own=0). 해법=`balance_cuts`: **모든 L0 컷을 최내곽 박스 span 내부에**
  배치(nesting ⇒ 전 rank가 전 레벨 소유 보장) + 업데이트밀도 quantile 배치(밸런스).
  farfield40 NR=4: bounds=[0,108,121,133,240], worst share 0.266(이상 0.25). `choose_axis_balanced`.
- **로컬 실 mpirun(MPICH host-staged, 1 GPU 2 프로세스) 검증**: bench5 pure-ALM/archB 2-rank
  `--verify` **전 레벨 bit-identical**. 4-rank 로컬은 동시빌드 4개로 비현실적(스로틀) — 파티션
  타당성은 드라이 체인체크로 확정.
- 남음(사용자 클러스터, runbook 18): OpenMPI+UCX cuda-aware 실검증, 4-rank verify, farfield40
  1-rev 스모크+스케일링. 이후 검토용 철학 보고서 작성 예정.
