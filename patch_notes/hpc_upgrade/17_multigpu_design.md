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
