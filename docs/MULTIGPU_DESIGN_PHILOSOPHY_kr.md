# Multi-GPU LBM 솔버: 설계 철학과 검증 체계 (검토용 보고서)

작성: 2026-07-12 · 대상 코드: `src/parallel/` + `main_mpi.py` (커밋 `2c40712` 기준)
관련 설계/구현 기록: `patch_notes/hpc_upgrade/17_multigpu_design.md` (M1–M5 전 단계 로그),
게이트 스크립트: `patch_notes/hpc_upgrade/gates/mgpu_m*.py`

본 문서는 D3Q27 cumulant LBM(Esoteric Pull 단일버퍼 스트리밍) + 5-레벨 MLG(multi-level
grid) + ALM(actuator line) 솔버를 MPI 다중 GPU로 분산한 작업의 **설계 철학, 핵심 결정과
기각된 대안, 검증 체계**를 외부 검토자가 독립적으로 평가할 수 있도록 정리한 것이다.

---

## 0. 요약

- **분산 대상**: 91.6M 셀 HVAB 로터 호버 케이스(5-레벨 MLG, dyn. Smagorinsky, eq/sponge
  BC, 4-블레이드 ALM + BEM)를 1D 슬랩 분해로 N-GPU에 분산. 축은 자동 선택(축-일반적).
- **정확도 기준**: 분해 실행의 소유(owned) 셀은 단일-GPU production 실행과
  **비트 단위 동일(bit-identical)** — ALM 부분합 재결합 한 곳만 원리적으로 last-bit
  허용(실측은 bench5에서 bit).
- **검증 상태**: 로컬(loopback + 실제 MPICH mpirun) 및 클러스터(anode1 4×4090, OpenMPI
  5.0.5 + UCX CUDA-aware)에서 2-rank/4-rank 모두 전 레벨 bit 확인. 이 과정에서
  발견·수정된 라이브러리 비결정성 1건은 §6의 사례 연구 참조.
- **성능 설계**: v1 물리밴드 halo(단순·증명 용이)가 기본값. 6× 트래픽 절감의 v2
  슬롯교환은 수학적으로 증명(3축 bit)됐지만 **실측이 정당화할 때까지 미결합**.
- **최종 실측**(D40 4×4090, 25-rev 풀런): **0.671 s/step** — 밸런스 이상치(worst share
  0.266 → 3.76×) 대비 병렬효율 ~100%. 잔여 시간의 구성은 병렬화 손실이 아니라 복제 ALM
  0.275(Amdahl)·lockstep 스큐 0.18·계산 0.20. 물리 동일성: CT +0.027%(§2 검증 완결).
  ※ 이 수치는 4개 최적화 pass의 산물 — 상세와 각 pass의 교훈은 부록 C 하단 및 patch 17.

---

## 1. 설계 철학 (원칙과 그 이유)

### 1.1 정확도의 통화(currency)는 bit-parity다

분산 병렬화 검증의 일반 관행은 "물리량이 허용오차 안에서 일치"이다. 우리는 이를
기각하고 3단계 사다리를 세웠다:

| 등급 | 기준 | 허용 대상 |
|---|---|---|
| **bit** | 부동소수점 비트열 완전 동일 | 순수 재배열/치환 연산, 셀별 독립 커널 |
| **fp last-bit** | ~1e-7 (f32 1 ulp) | 동일 수학의 결합순서 변화(예: 부분합 재결합) |
| **CV-band** | median<1e-3 ∧ max<3e-2 | 자유후류 카오스(발산 지수적 증폭) 하의 궤적 비교 |

원칙: **각 연산이 어느 등급에 속하는지 사전에 논증하고, 예측과 다른 결과는 등급 안이라도
반드시 원인을 규명한다.** 이 원칙의 실효성은 §6(1-ulp 사건)에서 실증됐다 — tolerance
게이트만 있었다면 "PASS"로 지나갔을 라이브러리 비결정성을 bit 예측 위반이 잡아냈다.

bit-parity가 가능한 이유는 LBM의 구조에 있다: collision은 셀별 독립(per-cell)이고
streaming은 순수 치환(permutation)이다. 따라서 **분해가 도입하는 모든 연산을 "정확한
치환"과 "셀별 동일 산술"로만 구성하면** 소유 셀의 궤적은 정의상 단일 실행과 동일하다.
설계의 대부분은 이 성질을 깨지 않기 위한 여백(margin) 논증이다.

### 1.2 축-일반성(axis-generality)은 처음부터, 특수화는 나중에도 하지 않는다

현재 검증 케이스는 x축 로터지만, 유입류가 −y/+z/−x인 케이스가 예정돼 있다. 분해 축,
halo 교환, MLG 커플링, ALM 훅 모두 축 인덱스를 매개변수로 받으며 **어디에도 특정 축
가정이 없다**. 게이트는 3축 전부를 대칭적으로 검증한다(G-M1, G-M4). 축 "선택"은
성능 문제로 분리되어 §3.2의 원칙적 규칙으로 처리된다.

### 1.3 분산 훅은 production 코드에 정식으로, fork는 만들지 않는다

분산 실행을 위한 별도 솔버 사본을 만들지 않았다. 대신 production 클래스에
**미설정 시 완전 no-op인 훅**을 심었다:

- `ActuatorLineModel._grid_offset` — 그리드 상대 연산(샘플링/스프레딩)만 로컬 좌표로
  이동. 로터 물리·후류 기하는 global 좌표 유지(상대거리 이동불변이라 보정행렬 무변경).
- `ActuatorLineModel._velocity_sampler` — `step()` 내부의 속도샘플 주입점. MPI 구현이
  이 자리에서 Allreduce한다.
- radial-truncation의 전역 scale 컨텍스트 — 재정규화 스케일은 전 rank 동일해야 하므로
  전역 기하로 계산하고, 커널의 셀별 radial cut만 로컬 좌표를 쓴다.

1-rank 회귀는 "ALM smoke 궤적의 출력 자릿수 전부 보존"으로 확인했다. 게이트가 구동하는
코드와 클러스터가 실행하는 코드가 **같은 코드**라는 것이 이 방식의 핵심 가치다.

### 1.4 증명된 프리미티브의 재사용, 물리 수식의 복제 금지

- 분해의 만능 다리는 `esoteric_gather/scatter_std_region`(영역-범위 esoteric↔표준 변환,
  정확한 치환) 하나다. halo 교환, MLG 커플링, 체크포인트가 전부 이것 위에 선다.
- MLG 커플링의 수학(융합 rescale 커널, cubic upsample, f_neq 필터)은 global
  `GridCoupling` 인스턴스를 **감싸서** 로컬 슬라이스로 호출한다(`RankLocalCouplingV1`).
  물리 수식이 두 곳에 존재하지 않으므로 단일-GPU 물리 개선이 자동으로 분산에 반영된다.

### 1.5 결정적 복제 빌드 (분산 초기화 대신)

각 rank가 **동일한 전체 케이스를 결정적으로 빌드**한 뒤 자기 슬랩만 남기고 해제한다.
분산 초기화기를 새로 작성하는 대신 이미 검증된 단일 빌드 경로를 그대로 쓰는 것이다.
- 근거: 초기화는 결정적(모든 rank에서 비트 동일 상태)이며, 케이스가 단일 GPU에 들어가는
  한(rank마다 자기 GPU에서 피크 발생) 메모리상 문제가 없다. D40(91.6M)은 초기화 피크
  20.6GB < 24GB로 충족.
- **명시적 한계**: 단일 GPU 용량을 넘는 케이스는 분산 초기화가 필요하다(§7 로드맵).
  이 한계는 숨겨져 있지 않고 러너 docstring과 런북에 문서화되어 있다.

### 1.6 게이트 사다리: 한 번에 한 단계, 통과 전 다음 단계 없음

M1(단일격자 halo) → M2a(합성 2-레벨 커플링) → M2b(실제 bench5 5-레벨+SGS+BC) →
M3(ALM) → M4(v2 halo) → M5(MPI 배선·클러스터). 각 단계는 전용 게이트가 production
레퍼런스와 대조하며, **이전 단계의 결과를 가정하지 않고 매번 전체를 재검증**한다
(예: G-M3는 5-레벨 필드 전체를 다시 bit 대조한다).

`LoopbackTransport`(프로세스 내 우편함)는 "전선만 없는 MPI"로, 게이트가 클러스터 없이
N-rank 코드 경로 전체를 구동하게 한다. 전송 계층만 갈아끼우면 실제 MPI가 된다 —
로컬 게이트와 클러스터 실행의 코드 경로 차이는 `Transport` 구현 하나뿐이다.

### 1.7 실측 후 최적화 (성능 추측 금지)

v1 물리밴드 halo는 트래픽이 크지만(면당 2셀×27슬롯) **증명이 단순**하다. D40 4-rank
실측 예상 오버헤드 ~1%에서 v1로 충분하므로 v1이 production 기본값이다. 6× 절감의 v2는
프로토콜 수학을 게이트로 증명해 두되(§3.4), 음향급 강스케일링 실측이 필요를 보일 때
결합한다. "더 빠른 것"이 아니라 "필요한 만큼 빠른 것 중 가장 단순한 것"을 기본값으로 둔다.

---

## 2. 분해 구조

### 2.1 1D 슬랩 분해를 선택한 이유

- 통신 상대가 rank당 최대 2(양옆)로 고정 — PCIe(no-NVLink) 클러스터에서 halo 볼륨이
  지배하므로 통신 그래프 단순성이 곧 성능 예측 가능성이다.
- MLG 중첩 박스와의 교차 로직이 1축 구간 산술로 닫힌다(§2.3의 파생 컷).
- 한계: rank 수가 커지면 슬랩이 얇아져 halo 중복계산 비율이 커진다. 4~8 GPU 규모에서는
  1D가 우월하고, 그 너머는 §7의 미래 과제다(현 하드웨어 4×4090에 맞는 선택).

### 2.2 Partition1D와 파생 컷(derived cuts)

레벨 0의 컷만 자유 변수다. 레벨 k의 컷은 **좌표 매핑으로 파생**된다:
`fine = 2·(coarse − box_lo)`, 박스 끝을 소유한 rank가 co-located 격자의 +1 노드를
흡수한다. 이렇게 하면 **커플링 영역(strip/excised)이 rank마다 겹침도 빈틈도 없이
분해**되고, C2F/F2C가 rank-로컬 연산이 된다(커플링 통신 자체가 소멸).

### 2.3 균등분할의 함정과 balance_cuts

균등 L0 분할은 farfield40 케이스 4-rank에서 **모든 축이 실행 불가능**했다: 중첩 fine
박스가 도메인 중앙에 몰려 있어 바깥 rank들이 L4 셀을 하나도 소유하지 못한다(own=0).
해법은 own=0 지원이 아니라 컷 배치다:

> **모든 L0 컷을 최내곽(L4) 박스의 L0-span 내부에 둔다.** 박스는 중첩이므로
> (L4 ⊂ L3 ⊂ … ⊂ L0) 이 조건만으로 모든 rank가 모든 레벨을 소유함이 보장된다.

동시에 L3+L4가 전체 업데이트의 86%(farfield40)이므로, 업데이트 밀도의 quantile에 컷을
두면 부하 균형이 함께 달성된다(4-rank worst-share 0.266, 이상값 0.250). 컷 후보는
정확한 체인 시뮬레이션(`chain_owns` — 러너와 동일한 산술)으로 feasibility를 검사한다.

### 2.4 축 자동선택

원시 extent 비교는 함정이다(bench5에서 x extent가 가장 크지만 중첩 박스에 의해 x-체인이
퇴화 — L3/L4가 한 rank에 독점된다). 규칙: **각 축에 대해 balanced 컷 체인을 실제로
시뮬레이션하여 worst-rank 업데이트 점유율을 비교**하고, 동률이면 halo 평면의 메모리
연속성 내림차순(x>y>z; 배열이 (Q,x,y,z) C-order)으로 깬다. farfield40은 y/z 동률 →
y 선택(원리적 tie-break이지 물리적 선호가 아님).

---

## 3. Halo 교환

### 3.1 Esoteric Pull과 분산의 긴장

Esoteric Pull(단일버퍼 in-place 스트리밍)은 f-메모리를 절반으로 줄여 D40을 단일 24GB에
넣은 핵심이지만, 메모리 상태가 시간 parity에 따라 슬롯-스왑되어 있어 "물리적 f"가
직접 존재하지 않는다. 분산 교환은 두 방식으로 이 긴장을 푼다.

### 3.2 v1: 물리밴드 프로토콜 (production 기본)

매 스텝 **전에**, 소유 경계 밴드(ghost 폭만큼)를 `esoteric_gather_std_region`으로
물리 표준 순서로 풀어 보내고, 받은 쪽은 `esoteric_scatter_std_region`으로 자기 ghost
밴드에 심는다. 커널은 ghost 포함 전체 로컬 배열을 돈다:

- **ghost 층 1은 중복계산된다**: 동기화된 입력에서 계산되므로 그 STORE가 소유 edge에
  쓰는 값은 이웃의 실제 셀이 썼을 값과 비트 동일하다.
- 바깥 ghost 층은 부분적으로 쓰레기를 계산하지만 다음 교환이 덮어쓴다.
- 따라서 소유 셀의 진화는 단일 실행과 정의상 동일 — 이것이 G-M1/M2b의 bit 결과다.

**ghost 폭**: 순수 LBM은 2, SGS(dyn. Smagorinsky) 실행은 3. 후자는 게이트가 잡아낸
발견이다. 기제(검토 ①의 정리를 따름): 동기화된 ghost의 population 집합은 완전하므로
u 자체는 ghost 전 층에서 유효하고, 오염은 dyn_smag의 반경-2 스텐실이 로컬 wrap 경계를
넘을 때만 발생한다 — 즉 ghost=3은 마진 0으로 정확히 충족이다. 일반 규칙으로 명문화한다:
**ghost ≥ 1 + (커널의 비국소 입력 총 반경)**. 향후 더 넓은 test filter 등 커널 추가 시
이 식으로 재검토한다.

### 3.3 왜 커플링 통신이 없는가

MLG C2F/F2C의 읽기 여백(C2F 블록은 소유 범위 +2 coarse행, F2C strided 읽기는 +1
coarse행)은 ghost 신선도 스케줄(매 서브스텝 교환) 안에 들어온다. 파생 컷(§2.2) 덕에
쓰기 영역은 rank 간 분할 정확 — 결과적으로 **레벨 간 통신은 존재하지 않고, 레벨 내
halo 교환만 남는다**. 이 여백 논증의 상세는 `src/parallel/mlg_coupling.py` docstring과
patch 17 §M2에 있다.

### 3.4 v2: raw parity-slot 교환 (증명 완료, 결합 보류)

v1의 트래픽(면당 2×27 슬롯평면)을 9/54 = 1/6로 줄이는 프로토콜. ghost=1이고 ghost 셀은
SOLID로 마킹되어 **계산되지 않는 순수 우편함**이다. 스텝 **후에** 면당 정확히 "축을
횡단한 population 9개/셀"만 교환한다:

- **A그룹**: 우리 edge 셀의 STORE가 우리 ghost 평면에 예치한 유출분 → 이웃의 소유 edge,
  **같은 슬롯 인덱스로** 복사.
- **B그룹**: 이웃의 다음 LOAD가 원격으로 읽을 우리 edge 평면 상주분 → 이웃 ghost,
  역시 같은 슬롯으로.

재라벨링이 불필요한 이유: esoteric parity swap이 스트리밍을 이미 인코딩하고 있어,
송신 시점의 슬롯 라벨과 수신측 다음-parity LOAD가 읽는 슬롯이 대수적으로 일치한다.
또한 수신 슬롯과 수신자 자신의 STORE 슬롯은 pair 내에서 상보적이라 충돌이 원리적으로
불가능하다. G-M4: 3축 bit + 실측 트래픽 정확히 6.00×↓.

v2의 SGS 결합(u 평면 교환)과 MLG 결합(커플링 시점만 v1 밴드), 통신/계산 오버랩
(interior/edge 분할 런치)은 설계만 남기고 **실측이 정당화할 때 진행**한다(§1.7).

---

## 4. 분산 ALM

### 4.1 문제 구조

ALM `step()`은 회전 전진 → 마커 속도 샘플링 → BEM/보정 solve → 힘 스프레딩의
모놀리스다. 그리드 접점은 샘플링(읽기)과 스프레딩(쓰기) 둘뿐이고, 그 사이의 solve는
마커 상태만의 함수다. 여기서 프로토콜이 나온다:

> **각 rank는 자기 소유 셀에 대한 Gaussian 가중 분자/분모 부분합만 계산하고
> (마커당 4 스칼라, 전체 ~16KB), Allreduce 후 나눗셈한다. u_markers가 전 rank
> 동일해지므로 BEM/보정 파이프라인 전체가 통신 없이 복제 실행된다.**

- **소유권의 정확성**: 마커 스텐실 박스가 소유 슬랩 경계에서 자연 클립되므로 ghost
  이중계상이 구조적으로 불가능하다. 스텐실이 한 rank 안에 완전히 들어가는 마커는
  부분합이 단일-rank 합과 비트 동일하고, 경계에 걸친 마커만 결합순서가 달라진다
  (fp last-bit 등급 — 실측은 bench5 32 서브스텝에서 bit 유지, F_grid rel ~6.5e-17).
- **복제 solve의 타당성**: 동일 입력(u_markers) + 결정적 파이프라인 = 동일 출력.
  rank별 발산이 원리적으로 없으므로 "rank 0이 solve하고 브로드캐스트" 같은 비대칭
  구조(추가 통신 + 코드 분기)를 기각했다.
- **스프레딩**: 각 rank가 로컬 F_grid에만 쓴다(마커 힘은 전 rank 동일하므로 소유
  영역별 쓰기가 곧 분할). radial-truncation 재정규화 스케일만 전역 기하로 계산한다
  (§1.3의 scale 컨텍스트) — 스케일이 rank마다 다르면 힘 보존이 깨지기 때문.
- **미지원의 명시**: kleine free-wake(후류점 속도 샘플링의 분산 미구현)와 비-gaussian
  연구용 샘플러는 fail-fast로 막았다. production은 straight wake라 영향이 없다.

### 4.2 게이트 방식

G-M3는 rank마다 **실제 `ActuatorLineModel`**(별도 결정적 빌드에서 추출)을 스레드로
구동한다 — barrier 기반 `ThreadAllreduce`가 MPI.Allreduce와 동형의 제어 흐름을
프로세스 내에서 재현한다. 즉 게이트가 검증하는 것은 모형이 아니라 production 객체의
분산 구동 그 자체다.

---

## 5. SPMD 러너와 MPI 계층

- `DistributedMLGRunner`: 게이트에서 증명된 재귀(레벨별 sync → advance → C2F half/full
  → F2C)의 SPMD 재구성. rank별 루프를 제거했을 뿐 시퀀스가 게이트와 동일하므로,
  게이트의 bit 증명이 러너에 그대로 이전된다.
- `MPITransport`: CUDA-aware(UCX device-direct)와 host-staging 폴백을 하나의 인터페이스
  로. 수신 크기가 양쪽에서 결정적(밴드/팩 shape을 파티션이 알고 있음)이라 **와이어에
  크기 협상이 없다**. `mpi4py.rc.thread_level='serialized'`를 import 전에 설정해야
  OpenMPI 5.x가 UCX PML을 유지한다는 함정은 엔트리(`main_mpi.py`)가 처리한다.
- 예외는 `comm.Abort`로 즉사시킨다(행 방지). `--verify`는 소유 조립본을 rank 0에 모아
  단일-rank 신규 빌드 레퍼런스와 대조하는 **클러스터 현장 게이트**다.

---

## 6. 사례 연구: 4-rank 1-ulp 사건 (검증 철학의 실증)

클러스터 §1 검증에서 4-rank만, L0만, 정확히 7.451e-09(= 2⁻²⁷, 2/27-가중 population의
f32 1 ulp) 차이가 났다. tolerance 게이트는 PASS였지만 예측("순수 LBM은 전 레벨 bit")
위반이므로 원인 규명을 진행했다:

1. 로컬 3090에서 동일 컷으로 loopback 재현 → **bit** (분산 프로토콜 무죄)
2. GPU 4장 모델 동일 확인 (이기종 가설 기각)
3. `--devices 0,0,0,0`(4 rank를 GPU 0 한 장에) → **동일 재현** (같은 디바이스에서
   분해-vs-단일이 다름 = shape 의존)
4. verify의 diff 위치 출력 → **가장 얇은 두 슬랩(rank1 own=6, rank2 own=5)의 소유행
   전체에만** 국한

원인: 커플링의 모멘트 계산이 `xp.sum`/`einsum`을 사용했고, CuPy reduction/cuBLAS는
**배열 shape과 디바이스 SM 수에 따라 누적 분할 전략을 선택**한다. L0 F2C 블록만
유일하게 작아(얇은 행 × 48² transverse) 4090(128 SM)의 전략 경계를 넘었다 — 3090
(82 SM)은 경계가 달라 로컬 재현이 안 됐던 것까지 정합.

수정: c ∈ {−1,0,+1}이므로 곱셈 없는 **고정순서 ± 직렬 누적**으로 교체 — 어떤 shape,
어떤 디바이스에서도 결합순서가 동일해져 bit 보장이 구조적으로 회복됐다. 클러스터
재검증: 4-rank 전 레벨 bit.

교훈: (i) tolerance 기준만으로는 라이브러리 비결정성이 영원히 숨는다. bit 예측과
그 위반의 추적이 실질적 버그 클래스를 잡는다. (ii) 분산 경로에 파이썬 수준 라이브러리
리덕션이 하나라도 남아 있으면 shape 의존성이 생긴다 — 리덕션은 커널 내 직렬이거나
고정순서여야 한다.

---

## 7. 현재 한계와 로드맵 (검토자가 알아야 할 것)

| 한계 | 상태 | 계획 |
|---|---|---|
| MPI 러너 출력 = thrust CSV만 | VTK/체크포인트 rank0 조립 미배선 | M5 검증 후 후속 |
| 복제 빌드의 메모리 천장 = 단일 GPU 용량 | D40은 충족(피크 20.6GB) | 초과 케이스 시 분산 초기화 |
| free-wake·비gaussian 샘플러 분산 미지원 | fail-fast로 명시 차단 | 필요 시 후류점 샘플링 분산 |
| v2 halo production 미결합 | 프로토콜은 bit 증명 완료 | 강스케일링 실측 후 결정 |
| 1D 분해 한정 | 4~8 GPU 규모에 적정 | 그 이상 규모에서 재평가 |
| >2 rank ALM Allreduce 결합순서 | fp last-bit 등급(원리) | 게이트 기준에 반영됨 |
| ~~ALM 부분합 내부의 CuPy 리덕션 (F-1)~~ | **해소**(백로그 #3): 마커별 고정순서 RawKernel — 라이브러리 전략 의존 소멸 | 잔여=프로토콜 재결합(선언된 lastbit 등급); 폴백 경로(LBM_ALM_SAMPLE_KERNEL=0)에서만 복귀 |

## 8. 검토 요청 포인트

외부 검토에서 특히 따져봐 주기를 요청하는 논증들:

1. **중복 ghost 계산 논증**(§3.2): "ghost 층 1의 STORE가 소유 edge에 쓰는 값은 이웃
   실제 셀의 값과 동일"의 전제 — 커널의 비국소 입력이 SGS u-구배뿐인가? (우리는 ghost
   3으로 해소했지만, 향후 커널 추가 시 재검토 규칙이 필요하다)
2. **MLG 커플링 여백**(§3.3): C2F +2행 / F2C +1 coarse행 여백이 cubic/f_neq 필터
   스텐실을 모든 컷 위치에서 커버한다는 주장(`mlg_coupling.py` docstring).
3. **balance_cuts의 보장**(§2.3): "컷이 최내곽 박스 안에 있으면 전 rank 전 레벨 소유"
   — 박스 중첩이 항상 성립하는 격자 구성인지(비중첩 다중 박스 지원 시 재설계 필요).
4. **ALM 소유권 정확성**(§4.1): 스텐실 자연 클립 논증과, 경계 마커의 fp 재결합이
   유일한 비-bit 원천이라는 주장.
5. **v2 슬롯 대수**(§3.4): parity 스왑 하의 A/B 그룹 슬롯 라벨 유도
   (`src/parallel/halo_v2.py` 모듈 docstring + `_slot_tables`).

---

## 부록 A. 검증 결과 총괄

| 게이트 | 내용 | 결과 |
|---|---|---|
| G-M1 | 단일격자 TGV, 2-rank, 3축 | f/ρ/u 전부 bit |
| G-M2a | 합성 2-레벨(f_prev half-step 포함) | bit |
| G-M2b | **실제 bench5 5-레벨**(dyn_smag+eq/sponge) vs `mlg.advance()` | y/z 전 레벨 bit |
| G-M3 | 분산 ALM(pure + archB/kleine-straight), 실제 모델 스레드 구동 | 필드 bit, F_grid rel ~6.5e-17 |
| G-M4 | v2 슬롯교환, 3축 | bit, 트래픽 6.00×↓ |
| 로컬 mpirun | MPICH host-staged, 2-rank, pure/archB | 전 레벨 bit |
| 클러스터 §1(a) | 2-rank ALM, UCX cuda-aware | 전 레벨 bit |
| 클러스터 §1(b)(c) | 4-rank 순수/ALM (coupling 수정 후) | 전 레벨 bit |
| 클러스터 §2 | farfield40 D40 4-rank, 25-rev 풀런 | **완료**: 0.671 s/step(단일 3.1 대비 4.6×; 밸런스 이상치 3.76× 대비 효율 ~100%), 풀런 ~6h. **물리: last-5-rev CT +0.027% vs 단일**(rev-내 σ ±0.6%의 1/20), 팁 max\|ω\| +3.5%(카오스 폭 내), VTK/ckpt production 포맷 조립 |

## 부록 B. 파일 지도

```
src/parallel/
  partition.py    Partition1D, choose_axis(+balanced), balance_cuts, chain_owns
  halo.py         LoopbackTransport / MPITransport / HaloBandExchangerV1 (v1)
  halo_v2.py      SlotHaloExchangerV2 (v2, 결합 보류)
  mlg_coupling.py RankLocalCouplingV1, fine_range_from_coarse
  alm_dist.py     부분합, ThreadAllreduce(게이트)/MPIAllreduce, 샘플러 팩토리
  local_level.py  extract_level / LocalLevel (게이트 검증 코드의 승격)
  runner.py       DistributedMLGRunner (SPMD)
main_mpi.py       엔트리 (--verify, --devices, comm.Abort)
patch_notes/hpc_upgrade/17_multigpu_design.md   설계 + M1~M5 전 단계 로그
patch_notes/hpc_upgrade/18_m5_cluster_runbook.md 클러스터 절차
patch_notes/hpc_upgrade/gates/mgpu_m*.py         게이트 원본
```

---

## 부록 C. 외부 검토 반영 (2026-07-13)

검토 결과: **승인** (`docs/MULTIGPU_REVIEW_kr.md`) — 검토 포인트 5건 전부 성립,
게이트 4종 독립 재현, 핵심 논증 3건 독립 재유도 확인. 발견 사항 처리:

| 발견 | 처리 |
|---|---|
| F-1 ALM 부분합의 CuPy 리덕션 잔존 | §7 표에 명시 + `alm_dist.py` docstring에 KNOWN RESIDUAL 기록. fp-lastbit 등급 내 |
| F-2 러너 이중 sync (2× halo 트래픽) | 수정 완료(294516a) — 검토 시점 HEAD 이후. 멱등이라 bit-중립 |
| F-3 메시지당 deviceSynchronize | 수정 완료(294516a) — staged commit + 라운드당 1회 stream sync + persistent 버퍼 |
| F-4 verify --strict-bit | 추가 완료 — 순수-LBM 케이스는 bit로만 PASS 가능 |
| ①의 ghost 규칙 정밀화 | §3.2에 "ghost ≥ 1 + 비국소 입력 총 반경"으로 명문화 |

검토 후 성능 실측(§2 프로파일)이 §6의 교훈을 한 번 더 실증했다: 커플링의 고정순서
elementwise 체인(§6 수정의 1차 형태)이 D40 4-rank에서 1.0 s/step를 차지함이 드러나
**융합 RawKernel(셀별 직렬 = 결정성과 성능을 동시에)**로 재수정되었고, 이 과정에서
rank-로컬 f2c가 모멘트 시퀀스를 인라인 재기술하고 있던 것(§1.4 원칙의 사각지대)도
`_feq_fneq` 프리미티브 공유로 교정되었다 — 두 경로가 연산자를 공유하지 않으면
kernel-vs-elementwise 라운딩으로 갈라진다는 G-M2b 실증 포함.

## 부록 D. §2 실측 완결 (2026-07-13) — 검토 유보사항 해소

검토의 유일 유보("성능 주장이 견적")는 다음 실측으로 해소되었다:

- **성능**: D40 91.6M셀 4×4090, 25-rev 풀런 0.671 s/step (단일 3.1 → 4.6×; region 커널이
  단일 경로에도 이득이라 공정 비교는 단일 재기준 ~2.5s 대비 ~3.7×, 밸런스 이상치 3.76×의
  ~100%). 프로파일: alm 0.275(복제) / halo 스큐 0.18 / kernel 0.103 / coupling 0.096.
- **성능 수렴 과정 자체가 §1.1·§1.4 원칙의 추가 실증**: pass 2에서 G-M2b가 rank-로컬 f2c의
  연산자 미공유(수식 인라인 재기술)를 3.9e-07로 즉시 검출; pass 4의 단일-런치 region 커널은
  "순수 치환 = bit-by-construction"이라 게이트 전체가 무수정 통과.
- **물리**: 동일 config 25-rev, last-5-rev CT 단일 0.010399±6.0e-5 vs MPI 0.010402±5.7e-5
  (**+0.027%**), 팁 환대 max|ω| +3.5%(탈상관 스냅샷 변동 폭), 방위각 평균 유입류 편차 2.3%.
  분석: `aeromechanics_workshop/HVAB/0713_multi_gpu/analyze_mpi_vs_single.py`.

## 부록 E. 성능 재정산 (2026-07-13, 단일GPU 재기준 후 정정)

부록 D의 "4.6×"는 **구코드 단일(3.1s) vs 신코드 4-rank(0.671s)의 혼합 비교**였다.
M5 성능 수리 중 도입된 region gather/scatter 커널은 단일-GPU 커플링 경로도 동일하게
가속하므로(로컬 3090 실측: production 단일 1.633 s/step; NR=1 러너 1.734 = 오버헤드 +6%),
동세대 코드 기준의 정직한 분해 이득은:

| 비교 | 값 |
|---|---|
| 총 개선 (지난주 단일 3.1 → 현 4-rank 0.671) | 4.6× (실사용 관점) |
| **동코드 분해 이득** (4090 단일 추정 ~1.1 → 0.671) | **~1.6-1.8×** |
| 병렬화된 부분(kernel+coupling)의 분할 효율 | ~4× (이상적) |

구성 검산: 0.671 = 계산 0.20(단일 ~0.79의 1/4) + ALM 0.275(**비병렬 복제** — 단일에도
동일 존재) + halo 0.20. 즉 낮은 총효율의 원인은 분해 품질이 아니라 **Amdahl(ALM) +
halo 상수**이며, 이것이 백로그 #3(ALM 로컬화)·#5(v2+오버랩)의 정량적 동기다.
(#3·#5 완료 시 추정 ~0.35-0.45 s/step → 동코드 ~2.5-3×.)

교훈: 성능 주장에도 bit-parity 원칙과 같은 규율이 필요하다 — 비교 기준(코드 세대)을
명시하지 않은 speedup은 검증 불가능한 수치다. 클러스터 단일 재기준 명령:
`mpirun -n 1 ... main_mpi.py --config ..._mpi4.py --steps 64 --profile` (러너 오버헤드 +6% 포함).

## 부록 F. 2차 검토 반영 (2026-07-13)

2차 검토(HEAD ~c792ba1 시점, 게이트 전수 재현 + 성능 패스 4종 정확성 코드 확인)의
지적과 실제 진행의 대응:

| 검토 지적 | 처리 |
|---|---|
| F-5: --verify 파손 (build 튜플 미언팩) | 검토 시점 직후 커밋 `8a444b1`에서 이미 수정 (#3 작업 중 동일 오류를 직접 밟음) |
| 메타 교훈: 게이트 없는 진입점이 사각지대 | **G-verify 게이트 추가** — 런북 §1(a) 명령 형태를 그대로 구동, exit 0 + 5레벨 bit + PASS + --strict-bit(F-4) 경로까지 검사 |
| 경고 1: ALM 로컬화는 F-1과 충돌 — 고정순서 커널 먼저 | **처방과 동일 순서로 이미 구현됨**(#3): alm_sample_markers = 마커별 고정순서 커널 → F-1의 라이브러리-전략 의존 자체가 해소. 마커 부분집합 분할(BEM 분산)은 하지 않음 — BEM 복제 유지 |
| 경고 2: 로컬화 스큐 + v2 우선순위 재고 (스큐는 트래픽 절감으로 안 줄어듦) | **동일 결론으로 수렴**(#5): v2는 차단 분석과 함께 이연, fresh-skip(멱등 라운드 30% 소멸) + early-post/Irecv 오버랩 우선. ALM 행-비용의 balance_cuts 반영은 샘플링이 ~0.02-0.03 s/step/rank로 줄어 현 수치에선 불요 — 모니터링 항목으로 유지 |

### 부록 E 갱신 (2026-07-13 재측정)

백로그 #3(ALM 샘플링 커널)+#5(fresh-skip/오버랩) 반영 후 클러스터 확정치:

| 항목 | 값 |
|---|---|
| 4-rank D40 | **0.442 s/step** (0.671→; 25-rev ≈ 3.9h) |
| 내역 | alm 0.117 / halo_c 0.116 / kernel 0.103 / coupling 0.096 / halo_p 0.009 |
| 단일(NR=1, --dist-init 경로) | 측정 대기 — replicated NR=1은 native 24GB 불가로 판명(최대레벨 source+슬랩 동시보유), --dist-init로 공식화 |

NR=1 OOM 사건은 §1.5의 "복제 빌드 한계"가 NR=1에서 이중으로 나타나는 경우로,
WSL2 oversubscription이 로컬 검증을 가려온 사례다(23GB 하드리밋 재현으로 확정).
교훈: 메모리 검증은 native-한도 에뮬레이션(pool set_limit) 하에서 해야 한다.
