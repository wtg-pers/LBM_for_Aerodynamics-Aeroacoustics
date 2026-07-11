# 단일 GPU 최적화 — 방법과 원리 (요약)

> 2026-07-11 기준. HPC 트랙(patch_notes/hpc_upgrade/ 00~15) 성과의 압축 정리.
> 상세: 각 패치노트 + `15_p1b_esoteric_pull_design.md` §11 구현로그.
> 커밋 체인: 80a64c9(속도 베이스라인) → 44f2f56 → 3b462ba → c7a1cd1 → 133bbe2 → e294e09 → 0d499c6 → 67a6a6d.

## 속도 (pure-LBM 1.48×, ALM 벤치 2.9×)

| 방법 | 원리 |
|---|---|
| **Memory coalescing** (cubic-z 보간 커널, patch 11) | GPU warp의 인접 스레드가 인접 메모리 주소를 접근해야 DRAM burst가 활용됨. fiber-per-thread(stride-Nz 접근)로 돌던 z-방향 보간을 iz-fastest 스레드 매핑으로 재작성. 순수 접근순서 변경이라 bit-identical. C2F.L4 1.36× |
| **Kernel fusion** (C2F rescale, patch 12) | LBM은 memory-bound — 시간 = 데이터 이동량. 시간보간→macroscopic→f_eq→f_neq rescale로 이어지던 elementwise 체인(각각 global memory 왕복)을 단일 per-node 커널로 융합해 read-once/write-once화 + 커널 런치 오버헤드 제거. 누적 391→264ms |
| **ALM 선택적 GPU 상주** (patch 04~07) | freewake Biot-Savart(360K 상호작용)만 GPU로. 작은 배열(n≈48~64) 연산은 커널 런치 오버헤드가 이득을 상회 → CPU 유지가 실측상 빠름. bench5 61→21분 |
| **64-bit 인덱싱** | `q*N+idx`가 int32 한계(2³¹)를 넘는 레벨(>79.5M셀)에서 오버플로 → long long. DGX fine 크래시 원인이었음 |
| (기각) CUDA Graphs (patch 08~09) | 런치 오버헤드 제거 가설 → 실측 1.006×. 병목이 launch가 아닌 compute/bandwidth임을 확인하고 미채택(코드는 dormant, 기본 OFF) |

## 메모리 (D40 91.6M셀: ~38GB → 실측 17.1GB, 피크 20.6GB → 단일 24GB 4090)

| 방법 | 원리 |
|---|---|
| **Esoteric Pull** (Lehmann 2022, in-place streaming; patch 15 a~b) | 표준 two-grid는 streaming의 read/write 충돌 방지를 위해 f 버퍼 2벌 필요. Esoteric Pull은 반대방향 분포쌍(cᵢ, −cᵢ)을 인접 슬롯에 짝지어, 홀/짝 스텝이 교대로 스왑된 슬롯 규약을 쓰면 각 메모리 주소의 읽기 스레드=쓰기 스레드가 되어 race 없이 **단일 버퍼**에서 streaming+collision 수행 → f_post 제거(PDF 메모리 ½). cumulant collision·SGS·ALM 2-pass·MLG 커플링과 통합. env `LBM_ESOTERIC=1` opt-in |
| **f_prev sub-volume화** (patch 15 e1) | MLG C2F 시간보간이 읽는 것은 coupling의 coarse sub-volume 슬라이스뿐. full-level 복사(D40 기준 7GB) 대신 그 서브볼륨만 저장(0.5GB). 읽는 값이 동일하므로 bit-identical — 표준 경로에도 동일 적용·동일 이득 |
| **Region-scoped gather/scatter** (patch 15 e2) | esoteric↔표준 레이아웃 변환을 full-field가 아니라 coupling이 실제 접근하는 영역만: C2F는 [코스 서브볼륨 읽기 + 6면 경계 스트립 쓰기], F2C는 [strided(0::R) 노드 읽기 + excised 블록 쓰기]. full-field 임시버퍼(~7GB transient) 제거 |
| **초기화 x-슬랩 청킹** (patch 15 e3) | `compute_equilibrium`의 (Q,N) 브로드캐스트 임시배열(~4×f 크기)이 init 피크 52.6GB의 원인(WSL2 oversubscription이 은폐, 네이티브 카드면 OOM). pointwise 연산이므로 공간 슬랩으로 나눠 호출해도 bit-identical, 임시는 슬랩 크기로 상한 + 레벨 간 메모리풀 반환 → 피크 20.6GB |
| **체크포인트 슬랩 스트리밍** (67a6a6d) | 전 레벨의 물리 복사본을 GPU에 동시 보유하지 않고, 레벨별로 x-슬랩 단위 gather→호스트 전송 → GPU transient ~0.5GB 상한(체크포인트가 런 고수위에 +0GB). 첫 클러스터 D40 런의 OOM 사후 수정 |

## 검증 체계
변경 성격에 따라 3단계 게이트:
- **bit-identical** — 순수 재배열·접근순서 변경 (coalescing, f_prev sub, region ops, 슬랩 청킹)
- **fp32 last-bit 등가(~1e-7)** — 동일 수학, 다른 연산 경로 (esoteric vs 표준: BGK/cumulant/SGS/ALM/MLG 5-level 전부)
- **CV-band ±3%** — fp 섭동이 카오스 증폭되는 경로 (free-wake ALM 추력; median<1e-3으로 계통편향 별도 검출)

게이트 8종이 `patch_notes/hpc_upgrade/gates/`에 상주 — 이후 변경도 동일 방식으로 회귀 검증.

## 결과 요약

| 항목 | 이전 | 이후 |
|---|---|---|
| pure-LBM step (bench5, 4090) | 391 ms | 264 ms (1.48×) |
| ALM 벤치 (bench5 2rev) | 61분 | 21분 (2.9×) |
| D40(91.6M셀) 메모리 | ~38GB (24GB 불가) | 17.1GB, 피크 20.6GB |
| D40 실행 환경 | DGX급 필요 | **단일 RTX 4090 (여유 ~3.4GB)** |
| 레벨당 셀 한계 | 79.5M (int32) | 해제 (64-bit) |
