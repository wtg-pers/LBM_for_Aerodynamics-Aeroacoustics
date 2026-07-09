# Phase 1c 재조정 후 첫 win — C2F cubic_interp_z coalescing — 2026-07-08

10의 nsys 계획을 사용자가 클러스터 실행 → 커널 랭킹 확보. 최대 병목이 **메모리
coalescing 문제**로 드러나 **bit-identical·10.5× 커널 수정**으로 순수-LBM **1.9×**.

---

## A. 클러스터 nsys 결과 (4090, bench5_pure_lbm, 10 coarse step)

`cuda_gpu_kern_sum` 카테고리 집계 (%GPU 커널시간):

| 카테고리 | % | 정체 |
|---|---|---|
| **cupy 원소별**(add/mul/div/sub/copy) | **~39.5** | coupling rescaling 체인 **unfused** (add 1950·copy 4290 등 다수 소커널) |
| **cubic_interp** z 24.4 + y 5.4 + x 4.2 | **~34** | coupling 공간 upsample |
| core LBM (cumulant/stream/macro/dyn_smag) | ~24.8 | 실제 LBM 커널 |
| sgemm (einsum moment) | ~1.5 | ← 시간상 무시(=graph capture 차단막일 뿐, 성능 아님) |

→ coupling(보간+rescaling)=~75%, 로컬 MLG_PROFILE(76%)와 정합. **collision 아님.**

## B. ★핵심: `cubic_interp_1d_z` = 24.4% (x의 5.7×, 같은 배열 크기)

`interpolation_d3q27.py`: x/y/z 세 1D 커널이 **동일 크기** 배열을 처리하는데 z만
876M ns vs x 152M. 원인 = **thread 매핑 coalescing**:
- **x**: thread per (q,iy,iz), iz(연속축)가 thread 최속 → warp가 연속 메모리 → **coalesced**.
- **z**(구): thread per (q,ix,iy) + 내부 iz 루프 → 인접 thread가 iy(stride Nz) → **uncoalesced**.

## C. 수정 = iz-fastest 매핑 (bit-identical)

z 커널을 thread per (q,ix,iy,iz), **iz 최속**으로 재작성(even-iz lane early-return).
연속 iz를 warp가 담당 → coalesced. **노드별 산술(가중치·이웃·순서) 불변 → bit-identical.**
launch도 Q·Nx·Ny·Nz로 갱신. (`src/kernels/interpolation_d3q27.py`)

## D. 측정 (RTX3090 로컬)

- **커널 격리**(L4 dims 27×57×297×297): z_old 23.6ms → z_new 2.25ms = **10.5×**,
  **bits-differ 0/135.75M = BIT-IDENTICAL**. 게이트 `gates/cubic_interp_z_coalesce_gate.py` PASS.
- **전체 pure-LBM**(MLG_PROFILE 12-step): **TOTAL 996 → 521 ms/step = 1.91×**.
  C2F.L4 525→186(2.82×), C2F.L3 188→68(2.76×), C2F 합 762→285.
  (무프로파일 steady ≈ 961→~500ms. 4090 환산: pure-LBM 391→~205ms 예상.)

## E. 영향 & 게이트
- **bit-identical → 참조 sha256 `ac910ff` 보존, 재기준선 불필요.** 순수 add-on 성능.
- ALM 런(bench5_baseline)은 L4 BEM 84.8% 지배라 이득 소폭(LBM ~15%만); production
  (207M, LBM=coupling 지배)선 큰 이득.
- 게이트: `cubic_interp_z_coalesce_gate.py`(구 fiber 커널 reference vs 현 커널 bit==0).

### E.1 ★클러스터 검증 (4090, bench5_pure_lbm, 사용자 실행)
z-fix ON(grep 확인) 런:
| 섹션 | before(z-fix 전) | after | 배율 |
|---|---|---|---|
| C2F.L4 | 160.76ms | **117.82ms** | 1.36× (−43ms) |
| C2F.L3 | 57.19ms | 42.26ms | 1.35× (−15ms) |
| pure-LBM TOTAL | ~391ms(Phase0) | **333ms** | **1.17×** |
| MLUPS | 302(Phase0) | **352.7** | 1.17× |

물리 정상(conservation −0.0000%). **4090은 1.17×**(로컬 3090 1.9×보다 작음: 4090 L2
72MB가 uncoalesced 접근을 상당 흡수 → 원래 z-커널이 3090만큼 나쁘지 않았음. 3090 L2 6MB).
절대 −58ms/step(C2F.L4+L3) = 4090 pure-LBM ~17%, **무료(bit-identical)**.

### E.2 ★★부수 발견 — 4090 pure-LBM은 run-to-run 비결정적
동일 z-fix 런 2회 = **다른 sha** (`92c05427` vs `21837a17`)인데 MLUPS/TOTAL/conservation은
동일. → fp reduction 비결정성이 1006 스텝 카오스(τ=0.5)로 증폭. (graph 조사의 6e-8
비결정 reduction과 정합; 3090 eager는 결정적이나 4090은 SM/occupancy로 atomic 경로 추정.)
★**함의: 4090서 pure-LBM sha256은 유효 bit-gate 아님 → CV-band/물리 게이트만.** z-fix
bit-identity는 커널 수준 로컬 증명(0/135M)으로 이미 확보. **⇒ Target 2(융합)의 재기준선
걸림돌도 무의미**(어차피 sha 재현 불가) → CV-band 게이트로 융합 진행 가능.

## F. 다음 타겟 (남은 병목)
1. **★Target 2 = coupling rescaling 융합** (4090 nsys cupy 원소별 **39.5%** = 최대 프라이즈).
   `_compute_macroscopic`+`_compute_f_eq`+f_neq+rescale+reconstruct+temporal interp를
   read-once/write-once **1 custom RawKernel**로 → bandwidth 직격. last-bit 변화이나
   **E.2로 4090 sha가 애초 재현 불가 → 재기준선 걸림돌 소멸, CV-band 게이트로 진행 가능.**
   게이트: 로컬 CV-band(융합 vs 기존 물리 동등, 소수 스텝 macro 필드 비교) + 클러스터
   conservation/C_T/MLUPS.
2. cubic_interp y/x (9.6%) 및 F2C filter — 잔여, 후순위.
3. z-fix 클러스터 검증 = **완료**(E.1: pure-LBM 1.17×, C2F.L4 1.36×, 물리 정상).

## G. 변경 파일 (이번 세션 누적, uncommitted)
`src/kernels/interpolation_d3q27.py`(★z coalescing, bit-identical),
`src/grid/multi_level_grid.py`(graph scaffolding OFF·dormant),
`patch_notes/hpc_upgrade/{09,10,11}.md + gates/{p1c_s1_graph_gate, nsys_purelbm_driver,
cubic_interp_z_coalesce_gate}.py`. coupling.py 등 수치코드 미변경(einsum 원복).
