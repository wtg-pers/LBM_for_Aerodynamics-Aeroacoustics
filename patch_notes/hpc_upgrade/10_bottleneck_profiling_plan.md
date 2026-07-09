# Phase 1c 종료 후 재조정 — 병목 nsys/ncu 확증 계획 — 2026-07-08

09에서 CUDA Graph가 **launch-bound 아님(compute/bandwidth-bound)**으로 이득 0임을
실증. 다음 최적화 타겟을 정하기 전에 **어느 커널이 실병목이고 compute냐 bandwidth냐**를
nsys/ncu로 확증(사용자 선택). 본 노트 = 로컬 확보한 섹션분해 + 클러스터 프로토콜.

---

## A. 로컬 섹션 분해 (RTX3090, bench5_pure_lbm, MLG_PROFILE 12-step)

per-section sync가 sum을 부풀리나(총 996ms vs 무프로파일 961ms = +3.6%뿐 → 귀속 신뢰),
**C2F(coarse→fine) coupling이 지배:**

| 섹션 | ms/step | % | 비고 |
|---|---|---|---|
| **C2F.L4** | 525.4 | **52.7%** | 16 calls × 32.8ms. ★단일 최악 |
| **C2F.L3** | 187.6 | **18.8%** | 8 calls × 23.5ms |
| C2F.L2 | 41.9 | 4.2% | |
| F2C(전체) | 95.3 | 9.5% | L4 64.7 + L3 22.9 + … |
| **advance 전체(collision+stream+BC)** | 127.2 | 12.8% | L4.adv 84.9 포함 |
| fprev 복사 | 12.3 | 1.2% | |

→ **C2F 합 76%**, C2F.L4+L3 = **71.5%**. 병목 = **coupling의 cubic 보간 + rescaling**
(macroscopic·f_eq·filter). advance(순 collision)는 12.8%뿐. graph가 0이었던 것과 정합
(= 실제 GPU 커널 시간). Phase 0(4090)의 C2F.L4 지배를 재확인·강화.

★타겟 확정: **C2F.L4 커널 구성**을 분해해 어느 커널(cubic interp? einsum rescaling?
filter? elementwise?)이 실병목이고 BW-bound인지 compute-bound인지 판별 → 최적화 방식 결정.

## B. ⚠ 로컬(WSL2) nsys/ncu 불가

- **nsys**: CUPTI 커널 트레이싱이 WSL2서 미동작(trivial cupy 커널조차 GPU 데이터 0). API만.
- **ncu**: `ERR_NVGPUCTRPERM`(GPU 성능카운터 권한). consumer 드라이버+WSL2 제약.
→ **클러스터(native Linux, 4090)서 실행.** 로컬은 MLG_PROFILE(cupy sync, CUPTI 불요)까지만.

## C. 클러스터 프로토콜 (사용자 실행)

드라이버 = `patch_notes/hpc_upgrade/gates/nsys_purelbm_driver.py` (warmup 3 → cudaProfiler
범위로 N step만 캡처, setup 제외). repo 루트서:

### C.1 nsys — 커널 시간 랭킹 + 이름 (가벼움)
```bash
PROF_STEPS=10 nsys profile -t cuda --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o bench5_purelbm_prof --force-overwrite=true \
    python patch_notes/hpc_upgrade/gates/nsys_purelbm_driver.py
# 커널별 총 GPU시간·호출수·평균:
nsys stats --report cuda_gpu_kern_sum --format table bench5_purelbm_prof.nsys-rep
# (선택) 커널별 grid/block:
nsys stats --report cuda_gpu_kern_gb_sum --format table bench5_purelbm_prof.nsys-rep
```
산출: Time% 상위 커널명(예상: cubic interp 3D, streaming_pull_d3q27, cumulant fused,
einsum→`*gemm*`/`*cutlass*`, reduction/elementwise). **상위 2–3개가 다음 ncu 대상.**

### C.2 ncu — 상위 커널 compute-vs-bandwidth 분류 (무거움, 타겟 한정 필수)
C.1에서 얻은 커널명 정규식으로 몇 launch만:
```bash
ncu --profile-from-start off --launch-count 3 --launch-skip 0 \
    -k "regex:<커널명일부>" \
    --metrics \
gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__occupancy_limit_registers,sm__warps_active.avg.pct_of_peak_sustained_active \
    -o cubic_interp_ncu --force-overwrite \
    python patch_notes/hpc_upgrade/gates/nsys_purelbm_driver.py
ncu -i cubic_interp_ncu.ncu-rep --page details | head -60   # 또는 ncu-ui
```
(GPU 카운터 권한 필요: `sudo`가 아니면 관리자에 `NVreg_RestrictProfilingToAdminUsers=0`
설정 요청. 클러스터 표준이면 그대로.)

### C.3 판정 규칙
각 상위 커널에 대해:
- **DRAM % 높음(≳60–70%) & SM % 낮음** → **bandwidth-bound**. 레버 = 트래픽 감소:
  Phase 1a cell-centred coupling(f_prev/시간보간 제거 = C2F가 읽는 배열↓), 보간
  메모리접근 개선(coalescing, 배열 레이아웃), 필요 정밀도만 이동.
- **SM % 높음 & DRAM % 낮음** → **compute-bound**. 레버 = 알고리즘: cubic→저차 보간
  (경계만 고차?), rescaling einsum/f_eq 연산량 축소, 불필요 재구성 제거.
- **둘 다 낮음(occupancy 낮음)** → **latency/occupancy-bound**. 레버 = block/register
  튜닝, 커널 융합, 작은-region 커널 통합.

## D. 다음 세션 액션
1. 사용자: C.1(nsys) → 커널 랭킹/이름 회신. Claude: 상위 커널 지목.
2. 사용자: C.2(ncu, 상위 커널) → BW%/SM%/occupancy 회신. Claude: C.3로 분류.
3. 분류에 따라 타겟 최적화안 설계(Phase 1a coupling vs 보간 커널 재작성 vs occupancy).
- 후보 방향(분류 무관 공통 유력): **Phase 1a cell-centred coupling** — C2F가 지배적이고
  f_prev+시간보간 제거는 트래픽·메모리·multi-GPU halo 모두 이득(Phase 0 P2 권장).

## E. 이번 세션 산출 (uncommitted)
`src/grid/multi_level_grid.py`(graph scaffolding, 기본 OFF·dormant — 이득 0 확정이나
정확·안전, 제거 여부 사용자 결정), `patch_notes/hpc_upgrade/{09,10}.md`,
`gates/{p1c_s1_graph_gate.py, nsys_purelbm_driver.py}`. **coupling.py 등 수치코드 미변경**
(einsum 변경은 §09 원복 → bit-참조 `ac910ff` 보존).
