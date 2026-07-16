# 시뮬레이션 실행 가이드 (단일/멀티 GPU 통합, 2026-07-13 기준)

`main_mpi.py`가 **단일·멀티 GPU 공통 진입점**이다. NR=1이면 halo/통신이 아예 없는
ghost-free 경로로 production 단일과 bit-동일하게 돌고(mpirun 불필요), NR≥2면 분산
러너가 된다. 기존 `main.py`(OutputManager 풀 파이프라인)도 유효하지만, 신규 런은
main_mpi 사용을 권장한다(재시작·VTK·마커·checkpoint 전부 지원, 포맷 동일).

## 1. 기본 실행

```bash
# 단일 GPU (mpirun 불필요! — NR=1은 통신 자체가 없음)
LBM_ESOTERIC=1 python main_mpi.py \
  --config configs/hvab/hvab_hover_c10_farfield40_eso_single0.py \
  --steps 31425 --log-every 64 --dist-init --devices 0 \
  --vtk-every 1257 --ckpt-every 6285 --csv run_single.csv

# 멀티 GPU (rank 수 = GPU 수; OpenMPI+UCX)
mpirun --mca pml ucx -x LBM_ESOTERIC=1 -n 4 python main_mpi.py \
  --config configs/hvab/hvab_hover_c10_farfield40_eso_mpi4.py \
  --steps 31425 --log-every 64 --cuda-aware 1 --dist-init \
  --vtk-every 1257 --ckpt-every 6285 --csv run_mpi4.csv
```

주의: **결과 디렉토리는 config의 run_tag로 결정**된다. 동시에 두 런을 돌리려면
run_tag가 다른 config 2개를 써라(예: `_single0` vs `_mpi4`) — 아니면 VTK/checkpoint가
서로 덮어쓴다.

## 2. 플래그 요약

| 플래그 | 의미 | 권장 |
|---|---|---|
| `--steps N` | 목표 coarse step (절대값; 재시작 시에도 절대) | 1257 = 1 rev (farfield40) |
| `--log-every N` | 진행 로그/CSV 간격 | 16~64 |
| `--dist-init` | 슬랩-스코프 초기화(디바이스 풀필드 무할당). 균일 IC 전용. **obstacle 지원**(solid mask 호스트 마킹; bit 검증) | **항상 켜기**. NR=1 필수급, 대형 케이스 필수 |
| `--devices a,b,..` | node-local rank→GPU id 매핑 | 명시 권장 |
| `--cuda-aware 1` | UCX device-direct (mpirun+UCX 환경) | 클러스터 멀티랭크에서 1 |
| `--vtk-every N` | rank0 조립 VTK+마커 VTP (production 포맷) | rev당 1회 = 1257 |
| `--ckpt-every N` | rank0 조립 checkpoint npz (~10GB/개 @D40) | 6285 |
| `--csv PATH` | 성능 CSV (ALM: thrust/CT/CP/FM, flow: rho/u_max) | |
| `--restart PATH` / `--restart-latest` | checkpoint 재개 (bit-정확: 로터 위상·ramp·parity 연속). dist-init와 병용 불가 | |
| `--verify` | 종료 후 단일-rank 레퍼런스와 owned 조립 대조 (bench5급 전용 — 풀필드 레퍼런스 빌드 필요) | 기능 검증 시 |
| `--strict-bit` | verify를 bit로만 PASS 판정 (순수 LBM) | |
| `--profile` | 섹션별(halo/kernel/alm/coupling) 시간 어트리뷰션 | 성능 진단 시 |
| `--axis x\|y\|z` / `--ghost N` | 분해축/고스트 수동 지정 | 기본 auto/3 유지 |

## 3. 진행 로그 (3-tier 자동)

`step, s/step, ETA`는 항상 출력. 이후는 케이스 자동 감지:
- **ALM 케이스**: `CT, CP, FM` (CSV: step,time_lu,thrust,torque,power,C_T,C_P,FM)
- **고체 경계 케이스**: `CD, CL, CS` — esoteric MEM-force 커널(halfway-BB, 표준경로
  mem_force_d3q27와 동일 규약; owned-배타 누적 + Allreduce = rank-불변). 정규화는
  config `force_calculation.reference`(rho/velocity/char_length/span_length —
  **body 레벨 lattice 단위**; 구는 span=π/4·D로 원면적 인코딩). 진단 tier(atomicAdd)
- **순수 유동**: `rho_mean, u_max` (finest 레벨 owned, rank-집합 정확값)

## 4. 메모리/용량 지침 (실측 기반)

| 구성 | 한도 | 근거 |
|---|---|---|
| 단일 24GB, replicated init | ~105-110M 셀 | 빌드 피크 ~210 B/셀 |
| 단일 24GB, `--dist-init` | ~125M 셀 | 정착 ~187 B/셀 (D40: used 16.8GB) |
| 4×24GB, `--dist-init` | **~450-500M 셀** | rank 정착 ~165 B/셀 × worst-share |
| 호스트 RAM | 17 B/셀 (BC 메타) + checkpoint 시 f 조립 108 B/셀 | |

메모리 검증은 반드시 하드리밋 에뮬레이션으로: `cp.get_default_memory_pool().set_limit(...)`
(WSL2 oversubscription이 OOM을 가린 전례 2건).

## 4b. tier별 스모크 config (동작 확인용, bench5 스케일 ~수 분)

```bash
# ALM tier (CT/CP/FM 라인)
LBM_ESOTERIC=1 python main_mpi.py --config configs/hpc_bench/bench5_purealm_m3.py \
  --steps 4 --log-every 2 --dist-init --devices 0 --verify
# 순수 유동 tier (rho/u_max 라인; 비영 유입 eq/sponge + dist-init u0!=0 경로)
LBM_ESOTERIC=1 python main_mpi.py --config configs/hpc_bench/bench_flow_uniform.py \
  --steps 4 --log-every 2 --dist-init --devices 0 --verify
# 고체 경계 tier (sphere HWBB, 5-level; CD/CL/CS 라인 + 컷-관통-solid bit; dist-init 가능)
LBM_ESOTERIC=1 python main_mpi.py --config configs/hpc_bench/bench_sphere_hwbb.py \
  --steps 4 --log-every 2 --dist-init --devices 0 --verify
```
셋 모두 `[verify] RESULT: PASS` + 전 레벨 bit=True가 정상. body tier의 진행 라인은
CL/CD 배선 전까지 rho/u_max 폴백(SOLID 마스킹 적용 — 미마스킹 시 미초기화 u가 새는
버그를 sphere 스모크가 검출·수정함).

## 5. 검증 게이트 (수정 후 실행 순서)

```bash
python patch_notes/hpc_upgrade/gates/eso_gather_scatter_gate.py   # 프리미티브 왕복 bit
python patch_notes/hpc_upgrade/gates/mgpu_m2b_gate.py             # 5레벨 MLG bit
python patch_notes/hpc_upgrade/gates/mgpu_m3_gate.py              # 분산 ALM
python patch_notes/hpc_upgrade/gates/mgpu_verify_gate.py          # --verify 진입점
python patch_notes/hpc_upgrade/gates/mgpu_restart_gate.py         # 재시작 왕복 bit
python patch_notes/hpc_upgrade/gates/eso_bench5_alm_smoke.py      # 물리 CV-band
python patch_notes/hpc_upgrade/gates/eso_mem_force_twin_gate.py   # MEM force 쌍둥이 (std↔eso, R3-2)
python patch_notes/hpc_upgrade/gates/cyl2d_re100_gate.py          # 2D end-to-end Cd 밴드 (~100s, R3-3)
```
규칙: 커널/커플링/러너 수정 → 전체. ALM만 → M3+smoke. 출력만 → verify 게이트.
2D(D2Q9) 경로 수정 → cyl2d 게이트. 고체경계(HWBB/eso solid) 수정 → twin 게이트 + sphere 스모크.
ALM 정규화 커널(alm_kernel/spreading/interpolation/보정) 수정 → β 게이트 3종:
`patch_notes/alm_beta_kernel/gates/gbeta{0,2,3}_*.py` (추상화 bit / 보정 유도 / bench5 A/B).

## 6. 알려진 한계 (fail-fast로 명시됨)
- `--dist-init`: 비균일 IC 미지원, restart 병용 불가 (obstacle은 지원됨)
- 2D D2Q9(익형 α-sweep 등)는 `main.py` 경로 유지 — nu-마이그레이션 완료(56개 config).
  hover-ALM 레거시 9개(configs/alm 구형)는 U_inf=0이라 수동 nu(팁속도 기준) 필요 상태로 유지
- 분산 ALM: kleine free-wake·비gaussian 샘플러 미지원 (production은 straight)
- `--verify`: 레퍼런스가 풀필드 빌드라 D40급에서는 메모리상 부적합 (bench5급 전용)
- mpirun 없이 멀티랭크 불가 (NR=1만 plain python 지원)

관련 문서: 설계·검증 = `docs/MULTIGPU_DESIGN_PHILOSOPHY_kr.md`,
전 단계 로그 = `patch_notes/hpc_upgrade/17_multigpu_design.md`,
클러스터 절차 = `patch_notes/hpc_upgrade/18_m5_cluster_runbook.md`.
