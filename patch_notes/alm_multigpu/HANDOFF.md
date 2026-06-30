# HANDOFF — 작업 순서 (2026-06-30 갱신)
다음 세션은 이 파일부터 읽고 **Task 1**부터 순서대로 진행.

## 작업 순서 개요
1. **클러스터 환경 설정** (mpi4py 설치 — multi-GPU 선행 조건)
2. **NASA C81 baseline 분석** (3케이스) — 팁 유도결손 *외* 물리 왜곡 발견
3. **kleine free-wake 수정·검증** (1패널 / 팁마커 only / tight-coupling 반영 확인)
4. **multi-GPU 구현**

---

## Task 1 — 클러스터 환경 설정 (FIRST)
목표: multi-GPU(Task 4) 구현 전에 클러스터 환경 준비.
- 상태: **MPI 설치됨, mpi4py 미설치.**
- 할 일:
  - mpi4py 설치: `MPICC=$(which mpicc) pip install mpi4py` (또는 `conda install -c conda-forge mpi4py`).
  - 검증: `python3 -c "from mpi4py import MPI; print(MPI.COMM_WORLD.size)"` +
    `mpirun -n 4 python3 -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"`.
  - CUDA-aware MPI 확인: `ompi_info | grep -i cuda` (OpenMPI) / `mpirun --version`.
    → **CUDA-aware면 device 포인터 직송, 아니면 host staging** (Task 4 halo 코드 경로 결정).
  - (CUDA-aware면) CuPy↔mpi4py device 버퍼 send/recv smoke.
- 환경: 1노드 **4×RTX4090 24GB, NVLink 없음 → PCIe P2P** (halo 최소화 필요).
- 산출: 환경 확인 로그 + CUDA-aware 여부 기록.

## Task 2 — NASA C81 baseline 분석 (3 케이스)
목표: NASA 덱(깨끗한 LUT) baseline에서 **팁 유도결손을 제외한 다른 물리성 왜곡** 탐색.
("약간 애매한 부분"이 보임 — 그 정체 규명.)
- 케이스: `configs/hvab/hvab_hover_c10_{pureALM,prtipR,epscorr}_nasa.py` (25rev).
- 상태: ~8rev NASA 데이터로 "NASA서도 CT 과대"=팁결손 LUT 아님 확인됨.
  **25rev 완주 후 정밀 분석 필요** (완주 여부 다음 세션서 확인).
- 분석: `hover_fm_post` + `compare_spanwise` (restart-aware). CT/FM + **스팬 분포**
  (하중/AoA/inflow φ/국부 Mach)에서 **팁 외 이상** 탐색 — 예: 중간스팬 하중, 루트 거동,
  내측 분리, 항력 레벨, Mach 보정 영향.
- 산출: 비교표 + "애매한 부분" 정체 규명 (어떤 비물리 왜곡인지).

## Task 3 — kleine free-wake 수정·검증
목표: free-wake를 **팁 전용 최소 모델**로 정리 + **tight coupling 실제 작동 검증**.
3개 항목:
1. **free wake = 정확히 1 wake 패널만** (`n_w`로 제어 — 현재 `_kleine_wake_nw=50` 기본 → 1).
   config에서 제어 가능하게 노출.
2. **free-wake 부착 spanwise 마커 지정 → 팁 마커에만** (애초 목적이 팁). 현재 스팬 전반 적용
   추정 → 팁-only로 제한 (부착 마커 인덱스/반경 지정 옵션).
3. ★**tight coupling 검증**: free-wake가 산출한 유도속도가 **동일 타임스텝의 LBM 결과에
   반영**되는지 확인. 반영 안 되면 LBM↔free-wake 결합이 무의미(free-wake 독자 결과만 사용) →
   tight coupling 되도록 수정. 데이터 흐름 추적: free-wake 유도속도 → ALM 유효속도 →
   force → LBM body force가 **1스텝 내 폐루프**인지.
- 대상: `src/actuator/actuator_line.py` (`_kleine_wake_mode`/`_kleine_wake_nw`/
  `_convect_and_shed_wake`/`_kleine_w_corr`), `src/actuator/smearing_correction.py` (`FreeWake`).
- 산출: 1패널·팁only config + tight-coupling 판정(+ 미반영 시 수정).

## Task 4 — multi-GPU 구현 (LAST, Task 1 완료 후)
목표: ALM multi-GPU → **DGX Watanabe(fine,단일) vs 클러스터 4×4090 비교** (HART2 workshop).
인프라는 long-term SU2 커플(가이드 §3 IN 투자)과 동일 → 재사용.
- 목표규모: Watanabe fine(~207M/~43GB)급 또는 조금 더 fine. 4분할 ~52M/GPU(~15-20GB, 적합).
  ★도메인분할로 per-GPU 최대레벨 <79.5M → int32 ceiling 자동 회피.
- 접근: 공간 도메인분할 + halo 교환. 단계:
  - **4a**: 단일레벨 균일격자 MPI halo (방향별 pack/unpack RawKernel + CUDA-aware send/recv
    또는 host staging). **검증: 단일=다중 bit/물리 일치.**
  - **4b**: MLG 확장 (coarse 분산; fine 레벨 분산 vs **복제**; GPU-경계 coarse↔fine
    bitmask coalescence).
  - **4c**: **ALM across GPU 경계** — 로터 마커 force-spread/velocity-sample이 경계 넘음 →
    로터 위치 gather/scatter. hub_center는 **L0 LU 글로벌 좌표** → 글로벌↔로컬 변환 주의
    ([[feedback_alm_hub_center_lu]]).
- 제약: 4090 PCIe → halo 최소화. multi-GPU config는 DGX Watanabe와 **물리 동일**(분할만 다름).
- 열린 결정: 분할 방향(x-슬랩 vs 3D블록; 로터 중앙→슬랩이면 부하불균형) / CUDA-aware vs host
  staging / fine 분산 vs 복제 / ALM 마커 소유(rank vs 복제+부분합).

---

## 공통 컨텍스트 / 제약
- **int32 fix 완료**(10개 d3q27 커널 64-bit + multi_level_grid f_prev 가드), DGX fine 정상.
  단 **전부 uncommitted working changes** (다음 세션 `git status` 확인).
- 무거운 production run은 사용자가 클러스터 직접 실행([[feedback_simulation_execution]]).
  Claude=config/smoke/구현/검증.
- 다단계 구현=stepwise patch notes([[feedback_stepwise_patch_notes]]). RawKernel ASCII만([[feedback_cuda_kernel_ascii]]).
- 로컬=RTX3090 1대 → 4-GPU 실검증은 클러스터, 로컬은 mpi4py 2-rank(CPU 또는 GPU공유) smoke만.

## 아티팩트 포인터
- 전략 가이드: `docs/SU2+LBM_coupling_guide_v0.{md,pdf}` (§9=고민 체크리스트)
- 메모리절감/multi-GPU 로드맵: `patch_notes/memory_multigpu_roadmap/ROADMAP.md`
- int32 패치: `patch_notes/int32_index_64bit/PLAN.md`
- DGX 기준선 config: `configs/hvab/hvab_hover_c10_fine_watanabe_nasa.py`
- 메모리: [[project_su2_coupling_direction]](long-term 목표) [[project_int32_kernel_ceiling]]
  [[reference_hvab_cfd_benchmarks]] [[project_next_session]]

## 백그라운드
- DGX: Watanabe fine 런(31425 steps) 구동 중. (Task 2 NASA 분석과 별개 기준선; ALM 후순위화로 우선도↓.)
