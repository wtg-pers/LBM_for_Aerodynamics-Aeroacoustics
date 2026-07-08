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

### Task 1 진행 (2026-06-30)
- **로컬(DESKTOP-VAR836F, RTX3090, conda env `lbm_study`)**: mpi4py 4.0.3·cupy 13.6.0 **이미 설치됨**.
  MPI = conda **MPICH 4.3.2 (ch4:ofi), CUDA-aware ✗** (`mpichversion` configure에 `--with-cuda` 없음).
  검증 OK: single size=1, 4-rank 출력, **host-staging CuPy halo PASS**; device-direct **SIGSEGV(예상)**.
- **★클러스터 설치 caveat (중요)**: conda-forge mpich = NOT CUDA-aware. 클러스터에 CUDA-aware 시스템 MPI가
  있어도 `conda install -c conda-forge mpi4py`는 그걸 덮어써 CUDA-aware를 잃음 → **클러스터에선 시스템 MPI에
  소스빌드**: `MPICC=$(which mpicc) python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py`.
- **검증 스크립트**: `patch_notes/alm_multigpu/task1_cluster_verify.py` (scp 후
  `mpirun -n 4 python3 ...` host-staging / `TEST_DEVICE_DIRECT=1 mpirun -n 2 ...` 로 CUDA-aware 판정).
- **남음(클러스터에서)**: ① 위 pip 소스빌드로 mpi4py 설치 ② verify 스크립트로 CUDA-aware 판정 →
  Task 4 halo 경로(device 직송 vs host staging) 확정.

### Task 1 클러스터 결과 (2026-06-30) — anode1, 4×4090
- **★공유 env 제약**: sonar(음향, `to_claude/sonar/`)와 LBM이 **동일 conda env 공유**(별도 venv 없음).
  sonar 멀티-GPU = **multiprocessing spawn + `CUPY_DEVICE` 핀, MPI 미사용** → mpi4py와 무충돌.
  유일 위험=env churn → **pip 소스빌드 `--no-binary=mpi4py --no-deps`로 격리**(mpi4py deps 0개).
- **설치 성공**: `mpi4py 4.1.2`, 시스템 **Open MPI v5.0.5**(root@anode0)에 링크. env 무손상 확인
  (numpy 2.0.2 / scipy 1.13.1 / cupy 13.6.0 그대로). 4-rank wiring OK.
- **★UCX/THREAD_MULTIPLE 이슈**: `pml_ucx.c:329 UCP worker does not support MPI_THREAD_MULTIPLE,
  PML UCX could not be selected` → OMPI5의 **CUDA-aware는 UCX PML 경유**인데, mpi4py 기본
  THREAD_MULTIPLE 요청 → MT 미빌드 UCX가 거부 → ob1 폴백(=CUDA-aware 빠짐). **해결: halo는
  rank당 단일스레드라 `mpi4py.rc.thread_level='serialized'`(또는 env `MPI4PY_RC_THREAD_LEVEL=serialized`)
  로 낮추면 UCX 선택됨.** verify 스크립트에 박음.
- **★Task 1 종료 판정 (CUDA-aware = YES)**:
  - serialized 적용 시 UCX 경고 소멸(UCX PML 선택). `ompi_info`: `mpi_built_with_cuda_support:value:true`,
    configure `--with-cuda=/usr/local/cuda --with-ucx=/usr/local`, accelerator:cuda · btl:smcuda · pml:ucx ·
    ext cuda/rocm. UCX 1.18.0 `--with-cuda` 빌드.
  - **device-direct 실측 PASS**(`RESULT host_staging=PASS device_direct=PASS`, GPU0↔GPU1, NVLink無 PCIe).
  - **→ Task 4 halo = CuPy device 포인터 직송**(UCX가 GPUDirect P2P/IPC 투명 처리). host staging 불요.
  - **운영 필수**: ① `mpi4py.rc.thread_level='serialized'`(import 전) 안 하면 UCX 안 떠 CUDA-aware 꺼지고
    ob1 느려짐 → production 런처에 박을 것. ② NVLink無 PCIe → halo 부피 최소화.
  - 환경 경로: 클러스터 LBM 루트 `~/00_LBM_solver/`, verify 스크립트 `~/00_LBM_solver/alm_multigpu/`.

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

### Task 2 결과 (2026-06-30) — ✅ 완료
- 데이터: `aeromechanics_workshop/HVAB/260630_results_nasa_c81/{pureALM,dag,prandtl_Rtip}_csv`
  (3케이스 25rev **연속 완주**, restart 무손상). 산출물: `aeromechanics_workshop/HVAB/task2_nasa_analysis/`
  (FINDINGS_kr.md + 재현 스크립트) + `hvab_nasa_c10_task2_annotated.png`.
- **적분**: CT 0.0104~0.0105(GT Fluent 0.00832 대비 **+26% 과대**), **FM 0.747~0.752**(이전 NeuralFoil
  0.60대보다 건강, 깨끗한 25rev). 보정 3종 적분차 미미(light 격자).
- **★"애매한 부분" 정체 = 익형 전환 r/R 0.825(RC4-10→RC6-08)의 sectional CL −16% 계단 불연속.**
  C81 직접조회로 Mach 보간 아티팩트 *반증*(RC4-10 내부 매끈) → 익형 차이 확정(동일 α/M서 RC4-10 0.475 vs
  RC6-08 0.400). 원인=`blade._get_airfoil_at_r` piecewise-constant 익형 배정(**스팬 블렌딩 없음**).
  실제 lofted 블레이드/NASA sectional은 매끈 → 0.825 노치가 비물리 아티팩트. (0.675 경계는 RC4-12=placeholder라
  무계단, 0.975는 팁과부하에 가림.) **저후회 fix=경계 ±Δr/R 인접 덱 CL/CD 선형 블렌드.**
- 부차: 루트 컷아웃 첫마커 α9°/CL1.04 스파이크(단정보류), 팁 CD 0.058 폭증(M0.65 압축성+과부하 복합).
- 덱 caveat: RC4-12(placeholder)·RC6-08T(tab근사)=내측40%·팁 근사 덱. CT+26% 과대 주범은 여전히 **F1 팁 유도결손**(F2 아님).

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

### Task 3 진행 (2026-06-30) — 코드 정밀독 완료
- **item 3 (tight coupling) ✅ 검증=작동**: `actuator_line.py` step() → `_compute_bem_forces`서
  `u_n += w_corr`(:819) → `recompute_velocity_triangle`(:821) → 폴라 재조회(:826) → 같은 스텝
  F→`spread`→body force→LBM. free-wake convect는 CFD속도(:653, Kleine §3.4). 독자결과 아님. **수정 불요.**
- **item 1 (n_w 노출) ✅ 이미 됨**: 로더 `:1414` `ec.get('n_w',50)`. ⚠️ n_w=1이면 free-wake 비활성
  (`:568` len(wake)>=2 필요→straight fallback). "1패널" 의도 재확인 필요.
- **item 2 (팁-only shedding) ✅ 구현·검증(2026-06-30, Q2)**: config `eps_correction.wake_markers`
  (all|tip|N|f|idx). `_shed_idx`+subset shed + `_kleine_w_corr` free분기 `G[shed_idx,:]`·`eps[shed_idx]`·
  dr축 정합(array dr 버그도 수정). 검증: tip→ring 1점(단일필라멘트), all+scalar byte-identical(maxabs 0).
  config `hvab_hover_c10_kleine_free_tip.py`. 상세 `patch_notes/alm_marker_distribution/04_q2_freewake_tiponly.md`.
- **★병행: ALM 근본문제 audit 문서 신설** `docs/alm_fundamental_issues_audit_kr.md` (파이프라인 6단계,
  Task2 익형블렌딩=F2 등 카탈로그, 사용자와 하나씩 진행).

## Task 4 — multi-GPU 구현 (LAST, Task 1 완료 후)
목표: ALM multi-GPU → **DGX Watanabe(fine,단일) vs 클러스터 4×4090 비교** (HART2 workshop).
인프라는 long-term SU2 커플(가이드 §3 IN 투자)과 동일 → 재사용.
- 목표규모: Watanabe fine(~207M/~43GB)급 또는 조금 더 fine. 4분할 ~52M/GPU(~15-20GB, 적합).
  ★도메인분할로 per-GPU 최대레벨 <79.5M → int32 ceiling 자동 회피.
- 접근: 공간 도메인분할 + halo 교환. 단계:
  - **4a**: 단일레벨 균일격자 MPI halo (방향별 pack/unpack RawKernel + **CUDA-aware device-direct
    send/recv** — Task 1서 확정, host staging 불요). thread_level=serialized 필수. **검증: 단일=다중 bit/물리 일치.**
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
