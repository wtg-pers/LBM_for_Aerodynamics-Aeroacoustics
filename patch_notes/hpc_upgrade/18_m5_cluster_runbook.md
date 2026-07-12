# M5 클러스터 실런 런북 (anode1 4×4090)

작성: 2026-07-12. 로컬(WSL2, 1 GPU, MPICH host-staged)에서 실 `mpirun` 2-rank가
bench5 pure-ALM/archB 양쪽 **전 레벨 bit-identical** 확인된 상태. 이 문서는
클러스터에서 사용자가 수행할 검증 시퀀스.

## 0. 환경
- task1에서 검증한 OpenMPI 5.0.5 + UCX(CUDA-aware) 환경 그대로 사용.
- `mpi4py.rc.thread_level='serialized'`는 `main_mpi.py`가 import 전에 설정함(수동 조치 불필요).
- CUDA-aware 경로: `--cuda-aware 1` (UCX device-direct). 문제가 생기면
  `--cuda-aware 0`(host-staging)으로 폴백 — **정확도는 동일**, 대역폭만 손해.

```bash
# 권장 mpirun 공통 옵션 (OpenMPI+UCX)
MPIRUN="mpirun --mca pml ucx -x LBM_ESOTERIC=1"
```

GPU 지정: 기본은 node-local rank → 0,1,2,... 순서 자동 배정.
특정 GPU를 쓰려면 `--devices` (node-local rank 순서대로 매핑):
```bash
# 예: 2-rank를 GPU 0,1에            예: GPU 2,3만 사용
... main_mpi.py --devices 0,1 ...   ... main_mpi.py --devices 2,3 ...
```
(`CUDA_VISIBLE_DEVICES`로도 가능하지만 런처별 env 전파 문법이 달라
— OpenMPI `-x`, MPICH `-genv` — CLI 플래그를 권장.)

## 1. 기능 검증 (bench5, 수 분)
```bash
# (a) 2-rank + ALM, bit 기대
$MPIRUN -n 2 python main_mpi.py \
  --config configs/hpc_bench/bench5_purealm_m3.py \
  --steps 2 --log-every 1 --cuda-aware 1 --verify

# (b) 4-rank 순수 LBM: 4-rank에서도 STRICT BIT 기대 (halo/coupling은 전부 exact)
$MPIRUN -n 4 python main_mpi.py \
  --config configs/hpc_bench/bench5_pure_lbm.py \
  --steps 2 --log-every 1 --cuda-aware 1 --verify

# (c) 4-rank + ALM: max|df| < 1e-4 기대 (bit 아님 — ALM 부분합의 MPI
#     Allreduce 결합순서가 4-rank에서 단일합과 재결합 차이 가능; 프로토콜상
#     마커별 fp last-bit 수준)
$MPIRUN -n 4 python main_mpi.py \
  --config configs/hpc_bench/bench5_purealm_m3.py \
  --steps 2 --log-every 1 --cuda-aware 1 --verify
```
판정: (a)(b) `bit=True` 전 레벨, (c) `RESULT: PASS`.

## 2. 실 케이스 (farfield40 D40 case1, 4-rank)
드라이 체크 결과(로컬 검증): axis=y 자동선택, L0 bounds=[0,108,121,133,240],
L4 owns=[148,208,192,133], worst-rank share 0.266 (이상 0.250, 밸런스 효율 94%).
빌드 피크 ~19.2GB/GPU(초기화 일시), 러너 구축은 피크 미증가(뷰 추출+레벨별 원본
해제 — 최초 버전의 레벨별 f 복사 OOM은 수정됨), 정착 3.7~5.0GB/rank(로컬 D40 프로브).

```bash
# 1-rev 스모크 (1257 coarse steps ≈ 단일GPU 대비 ~3.5× 빠를 것으로 기대)
# 전용 config (run_tag=farfield40_eso_mpi4 — 단일GPU case1 결과와 디렉토리 충돌 방지)
$MPIRUN -n 4 python main_mpi.py \
  --config configs/hvab/hvab_hover_c10_farfield40_eso_mpi4.py \
  --steps 1257 --log-every 16 --cuda-aware 1 \
  --csv mpi4_case1_rev1.csv

# 좋으면 풀런 (25 rev) — VTK/checkpoint는 production 포맷으로 rank0 조립
$MPIRUN -n 4 python main_mpi.py \
  --config configs/hvab/hvab_hover_c10_farfield40_eso_mpi4.py \
  --steps 31425 --log-every 64 --cuda-aware 1 \
  --vtk-every 1257 --ckpt-every 6285 \
  --csv mpi4_case1_full.csv
```
(성능 4-pass 후 실측 0.671 s/step → 25 rev ≈ 5.9h. VTK ~1.7GB/스냅샷·rev당 1회,
checkpoint ~10GB/개·6285步마다 — 디스크 ~60GB 확보. 출력 시점에 rank0로 host-staged
gather가 돌므로 해당 step은 수 초 느려짐.)
비교: 단일GPU case1 CSV의 같은 rev 구간과 thrust 곡선 겹쳐보기
(ramp 구간이라 초기 일치 뚜렷해야 함; CV-band ±3% 이내).

성능 판독:
- `[mpi] done: ... s/step` — 단일GPU D40 ≈3.1 s/step 대비 목표 ~0.85-1.0 s/step.
- GPU-Util 4장 모두 80%+ 인지, 특정 rank만 놀지 않는지(`nvidia-smi dmon`).

스모크가 좋으면 풀런: `--steps 31425` (25 rev).

## 3. 보고 항목
1. §1 (a)(b)(c) verify 출력 전체
2. §2 `[mpi]` 로그 + CSV + s/step + `nvidia-smi` 스냅샷
3. 이상 시: 전체 traceback (`comm.Abort` fail-fast가 걸려 있어 행 없이 죽음)

## 4. 현재 한계 (M5 스코프)
- ~~thrust CSV만~~ → **VTK(.vti/.vth)·checkpoint(npz) rank0 조립 지원**(--vtk-every/
  --ckpt-every; production 포맷 동일 = 기존 분석스크립트·단일GPU restart 그대로 사용 가능).
  잔여: MPI 러너 자체의 --restart 재개 배선, ALM 마커 VTP
- kleine free-wake·비gaussian 샘플러 분산 미지원(fail-fast) — production은 straight
- 케이스가 단일 GPU 메모리를 초과하면 분산 초기화 필요(후속; D40은 해당 없음)
- v2 slot halo(M4)는 게이트 검증만, production 미결합 — 강스케일링 실측 후 결정
