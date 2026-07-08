# Phase 1a — Stage A: boundary-only C→F — 2026-07-06

01_phase1a §2 (I1) 구현. **bit-identical, vertex 유지, 저위험** 속도 최적화.
Stage B(cell-centred + f_prev 제거)와 독립. 상태: **로컬 검증 완료, 클러스터 게이트 대기**.

## 무엇을 / 왜

Phase 0에서 C2F.L4 = 158 ms(9.9 ms/call ×16)가 pure-LBM의 단일 최악으로 측정됨.
원인(01_phase1a §2 I1): `coarse_to_fine`이 `_upsample_to_fine`으로 **fine_domain
전체** 배열 `zeros((Q,Nx_f,Ny_f,Nz_f))`을 할당·zero하고 3개 보간 커널을 전체볼륨에
돌리지만, **실제로는 6개 경계 strip(ow_f=4셀)만 기록**. L4=5M셀 생성→~1M 사용,
540 MB zero ×16/step.

## 어떻게 (bit-identical 근거)

C→F는 fine 경계 strip만 쓰므로 **면별로 얇은 coarse slab만 upsample**:
- 면-법선축으로 `cw = ow_f/2 + 2` coarse 노드, 접선 2축은 full.
- `cw`가 cubic 스텐실 반경(3 fine = 1.5 coarse)을 덮고, slab 외곽 edge가 실제
  도메인 경계와 일치 → 경계 스텐실도 일치 → **기록 strip이 전체볼륨과 바이트 동일**.
- fine 내부(=fine 솔버 소유, C→F 미기록)는 계산 자체를 건너뜀.

## 변경 파일

- `src/grid/coupling.py` (3D만):
  - `_upsample_to_fine` → **`_upsample_block(coarse_nodes)`** (임의 블록 upsample 프리미티브).
  - **`_upsample_boundary_into(coarse_nodes, f_fine)`** 신규 (면별 slab 기록).
  - `__init__`: `_bnd_face_specs` 6면 precompute (coarse-slab/fine-write/slab-read slice).
  - `coarse_to_fine`: boundary-only 경로 + full 폴백.
  - 모듈 플래그 `_C2F_BOUNDARY_ONLY` (env **`MLG_C2F_FULL=1`**로 전체볼륨 복귀=디버그).
- `coupling_2d.py`는 **미변경**(테스트 전용, Stage B서 3D/2D 동시 처리).

## 로컬 검증 (완료)

- **bit-identity 스윕** (`scratchpad/phase1a_stageA_{ab,robust}.py`, numpy CPU=실제 3D 코드):
  6 지오메트리 × {full, half} = **12/12 Δ=0.00e+00**. 커버: normal, **clipped min/max**
  (경계 접촉), **thin region**, **overlap_width 1/2/3**, half-step(시간보간)·full-step.
- **속도** (CPU numpy, L4형 thin-slab aspect fine 35×175×175): full 504 ms → boundary
  168 ms = **3.00×**. cubic 레벨(L2/L3)은 더 큼(shell 비율 작음); GPU는 540MB zero
  회피로 추가 이득 기대.

## 클러스터 게이트 (사용자 실행) — ★강한 gate

Stage A는 **완전 bit-identical**이므로 bench5 전체 1006-step 체크포인트가
**reference와 바이트 동일**해야 함:

```bash
MLG_PROFILE=1 python main.py --config configs/hpc_bench/bench5_baseline.py 2>&1 | tee bench5_stageA.log
sha256sum result_.../checkpoints/checkpoint_00001005.npz   # == ac910ff9…b3b3e61 이어야 통과
```
- **bit 게이트**: sha256 == `ac910ff914403acb20f958e90dc28e197ec0bc6e796012703e02a479fa3b3e61`
  (00_phase0 §A). GPU CUDA interp도 slab/full 동일 스텐실→동일 결과 예상, sha256이 확증.
- **속도 게이트**: `[MLG_PROFILE]`의 **C2F.L4 158 ms → 대폭↓**, 전 레벨 C2F 합 232 ms↓,
  TOTAL(pure-LBM 391 ms 대비) 확인.
- sha256 불일치 시: `MLG_C2F_FULL=1`로 회귀 재현 후 원인 격리.

## 다음

1. 위 클러스터 게이트(bit sha256 + C2F 타이밍) 통과 확인.
2. 통과 시 → **Stage B**(cell-centred explosion/coalescence + f_prev 제거, 01_phase1a §3).
   2D testbed 먼저(시간보간 제거 정당성 검증), coupling_2d 포함.
