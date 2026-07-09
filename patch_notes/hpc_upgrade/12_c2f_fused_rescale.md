# Target 2 — C→F rescale 융합 (fused RawKernel) — 2026-07-08

nsys(11 §A)의 최대 잔여 = **cupy 원소별 rescaling 체인 39.5%**(unfused). C→F의
temporal interp + macroscopic + f_eq + reconstruct를 **1 per-node RawKernel**로 융합해
read-once/write-once로 만들어 bandwidth 직격. **CV-band 등가**(fp32 last-bit).

---

## A. 배경 (왜 여기가 병목인가)
`coupling.coarse_to_fine`의 재스케일은 cupy 원소별 연산의 긴 체인:
```
rho=Σf(sum) · mom=Σc·f(einsum→sgemm) · u=mom/rho(div) · cu=c·u(einsum→sgemm)
· f_eq=w·rho·(1+cu/cs2+...)(다수 add/mul/div) · out=f_eq+factor·(f−f_eq)(sub·mul·add)
```
각 연산이 full (Q,region) 배열을 DRAM 왕복 → ~1 FLOP/2-3 word = **bandwidth-bound**.
`coupling.py`가 이미 주석으로 "macro/f_eq decomposition on the full coarse sub가 C→F
병목"이라 명시(_C2F_BOUNDARY_ONLY 기본 OFF 근거)한 바로 그 지점.

## B. 구현
- **`src/kernels/coupling_rescale_d3q27.py`** (신규): `CouplingRescaleKernelD3Q27`.
  per-node 커널(thread=spatial node, q=0..26 unroll, `#define NQ 27`로 f[27] 레지스터
  상주). f(+f_prev, half_step) 읽어 rho/u/f_eq 레지스터 계산 → `out=f_eq+factor·(f−f_eq)`
  1회 write(in-place 가능). D3Q27 fp32 전용, `--use_fast_math`.
- **`src/grid/coupling.py`**: `__init__`서 `_fused_rescale` 생성(cupy + Q27 + fp32,
  env `COUPLING_FUSED_RESCALE` 기본 **1**). `coarse_to_fine` 2-4단계를 fused 호출로 교체;
  **Python 경로는 정확한 fallback로 보존**(env=0 또는 비cupy/비Q27/비fp32). F2C·
  `_compute_macroscopic/_compute_f_eq`는 미변경(F2C는 stencil filter 포함=별건).

## C. 검증 (로컬 RTX3090)
### C.1 per-call 등가 + 속도 (`gates/coupling_fused_rescale_gate.py`)
- half=F/T 모두 **max|Δ|=8.9e-8, max rel=3.0e-7** = fp32 last-bit(**CV-band 등가**).
- rescale wall(30×150×150): python 5.79ms → **fused 0.40ms = 14.4×**(base.copy 포함).
### C.2 end-to-end (bench5_pure_lbm, 5 coarse step, fused vs Python)
- 전 레벨 **rel ~3.7e-6, finite=True**, **질량 보존**(L0 129024.008 동일). 4/4 coupling 활성.
### C.3 성능 A/B (bench5_pure_lbm, MLG_PROFILE 12-step)
| | fused OFF(z-fix만) | fused ON | |
|---|---|---|---|
| C2F.L4 | 186.5ms | **101.6ms** | 1.84× |
| C2F.L3 | 67.5ms | 37.9ms | 1.78× |
| pure-LBM TOTAL | 515ms | **390ms** | **1.32×** |

→ z-fix 포함 누적 **996→390 = 2.55×**(3090). 4090은 C2F.L4 117.8→~64ms 예상
(L2 흡수로 3090보다 배율 작을 수 있음, 클러스터 확인).

## D. 게이트 정책 & 영향
- **CV-band 등가**(3.7e-6). 4090 pure-LBM sha는 애초 run-to-run 비결정(11 §E.2)이라
  **재기준선 걸림돌 없음** — bit-gate 아닌 conservation/C_T/MLUPS로 판정.
- ALM 런(bench5_baseline)도 C→F 공유라 LBM 부분(~15%) 이득 + coupling sha 이동(무관, CV-band).
- 기본 ON. A/B용 `COUPLING_FUSED_RESCALE=0`으로 Python 경로 복귀(정확한 이전 동작).

## E. ★클러스터 검증 = 완료 (4090, bench5_pure_lbm, fused ON vs OFF)
| 섹션 | fused OFF(z-fix만) | fused ON | 배율 |
|---|---|---|---|
| C2F.L4 | 117.44ms | **70.77ms** | **1.66×** |
| C2F.L3 | 41.98ms | 26.21ms | 1.60× |
| pure-LBM TOTAL | 332.9ms | **264.2ms** | 1.26× |
| MLUPS | 352.8 | **444.0** | 1.26× |
| conservation | −0.0000% | **−0.0000%** | 동일(물리 보존 ✓) |

conservation이 OFF와 완전 동일 = **CV-band 물리 보존 확정**. 4090 세션 누적(원본→z-fix
→fusion): pure-LBM **391→264ms(1.48×)**, MLUPS **302→444**. 두 최적화 모두 production
GPU 검증 완료. (3090 로컬은 996→390=2.55×; 4090이 배율 작은 건 L2 72MB 흡수.)
재현 커맨드:
```bash
MLG_PROFILE=1 python -u main.py --gpu 0 --config configs/hpc_bench/bench5_pure_lbm.py 2>&1 | tee pl_fused.log
COUPLING_FUSED_RESCALE=0 MLG_PROFILE=1 python -u main.py --gpu 0 --config configs/hpc_bench/bench5_pure_lbm.py 2>&1 | tee pl_nofuse.log
```

## F. 남은 병목 & 다음
- F2C rescale(pointwise 부분) 융합 = Target 2b(filter는 stencil이라 pre/post 2커널로
  분리 융합). F2C.L4 40ms → 소폭. 후순위.
- cubic_interp x/y(coalesced지만 9.6%) — 잔여.
- 이후 Phase 1a cell-centred coupling(구조적) → Phase 2 multi-GPU.

## G. 변경 파일 (이번 세션 누적, uncommitted)
신규 `src/kernels/coupling_rescale_d3q27.py`, `gates/coupling_fused_rescale_gate.py`.
수정 `src/grid/coupling.py`(fused C2F rescale + fallback). 기존:
`src/kernels/interpolation_d3q27.py`(z-fix), `src/grid/multi_level_grid.py`(graph OFF),
notes 09-12 + gates. **coupling `_compute_*`는 F2C용으로 유지, einsum 원복 상태 그대로.**
