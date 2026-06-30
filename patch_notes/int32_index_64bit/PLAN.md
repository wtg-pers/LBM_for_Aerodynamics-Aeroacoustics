# Patch: D3Q27 커널 population 인덱스 64-bit화

## 문제
fine preset(D40, 5-level) DGX 실행 시 step0에서 `CUDA_ERROR_ILLEGAL_ADDRESS`
@ `coupling.py coarse_to_fine` `f_coarse[...].copy()`. **OOM 아님**(42.8GB 할당 성공).

## 근본 원인 (확정)
모든 D3Q27 RawKernel이 `int idx = ...; f[q * N + idx]` (N=Nx·Ny·Nz, q∈[0,26]).
`q*N`이 **int32**라 한 레벨이 **N ≥ 2³¹/27 ≈ 79.54M 셀** 이면 `q*N+idx`가 2³¹ 초과 →
wrap(음수) → illegal access. fine **L4=120M > 79.5M** → 크래시.
- 비동기라 실제 결함(L4.advance: cumulant/dyn_smag/mem_force)이 다음 동기점에서 표면화.
- `fine_mini`(D16, fine와 동일 5-level 비율, L4=7.7M < 79.5M) 로컬 통과 → 5-level 로직 무죄,
  **크기 의존** 단독 확정. 임계 79.5M가 "기존 최대레벨 ≤52M OK, fine L4=120M만 crash"와 일치.

## 영향 파일·사이트 (레벨당 >79.5M서 전부)
`src/kernels/*_d3q27.py` 중 `q * N + idx` / `q * NxNyNz` / Q범위 `tid` 패턴:
- cumulant_d3q27.py (L92, 397, 402; raw-string 변형 L536, 543)
- macro_d3q27.py (L57)
- bgk_d3q27.py (L82, 150, 208, 265, 332, 389)
- streaming_d3q27.py
- dyn_smag_d3q27.py (L26~)
- wale_d3q27.py (L24~)
- mem_force_d3q27.py (L32~)
- bounce_back_d3q27.py (L46, 48, 71, 72, 78, 109)
- interpolated_wall_d3q27.py (L72, 73, 90, 94, 120)
- interpolation_d3q27.py (L40, 51, 102, 107, 146, 150, 186, 190) — `int tid/rem`이 Q·N=3.25e9 범위, **곱셈 전부터 overflow**

## 수정 전략
Q 차원이 곱해지는 **flattened 인덱스만** 64-bit. cell-local(ix/iy/iz, src_idx≤N)은 int32 유지.
- `f[q * N + idx]` → `f[(long long)q * N + idx]`
- interpolation_d3q27: `int tid` → `long long tid`, `int rem` → `long long rem` (q*NxNyNz 빼는 식)
- ASCII 유지 (RawKernel raw string non-ASCII 금지 규칙).
- cupy 자체 연산(.copy 등)은 size_t → 무관, 손대지 않음.

## 검증 (로컬 RTX3090 24GB)
1. **before**: `ovf87` preset (D76, **단일레벨**, 87.78M 셀 > 79.5M, SGS off) →
   `CUDA_LAUNCH_BLOCKING=1` 로 illegal address 재현 + 실제 결함 커널 직접 확인.
2. **after**: 동일 repro 통과(2~3 step, drift 정상).
3. **regression**: `light`(production 기본) + `fine_mini`(5-level) 결과 **bit-identical** (인덱스 타입만
   바뀜 → 수치 무변). T_lu/C_T 동일 확인.

## 결과 (2026-06-29 완료)
- [x] **before 크래시 재현**: `ovf87`(D76 단일레벨 87.78M) → `CUDA_ILLEGAL_ADDRESS`
      @ `streaming_d3q27.py:112` (CUDA_LAUNCH_BLOCKING=1로 결함 커널 직접 확인).
      + 단일레벨 선결버그(`multi_level_grid.py:153` `_f_prev[0]` None) 가드 추가.
- [x] **64-bit 수정 적용** (10개 d3q27 커널):
      - param N: `const int N`→`const long long N` + launch `cp.int32(N)`→`cp.int64(N)`
        (cumulant, macro, mem_force, bgk×2, bounce_back×3, interpolated_wall)
      - local N: `int N=Nx*Ny*Nz`→`long long N=(long long)Nx*Ny*Nz`
        (streaming, dyn_smag, wale, bgk-streamvariant, interpolated_wall-link)
      - interpolation: `int NxNyNz`→`long long NxNyNz` (q*NxNyNz 승격). cell-local(idx/ix/iy/iz)는 int 유지.
      - 검증 grep: `const int N`/`int N=Nx`/`cp.int32(N)` 잔존 0.
- [x] **after 통과**: `ovf87`(87.78M) exit0, drift +0.000%. (수정 전 동일 케이스 크래시)
- [x] **회귀 bit-identical**: `fine_mini`(D16 5-level, SGS on, ALM) **T_lu=9.699473 /
      Q_lu=48.347881 / P_lu=0.037772 수정 전과 완전 동일** → 수치 무변 증명
      (N<79.5M서 int/longlong 산술 동일). 완전성 커널(bounce_back/interpolated_wall/bgk×3)
      RawModule 컴파일 OK.

## 남은 사항
- 완전성 커널(bgk/bounce_back/interpolated_wall)은 HVAB ALM 미사용 → 런타임은
  obstacle 케이스(cylinder/sphere) >79.5M에서 최종 확인. 컴파일은 통과.
- DGX `fine`(L4=120M) 재실행이 >79.5M 실사용 최종 검증. 잔여 리스크는 int32와 무관한
  L3→L4 nesting sep 0.12D(<0.15D 가이드) 단 하나 — NaN시 fine L3 lat 0.7→0.73.
- DEBUG 자산 유지: preset `fine_mini`/`ovf87`, config `_finemini_repro.py`/`_ovf87_repro.py`.
