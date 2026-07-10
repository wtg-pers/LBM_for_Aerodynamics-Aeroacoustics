# 15 — Phase 1b: Esoteric Pull 부활 + cumulant 재구현 (메모리 −91 B/node)

> Status: **DESIGN (구현 전)** · 2026-07-10
> 결정: 메모리 트랙 = **1b esoteric(Pull) → 멀티GPU** ([[feedback_long_term_structural]] 정도). cell-vertex 유지(결정 B).
> 정독 근거: `docs/ESOTERIC_PULL_STATUS.md`(blueprint) + git `8036317` (BGK Pull 커널+통합, 정독 완료).
> 목표: f_post 제거로 D40(91.6M)을 단일 24GB에 (~20.7GB, 여유 ~3GB). 정확도(f32) 유지, 음향 로드맵 무해.

---

## 0. 정독으로 확정된 출발점
- git `8036317`은 **이미 Esoteric Pull (Lehmann 2022)** — `src/kernels/esoteric_d3q27.py`(387줄, BGK Pull) +
  통합(`simulation.py` `_init/_advance_esoteric`·node_type BC·dispatch, `stream.py`, `initializer.py` parity,
  `output_manager.py` checkpoint, MLG gather/scatter 래핑). **과거 검증**(Sphere Cd 0.08%, MLG rho<3e-6, checkpoint parity).
- **cumulant Esoteric Pull = never committed=유실** → **재구현**. (blueprint의 `esoteric_cumulant_d3q27.py`·`esoteric_d2q9.py`
  는 커밋 안 됨.)
- Pull 메커니즘(단일버퍼): paired-opposite 방향(i,i+1 for i=1,3,..,25; rest=slot0), `is_odd=t&1`로 LOAD/STORE slot swap
  (odd: fhn[i]=f[i]@self, fhn[i+1]=f[i+1]@nbr_i; even: 반대), BC를 커널 내부 `node_type`으로, MLG는 gather/scatter로
  coupling에 **standard f** 제공(coupling.py 무변경). 참고: `scratchpad/esoteric_8036317_kernel.py`,
  `scratchpad/esoteric_8036317_integration.diff`.

## 1. 구현 단계 (a→d) — 각 단계는 **로컬 3090 게이트 통과 + 패치노트 체크포인트** 로 종료
### (a) BGK Pull 부활 + 현 코드 reconcile
- `esoteric_d3q27.py` 부활(git 8036317) → **ASCII 규칙**([[feedback_cuda_kernel_ascii]]) 재확인(raw kernel non-ASCII 금지).
- 통합 포트: `simulation.py`의 `_init_esoteric`/`_advance_esoteric`/dispatch(`_use_esoteric`), `initializer.py` parity,
  `output_manager.py` checkpoint extra(`esoteric_step`). **현 코드 diff와 reconcile**(simulation.py는 8036317 이후 fused/
  ALM/precision으로 크게 변경 — 3-way merge 아니라 스캐폴딩을 현 구조에 재배선).
- **게이트 a**: 소형 순수-LBM(BGK) 케이스, esoteric ON vs OFF **conservation/CV-band 등가**(과거 bit참조 無 → 물리게이트).
  `gates/eso_bgk_conservation_gate.py`.

### (b) cumulant Esoteric Pull 재구현 (핵심 신규작업)
- BGK Pull 커널의 **LOAD/STORE/macro/BC(node_type)/force 스캐폴딩 유지**, **collision 블록만** 현 `cumulant_d3q27.py`의
  fused cumulant(Forward Chimera→cumulant transform→relax[ω1/ωbulk/ωhigh]→backward)로 교체.
- **방향 매핑**: esoteric paired ordering → `K[cx+1][cy+1][cz+1]` bijective(blueprint 검증됨). cx/cy/cz는 CX_ESO 등.
- **body_force**(ALM): Guo velocity correction + source term + sign(현 cumulant 커널이 이미 지원 → 이식).
- 파일: `src/kernels/esoteric_cumulant_d3q27.py` 신규. `EsotericCumulantKernelD3Q27`.
- **게이트 b**: 소형 케이스에서 **표준 fused cumulant post-collision f vs esoteric-cumulant gather(physical) max_diff → 0**
  (blueprint가 max_diff 0.0 달성했던 검증). `gates/eso_cumulant_equiv_gate.py`. ★가장 중요한 정확성 게이트.

### (c) reconcile — MLG / ALM / 융합 / precision / f_prev
- **MLG coupling**: gather/scatter로 esoteric f → standard f → **현 coupling.py(cubic-z 코얼레싱 + fused C2F rescale,
  80a64c9)** 그대로 사용 → scatter. even/odd parity가 coupling 시점·서브스텝과 정합하는지(blueprint: f_prev는 physical
  layout 저장) 확인. **f_prev 유지**(cell-vertex temporal interp, 1a 폐기).
- **ALM 2-pass**: gather physical f → macroscopic → `_compute_body_force`(canonical-axis + BEM-GPU, 현재값) → esoteric
  cumulant kernel(body_force). `_advance_esoteric_with_alm`.
- **precision**: esoteric도 f32(현 표준). **CUDA graph**: esoteric은 advance 경로 변경 → graph capture 재검토(현재 graph
  dormant/OFF라 우선 무시).
- **게이트 c**: 2-level MLG esoteric rho 보존 < 3e-6(과거값); ALM 소형(constant Fx→profile) 대칭.
  `gates/eso_mlg_alm_gate.py`.

### (d) 검증 → HVAB (경계 주의: §3)
- **메모리 실측**: `gates/precision_dtype_probe.py` 확장으로 esoteric ON 시 B/node·D40 투영 < 24GB 확인.
- **빌드검증 + 초소형 HVAB smoke**(로컬 3090, D 축소): NaN 없음·발산 없음·에너지 정상. **← 여기까지 Claude 로컬 가능.**
- **진짜 HVAB 공력 검증(D40, 팁하중 vs KSAS/EXP)**: **대형 실행=사용자 클러스터**([[feedback_simulation_execution]]).

## 2. reconcile 델타 (8036317 이후 변경, 재배선 대상)
fused cumulant(float32) · cubic-z 코얼레싱 · C2F rescale 융합 · canonical-axis(ALM) · BEM-GPU 상주 · precision 처리 ·
2D fused 경로 · int32→64bit 인덱싱. → 8036317 스캐폴딩을 **현 구조에 재배선**(패치 아닌 재통합).

## 3. ★실행 모드 경계 (사용자 요청 "오토모드→HVAB smoke"에 대한 정직한 답)
- **표준 규칙**([[feedback_stepwise_patch_notes]]): 다단계=단계별 패치노트, **자동모드 금지**. 이번 요청은 이를 override.
  → 절충: **단계(a~d) 내부는 자율 구현/자가검토/디버깅, 단계 경계마다 [패치노트 + 통과 게이트]로 체크포인트**(정도의
  audit trail 유지). 각 단계 완료 시 게이트 결과 보고.
- **HVAB 공력 smoke(진짜 실행)은 Claude가 완결 불가**([[feedback_simulation_execution]]): 로컬은 빌드검증+초소형 smoke가
  천장. **D40 팁하중 검증 런은 사용자 클러스터.** Claude 산출 = 통과 게이트들 + ready-to-run config + 메모리 실측.
- 정직성: esoteric in-place + cumulant + MLG + ALM은 수치적으로 미묘 → **게이트로 정확성 증명, 막히면 은폐 없이 보고**.

## 4. 주요 리스크
- in-place race(paired-opposite parity가 틀리면 조용한 오염) → 게이트 a/b가 방어.
- even/odd parity가 **MLG 서브스텝 × coupling × f_prev × checkpoint** 4곳과 정합해야(blueprint가 다룬 부분, 재현 시 주의).
- cumulant K-매핑 버그(가장 흔한 실수) → 게이트 b(max_diff→0)가 유일한 진짜 방어.
- ASCII 커널 규칙, CUDA graph 재캡처.

## 11. 구현 로그
### ✅ Phase (a) BGK Pull 부활 — 완료·검증 (2026-07-10, 로컬 3090)
- `esoteric_d3q27.py` git 8036317에서 복원(ASCII 통과). `simulation.py` opt-in 배선(env `LBM_ESOTERIC=1`,
  BGK D3Q27, 단일버퍼=`_f_post` 미할당). 비-esoteric 경로 무변경(회귀 0 확인).
- ★**잠복 버그 수정: `init_f_esoteric`** — local swap만 하고 **스트리밍 roll shift 누락**. 커널의 even LOAD가
  paired 슬롯을 이웃(x+c_i)에서 읽으므로 init이 `f_mem[i]=roll(f[i+1],+c_i)`로 pre-stream 해야 step-0 LOAD가
  f0를 정확 재현. blueprint의 "max_diff 0.0"은 **단일 post-collision** 비교라 이 다단계 버그가 잠복했음.
  both-parity + ndim-robust로 수정.
- **게이트 결과**:
  - `eso_bgk_equiv_gate.py`(커널 vs 표준 BGK, TGV 24³ 40스텝): max|Δrho|=6.0e-7, max|Δu|=1.3e-7
    (float32 last-bit), 질량drift 2.8e-7, NaN無 → **PASS**.
  - `eso_sim_integration_gate.py`(실 Simulation, LBM_ESOTERIC=1): `_use_esoteric=True`, `f_post=None`,
    30스텝 질량drift 7e-8, NaN無 → **PASS**.
  - 회귀(precision_dtype_probe, cumulant MLG, esoteric off): 317 B/node 동일 → **무회귀**.
- 커밋 **44f2f56**. 다음 → Phase (b).

### ✅ Phase (b) cumulant Esoteric Pull 재구현 — 완료·검증 (2026-07-10, 로컬 3090)
- 신규 `src/kernels/esoteric_cumulant_d3q27.py`(`EsotericCumulantKernelD3Q27`): BGK Pull LOAD/STORE/BC 스캐폴딩에
  현 fused cumulant collision **steps 4-8 verbatim 이식**(K[3][3][3]는 cx 무참조라 순서 무관 → 그대로 복사),
  macro/K-binning/scatter/Guo-source만 CX_ESO로. **64-bit 인덱싱**(현 컨벤션; BGK 커널은 아직 int32=phase(c) 정리).
  CX_ESO/W_ESO/_fmt_array는 `esoteric_d3q27`에서 import(단일소스).
- `simulation.py` 배선 확장: `set_distribution` esoteric 분기에 CumulantCollision 추가, `_init_esoteric`가
  cumulant 시 `EsotericCumulantKernelD3Q27`+`omega_bulk`/`omega_3` 저장, `_advance_esoteric` cumulant 분기(force 인자).
  force=None(ALM 2-pass는 phase(c)).
- **게이트 결과**:
  - ★`eso_cumulant_equiv_gate.py`(esoteric cumulant vs 표준 fused cumulant, TGV 40스텝): **첫 실행 PASS** —
    max|Δrho|=7.2e-7, max|Δu|=1.3e-7(float32 last-bit), step-0부터 정확, 질량drift -3.5e-7. **전사 오류 0.**
  - `eso_sim_integration_gate.py`(BGK+cumulant, 실 Simulation): 둘 다 **PASS**(`is_cumulant` 정합, f_post None).
  - 회귀(cumulant MLG, off): 317 B/node 동일 → **무회귀**.
- 커밋 **3b462ba**. 다음 → Phase (c).

### ✅ Phase (c) reconcile — 완료·검증 (2026-07-10, 로컬 3090) — 검토시점 한계 5종 전부 해소
- **c1 BGK 커널 64-bit**: `esoteric_d3q27.py` 커널 인덱싱 int→long long(idx/N/j_i; cumulant와 동일 컨벤션).
- **c2 gather/scatter**(MLG 브릿지, 재구현): `esoteric_gather/scatter_{physical,std}` — LOAD 슬롯매핑의 역
  (`roll(-c_i)`), scatter=`init_f_esoteric`(단일구현). + `EsotericMacroKernelD3Q27`(LOAD+macro만; SGS pre-pass/ALM용).
- **c3 SGS**: esoteric cumulant 커널 템플릿화 — 표준 커널의 `_SGS_BLOCK_{OFF,SMAG,WALE}` **단일소스 재사용**.
  smagorinsky=inline(K국소), wale/dyn_smag=2-pass(esoteric macro→기존 WALE/DynSmag 커널→nu_t_in). BGK+SGS는 명시적 거부.
  버그 1건: `{{SGS_PARAM}}`을 라인주석 뒤에 삽입→콤마가 주석에 먹혀 nvrtc 실패. 주석 제거로 수정.
- **c4 ALM 2-pass**: `_advance_esoteric_with_alm` — esoteric macro pre-pass(비보정 u)→`_compute_body_force`
  (canonical-axis 파이프라인 무변경)→커널(force; Guo 보정+source 내부). `_advance_fused_with_alm` 미러.
- **c5 MLG**: `multi_level_grid.py` `_phys_f`/`_write_phys_f` — coupling·f_prev가 항상 **물리 f(표준순서)** 를 봄
  (비-esoteric은 identity=무회귀). coupling.py **무변경**(cubic-z/C2F 융합 80a64c9 그대로 사용). graph precheck에
  esoteric 거부 추가(parity가 스텝마다 바뀌어 whole-step 캡처 불가).
- **c6 checkpoint/sponge**: checkpoint는 **physical layout으로 저장**(`physical_f` 프로퍼티, output_manager 5개 사이트)
  → restart는 기존 fresh-convert 경로 그대로(디스크 포맷 parity-free·하위호환). sponge 6면 확장(y/z 추가;
  bc_uz=sigma 재사용이라 sponge 목표 w=0 가정 — hover엔 무해, 문서화).
- **게이트 결과**(전부 PASS): `eso_gather_scatter`(roundtrip **bit-exact** 4-parity + 동역학 참조 4.8e-7) ·
  `eso_sgs_alm`(A smag 커널 7.2e-7/nu_t 4e-8, B dyn_smag Simulation e2e 7.2e-7, C ALM 상수력 7.2e-7+가속확인) ·
  `eso_mlg`(**bench5 5-level HVAB-mini topology**: 5레벨 전부 esoteric, L0-L4 max|Δ|≤8.3e-7, mass drift 표준과 동일
  −1.49e-7) · 기존 3종(a/b/통합) 재실행 PASS.
- 남은 정직한 한계: MLG **동역학**(ALM 강제) 검증은 (d)의 bench5_baseline smoke에서; esoteric MLG gather/scatter는
  full-field(성능 미최적화, boundary-only는 추후); sponge 목표 w=0 가정.
- 커밋 **c7a1cd1**. 다음 → Phase (d).

### ✅ Phase (d) HVAB-mini ALM smoke + 메모리 실측 — 완료 (2026-07-10, 로컬 3090)
- f32 가드 추가(esoteric은 f32 전용, f64 유입 시 명시 거부).
- **`eso_bench5_alm_smoke.py` PASS** — bench5_baseline(HVAB D16, 5-level MLG+cumulant+dyn_smag+eq/sponge+회전 ALM),
  20 coarse steps std vs eso: 추력 |F| 추적 **median tail rel 7.7e-5**(계통 편향 無) / max 1.1e-2(간헐 스파이크
  = free-wake fp-카오스, repo 확립 CV-band ±3% 내; 게이트=median<1e-3 AND max<3e-2 카오스-인지형). NaN 無.
- **메모리 실측**: pure-LBM eso **2.556GB vs std 3.584GB = −1.03GB**(f_post 제거 예측치 정확 일치, L4 f_post=None);
  ALM eso **2.676 vs std 3.765 = −1.09GB**. leak 無(스텝 1→12 used 평탄). 측정 교훈: 동일 프로세스 연속 빌드는
  gc.collect() 필수(잔존 참조 ~1.4GB이 다음 측정 오염).
- ★★**D40 정직한 재산정 — 현 상태로는 24GB 불가**:
  - live(used) 성분합 @D40(91.6M): f 9.9 + rho/u 1.5 + bc 1.6 + **f_prev 7.0**(비-finest 65.2M×108B, full-level 복사)
    + dyn_smag 2.2 + ALM F_grid(f64) 0.6 + misc ≈ **~23GB**.
  - **transient**(MLG 브릿지 full-field gather/scatter): D16 실측 pool/used = 6.28/2.68(≈2.3×). D40 L4 gather 1회
    ≈2.85GB, C2F 중 동시 temp ~7GB → **peak ~30GB. OOM.**
  - ⇒ **Phase (e) 필요(2건, 구조적으로도 정당)**:
    (e1) **f_prev sub-volume화**: coupling은 `f_coarse_prev[coarse_sub_slices]`만 읽음 → full-level 복사(7.0GB)를
    child sub-volume(~0.5GB)로. **std 경로에도 동일 이득**(값 동일 → std bit-identical 기대). −6.5GB.
    (e2) **boundary-scoped gather/scatter**: coupling이 읽는 코스 서브볼륨/쓰는 스트립만 gather/scatter(+roll 할로 1셀)
    → transient ~7GB→~1GB. esoteric 전용.
  - (e1)+(e2) 후 D40 ≈ live ~16.4GB + transient ~1-2GB = **~18GB ✓ 여유 포함 적합**.
- 산출물: 게이트 7종(전부 PASS), HVAB-mini ALM 동역학 검증 완료. **D40 ready-to-run config는 (e) 완료 후**가 정직한 순서.
- 커밋 **133bbe2**. 다음 → Phase (e).

### ✅ Phase (e) D40-fit 최적화 — 완료 (2026-07-10, 로컬 3090) — **D40 단일 24GB 실증**
- **(e1) f_prev sub-volume화**: coupling이 실제 읽는 `coarse_sub`만 저장(`_f_prev_sub_src`; coupling에
  `f_coarse_prev_is_sub` kwarg). bench5 f_prev 레벨당 8~70MB(was 수백MB). **std 경로에도 동일 적용 —
  ALM smoke std 추력 trace가 (d)와 전 자릿수 일치 = bit-보존 실증.**
- **(e2) region-scoped esoteric 브릿지**: full-field `_phys_f`/`_write_phys_f` 제거 →
  `esoteric_{gather,scatter}_std_region`(wrap-정확 advanced indexing, strided 지원; 커널과 동일 `%` 규약).
  C2F=[코스 서브볼륨 gather + `strips_out`(경계 슬랩 업샘플)로 스트립만 scatter], F2C=[`0::R` strided gather
  (`f_fine_is_at_coarse`) + `return_excised`로 excised 블록만 scatter]. coupling에 opt-in kwargs(기본=기존 동작).
  **2D coupling 가드**(`_scoped`): 2D MLG는 legacy 경로 그대로(회귀 0).
- **(e3, 예정외 발견·수정) init 메모리 피크**: D40 첫 빌드에서 **pool 고수위 52.6GB**(WSL2 oversubscription이
  은폐; 네이티브 Linux 4090이면 init OOM). 범인=레벨별 monolithic `compute_equilibrium`의 (Q,N) 브로드캐스트
  임시(~4×f, L4 11GB)+레벨 간 pool 미회수. 수정: **x-슬랩 청킹 equilibrium**(`_equilibrium_lowmem`, pointwise라
  **bit-identical** — |F_grid| 합/ALM trace로 실증) + 레벨 간 `free_all_blocks()`. → **pool 52.6→20.6GB**.
  부수: D40 coarse step 21.4s→**3.1s**(스왑 제거).
- **게이트**: G1 region gather/scatter(**bit-exact** 4-parity×4-region 조합) · G2 scoped coupling
  (`eso_coupling_scoped_gate`: e1/e2/F2C 전부 **bit-exact** vs 원본, boundary-slab vs full-volume 포함) ·
  eso_mlg 재PASS(수치 (c)와 동일) · ALM smoke 재PASS(**std trace 전 자릿수 보존**, eso도 동일).
- **메모리 최종 실측**: bench5 pure eso 2.556→**2.032GB**(std 3.584→3.242); pool: eso 6.28→**2.84GB**.
- ★★**D40 로컬 실증(3090)**: `farfield40` preset(farfield의 D-상대 기하 그대로 D0=40) = **정확히 91.6M**
  (L0 15.0/L1 7.7/L2 17.4/L3 25.1/L4 26.4M — handoff §4 명세 재현). LBM_ESOTERIC=1 빌드+2 coarse step:
  **used 17.1GB, pool 고수위 20.6GB, NaN 無, ALM 정상** → **단일 24GB 4090 적합(여유 ~3.4GB)** ✓.
  (std는 f_post 포함 ~27GB live → esoteric이 enabler.)
- **전달물**: `configs/hvab/hvab_hover_c10_farfield40_eso.py`(4-case CASE 1: 순수 ALM 등방 gaussian+N64+보정off,
  NASA 덱, n_rev=25). 실행: `LBM_ESOTERIC=1 python main.py --config configs/hvab/hvab_hover_c10_farfield40_eso.py`.
  ⚠MLG_CUDA_GRAPH 금지(자동 거부됨). 나머지 3케이스(archB/KSAS)는 덱 배선(handoff §4) 후.

## 5. 게이트 목록 (`patch_notes/hpc_upgrade/gates/`)
`eso_bgk_conservation_gate.py`(a) · `eso_cumulant_equiv_gate.py`(b, ★) · `eso_mlg_alm_gate.py`(c) ·
`precision_dtype_probe.py`(메모리, 확장).
