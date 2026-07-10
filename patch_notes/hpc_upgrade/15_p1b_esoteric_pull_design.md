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
- 상태: **uncommitted**(80a64c9 위). 다음 → Phase (b).

## 5. 게이트 목록 (`patch_notes/hpc_upgrade/gates/`)
`eso_bgk_conservation_gate.py`(a) · `eso_cumulant_equiv_gate.py`(b, ★) · `eso_mlg_alm_gate.py`(c) ·
`precision_dtype_probe.py`(메모리, 확장).
