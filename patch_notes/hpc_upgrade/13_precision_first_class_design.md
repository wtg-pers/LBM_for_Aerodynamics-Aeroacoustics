# 13 — Precision 1급화 설계 (DPSP 기본값)

> Status: **DESIGN (구현 전)**  · 2026-07-10
> Supersedes: 루트 `handoff.md` §2 "Option B" 의 **레버 순서** (1a cell-centred → 1b esoteric 우선).
> 이 노트는 메모리 절감의 **1차 레버를 precision(DPSP)으로** 재확정한다. 1a/1b는 폐기가 아니라 §7로 재배치.
> 근거 논문: `to_claude/ref_papers/high_computing/2024_Markus_Holzer` §2.2.1(streaming), §5.1.2(layout), §9.7(precision).

---

## ★ Phase 0 감사 결과 (2026-07-10, 실측) — 전제 정정 (필독)
> `gates/precision_dtype_probe.py`(bench5_pure_lbm=HVAB base, D=16, 로컬 3090):
> **모든 f-버퍼가 이미 float32**(f/f_post/f_prev/rho/u). `grid_level.py` float64 하드코딩은 MLG에서
> **dead**(레벨=`Simulation` 객체, f는 `initializer`가 `compute_dtype`=float32로 할당). 실측 **317 B/node**(f32).
>
> ⇒ **precision(DPSP/SP)은 메모리 레버가 아니다** — 현재가 이미 f32라 DPSP·SP는 저장 동일→**0 절감**.
> 아래 §0의 "precision-first for memory"는 **"현재 f64" 오전제** 위에 있었다. **메모리 전략은 handoff
> Option B(1a f_prev제거 / 1b f_post제거)로 복귀**한다. 유일한 precision *메모리* 레버는 f16(SPHP)뿐인데
> 정확도↓라 음향목표와 상충.
>
> **precision 1급화의 재정의된 가치**(폐기 아님): **정확도/유연성** — (i) 음향단계 DP 스위치, (ii) f32
> collision round-off 개선용 DPSP, (iii) config↔grid_level dead code↔fused f32전용의 **drift 정리**.
> 즉 이 노트는 이제 **"메모리 노트"가 아니라 "precision 위생/정확도 노트"**. D40 메모리는 별도 트랙(1a/1b).
>
> **D40 메모리 재산정(317 B/node)**: 현재 ~29–31GB(초과) / 1a −7GB→~23.5GB(겨우) / 1a+1b→~13.6GB(여유)
> / SPHP→~17GB(정확도↓). → 다음 세션 재계획: Track M(메모리=Option B) vs Track P(precision=본 노트).

## 0. TL;DR / 결정  — ⚠ 위 Phase 0로 **메모리 전제 무효화**(정확도 트랙으로 재해석)
- 메모리 반감의 1차 레버 = **PDF 저장 dtype을 config 1급 노브로** 만들고 **기본값 DPSP**(저장 float32 / collision float64).
- 이유(구조): precision은 알고리즘을 안 건드리는 **직교 cross-cutting 파라미터**. 방금 커밋한 HPC 베이스라인(cubic-z / C2F 융합 / CUDA graph, `80a64c9`)의 **스트리밍·커플링 알고리즘을 전혀 안 바꾼다**. 실행 단위로 되돌릴 수 있다(음향=DP, 대형공력=DPSP).
- 부수효과지만 큰 것: 이건 신규 기능이 아니라 **현재 흩어진 precision 처리의 정리(cleanup)** 이며, D40-on-24GB도 최저위험으로 해결(단독 ~2×).

## 1. 문제 — precision이 표류 중 (정적 증거)
현재 precision이 모듈마다 애드혹이라 **정적 독해만으로 flagship(HVAB)의 실제 런타임 저장 dtype을 확정할 수 없다**:

| 위치 | 상태 | 파일 |
|---|---|---|
| config 노브 | `precision ∈ {float32,float64}` 존재, HVAB = **float32** | `configs/hvab/_hvab_hover_base.py:312` (+주석 "float32 ~410B/cell":217) |
| config→compute_dtype | 파싱되나 `compute_dtype`은 **lattice에만** 전달 | `src/solver/setup.py:423-434` |
| 저장 할당 | `GridLevel.f/rho/u` = **float64 하드코딩**, dtype 인자 없음 | `src/grid/grid_level.py:157,162,165,203` |
| 3D fused 커널 | cumulant D3Q27 = **float32 전용**(`const float*`, f32 레지스터), dtype 디스패치/캐스트 **없음** | `src/kernels/cumulant_d3q27.py:37,87`; `simulation.py:241-243` |
| fused 호출 | `f_in=self.f` 를 **astype 없이** 전달 | `src/solver/simulation.py:497-529` |
| 2D fused | 여긴 dtype 체크함(`is_f32 = f.dtype==float32`) — 3D와 비대칭 | `simulation.py:284` |
| CuPy default 경로 | dtype-follows-array (별도 collision object) | `simulation.py:447-476` |
| 인스턴스화 | `SimulationCore(` / `GridLevel(` 직접 생성 사이트가 grep에 없음 → **grid_level.py가 dead/legacy 의심**, 실제 f는 initializer가 할당(precision 준수?) | `setup.py:224,885` |

→ 즉 (a) config는 f32를 원하는데 (b) 한 할당 경로는 f64 하드코딩, (c) 커널은 f32 전용, (d) 실제 사용 f의 출처가 불명확. **이 모순/불명확성이 1급화의 직접 근거.** Phase 0에서 런타임 실측으로 확정한다.

## 2. 목표 아키텍처 — precision 1급화
논문(lbmpy/waLBerla)의 3계층 직교 분리를 손코딩 범위에서 채택:

```
저장 표현(store_dtype)   : DP / DPSP / SP / (SPHP)   ← config 1급 노브, 알고리즘 불변  [이 노트]
스트리밍 패턴            : two-grid pull / esoteric   ← 전략(§7, 1b)
격자 전이 layout         : cell-vertex / cell-centred ← 전략(§7, 1a)
```

**단일 진실 소스**: config `precision` → `SolverSetup`이 `store_dtype`(+`compute_dtype`) 도출 → **모든 f-버퍼 할당과 커널 dispatch가 이 값 하나를 참조**. 하드코딩 `float64`/`float32` 제거.

config 스키마(확정):
```python
simulation = {..., "precision": "dpsp"}   # {"dp","dpsp","sp","sphp"}  기본 "dpsp"
# 하위호환 매핑 없음(결정). 구 키 "float32"/"float64"는 loud error → 명시 마이그레이션 강제.
```
★**마이그레이션 필요**: 구 `precision:"float32"/"float64"` 사용 config 전부 신 키로 교체
(HVAB `_hvab_hover_base.py:312` "float32"→ **Phase A는 sp, Phase B부터 dpsp**; VTK writer의
`precision` 인자는 별개 I/O 옵션이라 무관). [[feedback_acoustic_scaling]]의 69개 config 마이그레이션과 함께.
| 정책 | store_dtype | compute | 메모리 | 용도 |
|---|---|---|---|---|
| dp | f64 | f64 | 1× | 음향/기준·회귀 |
| **dpsp (기본)** | **f32** | **f64** | ~0.5× | 공력 프로덕션(정확도 유지) |
| sp | f32 | f32 | ~0.5× | 최속(속도 우선) |
| sphp | f16(shift) | f32 | ~0.25× | 극한 메모리(기존 bgk 커널) |

## 3. DPSP 의미론 + 커널 레지스터 텐션 (중요)
DPSP = **글로벌 메모리는 f32, 레지스터 연산은 f64**(load 시 승격, store 시 강등). Skordos 배경밀도 감산(PDF를 0 주변으로; 우리 `bgk_d3q27.py:156-262`의 `f−w` shift가 이미 그 형태)이 round-off 억제에 유효.

**텐션**: 현 fused cumulant 커널은 **의도적으로 f32 레지스터**("Register budget (float32)")라 occupancy가 확보됨. DPSP로 f64 레지스터화하면 **레지스터 압력 2×→occupancy↓** 가능. 따라서:
- **CuPy default 경로**: DPSP 자명(collide 진입 시 f64 승격, 종료 시 f32 저장). 쉽다.
- **fused RawKernel 경로**(빠른 경로): 두 가지 선택
  - (B1) f64-레지스터 변형 커널 신설(load-cast/store-cast) — 정확도↑, occupancy 비용.
  - (B2) fused는 **SP-fast(f32 레지스터+f32 저장)** 유지, DPSP는 CuPy 경로에만 — 공력은 SP-fast, 정확도 민감 케이스만 DPSP.
- → **미결 결정 §8-(a)**: HVAB fused 경로를 SP-fast로 둘지 DPSP(f64-reg)로 갈지. 메모리 절감은 둘 다 동일(저장 f32), 차이는 collision 정확도 vs occupancy.

## 4. Touch-point 지도 (구현 시 수정 대상)
- **config/setup**: `precision` 파싱 확장(4정책), `store_dtype`/`compute_dtype` 도출·주입. `setup.py:423-434`.
- **저장 할당**: f, rho, u, `_f_prev`(`grid_level.py:157-207` 또는 실제 initializer 경로), `_f_post`(`simulation.py:183`), rho/u 버퍼(`simulation.py:487-489`), nu_t/rho/u fused 버퍼(`_advance_fused` `simulation.py:487`).
- **collision**:
  - CuPy object 경로: load 승격/store 강등(DPSP).
  - fused: §3의 B1/B2 결정에 따라 커널 변형 또는 SP 유지. `cumulant_d3q27.py`.
- **coupling(HPC 베이스라인!)**: `coupling_rescale_d3q27.py`(`const float* f_prev/f_now` — dtype 정합 확인), `interpolation_d3q27.py`(cubic-z), `coupling.py`. **알고리즘 불변, dtype만 정합**.
- **boundary 커널**: HWBB/IBB D3Q27, mem_force — f32 전용 여부 확인·정합.
- **ALM**: spreading이 float64 반환(`actuator_line.py:244 _F_grid float64`, CL/CD float64) → body_force를 store_dtype로 캐스팅하는 단일 지점 정리(현 `simulation.py:650-663` 임시 캐스트 대체).
- **I/O**: VTK는 이미 float32 옵션(`mlg_vtk_writer*`), checkpoint는 "f: float64" 주석(`io/checkpoint.py:258`) → **checkpoint 포맷에 dtype 태그**(DP↔DPSP 체크포인트 상호읽기/마이그레이션).
- **CUDA graph**: 캡처는 dtype 고정 → precision 변경 시 **재-warmup/재캡처** 필요(코드 아닌 런타임 주의; `multi_level_grid.py` warmup 경로).

## 5. 단계 계획 (stepwise; 자동모드 금지)
- **Phase 0 — 감사(코드/초소형 smoke)**: §1 모순의 **런타임 실측**. 각 버퍼의 실제 dtype 로그, grid_level.py dead 여부, HVAB가 현재 f32/f64 중 무엇으로 도는지, 실제 B/cell. `mlg_cellcount.py`로 D40 메모리 **재산정**(handoff 38GB/410/141 수치 불일치 해소). → 노트 §1 표 확정. **여기서 D40이 이미 f32면 목표는 사실상 달성 상태 확인**.
- **Phase A — 단일소스 plumbing (SP-capable)**: `store_dtype` 도출·전 할당 배선, 하드코딩 dtype 제거, 2D/3D dispatch 대칭화. **SP(f32 store+compute) end-to-end 통과 + 실제 메모리 절감 측정**. 최저위험, 여기서 ~2× 확보.
- **Phase B — DPSP compute (기본값화)**: cast-on-load/store로 collision f64화. CuPy 경로 우선, fused는 §8-(a) 결정 반영. `precision` 기본값 **dpsp**로. Skordos shift 필요 시 cumulant에 이식.
- **Phase C(옵션) — SPHP/FP16**: 기존 bgk FP16 shift 커널 재활용, 필요 시 cumulant 확장. **극한 메모리(D48+/multi-rotor) 전까지 보류.**

## 6. 검증 / 게이트
- **회귀**: DP(=현 f64) 대비 DPSP는 **bit-identical 아님**(정상) → 4090 pure-LBM은 이미 run-to-run 비결정(§handoff §7)이므로 **CV-band 게이트** 사용. 소형 케이스에서 DP vs DPSP 매크로장 상대오차 상한 설정.
- **공력 무결성**: HVAB 초소형 smoke로 CT/팁하중이 DP↔DPSP 간 유의미 shift 없는지(팁 M²cₙ). 프로덕션/대형은 사용자 클러스터(로컬 CPU온도 제약).
- **메모리**: Phase 0/A에서 D40 B/cell·총 GB 실측, <24GB 확인.
- **게이트 스크립트**: `patch_notes/hpc_upgrade/gates/`에 `precision_dpsp_cvband_gate.py` 추가.

## 7. 1a/1b 재배치 (메모리 레버 아님 → 각자 명분)
- **1a cell-centred**(논문 §5.1.2, Rohde): 진짜 가치 = **커플링 품질**(hanging node 제거, 암묵 보존, coalescence 암묵 필터=Nyquist). MLG 경계 artifact(`feedback_mlg_region_padding`)의 구조적 해법일 때 도입. f_prev 제거는 부수효과. ⚠ 버퍼삭제가 아니라 **C2F/F2C 재유도**(저차→linear explosion) = 고위험, 방금 커밋한 커플링 최적화와 충돌.
- **1b esoteric**(논문 §2.2.1): 진짜 가치 = **대역폭 + 최대 격자**. dense GPU엔 Twist(git history)보다 **Esoteric Pull(Lehmann)** 이 misaligned store↓. 도입 시 영구 유지보수세(모든 신규 BC/커널이 even/odd twist 처리). **최대 스케일 필요 시.**

## 8. 결정 (확정 2026-07-10)
- (a) **fused HVAB 경로** → **SP-fast 먼저**. Phase A는 f32-레지스터(현 커널)로 메모리+plumbing 확보.
  DPSP는 Phase B에서 **CuPy default 경로에 우선 적용**(자명·무비용). f64-레지스터 fused 변형은
  **공력 smoke에서 SP가 팁하중을 유의미하게 흔들 때만** 신설(occupancy 비용 성급히 안 냄).
- (b) **음향 정책** → 공력=**dpsp** / 음향=**dp** config 스위치로 확정. dp는 회귀 기준선으로 상시 유지.
- (c) **config 키** → `{dp,dpsp,sp,sphp}` 문자열, 기본 `dpsp`. **하위호환 매핑 없음** — 구 `float32/float64`는
  loud error(§2). 마이그레이션 강제.

## 9. 근거 (논문 페이지)
- §2.2.1(p12-13): in-place가 2번째 PDF 세트 제거로 메모리 반감(GPGPU 특히). Twist=indirect 최적, dense엔 misaligned store; Esoteric Pull(Lehmann)이 개선.
- §5.1.2(p73-74): cell-centred(Rohde) = explosion이 fine 2행 생성 → **temporal interp 불필요**(=우리 f_prev 근거), 단 저차→linear explosion.
- §9.7(p144): DP/DPSP/SP/SPHP 4정책. 저장 정밀도↓로 메모리·대역폭↓. Skordos 배경밀도 감산이 저정밀 round-off 억제 필수.
