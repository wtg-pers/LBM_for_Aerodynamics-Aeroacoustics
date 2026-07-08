# Phase 0 — 계측 (measurement) — 2026-07-06

PLAN.md §0 실행. 목표: bench5 reference 게이트 확정 + 1 coarse-step 분해로
**launch-bound vs kernel-bound 판정** → Phase 1c 조기착수 여부 결정.

상태: reference 런 수령·분석 완료, 계측 자산 구축 완료, **클러스터 측정 런 대기**
(baseline+profile / pure-LBM / nsys). Claude=config·계측·분석, production run=사용자.

---

## A. Reference 게이트 기준값 (bench5, 1×4090) — 확정

경로: `patch_notes/alm_multigpu/result_hvab_hover_c10.0_M650_mlg5_D16_bench5_bench5_baseline/`

| 게이트 | reference 값 | 비고 |
|---|---|---|
| **속도** | **32.70 MLUPS** · 3642.30 s / 1006 step · **3.62 s/coarse-step** | performance.csv |
| | updates/step = 118,380,590 · num_levels=5 | |
| **물리** | rev2 tail thrust_lu ≈ 5.05–5.28 → **CT ≈ 0.0100–0.0103** | Task2(0.0104~5)와 정합 |
| **질량** | domain_drift 최종 = **−1.150e-4** | mass_conservation.csv (step 1005) |
| **bit** | `checkpoint_00001005.npz` (844,249,179 B) | ↓ sha256 |

**bit-gate sha256** (동일 GPU/env 전제):
```
ac910ff914403acb20f958e90dc28e197ec0bc6e796012703e02a479fa3b3e61
```

⚠️ **PLAN.md §1 게이트 오타**: 파일명은 `_00001006.npz`가 아니라 **`_00001005.npz`**
(마지막 기록 스텝 = 1005; total_steps 1006은 0→1006 범위). PLAN 게이트 참조를 수정할 것.

## B. bench5 실측 토폴로지 (setup_log.txt)

| level | shape | cells | ×2^k = updates/coarse | dx | floor@216B [ms] |
|---|---|---|---|---|---|
| L0 | 56×48×48 | 129,024 | 129,024 | 1.0 | 0.03 |
| L1 | 73×89×89 | 578,233 | 1,156,466 | 0.5 | 0.25 |
| L2 | 105×137×137 | 1,970,745 | 7,882,980 | 0.25 | 1.69 |
| L3 | 89×201×201 | 3,595,689 | 28,765,512 | 0.125 | 6.16 |
| L4 (rotor) | 57×297×297 | 5,027,913 | 80,446,608 | 0.0625 | 17.24 |
| **합** | | **11,301,604** | **118,380,590** | | **25.57** |

- Σ(cells×2^k) = 118,380,590 → performance.csv와 정확히 일치(검산 OK).
- 실측 **11.3M nodes** (PLAN의 "9.05M"은 프리런 추정치 — 실측으로 갱신).
- ALM target = **L4** (rotor hub 포함 최상세). steps_per_coarse: L0=1,L1=2,L2=4,L3=8,L4=16.

## C. 이번 세션 구축한 계측 자산

1. **`src/utilities/step_profiler.py`** — env-gated MLG coarse-step 프로파일러.
   ALM_PROFILE 관례 미러링. **disabled 기본**(프로덕션 무영향; 검증: cupy 미임포트,
   `timed()`→공유 no-op CM, 출력 0). GPU-synced per-section wall + 100스텝마다
   running mean. env:
   - `MLG_PROFILE=1` — 켬
   - `MLG_PROFILE_EVERY=N` — 리포트 주기(기본 100 coarse step)
   - `MLG_NVTX=1` — NVTX 레인지 추가(nsys용, 동일 섹션명)
2. **`src/grid/multi_level_grid.py`** — advance/_advance_fine에 섹션 배선.
   섹션명(레벨당): `L{k}.advance` · `C2F.L{k}` · `F2C.L{k}` · `fprev.L{k}`.
   이름은 __init__서 precompute(disabled 경로 문자열 포맷 0).
3. **`configs/hpc_bench/bench5_pure_lbm.py`** — bench5_baseline과 **토폴로지·수치 100% 동일,
   `actuator_line.enabled=False`만 차이**(검증: regions/SGS identical, num_levels 5=5).
   → updates/step 동일 → MLUPS 직접 비교 가능 → wall 델타 = 순수 ALM 비용.

## D. 측정 프로토콜 (클러스터 1×4090, 사용자 실행)

셋 다 bench5 규모라 분 단위. env 앞에 붙여 실행:

```bash
# Run 1 — full step budget (MLG 레벨 분해 + ALM phase 분해)
ALM_PROFILE=1 MLG_PROFILE=1 python main.py --config configs/hpc_bench/bench5_baseline.py

# Run 2 — pure-LBM (ALM off): LBM-only wall.  ALM cost = Run1 - Run2
MLG_PROFILE=1 python main.py --config configs/hpc_bench/bench5_pure_lbm.py

# Run 3 (선택·확정용) — nsys 미세동기 없는 커널/갭 타임라인
MLG_NVTX=1 nsys profile -t cuda,nvtx -o bench5_phase0 \
    python main.py --config configs/hpc_bench/bench5_baseline.py
```

산출: Run1/Run2의 `[MLG_PROFILE]` 표 + `[ALM_PROFILE]` 표(sample/bem/spread) 캡처.

## E. Launch-bound 판정 기준 (핵심)

**bandwidth floor** = updates/step × B/cell ÷ 4090 BW(≈1008 GB/s).
- 216 B/cell(2-array fp32, PLAN 관례) → **25.6 ms/coarse-step**.
- 단 fused+stream이 **별도 2커널**(각각 f read+write) → 실 DRAM 트래픽 ≈2× →
  **현실 floor 25–50 ms/coarse-step**.

측정 wall(baseline) = **3620 ms/coarse-step**.

> **판정 규칙**: pure-LBM(Run2) wall ÷ floor 비율이 크면(≥10×) → 대역폭 아님 =
> **launch/sync/CPU-bound** → PLAN §1c(CUDA Graphs + ALM 오버랩 + sync 소탕)를
> 1a/1b보다 **먼저**. 각 `L{k}.advance` ms/step을 위 floor 열과 대조:
> L4가 17 ms floor인데 수백 ms면 런치/동기 지배 확정.
> baseline이 이미 3620 ms ≫ 25–50 ms(≥70×)라 **launch-bound가 유력**(가설과 정합) —
> Run2가 이를 LBM 단독으로 확증하고, MLG 표가 어느 레벨/결합/ALM에 쓰이는지 지목.

## F. 측정 결과 (2026-07-06, 클러스터 1×4090) — 확정

프로파일 perturbation 무시가능: baseline 프로파일 TOTAL 3580.8 ms ≈ 무프로파일 wall
3620 ms (≤1%). 물리 bit-일치(진행바 C_T=0.00915 = ref step975 azimuth 저점, drift 일치).
→ synced-wall 귀속 신뢰 가능.

```
coarse-step budget (bench5, 1x4090):
  wall/step (baseline, w/ ALM) = 3580.8 ms
  wall/step (pure-LBM)         =  391.1 ms
  => ALM 직렬 비용             = 3189.7 ms  (= 89% of baseline)  ★#1
     그중 freewake_influence(BEM) = 188 ms/call × 16 = 3014 ms (84% of wall)

  LBM 내부 분해 (pure-LBM 391 ms):
    collision+stream+BC (L0-4 advance) =  91.1 ms (23%)   ← floor 25-50 대비 1.8-3.6x (건강)
    coupling (C2F+F2C+fprev)           = 300.0 ms (77%)   ★LBM측 병목
       C2F 합 232.4 (C2F.L4 158.3=40%!) | F2C 합 58.3 | fprev 9.2
  => LBM은 collision이 아니라 **coupling(C2F.L4)** 이 6x 승수.
```

### 판정 (verdict) — PLAN §0 예상보다 정밀

두 개의 독립 병목이 pure-LBM 뺄셈으로 분리됨:

1. **ALM free-wake BEM = 89% (BEM 단독 84%)**. 정체 = `_kleine_w_corr`→
   `freewake_influence` Biot-Savart 영향행렬 rebuild, `rebuild_every=1`이라 fine
   substep마다(16×/coarse). **오버랩으로 못 고침**(ALM 3190 ≫ LBM 391 → 오버랩 상한
   = 3190 ms, 18%만 이득). → **알고리즘/GPU 포팅 필수**. 노브: `rebuild_every`(즉시,
   config만) / GPU 포팅(정확도 유지, ~10x 기대 → −2700 ms). PLAN §1c "BEM GPU 포팅"을
   **#1로 승격, 오버랩과 분리**.
2. **LBM은 collision이 아니라 coupling-bound**. collision+stream 91 ms(1.8-3.6x floor,
   양호)인데 coupling 300 ms(3.3x physics). C2F.L4 = 9.9 ms/call ×16 = 158 ms 단일
   최악. → **Phase 1a(cell-centred, f_prev+시간보간 제거)가 메모리뿐 아니라 속도
   레버**. compute-bound인지 launch-bound인지는 nsys(Run3)로 확정 → 후자면 1c CUDA
   Graphs 병행.

### ★스케일 caveat (multi-GPU 목표 관점)

bench5(11M, ALM-heavy)는 ALM을 과대표집. ALM 비용(마커~200·후류~1000)은 **격자와
무관 고정**인데 LBM은 격자에 비례 → Watanabe-fine(207M, ~×20)에선 LBM이 재지배,
ALM은 소수. 또 큰 격자 = 큰 커널 → 런치 오버헤드 amortize. **단 coupling은
multi-GPU서 GPU경계 halo(MPI)가 되므로 Phase 1a 효율화가 이중 중요.** slab5-smoke
(45.3M) 2차 앵커로 이 추세 확인 필요.

### 권장 순서 (Phase 0 반영)

- **P1**: freewake_influence GPU 포팅 (−2700 ms 잠재, 순수이득·다운사이드 無).
  선행 검증 = `rebuild_every` 민감도 런(config만).
- **P2**: Phase 1a cell-centred coupling (−200~250 ms + 메모리 + multi-GPU halo 대비).
  nsys로 C2F.L4 launch-vs-compute 판별.
- LBM collision 순수최적화는 후순위(이미 ~2x floor).
- **열린 결정**: multi-GPU production ALM 모델? free-wake면 P1 최우선, pure/Dağ면
  P2가 #1 (188 ms 소멸).

## G. Housekeeping — ✅ 완료 (2026-07-06)

- ✅ **PLAN.md §1 게이트 파일명** `_00001006` → `_00001005` + sha256 인라인.
- ✅ **.gitignore**: `patch_notes/**/*.{npz,vti,vtp,vth,vtu,vtm,pvd}` 제외 추가.
  검증: 체크포인트·vtk = IGNORED, CSV/log = TRACKED. bit-gate는 sha256 대체
  → 844MB 커밋 불요. (`**/**Zone.Identifier`는 이미 존재했음.)
- ✅ **Zone.Identifier junk 78개 삭제** (0 남음).
