# APC 18x8E — 싱글 로터 ALM 하버링, tip loss A/B × RPM 스윕 (2026-08-07)

APC 18x8E Thin Electric(2-blade, R=9.0 in) 하버링을 HVAB 파이널 런
(`hvab_hover_c10_farfield40_eso_archB_ksas_mlg4_shen_g030`) 포뮬레이션으로 해석.
축은 **tip loss function 하나만**: OFF vs Shen g=0.3(HVAB 테스트 조건 그대로).

## 런 매트릭스 (8 케이스)

| RPM  | tip loss OFF (GPU 2)              | Shen g=0.3 (GPU 3)                   |
|------|-----------------------------------|--------------------------------------|
| 2446 | `apc18x8e_hover_2446rpm_notl.py`  | `apc18x8e_hover_2446rpm_shen030.py`  |
| 3460 | `apc18x8e_hover_3460rpm_notl.py`  | `apc18x8e_hover_3460rpm_shen030.py`  |
| 4446 | `apc18x8e_hover_4446rpm_notl.py`  | `apc18x8e_hover_4446rpm_shen030.py`  |
| 5446 | `apc18x8e_hover_5446rpm_notl.py`  | `apc18x8e_hover_5446rpm_shen030.py`  |

★판정(2026-08-08, 2446): notl CT −5.5%/CP +8.1% vs shen030 CT −22.6% → **notl 채택**
(kleine 보정과 Shen의 팁 이중 차감; HVAB와 반대로 baseline이 실험 아래).

notl RPM 스윕 (GPU 2/3 두 큐 병렬; full-field VTK는 마지막 5 rev만 10°마다):

```bash
bash configs/apc18x8e/run_notl_sweep.sh
# 2446 기존 결과 보존 시: RPMS_GPU2="4446" RPMS_GPU3="3460 5446" bash configs/apc18x8e/run_notl_sweep.sh
```

단일 케이스 실행 (메인 디렉토리 기준, 케이스별 docstring에 동일 명령):

```bash
LBM_ESOTERIC=1 python main.py --mpi \
    --config configs/apc18x8e/apc18x8e_hover_2446rpm_notl.py \
    --gpu 2 --steps 31425 --log-every 64 --vtk-every 35 \
    --vtk-fields-last 180 --ckpt-every 31425 --csv apc18x8e_2446_notl.csv
```

## 공통 구성 (HVAB 파이널 런 이식, `_apc18x8e_hover_base.py`)

- 격자: farfield40_mlg4 프리셋 D-상대 동일 — D=40, **MLG 4-level**, ALM은 L3
  (dx_fine=1.43 mm), 도메인 6.5D×6D×6D(260×240×240 L0), blockage 2.7%.
  총 ~65.1M 셀 → esoteric ~13.5 GB/케이스 (24 GB GPU 1장 OK).
- ALM: iso gaussian sampling/spreading, ε=0.25c(2Δx floor), Kleine straight
  eps-correction(inviscid target), Merabet radial truncation, uniform n64,
  coeff_mode=rotorcraft, ramp 1 rev.
- 수치: convective scaling u_max=0.1 (1257 steps/rev), cumulant D3Q27,
  dyn_smag SGS, eq+sponge(outlet) BC. 25 rev = 31,425 steps.
- RPM은 dt_phys만 바꿈(steps/rev 동일): M_tip 0.172/0.243/0.313/0.383,
  Re_75 74k/105k/135k/165k.

## 지오메트리·폴라 (input_files/apc18x8e/)

- `apc18x8_chord_twist_distribution.csv` — 51 station r/R·chord·twist 전량 사용
  (twist=기하 피치각, 추가 collective 없음). x_qc(sweep)·t/c 열 미사용.
- `18x8E-PERF.PE0` (APC 공식) — hub transition 2.00 in → 공력 활성 r/R≥0.2222.
  **AIRFOIL SECTIONS: E63(r=2.00 in 시작) → APC12≡NACA4412(r=5.76 in 전이 완료)**
  → 전이 중간점 r/R=0.4311에서 E63→NACA4412 스위치.
  ※ 폴더의 naca0012/naca2416.dat은 PE0 블레이드 정의에 없어 미사용.
- 폴라: neuralfoil(asb) Re-보간 덱(10k~300k, 12-step) — APC 9x4.5MR 스윕 동일 경로.
  한계: t/c 스케일(내측 0.17~0.31) 미반영, Mach 보정 없음(M_tip≤0.38).

## 실험 대조

`apce_18x8_static_2184od.txt` (UIUC-style static, 프로펠러 convention
CT=T/ρn²D⁴): 요청 4점 = 실험 2446.667/3460.000/4446.667/5446.667 RPM 행.

| RPM(exp) | CT_exp   | CP_exp   |
|----------|----------|----------|
| 2446.667 | 0.088769 | 0.025939 |
| 3460.000 | 0.088198 | 0.025643 |
| 4446.667 | 0.091241 | 0.026248 |
| 5446.667 | 0.093322 | 0.026812 |

솔버 출력(rotorcraft convention) 변환: **CT_prop=(π³/4)·CT_rc≈7.7516·CT_rc,
CP_prop=(π⁴/4)·CP_rc≈24.352·CP_rc**.
