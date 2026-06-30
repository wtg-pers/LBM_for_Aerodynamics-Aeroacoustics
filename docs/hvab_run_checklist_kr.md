# HVAB ALM hover — 실행/검증 체크리스트 (사용자용)

> `configs/hvab/` config로 클러스터 hover sweep을 돌리고 NASA HVAB 실험과 비교하기 위한
> 확인 목록. 관련: `docs/hvab_geometry_kr.md`(geometry+provenance), `docs/alm_mach_pass_theory_kr.md`.
> smoke 통과 확인됨(2026-06-24): multi-airfoil 4 RC + Mach-pass + 테이퍼 배관 작동.

## A. 실행 전 — geometry/조건 cross-check
`docs/hvab_geometry_kr.md` ↔ **NASA/TM-2020-5002153** Table 1/2 (NTRS 20205002153)
- [ ] R=66.5in(1.6891m), σ≈0.1033, 4 blades
- [ ] twist 선형 **−14.05°/R @ 0.75R** (root fairing <0.252 무양력)
- [ ] chord 5.45in 등현 → **r/R 0.95023부터 3.27in 테이퍼(외측 ~5%만)**
- [ ] airfoil 밴드: RC(4)-12(<0.675) / RC(4)-10 / RC(6)-08 / RC(6)-08T(>0.975)
- [ ] 공력 활성 스팬 r/R 0.252~1.0
- [ ] 조건: M_tip=0.65, RPM≈1250.5(TM 1250.39), SLS(a=340.3, ρ=1.225, ν=1.461e-5)

## B. preset ↔ GPU 메모리 (cells/GB는 BYTES_PER_CELL=410 추정)
- [ ] **light** (D32, tip 6.3 cell, 30.5M, ~12.5GB) → **24GB GPU 안전, 기본**
- [ ] medium (D40, tip 7.9, 60.2M, ~24.7GB) → 24GB OOM 위험, ~32GB 필요
- [ ] **fine** (D40 5-level, **tip 15.7 cell≈16**, 183M, ~75GB) → **DGX(128GB)** = 사용자 "tip≥16" 기준 충족
- 선택: `build_config(collective_deg=8.0, preset="fine")`

## C. 실행 직후 (timestep 전, setup 검증)
- [ ] setup log: `Polar: RC4-12 + RC4-10 + RC6-08 + RC6-08T` (4종 로드 = multi+Mach-pass)
- [ ] `csv/blade_diagnostics/blade_geometry.csv`:
  - [ ] 마커별 airfoil 배정이 밴드와 일치
  - [ ] chord 외측 5%만 테이퍼(나머지 5.45in 등현)
  - [ ] twist 선형, eps_lu 분포

## D. 수렴
- [ ] thrust/torque history 평탄화(steady tail), CV% 작음 → `--avg-revs 3` 신뢰

## E. 물리 sanity
- [ ] 팁 section Mach ~0.65 (Mach-pass 정확성)
- [ ] spanwise 팁 φ/하중 — **CT에서 본 팁 과예측 경향** 재현 여부 인지하고 해석

## F. 검증 (목표) — NASA HVAB 데이터 필요
- [ ] **NASA HVAB 실험 데이터 다운로드**: rotorcraft.arc.nasa.gov/HVAB/ (sectional airloads + 성능)
- [ ] sectional M²·Cn(r/R):
  `python -m src.utilities.spanwise_post --result result_hvab_hover_c08.0_M650_mlg4_D32_light --mtip 0.65 --avg-revs 3`
  → NASA sectional과 overlay (CT엔 없던 ALM 직접 비교점)
- [ ] C_T / FM sweep:
  `python -m src.utilities.hover_fm_post --results "./result_hvab_hover_c*" --sigma 0.1033 --prefix hvab --avg-revs 3`
  → `hvab_fm_summary.csv`, `hvab_fm_vs_ctsigma.png` (prefix 옵션으로 hart2 결과와 분리)

## G. 사용자 action items (Claude 불가)
- [ ] NASA HVAB 실험 데이터 다운로드 (비교 기준)
- [ ] **RC(4)-12 / RC(6)-08T as-built 좌표**를 HVAB FileShare(plot3d)에서 → production 전 authentic 교체
      (현재: RC(4)-10/RC(6)-08 authentic, RC(4)-12=12%스케일 placeholder, RC(6)-08T=RC(6)-08 tab 무시)
- [ ] precone 값 NASA TM Table 1 확인 (원하면; 효과 ~0.2%, 현재 미적용)
- [ ] tip chord ≥16 검증은 DGX에서 `preset="fine"` 1~2점

## H. 알려진 한계 (결과 해석 시 염두)
- light tip chord 6.3 cell → **팁 ε floor 제한**(0.25c 미달) → 팁 하중 bias 가능. 16-cell은 fine(DGX).
- RC(4)-12(12% 스케일)·RC(6)-08T(tab 무시) 근사 → inboard/팁-tab 불확실.
- **sweep tip(30°)·precone 미모델** (ALM 직선 마커선 한계).
- **ALM 고유 팁 유도 과예측**(CT 진단: smeared-ALM 다운워시 결손) — HVAB 풀 sweep서 확인됨
  (CT +11~14%, CP +1.5~50%, FM peak 미포착). → **non-iterative smearing correction 구현 완료**
  (2026-06-25, JFM-2019 control-point 형식, docs/alm_tip_overprediction_record_kr.md §9).
  아래 §I A/B로 효과 검증.

## I. viscous-core smearing correction A/B (구현 완료, 검증 대기)
보정 = 팁 다운워시 결손을 lifting-line 유도속도로 회복 → 팁 α↓ → 팁 과하중 완화 목표.
- **ON config** (inviscid target + **Prandtl OFF**, 물리보정이 팁 de-loading 직접해상→Prandtl 중복방지):
  `configs/hvab/hvab_hover_c{06,08,10,1276}_epscorr.py`
- **baseline** = 기존 `hvab_hover_c{...}.py` (Prandtl ON, 보정 OFF) — 보유 결과 재사용
- smoke gate(통과 확인): `ct_hover_smoke_epscorr.py`(−0.32%), `hvab_hover_smoke_epscorr.py`(4 RC+Mach+corr)
- 해석 질문: 물리 보정이 경험적 Prandtl보다 CT/CP 팁 과예측을 더 줄이고 **FM peak를 회복**하는가
```bash
# ON sweep (light)
for c in c06 c08 c10 c1276; do
  python main.py --config configs/hvab/hvab_hover_${c}_epscorr.py
done
# A/B 후처리 (ON 결과 + 기존 baseline 둘 다 glob → Run44 overlay)
python -m src.utilities.hover_fm_post --results "./result_hvab_hover_c*" \
    --sigma 0.1033 --prefix hvab_epscorr --avg-revs 3 \
    --exp natural=aeromechanics_workshop/HVAB/performance_datas/hvab_M065_NATURAL_Run44_reference.csv
```

## 실행 커맨드 (요약)
```bash
# 클러스터 sweep (light, 24GB)
for c in c06 c08 c10 c1276; do
  python main.py --config configs/hvab/hvab_hover_${c}.py
done
# (DGX, tip-16 확인) python main.py --config <build_config(..., preset="fine") wrapper>

# 후처리
python -m src.utilities.hover_fm_post --results "./result_hvab_hover_c*" --sigma 0.1033 --prefix hvab
python -m src.utilities.spanwise_post --result <run_dir> --mtip 0.65
```
