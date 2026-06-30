# 04 — HVAB A/B/C 실험 config

**대상:** HVAB hover (실험 reference 보유: Run44/Run59 적분 FM). CT는 spanwise 실험
앵커 부재(surface Cp만; ALM 비교 불가)라 HVAB 단독 채택 (사용자 결정 2026-06-26).

## build_config 변경 (`configs/hvab/_hvab_hover_base.py`)
- 신규 인자 `sampling=None`. None/`{"mode":"gaussian"}` → bit-identical.
- `actuator_line["sampling"] = sampling` 주입 (eps_correction 직후).
- 전달 경로(검증됨): config['actuator_line'] → setup `self._al_cfg`(L319) →
  `al_cfg=deepcopy`(L1746) → `create_actuator_line_from_config`(L1772) → `model._sampling_mode`.

## 생성 config (전부 c10·M0.65·preset=light·**pure ALM**)

| 파일 | mode | 격리 |
|------|------|------|
| `sampler_A_gaussian_c10.py` | gaussian | A baseline (현행 ±3ε, off-disk 29.8%) |
| `sampler_B3_mask_c10.py` | mask_disk | B-iii (off-disk b1 제거) |
| `sampler_B1_point_c10.py` | point | B-i (b1+b2 제거) |
| `sampler_smoke_mask.py` | mask_disk | 6스텝 CPU smoke (검증용) |

**pure ALM 이유**: prandtl_loss=False + eps_correction=None → 팁 φ를 가리거나(Prandtl
=force만 스케일) 보정하는(Dağ=u_n에 downwash 추가) 요소를 모두 끔 → *샘플러만* 비교.
preset=light = off-disk 진단·기존 c10 spanwise와 동일 격자.
(B-ii aniso는 분해 결과 확인 후 production fix로; `sampling={"mode":"aniso","eps_r_factor":F}` 한 줄.)

## 실행 (사용자 클러스터; 3 런)
```
python main.py --config configs/hvab/sampler_A_gaussian_c10.py
python main.py --config configs/hvab/sampler_B3_mask_c10.py
python main.py --config configs/hvab/sampler_B1_point_c10.py
```
폴더: `result_hvab_hover_c10.0_M650_mlg4_D32_light_{sampA_gauss,sampB3_mask,sampB1_point}/`

## 비교 (spanwise_post / compare_spanwise.py)
지표: 팁 **φ·α·u_n 스팬분포**, **C_T·FM**.
- `A − B-iii` = **off-disk(b1)** 기여
- `B-iii − B-i` = **반경 smoothing(b2)** 기여
- **판정**: B-i/B-iii에서 팁 φ가 운동량이론(~3°) 쪽으로 회복되면 → (b) 실재 +
  Prandtl 없이 고칠 수 있음. 회복 미미하면 → (a)본질/해상도 지배 → ε·해상도·Dağ/Meyer-Forsting.
- 예상 시그니처: point/mask가 팁 α·하중↓ → **C_T가 baseline(gaussian)보다 낮아져 실험에 근접**.

## 검증 (smoke, PASS)
`sampler_smoke_mask.py` (D=16, 2-level, CPU, 6스텝): fine-level ALM(4×48, R=16 lu @L1)에서
mask_disk 무오류 완주, 보존 OK. → config→MLG→fine-ALM→`sample_velocity_alt` 실경로 정상.
