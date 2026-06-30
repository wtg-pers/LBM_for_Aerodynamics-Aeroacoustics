# 7단계 — Prandtl `R_tip_eff` 표준화 옵션 (감사 후속)

## 배경 (감사 결과)

Prandtl 구현 감사: 공식은 표준 정확(ALM=BEMT=교과서 `F=(2/π)arccos(exp(−f))`).
**유일 비표준 = `R_tip_eff = R_tip − ε_tip`** (BEMT/표준은 `R_tip`). `f=B·max(R_tip_eff−r,0)/…`
라 **r>R_tip_eff 외측 ε밴드를 hard-zero** (baseline ε=5.33 → 외측 4.2% 완전제거). 이게
(1)Prandtl을 과격하게 만들고 (2)ε에 커플링 → light+taper 역효과(외측마커 un-zero)의 근원.

## 변경 — 옵션 플래그 `eps_offset` (기본=레거시 유지)

`prandtl_loss` dict에 `eps_offset` 키 추가. **기본 True = 기존 R−ε 동작 그대로**(재현성),
False = 표준 `R_tip_eff = R_tip`.

```python
# config:
"prandtl_loss": {"enabled": True, "eps_offset": False}   # 표준 Prandtl
```

**편집 (`src/actuator/actuator_line.py`)**
- `__init__`: `self._prandtl_eps_offset = True` (기본).
- `_compute_prandtl_factor`: 분기 —
  ```python
  if self._prandtl_eps_offset:           # 레거시 (기본)
      R_tip_eff  = R_tip  - blade.marker_epsilon[-1]
      R_root_eff = R_root + blade.marker_epsilon[0]
  else:                                   # 표준 (BEMT 일치, ε-디커플)
      R_tip_eff  = R_tip
      R_root_eff = R_root
  ```
- `create_actuator_line_from_config`: `model._prandtl_eps_offset = prandtl.get('eps_offset', True)`.

## 검증 (ct_hover_smoke, CPU 6 step, Prandtl ON)

| 케이스 | T_lu |
|---|---|
| legacy (eps_offset 기본 True) | **0.080959** (원래 baseline과 bit-identical ✓) |
| 표준 (eps_offset False) | 0.125060 (덜 공격적 → 추력↑) |

- 기본값 회귀 통과(0.080959). 플래그 ON 시 Prandtl 거동 변화 확인.
- D=16은 ε=2가 R≈16의 12.5%라 차이 큼(+54%); D32 production은 ε=5.3/R≈128=4.2%라 더 완만 예상.

## Production 테스트 config (클러스터/DGX)

- `ct_hover_t08_m088_prtipR.py` — 표준 Prandtl, no taper → `result_..._light_prtipR`
- `ct_hover_t08_m088_prtipR_taper.py` — 표준 Prandtl + tip_taper → `result_..._light_prtipR_taper`

비교:
```bash
python main.py --config configs/caradonna_tung/ct_hover_t08_m088_prtipR.py
python main.py --config configs/caradonna_tung/ct_hover_t08_m088_prtipR_taper.py
python src/utilities/compare_taper_ab.py \
  --A result_ct_t08.0_M877_mlg4_D32_light_prtipR \
  --B result_ct_t08.0_M877_mlg4_D32_light_prtipR_taper \
  --mtip 0.877 --la stdPr --lb "stdPr+taper"
```
기대: (1) prtipR baseline은 light(0.00553)보다 ↑(덜 공격적 표준 Prandtl). (2) prtipR_taper는
prtipR보다 ↓ — 이번엔 R_tip_eff가 ε에 안 묶이므로 un-zero artifact 없이 테이퍼 −효과가
깨끗하게 stack.
