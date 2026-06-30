# 5단계 — 검증

스모크 덱: `configs/caradonna_tung/ct_hover_smoke.py` (CPU, D=16, 2-blade, prandtl on).
3개 일회성 변형(출력 폴더 분리)을 사용:
`_tmp_eps_baseline.py` (A, eps 키 없음), `_tmp_eps_default.py` (`epsilon_mode:"default"`),
`_tmp_eps_taper.py` (`tip_taper`, factor 1.0, start 0.7). 실행:
`python main.py --config <variant> --max-steps 6 --no-vtk --clear`.

> 정리(rm)가 거부되어 이 파일들은 디스크에 남아 있음. 제거하려면:
> `rm -f configs/caradonna_tung/_tmp_eps_*.py && rm -rf result_ct_smoke_eps{A,B,DEF}`

## 결과 1 — Default-OFF 회귀 게이트 (핵심 보장)

| 런 | T_lu | Q_lu | P_lu |
|-----|------|------|------|
| A (키 없음) | 0.080959 | 0.014371 | 0.000090 |
| `epsilon_mode:"default"` | 0.080959 | 0.014371 | 0.000090 |

- `rotor_performance.csv` A vs default → **BIT-IDENTICAL** (전정밀도 `diff`).
- `blade_geometry.csv` A vs default → **BIT-IDENTICAL**.
- ⇒ 플래그가 off일 때 무해함이 증명됨. 기존 HART2/CT 결과 재현성 보존.

## 결과 2 — `eps_lu` 진단 컬럼

- per-step `blade_diagnostics/<j>.csv` 헤더에 `eps_lu`가 7번째 컬럼(`chord_lu` 뒤)으로 존재.
  (6 step 런에서는 데이터 행 없음: blade 진단은 `output_interval`마다 로깅되는데 6보다 큼 —
  정상. 여기서 검증하는 건 header/writer 배선.)
- setup 시점에 쓰이는 정적 `blade_diagnostics/blade_geometry.csv`가 per-marker `epsilon_lu`를
  독립적으로 보고 — 짧은 런에서도 A/B에 유용.

## 결과 3 — D=16 스모크에서는 테이퍼가 no-op (이유 포함), production 해상도에서는 정확

D=16에서는 chord가 `chord_lu = 2.667`밖에 안 됨 → `chord/4 = 0.667 < 2.0` floor라 기준 ε이
이미 **모든** marker에서 `2.0`. 팁 목표도 `max(1.0·2.0, 2.0) = 2.0`이라 테이퍼가 2.0→2.0 =
변화 없음. 따라서 세 런의 T_lu가 동일하고 A·B의 `blade_geometry.csv`도 동일. **이는 올바른
동작** (ε은 LBM floor 아래로 절대 안 내려감), 단지 테이퍼를 보여주지 못할 뿐.

`chord/4 > floor`인 곳에서 테이퍼를 실제로 보기 위해, 직접 `Blade` 테스트로 production CT
스케일(`chord_lu ≈ 21.3`, `chord/4 ≈ 5.3`)을 재현하여 정확한 런타임 경로
(`to_lattice_units` → `set_lattice_spacing(dx=1.0)`)를 실행:

```
 marker | r/R   | chord_lu | eps(default) | eps(tip_taper)
   ..    <0.70    21.300     5.3250         5.3250          (inboard: 불변)
   12    0.700    21.300     5.3250         5.3250          (taper_start: t=0)
   13    0.740    21.300     5.3250         4.8817   <-- 테이퍼 시작
   ..
   18    0.940    21.300     5.3250         2.6650
   19    0.980    21.300     5.3250         2.2217   <-- 팁 → floor 2.0 쪽으로
```
- `default`: ε 균일 = `chord/4` (공식 불변).
- `tip_taper`: `r/R = 0.7` 안쪽은 ε 동일, 이후 floor 걸린 팁 값까지 매끄러운 **선형** 블렌딩.
  팁 marker는 `r/R = 0.98`(cell-centered)이라 ε = 2.22; 정확히 팁에 있는 marker라면 2.0.
  공식 `(1-t)·5.325 + t·2.0`과 일치.

## 결론

- **Default 경로 bit-identical** ✓ (회귀 안전).
- **테이퍼 수학 정확** ✓ (inboard 불변, `taper_start`부터 선형 테이퍼, 팁 floor).
- **진단 연결** ✓ (`eps_lu` 컬럼 + 기존 `blade_geometry.csv`).
- **커널 변경 불필요** ✓ (per-marker ε이 이미 end-to-end 지원; 비균일-가능 ε 경로로 런 정상 완료).

## 다음 (사용자, 클러스터 — 본 세션 범위 밖)

production CT M0.877 `light` 덱을 `epsilon_mode:"tip_taper"`로 실행하고 비교:
`C_T` vs 실험(0.00473)과 팁 φ 회복을 `spanwise_post.py`(`eps_lu` 플롯)로.
이후 B로 충분한지 vs Stage C(filtered lifting-line, JFM-2019 — hover 타당성 caveat 포함)가
필요한지 결정; HART2 FM 재평가.
