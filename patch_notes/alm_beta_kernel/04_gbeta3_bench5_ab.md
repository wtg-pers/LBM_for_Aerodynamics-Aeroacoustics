# β Kernel 04 — G-β3: bench5 A/B (gaussian ↔ wendland) + D40 매트릭스 준비 (2026-07-16)

01_design §14.4의 수행. 게이트 = `gates/gbeta3_bench5_ab_gate.py` **PASS**
(로컬 3090, esoteric production 경로, 20 coarse step, ramp 초기 구간).

## 1. A/B 구성

| 쌍 | gaussian | wendland | 검증 대상 |
|---|---|---|---|
| pure | bench5_purealm_m3 | **bench5_purealm_wendland**(신규) | 스프레딩+샘플링만 (보정 없음) |
| archB | bench5_archb_m3 | **bench5_archb_wendland**(신규) | **풀 β 체인**: +radial-trunc 재정규화(커널 일관 scales) + kleine-straight의 유도된 Wendland deficit K |

wendland config = m3 config + `actuator_line.kernel={"type":"wendland"}`
한 줄 (ε-등가 support √7.5는 support factor가 자동).

## 2. 결과

| 클레임 | pure | archB |
|---|---|---|
| [K] 커널 실적용 (n_cut) | g 3.0000 / w 2.7386 ✓ | 동일 ✓ |
| [A] 안정성 (NaN) | 없음 ✓ | 없음 ✓ |
| [B] 보존 (deposited = −ramp·Σmark) | rel 1.1e-5 ✓ | rel 1.3e-5 ✓ |
| [C] 추력 궤적 tail rel diff | median **4.4e-4** | median **3.5e-2** (max 5.1e-2) |
| [D] 스팬 F_n corr / 피크영역(0.85–0.97R) mean ΔF_n | 1.00000 / −2.4e-5 | 0.9646 / **+0.30%** |

## 3. 해석

- **pure 쌍이 4.4e-4 수준으로 일치하는 것은 ε-등가의 예측 그대로다**:
  20-step ramp 초기에는 샘플링되는 유동장이 매끈해서(유도유동 미발달)
  2차 모멘트가 같은 두 커널의 하중이 거의 동일하다. 커널-형태 효과는
  유도유동이 발달한 뒤(회전수 누적) 나타난다 — 그것이 D40의 판정 대상.
- **archB 쌍의 ~3.5% 차이는 보정 경로의 실질 가동 증거**: deficit K의
  형태 차이(03 §6 — d≳2ε에서 wendland가 빠르게 0)와 커널 일관 radial-trunc
  scales가 하중 분포에 작용한다. 안정·유계·보존 유지.
- 게이트의 등급은 **안정성/보존성**(물리 차이는 기대되는 것) — [C]/[D]의
  수치는 판정이 아니라 기록이다.

## 4. 게이트 제작 중 배운 것

- 보존 검사는 **ramp를 고려해야 한다**: `_F_grid`는 스프레딩 후
  `_ramp_factor`가 곱해지지만 `_last_forces_global`은 unramped —
  deposited = −ramp·Σmark가 올바른 항등식 (첫 판은 이걸 놓쳐 rel 0.96).

## 5. D40 매트릭스 준비 (클러스터 실런용, 신규 config 2종)

- **case 1′**: `configs/hvab/hvab_hover_c10_farfield40_eso_wendland_mpi4.py`
  (pure β, 4-rank run_tag 분리, 런북 §2 명령 헤더 포함)
- **case 4′**: `configs/hvab/hvab_hover_c10_farfield40_eso_archB_ksas_wendland.py`
  (archB+KSAS+β 풀 체인)
- 파싱 검증: kernel 키 관통·radial_trunc/eps_corr 플래그 확인 완료.
- 성공 기준(00_handoff §5.4): 피크영역 +13~17% → 한 자릿수 %,
  CT +15% → +5% 이내, **팁 dip 0.215 유지**.
- 산출물 규약: 스팬 M²cₙ + CT 수렴 + 팁와류 FWHM(기준값: wendland
  사영코어 1.7540ε, gaussian 1.6651ε — 03 §5).

## 6. 상태

G-β0(추상화 bit) → G-β2(보정 유도 검증) → **G-β3(bench5 A/B 안정성) 완료**.
β 사다리의 로컬 구간 전부 통과 — 남은 것은 D40 실런(사용자 클러스터)과
그 판정이다.
