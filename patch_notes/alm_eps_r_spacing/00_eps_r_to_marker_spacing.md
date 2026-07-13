# ALM anisotropic ε_r → marker spacing δr (Churchfield 2017 §II.A.2)

작성: 2026-07-09. 격자 재설계 논의 중 ALM 이산화 일관성 점검에서 도출.
관련: [[project_farfield_resolution_survey]], `docs/papers_kr/2017_MartinezTossas_advanced_ALM_kr.md`

## 동기 (버그성 mis-scaling)

Churchfield et al. 2017("Advanced Actuator Line Method", AIAA SciTech, = Natelson Ref.31)
§II.A.2, Eq.(2) 직후는 비등방 Gaussian 폭을 블레이드 기하에 이렇게 연관:
- ε_c = a_c·c   (chord)
- ε_t = a_t·t   (max thickness)
- **ε_r = a_r·δr (δr = actuator element width = marker 간격)** ← chord 아님

우리 구현은 세 축 모두 `factor × ε_iso`(=`max(0.25c, 2Δx)`)였음 → **ε_r이 chord에 비례**.
결과: 마커 간격 Δr > ε_r 인 팁에서 인접 마커 Gaussian이 안 겹쳐 **연속 line이 blob(aliasing)**
으로 깨짐. HVAB 목표(cells/R 256, N48): 팁 ε_r=0.25c_tip=3.15lu < δr=4.0lu → ε_r/δr=0.79 blob.

## 변경

ε_r 스케일 기준을 **marker 간격 δr**로 (Churchfield 규정대로). `r_ref` 플래그로 제어:
- `r_ref="spacing"` (**기본, 신규 물리**): `ε_r = max(a_r·δr, 2Δx)`  (2Δx = 2.0 lu floor; a_r = 기존 `r` 인자)
- `r_ref="chord"` (**legacy**): `ε_r = a_r·ε_iso` (2026-07 이전 동작, 재현용)

`r=1.0` + spacing → ε_r=δr = 마커가 정확히 맞닿아 겹침(중점 커널 0.78·peak) → 연속 line 보장,
마커 수와 무관. 등방 경로(`_aniso is None`)는 **bit-identical**(미변경).

### 편집 3곳
1. `src/actuator/actuator_line.py` `_compute_body_force` _aniso 구성:
   `eps_r = fr*epsilon_all` → r_ref 분기(spacing: `max(fr·δr, 2.0)`, δr=`rotor.get_marker_spacing()` [lu]).
2. `src/actuator/actuator_line.py` loader: `model._aniso`에 `r_ref` 추가(기본 "spacing").
3. `configs/hvab/_hvab_hover_base.py` build_config: `anisotropic` dict에 `r_ref` 전파(기본 "spacing").

## 검증
- py_compile 3파일.
- 수치: HVAB cR256 N48 — old ε_r(chord) 팁 3.15lu(blob) → new ε_r(spacing) 4.0lu(=δr, overlap).
- 등방 경로 불변(aniso 가드).
- (사용자) aniso config end-to-end 스모크: ε_r 배열이 δr 추종, 팁 line 연속.

## Step 2 — 마커 수 재유도 (2026-07-09 결정)

★ε_r=δr 수정으로 **비등방은 겹침 자동 보장**(임의 N). 단 **등방(Phase-A α baseline)은
반경 ε=0.25c 그대로**라 팁 겹침에 `δr ≤ 0.25c_tip` 필요. 두 경계:
- 하한 `δr ≥ 2Δx`(격자 spread 해상, 초과-해상 방지) → N ≤ span/2Δx
- 상한 `δr ≤ ε_c,tip=max(0.25c_tip,2Δx)`(등방 겹침/비등방 해상대칭) → N ≥ span/ε_c,tip

**자동 재유도(규칙 A: δr=ε_c,tip) = ceil(span_lu/ε_c,tip)**. HVAB(span 0.748R, c_tip/R 0.04917):
- cR<163(팁 ε floored): N 스케일 → **cR128 → 48** (=현 N_RADIAL, light에 맞춰졌던 값)
- **cR≥163(0.25c 해상): N = span/(0.25·c_tip) = 0.748/(0.25·0.04917) = 61 CONSTANT** (span·ε 동시 비례로 scale-invariant). cR256·512 모두 61/blade.

**★결정(2026-07-09)**: **지금은 N=64/blade 고정**(=61 자동값 + 5% 겹침마진, δr=2.99 vs ε_c,tip=3.15).
- 적용: **새 cR256 far-field config에 `n_radial=64`로만**. **전역 `N_RADIAL=48`은 불변**
  (light cR128엔 48이 규칙-A 정답; 전역 변경 시 light 과잉-해상 δr<2Δx).
- 현 N48 @ cR256 = 등방 팁 blob(δr=4.0 > 0.25c_tip=3.15)이 문제였음.

**★계획(deferred)**: ALM 검증·구현 성공 종료 후 **자동 배선(규칙 B)** 전환 — build_config 또는
blade.generate_markers가 격자(finest cR)+geometry에서 `N=ceil(effective_span_lu/δr_target)`
자동 산출(δr_target∈[2Δx, ε_c,tip]). coarse 격자서 48로 자기조정, hardcode 제거. [[project_farfield_resolution_survey]]
