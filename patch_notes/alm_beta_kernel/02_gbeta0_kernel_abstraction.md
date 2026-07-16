# β Kernel 02 — G-β0: 커널 추상화 리팩터 (2026-07-16, 완료)

설계 근거: `01_design.md` §13 (합의 확정: A=Wendland + (a)닫힌형, D=A/B 대조군).
원칙: **물리-코드 일치** — 스프레딩/샘플링/재정규화/보정이 같은 η 객체에서
유도되도록, Gaussian 하드코딩 전 지점을 단일 팩토리로 통일. kernel="gaussian"
(기본)은 **리팩터 전과 bit-identical**.

## 1. 신규 모듈 — `src/actuator/alm_kernel.py`

`ALMKernelSpec` (family = gaussian | wendland | winckelmans):

| 필드/메서드 | 역할 |
|---|---|
| `support_factor(n_cut_cfg)` | 컷오프/support 반경 [ε 단위]. gaussian=config `gaussian_cutoff` 그대로, wendland=**√7.5 고정**(ε-등가 support, config 무시), winckelmans=config(대수 꼬리라 자연 컷 없음 — 01_design §7b) |
| `norm_const` | η = norm_const/ε³ 의 상수 (gaussian 1/π^{3/2}, wendland 21/(2π·7.5^{3/2}), winck 15/(8π)) |
| `np_shape(xp, d²، ε²)` / `np_shape_arg(xp, arg)` | 비정규화 shape (등방/비등방). gaussian은 기존 표현식 그대로 (bit 계약) |
| `np_norm(ε)` / `np_norm3(εc,εt,εr)` | CPU 사이트 정규화 (gaussian은 `1/(π^1.5·ε³)` 원문 표현식 보존) |
| `deficit_K(xp, d, ε)` | 보정 결손 커널. gaussian=exp(−(d/ε)²), winckelmans=1/(1+(d/ε)²)² (01_design §7b 유도), **wendland=NotImplementedError**(02_correction에서 유도 예정 — 유도 없이 β 물리를 돌리는 사고를 구조적으로 차단) |
| `cuda_w_sample / cuda_eta_iso / cuda_eta_radial / cuda_eta_aniso` | RawKernel 템플릿에 베이크되는 CUDA 소스 조각. **gaussian 조각은 리팩터 전 소스를 byte-for-byte 재현** |

핵심 설계: `n_cut` 파라미터가 **support factor로 일반화** — 기존 커널들의
컷오프 산술(`r_cut = n_cut·eps`, `rc2`, `half = ceil(n_cut·ε_max)+1`)이
무수정으로 전 family를 지원한다 (wendland는 r_cut=R_s가 되어 커널 내
q = d/r_cut까지 공짜). CUDA 'inv_pi32' 파라미터 슬롯은 family norm 상수
운반용으로 재사용(이름은 gaussian byte-parity 위해 유지).

## 2. 리팩터된 지점 (kernel_spec=None → gaussian, 기존 호출 전부 무변경)

- `spreading.py`: RawKernel 3종 템플릿화({{ETA_ISO/RADIAL/ANISO}}) +
  `_bake`(ASCII 가드), 게터 family-키 캐시. CPU/GPU 폴백 전 경로
  (single_marker(+aniso)/uniform/method_a/그리드 루프), radial scales
  (reference+batch — norm은 S_all/S_kept 비율에서 소거라 등급 불변).
- `interpolation.py`: `alm_sample_markers` 템플릿화({{W_SAMPLE}}),
  varying/uniform GPU 경로, CPU reference(§2/§3/§4), dispatcher
  `interpolate_velocity_batch_gpu(kernel_spec=...)`.
- `actuator_line.py`: 생성자 `kernel` dict → `self._alm_kernel`,
  `self.n_cut = spec.support_factor(gaussian_cutoff)`. 샘플러/스프레딩
  호출에 spec 전달. `_viscous_core_correction`·("opt" 타깃 = 두 폭의
  deficit_K 차) 커널화. config `actuator_line.kernel={"type": ...}`
  (single + multi-rotor 로더).
- `smearing_correction.py`: `influence_matrix(kernel_spec=...)` —
  straight-Kleine의 A 행렬이 같은 deficit_K에서 유도(연산자 공유).
- `alm_dist.py`: 분산 샘플러/부분합에 kernel_spec 관통(분산 β 지원).
- **fail-fast** (비-gaussian 시): kleine free-wake(erf 해석형),
  비-gaussian 샘플링 모드(자체 가중 기계) → loud error.

## 3. G-β0 게이트 — `gates/gbeta0_kernel_abstraction_gate.py` PASS

| 클레임 | 결과 |
|---|---|
| [S] 생성된 gaussian CUDA 소스 4종 = baseline_0716 원본 | **sha256 4/4 byte-identical** (동일 소스 ⇒ 동일 SASS ⇒ bit) |
| [G] golden(고정 합성입력, 리팩터 전 산출) 대조 | 샘플링 num/den·radial scales·deficit K = **bit(diff 0)**; F_grid 3종 rel ≤ 1.7e-16 (f64 atomicAdd 순서 등급 — G-M3의 6.5e-17과 동급) |
| [U] 신규 family 단위 | ∫η dV=1 (2e-16/0), M₂=(3/2)ε² ε-등가 (3e-16), wendland support 밖 정확히 0, winck deficit ↔ 필라멘트 닫힌형 1.7e-16 |
| [E] 신규 family e2e 스모크 | wendland 보존 2.3e-5, winck 4.2e-3(꼬리 절단 문서화 값과 정합) |

golden: `gates/data/gbeta0_golden.npz` (baseline_0716 코드로 생성).

## 4. production 회귀 (전부 PASS, 로컬 3090)

- **G-M3** (분산 ALM, pure+archB): field max|df|=0.0, F_grid rel 6.7e-17
- eso_bench5_alm_smoke: median 2.0e-4 / max 8.4e-3 (CV-band 내)
- alm_radial_scales_gate: 3.97e-16 / eso_sgs_alm_gate: PASS

## 5. 다음 (01_design §14 사다리)

1. **02_correction**: Wendland deficit K의 닫힌형 유도(2D 사영 Γ_enc)
   → `deficit_K` 구현 + G-β2(고립 와류)로 유도식 검증.
2. G-β1 단위게이트는 [U]로 선반영 — 이산 재정규화 거동/등방성 항목만
   G-β2에 흡수 예정.
3. G-β3 bench5 A/B (gaussian ↔ wendland, ε-등가) → D40 매트릭스.
