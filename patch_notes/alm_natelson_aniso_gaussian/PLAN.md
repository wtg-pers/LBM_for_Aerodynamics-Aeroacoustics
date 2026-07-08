# ① 비등방 Gaussian force projection (Natelson 2026 Eq.7 / Churchfield 2017)

목표: 등방 Gaussian(단일 ε, 구형)을 **블레이드 로컬 3축 비등방**으로 확장.
$$g=\frac{1}{\pi^{3/2}\varepsilon_c\varepsilon_t\varepsilon_r}\exp\!\Big(-\frac{d_c^2}{\varepsilon_c^2}-\frac{d_t^2}{\varepsilon_t^2}-\frac{d_r^2}{\varepsilon_r^2}\Big)$$
c=chord, t=thickness, r=radial(span). **ε_c=ε_t=ε_r → 등방으로 정확 환원**(검증 포인트).
팁 과부하 레버 최대(force 분포가 실제 익형 형상에 가까워짐).

## 로컬 프레임 (per-marker, 매 스텝 회전)
{ê_θ(θ), ê_n, ê_r(θ)}는 정규직교(HAWT). twist=기하 피치일 때:
- ê_c = cos(twist)·ê_θ + sin(twist)·ê_n   (chord)
- ê_t = −sin(twist)·ê_θ + cos(twist)·ê_n  (thickness, ⊥chord in section plane)
- ê_r = radial_vector(θ)                    (span, 블레이드 공유)
→ {ê_c,ê_t,ê_r} 정규직교. (sweep 회전은 1차 구현서 무시 — 문서화)

## 폭 (config)
등방 ε_iso=max(chord/4,2Δx)에 배율: ε_c=c·ε_iso, ε_t=t·ε_iso, ε_r=r·ε_iso.
config `spreading.anisotropic={"enabled":T,"c":1.0,"t":..,"r":..}`. 기본 OFF.
c=t=r=1 → 등방 환원(검증). 물리 권장: thickness 좁게(t<1).

## 커널 (isotropic → aniso)
d=node−x_j; d_c=d·ê_c, d_t=d·ê_t, d_r=d·ê_r.
arg=(d_c/ε_c)²+(d_t/ε_t)²+(d_r/ε_r)²; norm=1/(π^{3/2}ε_cε_tε_r); η=norm·exp(−arg).
cutoff=타원체 arg≤n_cut²; bounding box=n_cut·max(ε_c,ε_t,ε_r).

## Step A — CPU + 프레임 + config  ← 이번
- `spread_force_single_marker_aniso`(신규, 등방 경로 불변), batch에 aniso 분기.
- `rotor.get_all_marker_aero_frame()`→(ec,et,er) (N,3).
- step()이 self._aniso 설정 시 프레임·폭 계산 후 spread에 전달.
- config 파싱, 기본 OFF. GPU(cupy)+aniso는 Step B까지 NotImplementedError(무음 오답 방지).
- 검증: (1) c=t=r=1 → 등방 spread와 일치, (2) force 보존 Σ=marker force,
  (3) t<1 → thickness 방향 좁아짐(형상 확인).

### Step A 상태: 완료 (2026-07-06, CPU)
- `spreading.spread_force_single_marker_aniso`(신규), `spread_forces_to_grid(aniso=)` 분기.
  `spread_forces_to_grid_gpu`: numpy→CPU 위임(aniso 처리), cupy+aniso→NotImplementedError(Step B).
- `rotor.get_all_marker_aero_frame()`→(ec,et,er). `actuator_line` `_aniso` 속성 +
  step() 프레임/폭 계산 + config `spreading.anisotropic={"enabled","c","t","r"}` 파싱.
- 검증(test_aniso.py): (1)등방환원 회전프레임+등폭 1.3e-18, (2)force보존 0.04%(cutoff),
  (3)ε_c=2ε→σ비 2.000, (4)프레임 정규직교 5e-17. 기본 OFF→기존 run byte-identical.
- 미결: sweep 회전 미반영(문서), radial-trunc와 미조합, GPU=Step B.

## Step B — GPU RawKernel aniso
- `_SPREAD_KERNEL_SRC` aniso 변형(ASCII only!). ec/et/er/eps_c/t/r 전달, 타원체 cutoff.
- radial-trunc와 조합은 별도(초기엔 상호배타 문서화).
- GPU==CPU 일치 검증(verify_gpu_spreading 확장).

### Step B 상태: 완료 (2026-07-06, GPU RawKernel)
- `_SPREAD_KERNEL_ANISO_SRC`(타원체 Gaussian, **ASCII-only 확인**), `_get_spread_kernel_aniso`,
  `_spread_rawkernel_aniso_gpu` 런처. `spread_forces_to_grid_gpu`: NotImplementedError 제거,
  active 필터 후 aniso rawkernel 호출(+실패 시 CPU-host 폴백). radial-trunc와 미조합(CPU와 동일).
- 검증: (1)3 커널소스 ASCII-only, (2)**커널 로직 numpy 에뮬 vs CPU aniso 6.9e-18**(스텐실
  decode·round·투영·타원체 cutoff·flat idx·정규화 전부 일치), (3)CPU aniso 무회귀.
- **미검증(사용자/클러스터)**: 실제 cupy 컴파일 + GPU==CPU 수치일치(로컬 GPU 없음).
  로직·ASCII 확인 완료라 리스크 낮음.

## Step C — 기본값·검증·config 마감
- 물리 권장 배율 확정(문헌), HVAB smoke A/B(등방 vs 비등방 팁 하중).

### Step C 상태: 완료 (2026-07-06)
- **문헌 권장 배율 확정**: c=1.0, t=0.5, r=1.0.
  근거: Martínez-Tossas 2017(등방 최적 ε=0.25c=우리 ε_iso → c=1.0) +
  Churchfield 2017(비등방). 팁서 물리 ε_t=0.25·두께=0.08·ε_c(sub-grid) → 최소해상
  2Δx≈0.5ε_iso → t=0.5(해상도 정합). span r=1.0.
- `build_config(anisotropic=)` 파라미터(True=권장기본, dict=커스텀) → `spreading`.
- A/B config: `hvab_hover_c10_aniso_iso.py`(등방 대조) / `hvab_hover_c10_aniso.py`
  (c=1,t=0.5,r=1). 순수 ALM(prandtl/eps_corr/sweep OFF), light preset, NASA 덱, 25rev.
- 검증(test_stepC.py): (A)A/B actuator_line은 spreading 외 완전 동일, (B)step() 등방
  vs 비등방 force field 상이(5.4e-5)·**총추력 보존**(0.01232 일치, 재분배만).
- **실행(사용자/클러스터)**: 두 config GPU run(25rev) → compare_M2Cn로 팁 M²Cn 비교
  (EXP 0.146 / 등방 pure 0.393 앵커). GPU 비등방 커널 초회 실사용.

## 원칙
각 Step 검증·정리·체크인. 자동 진행 금지. 기본 OFF로 기존 run byte-identical.
