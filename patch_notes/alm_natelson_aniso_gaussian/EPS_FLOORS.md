# EPS_FLOORS — 비등방 커널 폭의 이산화 하한 (2026-07-21)

## 배경 (factorial 리플 판정)
0721 factorial에서 s2/s3/s4 스팬 하중에 물결무늬 관측. inboard(0.28–0.6R)
Γ 리플 대역분해(블레이드별 RMS %):

| | 1–3.5마커 | 4–8마커 | >8마커 |
|---|---|---|---|
| s1 iso | 0.7 | 2.2 | 1.9 |
| s2 aniso샘플링 | 4.2 | 11.2 | 6.0 |
| s3 aniso스프레딩 | 1.4 | 3.0 | 4.3 |
| s4 both | 2.4 | 3.9 | 6.3 |

## 이론 (두 하한)
1. **격자 앨리어싱**: Gaussian 스펙트럼의 Nyquist(π/Δx) 잔여
   exp(−π²ε²/4Δx²) = 8.5%(ε=Δx) / 0.4%(1.5Δx) / 5e-5(2Δx) → **ε ≥ 2Δx**.
2. **마커 겹침(Poisson summation)**: 간격 δr 등폭 Gaussian 합의 라인 리플
   = 2exp(−π²ε²/δr²) = ±17%(ε=0.5δr) / ±1.6%(0.75δr) / ±0.01%(1.0δr)
   → **ε_r ≳ δr**.
3. 샘플링 필터 체적: N_eff ≈ π^{3/2}ε_cε_tε_r/Δx³ — factorial aniso는
   iso 대비 5.6× 작아 난류 샘플 분산 ~2.4×↑ (관측 랜덤성분 4×와 정합).

## factorial 위반 내역 (δr=3.74, tip chord 16.7 lu 기준)
- 스프레딩: ε_r은 **이미 2Δx floor 있었음**(=2.0 적용됨, 겹침비 0.53만 위반).
  ε_t=0.1c=1.67Δx는 floor 없음(위반).
- 샘플링: floor 전무 — ε_t=1.67Δx, ε_r=0.5δr=1.87Δx 둘 다 위반.
  → s2(샘플링 aniso)가 최악이었던 이유.

## 코드 변경 (src/actuator/actuator_line.py)
스프레딩 `_aniso` 구성과 `_build_sampling_aniso()` **양 경로 3축 전부
무조건부 `max(·, 2Δx)` floor** (스프레딩 ε_r의 기존 무조건부 floor 선례
확장). iso-환원(c=t=r=1)은 ε_iso=max(0.25c,2Δx)≥2Δx라 bit-identical 유지.
게이트: gbeta0_kernel_abstraction_gate PASS (spread iso/radial/aniso +
sample byte-identical, F_grid ≤1e-14).

**★레거시 비호환**: fact_s2/s4(샘플링 sub-floor 폭), fact_s3/s4(스프레딩
ε_t sub-floor)는 재실행 시 결과가 달라짐(의도된 변경). 기존 결과 판독은
기각 완료라 영향 없음.

## 신규 config (afix = aniso fixed)
`hvab_hover_c10_farfield40_eso_afix_{s2_anisosamp,s4_bothaniso}.py`:
- **n_radial 64→128** (δr 3.74→1.85Δx; 2026-07-09 crossover 연구로 마커수는
  M²cn 비교란 확인됨), **r-factor 0.5→1.0** → ε_r=max(δr,2Δx)=2.0 lu =
  기존 절대 선예도 유지 + 겹침비 0.5→1.08.
- ε_t: 팁에서 1.67→2.0 (inboard 0.1c=2.6은 그대로), ε_c=0.25c 불변.
- fact_s4 대비 config diff = {sampling.r, spreading.anisotropic.r,
  rotor.grid.n_radial} 3개뿐(검증됨).
- 드라이버 `run_afix.sh` (2체인 2-rank, l4wake 종료 후 실행).

## 판정 기준
1. inboard 리플이 s1 수준(≤1%대)으로 소멸하는가.
2. s4의 팁 M²cn roll-off가 합법 계수에서도 살아남는가 (리플/언더샘플링
   부산물이었는지 판별).
3. CT/CP 인플레이션(s4 +23% peak) 해소 여부.

## 마커 eps 기록 (2026-07-21 추가, 사용자 요청)
aniso 폭이 파일에 안 남던 문제: 마커 VTP에 `eps_lu`+스프레딩
`eps_c/eps_t/eps_r`+샘플링 `eps_samp_c/t/r` 7배열 추가, blade_diagnostics
CSV에 동일 6열 추가(맨 뒤, 기존 열 위치 불변). iso 경로는 3축=eps_lu
fallback. 구현: actuator_line이 스텝마다 `_last_eps_spread`/`_last_eps_samp`
스태시 → writer/진단이 읽음. 상세·검증은 patch_notes/hpc_upgrade/
20_mpi_result_csv_nut_vtk.md (CSV 복구·nu_t VTK와 동일 세션).

## 남은 논의 (미반영)
- ε_c=0.17c(Martinez-Tossas 2D 최적, 팁 2.8Δx 합법) 시도 여부 — 별도 레버.
- Martinez-Tossas ε_opt는 2D 병진 단면 유도(로터 회전효과·팁 3D 미포함),
  팁 거동 근거로는 한계. 팁 3D는 FLLC 계열이 정답 방향.
