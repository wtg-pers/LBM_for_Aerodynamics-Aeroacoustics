# 21 — esoteric BGK: ALM body force + dyn_smag SGS (2026-07-22)

## 목적
collision A/B 대조군: 팁와류 in-level 감쇠의 비-SGS 몫(~84%, half-life
cumulant 263° w/o SGS)이 cumulant 고유인지 collision-일반(=해상도 바닥)인지
판별. BGK+SGS half-life ≈260° → collision 무관 / 크게 다름 → collision 레버.

## 변경
- `src/kernels/esoteric_d3q27.py` (_ESOTERIC_BGK_KERNEL + wrapper):
  - **ALM body force**: cumulant 커널과 동일 스킴 미러 — half-force 속도
    시프트(u += F/2ρ) + Guo 소스항 Si=(1−ω/2)w_q[3(c·F−u·F)+9(c·u)(c·F)]
    (FLUID; sponge는 force=0 영역이라 미적용).
  - **SGS**: nu_t_in(dyn_smag 프리패스)로 국소 ω_eff=1/(3(ν₀+ν_t)+0.5),
    nu_t_out 패스스루(VTK 진단). 신규 CUDA 인자는 말미 추가(기존 위치 불변),
    래퍼 kwargs 기본 None → 기존 호출 소스호환.
- `src/parallel/local_level.py`: from_level에 bgk 플래그(type(lev.collision))
  + _eso_omega_* getattr 폴백; LocalLevel이 collision별 커널 선택 + advance
  분기(BGK는 ob/oh/Cs 없음). BGK+{smagorinsky,wale}는 fail-fast(dyn_smag만).
- `src/solver/simulation.py`: 단일-GPU eso BGK+SGS fail-fast 메시지 갱신
  (MPI 러너 경유만 지원 명시; 프리패스 배선 안 함 — 스코프 제한).
- configs: `bench5_bgk_sgs.py`(스모크) + `hvab_hover_c10_farfield40_eso_
  mlg4_bgk.py`(본런, mlg4+dyn_smag 동결·collision만 교체).

## ★함정 발견: collision 선택 키 우선순위
`setup._create_lbm_components`: **`numerics.collision`이
`simulation.collision_model`보다 우선**. build_config가 numerics에
'cumulant'를 넣으므로 simulation 키만 덮으면 **조용히 마스킹**됨.
실제로 1차 스모크가 cumulant로 돌았고, **쌍둥이 대조 bit-identical 체크로
적발**(CT/CP 완전일치=스위치 미작동의 지문). 교훈: collision 오버라이드는
`config["numerics"]["collision"]` 필수(우리 config는 둘 다 세팅).

## 검증 (로컬 3090, bench5 200스텝, 프로덕션 플래그 LBM_ESOTERIC=1 + --dist-init)
- BGK+dyn_smag+ALM: 안정 완주, NaN 無, nu_t VTI 기록 ✓, CT 1.88e-2 sane
  (force 경로 정상 — Guo 계수 오류면 %급 이탈).
- 동일경로 cumulant 쌍둥이와 차이 성장(step50 ΔCT 2.0e-7 → step200
  6.8e-7) = BGK 실활성 + 초기 램프서 collision 차이는 미소(기대대로).
- 부수 확인: eso-cumulant ≡ fused-cumulant CT prints bit-identical.

## 리스크/미결
- **BGK τ≈0.5 장주기 안정성 미증명**(200스텝≠20rev): 정온 영역은 ν_t=0
  bare BGK. mlg4_bgk 본런이 곧 안정성 시험 — 발산 시 ν 상향 운전점 필요.
- 단일-GPU eso 경로의 BGK+SGS 미배선(의도적 스코프 제한, fail-fast 유지).
- omega_high 스윕 노브 확인: config["simulation"]["omega_high"] →
  CumulantCollision.omega_3 → dist-init metadata_host가 레벨로 전파 ✓
  (스윕 config는 이 키만 세팅하면 됨).
