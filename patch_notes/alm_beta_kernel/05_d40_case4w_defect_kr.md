# 05 — D40 case4′ 팁 결함: 판별 세션 기록 (2026-07-19)

## 1. 결함 관측 (D40 매트릭스 분석, `0718_beta_kernel/analyze_beta_d40.py`)

case4′(archB+KSAS+Wendland, 2-rank GPU0,1, 25rev)에서만:

- **블레이드 1,2만** 팁 4마커 α 톱니: mk60→63 = +8/+10 → +2.5 → **−15°**
  (mk63: φ≈+22°, CL<0, F_n<0, w_corr = gaussian 대비 4×·부호반전).
- 블레이드 0,3은 gaussian case4와 거의 일치 (마커별 α 차 ≤0.2°).
- **첫 스냅샷(step 1257, ramp 중)부터 25 rev 내내 고정** — 시간 진동 아님.
- case1′(pure Wendland, 동일 2-rank)은 4블레이드 대칭 clean
  → 스프레딩/샘플링 커널 스왑 자체는 무죄.
- CT −2.5%(vs gaussian case4)는 파손 팁 음(−)하중 아티팩트.
- 판정 영향: §5.4 게이트(00_handoff) **판정 불가** — 재실행 필요.

안전망 관련: kleine 폴백 한계 |w| > 0.5·max|u_tan| = 0.05 인데 파손 상태의
w ≈ 0.04 → 한계 바로 아래에서 지속 (폴백 미발동).

## 2. 로컬 판별 사다리 (bench5, 실제 main_mpi 경로, RTX3090 GPU0 공유)

`aeromechanics_workshop/HVAB/0718_beta_kernel/smoke_mpi_archb/`
(run_pair_a/b.sh, run_d.sh, readout.py; 754step=1.5rev, --dist-init,
cuda_aware=0/MPICH, axis=y 자동 = 디스크 관통 분할 확인)

| run | 조합 | 팁 α 스프레드 | 판정 |
|---|---|---|---|
| a0 | gaussian archB × 1-rank | 0.01° | clean |
| a | gaussian archB × 2-rank | 0.20° | clean (fp-재결합층) |
| b0 | wendland archB × 1-rank | 0.01° | clean |
| b | wendland archB × 2-rank | 0.01° | clean |
| d | wendland archB+KSAS덱 × 2-rank | 0.12° | clean |

Wendland 활성 검증: (b)↔(a) CT 상대차 max 3.4% (G-β3 archB ~3.5%와 부합).
결함 시그니처(~18°) 대비 두 자릿수 여유로 전부 clean.

기각된 가설: ①compact-K 희소커플링(R_s/δr) — 실측 D40=3.05 >
bench5=2.34 (D40이 오히려 조밀; 전제 붕괴, pair C 취소) ②KSAS 덱 ③분할
기하(축/rank 소유) 자체.

## 3. 오프라인 워밍스타트 맵 A/B (`sawtooth_map_ab.py`)

코드가 자경(自警)하는 채널("w ∝ −Γ″ sawtooth ... feed back through Γⁿ⁻¹",
`_smooth_active` docstring)을 고립 시험: production `influence_matrix` +
`correct_noniterative` + 3점 smoother(2패스)를 실기하(healthy VTP에서
twist=α+φ 역산)로 frozen-inflow 반복.

- {gaussian, wendland} × {D40, bench5}, 시드 1e-3/0.1, 실속 tanh polar
  포함 → **전부 강수축(계약)**: 교란이 ~1e-16까지 소멸.
- 결론: **solve 내부 루프 단독으로는 sawtooth 유지 불가** — 고착에는
  유동 피드백(spread force→LBM→u_n 재샘플)이 필수. 관측된 φ 프로파일
  (팁 +22°/0.97R ≈0°)도 팁에 정체 와류가 실재함을 지시.

## 4. 미소거 잔여 인자 (bench5 ↔ D40 차이)

1. **절대 스케일/격자**: ε_tip/Δx = 4.17 (bench5 1.7) — 보정·와류코어의
   격자상 크기가 2.5×; L4 26.4M cells; R=320 fine; 파장 발달 25rev.
2. **cuda-aware=1 UCX/OpenMPI** (로컬은 MPICH cuda_aware=0).
3. 위 인자들 × 유동 피드백의 조합 (오프라인 실험이 유동 결합 필요성 입증).

## 5. 다음 = 클러스터 판별 런 (n_rev=3, ~19–40분/런, case4′ config 그대로)

사분면 설계 — R1로 단기재현 확인 후 R2/R4가 축을 가른다:

| run | 변형 | 목적 |
|---|---|---|
| R1 | case4′ 그대로, n_rev=3 | 단기(≤3rev) 재현 확인 = 저렴한 판별 하네스 검증 |
| R2 | case4′ **1-rank** (A100 80GB, 21GB fits) | rank 축 |
| R4 | **case4 gaussian** 2-rank | 커널 축 (gaussian×2-rank×D40은 미실행 조합!) |
| R3 | case4′ 2-rank **--cuda-aware 0** | R2가 clean일 때 transport 세분 |

해석: R4 clean + R2 broken → Wendland 커널측(rank 무관) /
R4 clean + R2 clean → Wendland×2-rank 교호(→R3) /
R4 broken → MPI×archB (β 무죄; R2로 세분).

## 5b. 클러스터 사분면 결과 (2026-07-19, 3rev 단축런) — **판별 완결**

| 런 | 조합 | 팁 스프레드(mk58-63) | 판정 |
|---|---|---|---|
| R2′ | wendland archB+KSAS **1-rank** | 18.7° (b1 mk63 α=−16.5°) | **BROKEN** |
| R4′ | gaussian archB+KSAS 2-rank | 0.07° | clean |

(+기존: gaussian archB+NASA 2-rank 22rev clean = sweep_archb c10)

**최종 결론: 결함 = Wendland β-체인 고유, rank/전송/덱 무관.**

발현 동역학 (R2′ mk63 α 스냅샷별): step629(ramp중) **4블레이드 전부
−16.4°** → 1258에서 b0,b3 회복 → 1887에서 b2 회복 → 2516+ b1만 영구
고착. 즉 ①램프 중 Wendland 보정이 **결정론적으로** 모든 팁을 심저
α≈−16°로 과보정 ②유동 발달과 함께 블레이드별 회복 ③회복이 쌍안정이라
일부가 팁 정체와류 상태에 갇힘(25rev 2-rank=2개, 3rev 1-rank=1개 —
rank 비대칭은 '누가 갇히나'만 편향). §3 오프라인 수축 결과와 정합:
건강 상태 앵커에선 수축(=회복 방향), 램프 과도가 진입 경로.

다음 = 원인 수정 세션: 램프 초기 Wendland 팁 과보정의 코드 지점 —
후보 ①deficit K_W의 팁 closure edge(−Γ_tip) 근거리 응답 vs gaussian
②kernel-consistent radial-trunc scales의 팁 재정규화(서포트 √7.5ε vs
3ε 절단분 차이) ③03 §1b spanwise-varying ε의 edge 외삽. 재현 하네스 =
D40 3rev 1-rank(R2′, ~40분)로 확립됨. 오프라인 맵을 램프 앵커(저추력
u_n, Γ~0에서 출발)로 재실행하면 GPU 없이 과보정 재현 가능성 높음.

## 5c. 오프라인 감사 + 램프 불일치 발견 (2026-07-19 후속)

램프 앵커 오프라인 맵(`ramp_anchor_ab.py`, u_n=f·u_n_healthy 스윕
f:0→1 워밍스타트 연속): **미재현** — 양 커널 동일하게 온화(w≈0.006 ≪
고착 필요치 0.04), 이력현상 없음. frozen-inflow 근사로는 부족 = 고착의
증폭 루프는 유동측(shed vortex 응답) 경유 확정.

대신 커널 의존 연산자 **전수 정량 감사 완료 — 전부 무죄**:
| 연산자 | gaussian↔wendland 차이 |
|---|---|
| deficit K 영향행렬 (팁 행) | max 1.2% |
| radial-trunc 재정규화 스케일 (mk63) | 0.8% (1.439 vs 1.451) |
| ETA_RADIAL CUDA vs numpy 레퍼런스 | bit-일치 (rel ~3e-16, 양 커널) |
| 스프레딩 보존성 | wendland 정확 보존, gaussian −0.044% (설계값) |

→ 퍼센트 수준 차이의 커널이 정성 반전을 만든 것 = 시스템이 램프 중
**임계 근방**이라는 뜻. 그리고 그 임계 창의 정체 발견:

**램프 불일치 (구조 결함, 커널 무관)**: `actuator_line.py:631` — 램프는
스프레딩 후 `_F_grid`에만 적용. BEM/kleine 보정은 램프를 모른 채 **풀
Γ의 w_corr(=풀 하중의 스미어링 결손)를 램프 중에도 전액 가산**. 물리적
정합은 "격자에 실제 침적된(램프된) 힘의 결손" = w_corr×ramp. 현재는
초기 램프에서 보정이 실제 침적력 대비 1/ramp 배 과대 → 팁 심저 진입
창. 타임라인 증거: R2′ 다이브=램프 중(step 629), 회복 시작=램프 종료
직후(1258). gaussian은 이 창에서 아임계로 살아남고(R4′ 629 무증상),
wendland은 %-수준 이득 차로 초임계 → 4블레이드 전부 진입.

**수정 후보 (설계 결정 필요)**: ①w_corr에 현재 스텝 램프 팩터 게이팅
(수렴 후 bit-identical, 진입 창 제거; 물리-코드 일치) ②보조: 팁 안전망
강화(현 0.5·u_tan 한계를 고착 상태 w≈0.04가 통과). 검증 플랜: 수정 후
R2′ 하네스(3rev 1-rank, ~40분)로 무진입 확인 → case4′ 25rev 재런 →
§5.4 게이트 판정.

## 6. 부수 개선 항목

- setup 로그에 `actuator_line.kernel.type` echo 없음 → 1줄 추가 권장.
- kleine 폴백 카운터(`_kleine_fallback_count`) 주기 로그 출력 권장.
- main_mpi 경로는 OutputManager CSV(블레이드 진단·rotor_performance)를
  기록하지 않음 — 스팬 분석은 마커 VTP 의존(handoff §6에 이미 명시).
  분석 파서/도구: `analyze_beta_d40.py`(raw-appended VTP 파서 포함).
