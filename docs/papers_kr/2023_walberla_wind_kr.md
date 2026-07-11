# waLBerla-wind: LBM 기반 풍력 ALM 고성능 솔버 — 리뷰 (2026-07-10)

> 원문: "WALBERLA-WIND: a lattice-Boltzmann-based high-performance flow solver for wind
> energy applications" (preprint, ~2023). `to_claude/ref_papers/alm/`.
> 맥락: waLBerla+lbmpy = Holzer 2024 박사논문 프레임워크(우리가 HPC 검토에 사용한 그것).
> **최초의 멀티노드 ALM+LBM** 주장. 관련: Asmuth 2019(단일GPU ALM+LBM), Rullaud 2018(최초 결합).

## 1. 방법 요약
- **유동**: cumulant D3Q27 (Geier), **SGS 없음(implicit LES)**, Mach 0.05 (0.05↔0.10 무차이,
  0.15부터 차이 실측 — pre-study).
- **ALM**: 마커 50/blade linear, 샘플링 **trilinear**, 스프레딩 = **Roma-Peskin discrete delta**
  (compact ~3셀; 컨볼루션 회피 + Σw=1 강제 → 힘 보존). **Gaussian 아님.**
- **tip-loss: Glauert 비활성** — "ALM+LBM 커플링 자체 평가 + 커뮤니티 합의 부재" 명시.
  팁 손실은 **격자 해상으로**(tip vortex resolution) 얻는 전략.
- **격자**: uniform만(32/64/128 cells/**D**) — waLBerla가 refinement 지원함에도 미사용.
- **검증**: NewMexico(D=4.5m, DNW 풍동, IEA Task29), TSR≈10/6.6/4, 대조 = 실험 + Castor
  (inviscid free-wake lifting-line; 팁손실 내재).
- **HPC**: 블록구조+Hilbert SFC 로드밸런싱(터빈 블록 가중), actuator 힘 통신 = 마커의 이웃
  서브도메인 침투 **예측** 후 marking→buffered MPI. GPUdirect는 **PDF만**(터빈 데이터는 메시지가
  작아 비효율). A100 1677→1866 MLUPS(터빈 off, fp32) — 스스로 "Holzer 82% 대비 22%" 미최적 인정.
  터빈 모듈 오버헤드 ~10%. weak scaling GPU당 준일정(74.46 GLUPS@다수GPU).

## 2. 결과 요지
- 32→64→128 cells/D로 갈수록 팁 힘 피크가 줄어 128에서 매끈 → Castor에 수렴. 고TSR(10/6.6)에서
  실험·Castor와 우수한 일치. 저TSR(4, 실속·스팬방향 유동)은 ALM 계열 공통 한계로 불일치.
- near-wake 속도 프로파일 우수(허브 근방 제외 — hub/nacelle 미모델).
- ★★**미해결 자인**: *"normal/tangential forces globally decreasing when refining the mesh ...
  not observed when using wider force-spreading kernels — needs further investigation"* —
  좁은 delta 커널은 ε_eff∝Δx라 격자 세분 시 힘이 계속 감소(= Martinez-Tossas FLLC가 예측하는
  ε→0 거동). **보정 없는 compact 커널은 격자수렴 안 됨을 그들 데이터가 보여주고 open problem으로 남김.**

## 3. "128이면 괜찮다"의 조건부 해석 (우리 관점)
1. **단위**: 128 cells/D = **64 cells/R**. 우리 D40 = 320 cells/R(=640/D) — 그들 최고 해상의 5배.
   그들의 "충분"을 우리 척도로 옮기면 D8 수준 — 우리 D-sweep에서 이미 팁 flatten 확인된 영역.
2. **메커니즘**: 팁이 좋아지는 건 보정이 아니라 **ε_eff가 격자에 묶여 자동 축소**(delta support 고정)
   → 팁 와류 해상. 팁 하중 완만한 풍력(고TSR·저Mach 0.05·저하중)이라 성립.
3. **HVAB 호버는 다른 문제**: 팁 하중 집중(M²cₙ peak) + tip M 0.65 + 강한 나선 자기유도 →
   해상-기반 전략의 난이도가 근본적으로 높음(Merabet/Mali가 보정을 필요로 한 이유).

## 4. 우리 프로젝트 시사점
- ★**커널 노벨티(handoff §1)의 선행+공백 확정**: compact delta를 이미 실전 사용했으나 **무보정**
  → 수렴 문제 open. 우리의 β(새 커널로 FLLC 재유도)가 정확히 그 공백을 닫음 — **노벨티 정당화 문헌**.
- **Roma delta를 커널 비교군에 포함 권고**: 구현 매우 쉬움(3셀, 컨볼루션 무). "delta 무보정(그들) vs
  새 compact 커널+FLLC 재유도(우리)" 대비가 논문 그림이 됨.
- Σw=1 보존 논거 = 우리 `_radial_trunc` 재정규화와 동일 문제의식(인용 가치).
- tip-loss off 검증 설계 = 우리 4-케이스 CASE1과 동일 철학 → 결과 해석 시 나란히 배치.
- trilinear 샘플링으로 그들 스케일에선 충분(우리 ①샘플러 질문의 데이터 포인트).
- Mach 0.05~0.10 무차이/0.15 차이 — 스케일링 선택 참고.
- **멀티GPU 설계노트(16) 직접 참고**: actuator 침투-예측 buffered 통신, GPUdirect PDF-한정,
  터빈블록 가중 SFC 로드밸런싱, "터빈 모듈 미분산 = serial fraction" 실패 교훈.

## 5. 약점
- uniform only(정작 waLBerla refinement 미사용) → far-field 포함 시 비용 폭발(우리 MLG 우위 지점).
- ε 독립 제어 불가(격자 종속) → 격자수렴·ε수렴 얽힘(우리 D-sweep caveat와 동일한 함정).
- 정량 오차표·적분하중(CT/CP) 비교 없음(그림 위주). 중간스팬 실험 신뢰성 논란 자인. preprint.

## 6. 액션
1. 4-케이스 CASE1 해석 시 이 논문 프레임(무보정 ALM+LBM 계열의 로터 확장)과 나란히.
2. 커널 노벨티 단계에 Roma delta 비교군 추가.
3. 멀티GPU 노트 16에 §4의 통신/로드밸런싱 패턴 인용.
