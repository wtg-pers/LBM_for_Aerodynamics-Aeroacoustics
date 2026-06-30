% SU2 + LBM 커플링 설계 가이드 (v0 초안)
% LBM 솔버 개발 — 전략 방향 정리
% 2026-06-30

---

## 0. 이 문서의 목적

LBM 솔버의 **long-term 목표 = SU2(FVM URANS)와의 커플링**으로 확정됨에 따라,
메모리/병렬화/아키텍처 의사결정을 그 목표 기준으로 재정렬한 v0 설계 메모.
구현 명세가 아니라 **방향·범위·열린 질문**을 담는다. (구현은 별도 patch_notes로.)

---

## 1. 아키텍처 결정 (토폴로지)

**Near-body / Off-body 존(zonal/hybrid) 커플링** — Helios(NSU3D/OVERFLOW 근접 +
Cartesian 외부)류.

| 영역 | 솔버 | 담당 | 강점 근거 |
|------|------|------|-----------|
| 근접장 (body-fitted) | **SU2 URANS** | 경계층, 표면/블레이드 하중 | RANS=벽경계 유동에 강함 |
| 외부장 (Cartesian/MLG) | **LBM** | 후류 대류, **음향 전파** | LBM=저소산·explicit, 공력음향에 최적 |

- 정보 교환은 **두 영역이 겹치는 overlap 경계에서만** (boundary-only).
- 따라서 CPU(SU2)↔GPU(LBM) 전송량은 작음 → 데이터 전송은 병목이 아님.
- **단, 1 step 연산이 매우 빨라야 함** (tight 2-way의 생명선).

---

## 2. 역할 분담 & 교환 데이터

- **SU2 → LBM**: 내부 overlap 경계에 macroscopic 상태 (u, p 또는 ρ) 제공
  → LBM은 이를 Dirichlet 속도/압력 BC로 받아 **populations 재구성**(f = f_eq + f_neq).
  - 재구성 로직은 **현재 MLG coupling(coarse→fine)에 이미 존재** → 재활용.
- **LBM → SU2**: 외부 overlap 경계에 macroscopic (u, p) 제공 → SU2의 외부 경계조건.
- = 분할형 **Dirichlet–Neumann** (또는 D–D) 교환.

---

## 3. 투자 범위 (어디까지) — IN / OUT

핵심 통찰: **multi-GPU halo 교환 인프라 = SU2 경계장 교환 인프라.**
한 번 만들면 둘 다 충족한다. 따라서 투자는 그쪽으로 수렴한다.

| OUT (동결/폐기) | 이유 |
|---|---|
| ALM 팁해상 / fine preset / chord-16 | ALM은 long-term 폐기 예정 (§9) |
| esoteric 단일-population | 커플 구도서 per-GPU 메모리 압력 ↓ (LBM=중간해상 외부장) |
| cell-centred 결합 재작성 | 격자 표현 재작성(node_map/BC/ALM/VTK 파급) = 최고위험, 이득 작음 |
| FP16 storage | 논문(Holzer p.146) 비추천(절단 심함); cumulant FP32가 sweet spot |

| IN (목표 정렬 투자) | 비고 |
|---|---|
| **MPI multi-GPU 도메인분할** | 외부장이 큼 → 필수. SU2도 MPI라 커플은 본질적 multi-node |
| **범용 "경계장 교환" 레이어** | halo(레벨내/GPU간) + SU2 인터페이스 겸용 (§5) |
| **per-step 속도** | tight 2-way 핵심 (§7) |
| (선택) f_prev strip-only | 저위험 ~5GB/GPU 보너스, vertex 격자 유지 |

> per-GPU 메모리 극한최적(esoteric/cell-centred)은 **스케일아웃 후에도 특정
> 외부장 케이스가 메모리 binding일 때만 재고**. 기본은 "GPU를 더 붙인다".

---

## 4. 경계장 교환 인터페이스 (저후회 첫 투자)

어떤 커플 강도를 고르든 필요한 공통 토대. 추상화:

```
BoundaryExchange:
  inject(region, u, rho/p):   외부 macroscopic → 해당 경계 셀에 populations 재구성
  extract(region) -> u,rho/p: 해당 경계 셀 populations → macroscopic
```

- 이 한 추상화가 (a) GPU간 halo, (b) MLG 레벨간 결합, (c) SU2 인터페이스를 모두 표현.
- 재구성 = f_eq(ρ,u) + f_neq (비평형 부분은 점성비로 rescale) — MLG coupling에 구현됨.
- 커플러: **preCICE** 권장 (SU2 공식 adapter 보유; MPI rank간 non-matching mesh
  매핑(RBF/투영) + 시간 서브사이클링 처리). LBM측에 preCICE adapter만 작성.
  - non-matching mesh를 preCICE가 처리하므로 **LBM을 cell-centred로 바꿀 필요 없음**.

---

## 5. 2-way tight 커플링 스킴 (열린 설계 공간 — 깊은 논의 필요)

"매 스텝 2-way"의 구체화. 여기가 **fast-step vs tight 안정성** 트레이드오프의 핵심.

1. **시간 동기 / 서브사이클링**
   - LBM Δt (음향 CFL) ≪ SU2 URANS Δt (implicit, 큼).
   - 자연스러운 cadence = **SU2 1스텝당 LBM N스텝 sub-cycle**.
   - "매 LBM 스텝마다 2-way"는 implicit RANS엔 비현실적 → 교환은 SU2 스텝 경계 기준.

2. **explicit(loose) vs implicit(tight) 커플**
   - explicit: 스텝당 1회 교환 → 빠름. 단 강결합/저질량비서 불안정 가능.
   - implicit: SU2 스텝 내 서브반복(+Aitken under-relaxation) → 안정. 단 SU2 여러 번 → 느림.
   - **"fast step" 요구와 "tight 2-way 안정"이 여기서 충돌** → 케이스로 결정.
   - preCICE가 둘 다 지원.

3. **인터페이스 조건**: Dirichlet–Neumann (SU2가 u 제공→LBM 속도BC; LBM이 u,p→SU2).
   우리 eq/neumann BC로 LBM측 수용 가능.

4. **overlap 존(Schwarz) vs 날카로운 인터페이스**
   - LBM–NS 커플엔 **겹침 존**이 훨씬 안정적 (우리 MLG overlap 개념과 호환).
   - 존 두께·완화(blending) 설계 필요.

---

## 6. per-step 성능 — 진짜 병목은 데이터가 아니라 오케스트레이션

경계-only라 전송량은 작다. tight 2-way에서 위험은 **스텝당 Python/CuPy 오버헤드**
(커널 런치 지연·host 동기화가 수많은 작은 스텝에 누적).

- 대응 1: LBM 스텝 **완전 GPU-resident** (host 동기 제거, .item()/.copy() 금지 경로).
- 대응 2: **CUDA Graph 캡처**로 스텝당 런치 오버헤드 제거.
- 대응 3(장기): 컴파일형 백엔드 (§8).

---

## 7. build vs adopt (의식적 포크)

| 옵션 | 장점 | 단점 |
|---|---|---|
| 자체 CuPy 솔버 확장 | ALM·검증된 물리 보유, 완전 제어 | multi-GPU+커플 인프라 직접 구현(수개월), 스텝당 Python 오버헤드 |
| **waLBerla/lbmpy** 외부장 채택 | GPU+격자세분+preCICE 예제 보유, 스텝당 오버헤드~0, exascale 검증 | 학습/이식 비용, ALM 재구현 |

→ 권고: 자체 솔버를 **물리검증·프로토타입 vehicle**로 유지하되, **tight·fast·multi-GPU·
coupled 외부장 production**에는 waLBerla 채택을 한 번은 진지하게 비교. (지금 결정 강요 아님.)

---

## 8. ALM 위상 (near-term vs long-term)

- **long-term**: SU2 body-fitted가 블레이드 해상 → **LBM ALM 폐기** (sub-option 동결,
  추후 SU2-ALM 업그레이드는 후순위). ALM은 "다른 기법과의 커플링 연습"으로 역할 완료.
- **near-term (지금)**: **HART2 로터 workshop**용으로 ALM 유지·활용.
  - 다음 작업 = **ALM multi-GPU 버전** → DGX의 Watanabe(fine, 단일노드) 조건 vs
    클러스터 multi-GPU 버전 비교.
  - 이 multi-GPU 작업은 **§3의 IN 투자(MPI 도메인분할)와 동일 인프라** → 버려지지 않음.
    즉 HART2 deliverable과 long-term 커플 인프라가 multi-GPU에서 만난다.

---

## 9. ★ 당신이 고민해야 할 항목들 (의사결정 체크리스트)

**커플링 스킴 (§5)**
- [ ] 교환 cadence: SU2 1스텝당 LBM N스텝 — N을 무엇으로 정할지(고정 vs Δt 비율 자동)?
- [ ] explicit(loose) vs implicit(서브반복)? 안정성과 "fast step" 중 무엇을 우선?
- [ ] implicit이면 under-relaxation 방식(상수 vs Aitken)과 수렴 판정?
- [ ] overlap 존 두께 / blending 함수 / 인터페이스 BC 종류(D-N vs D-D)?
- [ ] 압력/밀도 변환: LBM 압축성(약압축) ↔ SU2(압축/비압축?) — 음속·기준밀도 정합?

**경계장 교환 (§4)**
- [ ] preCICE 채택 vs 자체 경량 소켓/파일 교환? (preCICE 학습비용 vs 표준화 이득)
- [ ] non-matching mesh 매핑 방식(RBF/nearest/conservative) — 보존성 요구 수준?
- [ ] SU2 adapter 존재 확인 + LBM adapter 작성 범위?

**병렬/하드웨어 (§3, §6)**
- [ ] LBM=GPU + SU2=CPU 이종 → 같은 노드 배치 vs 분리 노드? 통신 경로?
- [ ] LBM multi-GPU 분할을 SU2 분할과 독립(preCICE가 bridge) vs 정렬?
- [ ] per-step 속도 목표치(예: 외부장 1 sub-step < X ms)? CUDA Graph 도입 시점?

**전략 (§7, §8)**
- [ ] 자체 솔버 확장 vs waLBerla 채택 — 결정 시점/기준(예: multi-GPU 성능 벤치 후)?
- [ ] ALM을 어디까지 유지? (HART2 후 동결 vs SU2-ALM로 이관?)

**검증**
- [ ] 커플링 검증 케이스(예: 단순 분할 도메인 Taylor-Green / 채널 / 실린더 후류)?
- [ ] 인터페이스에서 보존성(질량/운동량) 및 음향 무반사(반사 최소화) 점검 방법?

---

## 10. 참고 & 다음 단계

- 참고: Holzer 2024 PhD thesis (lbmpy/waLBerla, exascale; D3Q27 K17 cumulant→CuPy);
  preCICE (분할형 multiphysics 커플러, SU2 adapter); Helios near/off-body 패러다임.
- 관련 메모리: `project_su2_coupling_direction`, `project_int32_kernel_ceiling`,
  `reference_hvab_cfd_benchmarks`.
- **즉시 다음 작업(사용자 지정)**: HART2용 **ALM multi-GPU** 구현 → DGX Watanabe(fine)
  vs 클러스터 multi-GPU 비교. (이 인프라가 §3 IN 투자와 동일.)
- **저후회 병행 투자**: §4 경계장 교환 추상화 설계.

*(v0 — 커플링 스킴 §5 및 체크리스트 §9는 추후 논의로 갱신 예정.)*
