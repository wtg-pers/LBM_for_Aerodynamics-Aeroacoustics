# 14 — Phase 1a: cell-centred coupling 설계 (f_prev 제거 + 커플링 품질)

> ★★**SUPERSEDED / 보류 (2026-07-10)** — §10 원문검증으로 **결정 B 채택: cell-vertex 유지**.
> 이유: 음향(주요 goal)용으로 특별히 발전된 라인은 **cell-vertex+Astoul et al.**(p73)이고 우리 현재 스킴이
> 그 계열. cell-centred는 프레임워크 표준이나 음향 우수성 근거가 아님 → 음향 로드맵과 상충 위험.
> ⇒ **1a(cell-vertex→cell-centred) 폐기.** 본 노트는 (a) cell-vertex geometry 정밀감사(§9), (b) Astoul/
> cell-centred 원문근거(§10) **참고자료로 보존**. 후속: cell-vertex 유지하 **메모리 트랙 재계획**(1a 제외)
> + **Astoul 개선 조사**(음향/계면 품질 트랙). 아래 §0~§8은 폐기된 cell-centred 설계(기록).

> Status: ~~DESIGN~~ **SUPERSEDED** · 2026-07-10
> 결정 근거: Phase 0 감사(patch13)로 precision=메모리레버 아님 확정 → **메모리 트랙=Option B 복귀**,
> 사용자 선택 = **1a cell-centred 단독**(메모리 −7GB + 커플링 품질). 근거논문 Markus2024 §5.1.2 / §5.2.
> ⚠ 메모리 여유 빠듯(D40 ~24GB) → §6 margin budget 포함해야 실제 성공.

---

## 0. TL;DR
현재 MLG 커플링 = **cell-vertex(co-located)**(Lagrava Sandoval/Cheylan). 이를 **cell-centred(Rohde,
explosion/coalescence)** 로 재유도한다. 효과: (i) **f_prev 제거**(temporal interp 불필요 → 메모리 −7GB),
(ii) hanging node 소멸·암묵적 보존, (iii) coalescence=암묵적 Nyquist 필터(명시적 filter 대체). 대가:
방금 커밋한 **cubic-z/C2F 융합(80a64c9)이 C→F 경로에서 무용**해지고, 기본 explosion은 저차 → **linear
explosion 필수**. 대공사이며 24GB는 여전히 빠듯.

## 1. 현재 스킴 = cell-vertex (코드 실측, `src/grid/coupling.py`)
- **layout**: coarse·fine 노드 **동일위치**. F→C `f_fine[:, 0::R, 0::R, 0::R]`(:308)로 coarse-coincident
  fine 샘플. C→F `f_fine[:, 0::2,0::2,0::2]=coarse_nodes`(:367) 짝수 인덱스에 coarse.
- **C→F**(`coarse_to_fine` :223): ①coarse sub 추출 ②**temporal interp** `0.5*(f_prev+f_now)`(:261, half-step)
  ③macro→f_eq→f_neq ④rescale `factor_c2f`(:265) ⑤**cubic upsample**(`_upsample_block` :355, 짝수=coarse,
  홀수=cubic 보간) → 6면 strip write. **fused 경로**(`coupling_rescale_d3q27` :250-256)가 ②-④ 융합.
- **F→C**(`fine_to_coarse` :283): ①0::R 샘플 ②macro→f_eq→f_neq ③**명시적 low-pass filter**(`_filter_f_neq`
  :400, 7/19-point) ④rescale `factor_f2c` ⑤excised에만 write(overlap strip은 native 유지).
- **HPC 베이스라인(80a64c9)이 최적화한 대상** = 이 경로의 cubic-z coalescing(`interpolation_d3q27`)
  + C2F rescale 융합(`coupling_rescale_d3q27`). → 1a가 이 경로를 바꾸면 **그 최적화는 C→F서 무용**.

## 2. 목표 스킴 = cell-centred (Rohde; 논문 §5.1.2, §5.2)
- **layout**: fine 노드가 coarse 대비 **반 셀 오프셋**(FV 뷰). coarse·fine **비동일위치** → hanging node 無,
  거시량 **암묵적 보존**. overlap = coarse 1셀 ↔ fine 2셀(2D=반경, 논문 Fig 5.2).
- **C→F = explosion**(Eq 5.3): `f_fine(child) = f_coarse(parent)` — coarse 셀 PDF를 2^D fine 자식에 분배.
  기본=0차(nearest). **linear explosion(Chen [170]/Schornbaum waLBerla [29])** = 1차, 계면정확도 회복 **필수**.
- **F→C = coalescence**(Eq 5.4): `f_coarse = (1/2^D) Σ_children f_fine` — 2^D 평균 = **암묵적 저역통과**
  (명시적 `_filter_f_neq` 대체). 조기 cell-vertex의 Nyquist 위반을 구조적으로 해소.
- **temporal interp 불필요**: explosion이 fine 2행(=두 서브스텝분)을 생성(§5.1.2) → **f_prev 제거**(메모리 목표).
- **f^neq τ-rescale 유지**(layout 무관): explosion 후 fine에 `factor_c2f`, coalescence 후 coarse에 `factor_f2c`.
  (평형부는 rescale 안 함 — 현행과 동일 원칙.)

## 3. 트레이드오프 (정직)
1. **cubic-z/C2F 융합(80a64c9) C→F서 무용화**: cell-centred는 explosion(copy/linear)이라 cubic 불요.
   회귀 아님(explosion이 더 쌈)이나 그 HPC 작업은 미사용. rescale 융합은 explosion+rescale 경로로 **재타겟 가능**.
2. **정확도**: 기본 explosion=0차(<현행 cubic). linear explosion 미구현시 계면 정확도 하락 →
   rotor 계면 artifact([[feedback_mlg_region_padding]]) 악화 위험. **linear explosion을 1a 필수 산출물로.**
3. **기하 재작성**: `OverlapRegion`(반 셀 오프셋), 전 slice(coarse_sub/fine_bnd/excised/`0::R`) 재유도.
4. **메모리 여전히 빠듯**(~24GB): §6 margin budget 필수. 부족시 1b(f_post) 재소환.
5. **filter_level config 소멸/재정의**: coalescence가 필터를 흡수 → `filter_level` 무의미해짐.

## 4. Touch points
- `src/grid/coupling.py` — C→F(explosion+rescale), F→C(coalescence+rescale), `_filter_f_neq` 제거,
  `_upsample_block`/boundary specs 대체, 전 slice 재유도. **주 개편**.
- `src/grid/overlap_manager.py` (`OverlapRegion`) — 반 셀 오프셋 기하 재정의(비동일위치). **먼저 정독**.
- `src/grid/multi_level_grid.py` — `_f_prev[]` 할당/copy(:115-215,358-388) 제거, `is_half_step`/
  `f_coarse_prev` 인자 제거, recursive timestep의 temporal-interp 로직 삭제.
- `src/kernels/interpolation_d3q27.py` (cubic-z) — C→F서 미사용(legacy 유지 여부 §7).
- `src/kernels/coupling_rescale_d3q27.py` — explosion+rescale로 재타겟 or 대체.
- `src/solver/setup.py`, config — `filter_level` 처리, overlap width(=coarse 1) 조정.

## 5. 단계 (stepwise; 자동모드 금지)
- **1a-0 (본 노트)**: 설계 + `overlap_manager.py` 정독 후 기하 스펙 확정.
- **1a-1**: cell-centred 기하(OverlapRegion 오프셋) + F→C **coalescence** + C→F **nearest explosion(0차)**.
  목표=**동작 + 보존**. cell-vertex(80a64c9)와 bit 아님(새 스킴) → **CV-band + 질량/운동량 보존** 게이트.
  소형 MLG(`cyl_Re90_ibb_mlg3` 또는 bench5)로 검증.
- **1a-2**: **linear explosion**(1차) → 계면 정확도 회복. 소형서 계면 매끄러움/보존 재확인.
- **1a-3**: **f_prev 제거** end-to-end(메모리) + §6 margin budget. probe 확장으로 D40 메모리 **<24GB(+헤드룸)** 실측.
- **1a-4**: 공력 검증 — HVAB 초소형 smoke로 계면 artifact↓ 및 팁하중 불변/개선 확인. 대형=사용자 클러스터.

## 6. Margin budget (24GB 성공 위해 1a에 포함)
현 D40 ~31GB, 1a(f_prev) −7GB → ~24GB(헤드룸 無). 값싼 추가절감으로 여유 확보:
- **ALM `F_grid` f64→f32**(`actuator_line.py:244`): L4(26.4M)×24B×0.5 = **−0.32GB** + precision drift 정리.
- **CUDA graph OFF 확인**(D40): 이중버퍼 없음(handoff §7 = graph dormant). 확인만.
- dyn_smag SGS work 버퍼(rho_buf/u_buf/nu_t_in) 재사용/공유 검토 = 잠재 −수백MB.
- 그래도 헤드룸 <2-3GB면 **1b(f_post) 최소부분(예: work buffer 공유)** 또는 4-case SGS 경량화 재론.

## 7. 미결 결정 (사용자)
- (a) explosion 차수: nearest(먼저 shipping) → linear(정확). **linear이 4-case 필수인지**(계면 정확도 요구).
- (b) cubic-z 커널(`interpolation_d3q27`) legacy 유지 vs 제거(cell-vertex 경로 완전 폐기 여부).
- (c) 24GB 헤드룸이 §6로도 부족하면 1b 소환 허용할지(1a 단독 고수 vs 안전 우선).

## 9. 1a-1 기하 감사 결과 (`overlap_manager.py` 정독, 2026-07-10)
**현재 geometry = cell-vertex (node 기반 co-location) 확정.** 핵심 코드:
- `fine_shape = fdx*REFINE_RATIO **+1**`(:221-225, fdx=fine_domain_coarse 축별 coarse *간격수*). **+1 = co-location**
  (coarse node가 짝수 fine index에 정확히 얹힘).
- `coarse_to_fine(ic)=(ic-origin)*R`(:269) → **짝수 fine index**(coarse-coincident). `fine_to_coarse`(:300)=역.
- `is_coarse_coincident`= 모든축 짝수(:318). F→C `f_fine[0::R]`(coupling.py:308)= 이 coincident 노드 샘플.
- overlap: `fine_domain_coarse = fine_region.expanded(overlap_width)`(:208), `overlap_width=2`(cubic stencil용).
- `excised=fine_region`(:230, coarse 내부 미계산), coupling write 면 = fine 경계(ow_f=ow*R 두께)/coarse overlap strip.

**cell-centred 재유도 시 OverlapRegion 변경점 (스펙):**
| 항목 | cell-vertex(현재) | cell-centred(목표) |
|---|---|---|
| `fine_shape` | `fdx*R + 1` | `fdx*R`(+1 제거; parent 1셀→child 2셀, 비-co-located) |
| coarse↔fine 매핑 | `*R`(짝수=coincident) | **parent-child**: coarse셀 i → fine {2i,2i+1}, 반셀 오프셋 |
| `is_coarse_coincident` | 짝수 인덱스 | **폐기**(coincidence 없음) |
| F→C 샘플 | `f_fine[0::R]` | **coalescence**: (2^D children 평균) reshape→mean |
| C→F | 짝수=coarse+cubic 홀수 | **explosion**(nearest→linear) |
| overlap_width | 2(cubic) | 1로 축소 가능(explosion stencil 좁음; 메모리 소폭↓) |
| `_factor_c2f/f2c` | τ비 | **불변**(layout 무관) |

**★새 위험 — 좌표계 리플(coupling 밖으로 blast radius 확장)**: cell-centred는 fine 노드가 coarse 대비 **반
fine-셀 오프셋**(비-co-located). fine 레벨의 물리좌표 매핑(`geometry.py` `arange(N)`, obstacle mask **cell-center
PIP**[[feedback_vof_mask_sampling]], ALM `hub_center`[[feedback_alm_hub_center_lu]], BC/sponge 위치)이 "fine node i
= fine_origin + i·dx (coarse와 co-located)" 를 가정하면, 오프셋만큼 obstacle/ALM/BC가 어긋남. → **fine 레벨
좌표 origin에 반-fine-셀 오프셋을 단일지점 반영**하면 중앙에서 해소 가능(단, 레벨 좌표 origin의 *단일 소스*
존재 여부를 1a-1 다음 단계서 확인 필요 = `grid_level.units`/`level_scaling`/`setup._build_mlg`).
  - 참고: mask가 이미 cell-center PIP라면 cell-centred와 **오히려 정합적**(vertex 가정보다 자연스러움).
- **정확 노드수(2·fdx vs 2·(fdx+1))**: coarse "간격 vs 셀" 해석에 달림 → explosion/coalescence stencil과
  맞춰 1a-1서 확정(본 노트는 원리만; +1 co-location 제거가 요지).
- 불변: `level_scaling` rescale 계수, IndexBox 유틸(expanded/clipped 등), excised 개념.

**1a-1 다음 착수**: fine 레벨 물리좌표 origin의 단일 소스 추적(좌표 리플 위험 정량화) → OverlapRegion
cell-centred 분기 작성 → coupling.py explosion/coalescence.

## 10. 논문이 cell-centred를 실제 "채택"하는가? (원문 검증, 2026-07-10)
**예 — 분류 소개가 아니라 thesis/waLBerla가 실제 구현·사용하는 방법.**
- p75: "...relies on the **cell-centred grid layout, originally developed by Rohde et al.**; later implemented
  in **WA LB ERLA by Schornbaum**; ...**Hennig builds on Schornbaum**". thesis 본인 기여도 "Building on Hennig,
  we implemented the new algorithm"(p71). 모든 결과(난류·초대형 scaling)가 이 경로.
- p바디: **Explosion**(Eq5.3) + **Coalescence**(Eq5.4)가 실제 절차. waLBerla는 **linear explosion**(Chen,
  Schornbaum 구현)까지 사용 → 우리 §5 "linear explosion 필수" 판단과 정합.
- Geier **combined/compact interpolation**(고차지만 overlap 큼)은 **대안으로만** 언급; "a **simpler explosion**
  method is implemented in waLBerla"(p~) = waLBerla는 combined 대신 cell-centred explosion 채택.

**★단, 논문의 실제 구현은 §5.2 교과서적 explosion/coalescence보다 훨씬 정교** — 우리 1a 난이도의 실체:
- **2 ghost-layer state A/B** 스킴 + **combined stream-collide**, 서브스텝 후 explosion 재분배.
- **bit-masked partial coalescence**(streaming으로 coalescence 영역에 들어온 population만 선별 합산; pystencils
  생성 `zeroCoalescenceRegion` 등). 계면 population 정확성의 핵심이자 가장 까다로운 부분.
- ⇒ 우리 1a가 "보존까지 맞는" 수준이 되려면 nearest explosion+단순평균 이상으로 이 계면 처리가 필요.
  §5의 1a-1(nearest)→1a-2(linear)에 **1a-2.5: 계면 partial-coalescence/ghost 처리**를 추가해야 정확.

**★★ 아키텍처 주의 — 음향(사용자 최종목표) 관점의 반전 가능성:**
- p73: cell-**vertex** 계열은 co-located 복사가 Nyquist 위반→spurious noise라 interpolation+filter 필요했고,
  **"Astoul et al.의 개선이 고난류에서 aeroacoustic 시뮬레이션을 가능케 했다"** 고 명시. 즉 **음향용으로 특별히
  발전된 건 cell-vertex(Astoul) 라인**. 우리 *현재* 스킴(cell-vertex+filter, Lagrava/Cheylan)이 그 계열.
- cell-centred는 coalescence로 Nyquist를 **구조적으로** 회피(장점)하나, 논문이 이를 채택한 건 범용 목적이지
  음향 우수성 근거로 든 게 아님. **따라서 "음향엔 cell-centred가 낫다"는 단정 불가** — 두 계열 모두 음향판이
  존재. 현 검증단계는 공력이라 무관하나, **장기 음향 목표엔 cell-vertex(Astoul 개선) 유지가 오히려 정석일 수
  있음** → 1a로 cell-vertex를 버리는 것이 음향 로드맵과 상충하는지 사용자 판단 필요(§7에 추가).

## 8. 근거 (논문)
- §5.1.2(p73-74): cell-vertex(co-located, hanging node, Nyquist 위반→filter 필요) vs **cell-centred**
  (Rohde, 비동일위치, 암묵보존, coalescence=암묵필터, explosion이 2행생성→**temporal interp 불필요**,
  단 기본 explosion 저차→**linear explosion** Chen/Schornbaum).
- §5.2(p74+): explosion Eq 5.3 / coalescence Eq 5.4, overlap=coarse1↔fine2 (Fig 5.2).
