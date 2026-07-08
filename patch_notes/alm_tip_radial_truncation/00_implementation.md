# 00 — 팁/root Gaussian 반경 절단 + 재정규화 (Merabet 2021)

날짜: 2026-07-06 · 대상: `src/actuator/spreading.py`, `src/actuator/actuator_line.py`,
`configs/hvab/_hvab_hover_base.py` · 배경: 판별(B) `260706_5level_test/fieldviz_B_findings_kr.md`

## 가설 (판별 B에서)

Merabet & Laurendeau 2021(S-76 hover ALM)는 별도 팁보정 없이 팁 하중을 맞춤. 레시피
중 하나 = **소스항을 블레이드 기하 끝[r_root, r_tip]에서 반경 절단 + 재정규화**. 우리
`spreading.py`는 구형 3ε cutoff만 있고 반경 절단 없음 → 팁 마커 힘의 **~30%가 r/R>1
(블레이드 밖)으로 침착**(코드+field 확인) → 팁 vortex 바깥으로 밀림 → 팁 다운워시
약화 → 과부하 기여. 판별(C)서 baseline은 힘 보존 정상(손실無) 확인 → 이 이슈는
force-PLACEMENT라 재분배로 교정 가능.

## 구현

**보존 원리**: 각 마커의 Gaussian을 [r_root, r_tip] 밖 노드에서 0으로 하고, 남은 노드를
`scale = S_all/S_kept`로 재정규화 → 마커 총 침착힘 불변(안쪽 재배치). 내부 마커는
S_kept=S_all → scale=1 → **무변화(bit-identical)**.

- `spreading.py`:
  - `_cylindrical_radius(gx,gy,gz,axis,hub)`: 노드의 축 수직거리.
  - `compute_radial_scales(...)`: 마커별 scale (CPU/GPU 공유 = 동일 결과 보장).
  - `spread_force_single_marker(..., radial=None)`: radial 주면 mask+scale 적용.
  - `spread_forces_to_grid(..., radial_trunc=None)`: scale 사전계산 후 마커별 적용.
  - GPU: `_SPREAD_KERNEL_RADIAL_SRC`(신규 커널, r_node 계산→[r_root,r_tip] 밖 skip→
    eta·scale), `_get_spread_kernel_radial`, `_spread_rawkernel_gpu(..., radial)`,
    `spread_forces_to_grid_gpu(..., radial_trunc)`. host서 compute_radial_scales로
    scale 계산 → 커널 전달(CPU와 동일).
- `actuator_line.py`: step()서 `_radial_trunc` True면 rotor {rotation_axis, hub_center,
  radius(=r_tip), blades[0].marker_r[0]−marker_dr/2(=r_root)}로 dict 빌드→spread 전달.
  로더 `create_actuator_line_from_config`: `config['spreading']['radial_truncation']`→
  `model._radial_trunc`. 단일·fine-level(`_create_fine_level_alm`) 경로 모두 통과.
- `_hvab_hover_base.py`: `build_config(..., radial_truncation=False)` → 참이면
  `actuator_line['spreading']={'radial_truncation':True}`. 기본 off.

## 검증 (전부 로컬)

1. **CPU 단위**: 기본(radial=None) bit-identical(array_equal). 절단 시 총힘 12.995
   불변(보존✓), **r>R 힘 0.999→0**(제거), 안쪽 재분배(1.73→2.49), scale 팁 2.28·내부 1.0.
2. **GPU 단위(RTX3090)**: 기본 GPU=CPU Δ2.8e-17, **절단 GPU=CPU Δ2.8e-17**, 보존✓, r>R 제거.
3. **end-to-end 스모크(CPU, 2-level, radial ON)**: 크래시無, 질량보존 +0.0000%, thrust
   finite. ON vs OFF torque 0.094416 vs 0.094419(≠0=active; 8스텝이라 field 미발달로
   총추력은 보존상 동일).
4. **config**: testC 2종 spreading 플래그有, 기존 baseline(testB_slab5_pure) spreading=None
   (bit-identical 보존). py_compile OK.

## 프로덕션 A/B (사용자 클러스터)

pure ALM 기준(판별 결정), **light 먼저→slab5**:
- `configs/0703_dag_edge_fix/testC_light_pure_radtrunc.py`(light 25rev) vs off baseline
  `260630/260630_results_nasa_c81/pureALM_csv`.
- `configs/0703_dag_edge_fix/testC_slab5_pure_radtrunc.py`(slab5 15rev, ε=0.25c 청정조건)
  vs off `260706_5level_test/slab5_pure_alm_csv`.
판정: M²Cn 팁이 실측 0.146 쪽으로 하락하는가 + 방위각평균 field 팁 inflow 다운워시 회복.
(slab5는 5-level이라 사전 smoke 권장.)

## 성능 (host 비용) — boundary-only 최적화

`compute_radial_scales`는 매 스텝 host(numpy) 실행(재정규화가 보존에 필요; scale은
방위각 불변 아님 — 격자 aliasing으로 팁 scale 2.09~2.28 ~8% 변동 → **캐시-once 하면
보존 깨짐** → 매 스텝 재계산 필수). 대신 **내부 마커는 scale≡1(절단 없음)**이므로
`r_m ± n_cut·ε`가 [r_root,r_tip]에 닿는 경계 마커만 stencil 계산. slab5 스케일서
48마커/블레이드 중 8개만 계산(40개 스킵) → **192→~32개, 6× host 절감**, 정확성 불변
(max|Δ|=0). GPU 커널은 모든 노드에 mask 적용(경미), scale은 host서 boundary만.
남은 host 비용이 여전히 GPU stall이면 = Phase 1c(ALM 오버랩/CUDA graph, 별도 HPC 트랙).

## latent (판별 C 기록) — 미구현
`alm_target_level`이 hub 마진만 확인(디스크+3ε 미검사). 현 config 무해하나, 반경 절단이
켜져도 팁 마커가 레벨 edge 밖으로 나가면 절단 전에 이미 도메인 clip 손실 가능 →
"디스크+3ε ⊂ ALM레벨" 가드 권장(별도).
