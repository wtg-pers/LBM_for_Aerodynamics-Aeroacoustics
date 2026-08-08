# Octo-8 vehicle — 8x APC 18x8E + STL body 호버 본 런 (2026-08-08)

`assembly_full.stl` 기체(CATIA ASCII·mm·352,960 facets, 전장 2.07m × 폭 3.82m)
+ APC 18x8E 8기 ALM 하버링. RPM 5000, 40 rev, 단일 GPU.

## 실행 (메인 디렉토리, 단일 GPU 전용 — `--mpi` 금지)

```bash
LBM_ESOTERIC=1 python main.py \
    --config configs/octo8/octo8_hover_5000rpm_hwbb.py --gpu 2
```

## 레이아웃 (CAD mm, 노즈 원점; 회전방향=위(+z)에서 볼 때, 0808 2차 확정)

사용자 기술("노즈가 화면 아래·왼→오": 전방 CCW,CW,CCW,CW / 후방 CW,CCW,CW,CCW)
을 CAD로 변환(위에서 볼 때 화면 왼쪽=−y) → **체커보드**:

| rotor | x | y | 방향 | rpm |
|---|---|---|---|---|
| f1 | −302.24 | −1288.5 | CCW | −5000 |
| f2 | −302.24 | −548.5 | CW | +5000 |
| f3 | −302.24 | +548.5 | CCW | −5000 |
| f4 | −302.24 | +1288.5 | CW | +5000 |
| a1 | −1325.24 | −1288.5 | CW | +5000 |
| a2 | −1325.24 | −548.5 | CCW | −5000 |
| a3 | −1325.24 | +548.5 | CW | +5000 |
| a4 | −1325.24 | +1288.5 | CCW | −5000 |

- 검증: `octo8_preview_step0.py`(step 0/105 두 프레임) 마커 VTP로 ParaView에서
  회전방향 육안 확인 가능.
- z=+133.711mm 평면, 추력 +z(호버), 후류 −z → zmin sponge.
- 핸디드니스 인코딩(프로브 실증): 전 로터 `rotation_axis=[0,0,−1]`,
  `thrust_direction=[0,0,1]` 공통, **rpm 부호 = 방향**(+CW/−CCW), 양방향 모두
  +z 추력 확인. `theta_0` 로터별 π/8 스태거(blade-passage 동기화 방지).

## 지면 (IGE, 0808 추가)

- 호버링 높이 955.5mm = **지면 → 로터면(CAD z=+133.711mm)** (사용자 확정)
  → 지면 z_cad = −821.789mm, **z/D = 2.09** (기어 최하단은 지면 위 689.7mm)
- 지면 = zmin **hwbb no-slip 벽** (half-way라 벽면=노드0 −0.5셀; O_z=71.398로
  로터면→벽면 정확히 955.5mm 검증)
- 다운워시 → 지면 radial wall jet → **측면 4면 sponge**(두께 20, 강도 0.1)로
  흡수, 유입은 zmax eq

## 격자/수치

- **uniform 단일 레벨** (MLG off), dx = D/40 = 11.43mm, 도메인 324×496×200
  = **32.1M 셀** (eso ~6.7GB / std ~13.2GB). z 상단 = 로터면 위 2.92D
- CAD 원점 → LU (266, 248, 172); STL은 bbox 중심→center_lu 변환으로 역산
  (bbox 하드코딩 — **STL 교체 시 빌더의 _BBOX_* 갱신 필수**)
- u_max=0.1, 1257 steps/rev, 40 rev = 50,280 스텝
- 포뮬레이션 = 단일로터 notl 확정 구성: kleine straight + radial truncation
  + iso gaussian, tip loss OFF, dyn_smag, ramp 1 rev
- 마커 n_radial=32 (δr=0.625Δx ≤ ε), ε=0.25c → 전 스팬 2Δx floor(≈0.1R)
- 바디 **IBB**(Bouzidi 보간 bounce-back, 0808 사용자 지정) — 계단벽 hwbb 대비
  아웃워시↔기체 간섭 재현에 유리. 스모크: 402,598 wall link, q sentinel 0개

## VTK / 후처리

- `output_interval=105` = 30.07° 간격. **마커 VTP는 전 구간**, full-field는
  `output.vtk.fields_start_step`(=마지막 5바퀴, step 43995~)부터만
  → field 60장 × ~1.16GB ≈ 70GB
- CT/CP 요약: `python src/utilities/prop_hover_post.py --results
  "./result_octo8_*" --avg-revs 5` (rotor_performance.csv는 로터별 행 —
  멀티로터 출력 형식 확인 후 필요시 도구 보강)

## ★MLG 사용 금지 사유 (0808 실측 확인)

이 케이스가 **uniform 단일 레벨**인 이유(선택이 아니라 제약):

1. **같은 레벨에 박스 여러 개 = 미지원.** MLG는 레벨당 정확히 박스 1개인
   **엄격 중첩 텔레스코프**(L1 ⊇ L2 ⊇ L3). `levels[k]['region']`은 단일 dict,
   `OverlapManager.add_level_pair`는 레벨쌍마다 1개 region만 등록
   (`src/grid/overlap_manager.py:585`). 로터 8기에 각각 작은 fine 박스 = 불가.
2. **★멀티로터 + 로터를 덮는 fine 박스 = 로터 추력이 조용히 소실.**
   멀티로터 ALM은 L0 고정(`setup.py:1519`), 그런데 F2C가 "coarse excised
   region을 fine 데이터로 덮어씀"(`multi_level_grid.py:533`) — fine 레벨은 ALM
   힘을 모르므로 L0에 주입된 운동량이 매 스텝 지워짐.
   실측(2로터 D16 프로브, uniform vs L1 박스가 로터 덮음, 300스텝):
   | | 마커 mean\|u_n\| | T_lu |
   |---|---|---|
   | uniform | 7.99e-4 | 0.0318 |
   | MLG(로터 덮음) | **0.000e+00** | 0.0414 (+30%) |
   유도 다운워시가 정확히 0 = 힘이 유체에 전혀 들어가지 않음. **경고 없음**
   (게다가 배너는 `@L1`로 잘못 표기 — `setup.py:2340` 표시 결함, ALM은 실제 L0).
   → 해상도를 올리려면 **dx를 전역으로 낮추는 방법뿐**.

## 솔버 제약 (0808 코드 확정) + 이번에 넣은 src 패치

- 멀티로터 ALM = **MPI 미지원**(단일 GPU 강제), MLG fine-level 미지원(→ uniform)
- 패치(0808): ① setup: `rotors` 리스트도 hub_center LU→m·rpm→omega 변환(단일과
  동일 컨트랙트) ② UnitConverter: `rotors` 인식(최대 |rpm|·R 앵커, abs(rpm))
  ③ create_multi: eps_correction/spreading/prandtl_loss/ramp_steps 전달(기존
  누락) ④ output: `vtk.fields_start_step` 게이트(단일 GPU용 fields-last 대응)

## STL 수리 (0808, `assembly_full_repaired.stl` 사용)

원본 `assembly_full.stl`은 로더 거부(비수밀: 역와인딩 나사 32개 + 캡 없는 로드
4개 + 단면 링 8개). 수리 레시피(trimesh+manifold3d, 스크립트 재현 가능):
나사 제거(2.9mm — dx=11.4mm 서브그리드) → 로드를 캡 실린더로 재구성(반경
+0.3mm·양단 +2mm 연장 = 접선 접촉을 횡단 교차로 만들어 union 슬리버 방지) →
manifold 불리언 union → **로더 게이트(f32 exact-weld) 직접 통과 확인**.
bbox 원본 동일(−2125.79~−55 / ±1911 / −132.04~505 mm). STL 교체 시 동일 수리
+ 빌더 `_BBOX_*` 갱신 필요.

## 스모크 검증 (0808, 로컬 3090, 200스텝)

- 빌드+실행 클린(NaN 없음), 8로터 전원 등록, hwbb 마스크 정상
- **회전방향 실증**: CCW 4기 Rev −0.16·Q<0, CW 4기 Rev +0.16·Q>0(반토크 상쇄
  잔차 ~2%), 전 로터 T>0(+z 추력), 로터간 T 편차 <4%
- 성능: ~1.09 step/s (3090, 48.2M+8ALM+hwbb+eso) → 50,280스텝 ≈ **13시간/런**

## 한계 명시

- 로터 해상 dx=D/40 (단일로터 캠페인은 D/320): 절대 CT 재기준선 필요 —
  단일로터 notl 결과(CT −5.5% @2446)가 앵커, 여기는 기체 통합/상호작용 목적
- hwbb 바디: τ→0.5 정류 펌핑 병리(patch_notes/stl_body 13) — 바디 힘 판독은
  참고치. 정밀 바디 힘이 필요해지면 surfel 트랙으로 격상
- 폴라 한계는 단일로터와 동일(neuralfoil, t/c 미반영, Mach 보정 없음)
