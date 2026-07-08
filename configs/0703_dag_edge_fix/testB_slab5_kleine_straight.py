"""TEST B — slab-L4 grid (light_slab5), Kleine Phase-1 STRAIGHT wake, 15 rev.

★ Lead-1 격리 런: free-wake 팁 부호 뒤집힘이 free-wake '기하'에서 오는지 확정.

testB_slab5_kleine_free.py 와 **`wake` 키 하나만** 다름 ("free" → "straight").
method(kleine)·비반복 solve·smooth·target·격자(light_slab5)·NASA덱·gaussian·
uniform·prandtl OFF·15rev 전부 동일 → 팁 부호 차이의 유일 변수 = wake 기하.

배경(진단, _jobs/0707_analysis_update_process/HVAB_overprediction_analysis_kr.md):
  slab5_kleine_free(Phase-2 free)는 r/R≥0.95에서 Δα=+0.4~0.9° (upwash-signed,
  팁을 오히려 up-load) — 내측은 정상(Δα<0). Phase-1 straight 커널은 패치노트
  alm_dag_edge_fix에서 팁 downwash +0.0186(정상 부호)로 검증됨(influence_matrix).

판정(같은 slab5 격자에서 대조):
  · straight가 팁을 de-load(Δα<0, M²cₙ↓)하고 free만 up-load(Δα>0) →
    버그는 free-wake 기하(_convect_and_shed_wake, hover 가드 부재)로 100% 확정.
  · straight도 팁을 못 내리면 → 부호가 아니라 straight 커널 강도/해상도 문제.
비교쌍:
  free  = aeromechanics_workshop/HVAB/260706_5level_test/slab5_kleine_free_csv
  pure  = .../slab5_pure_alm_csv
  (참고) 독립 교차검증용 straight = testB_slab5_dag_straight.py (method=dag, 다른 solve)

주의: 5-level slab 격자는 첫 실행 전 짧은 smoke(~300 step, nan_trap ON) 권장.
      light_slab5는 도메인 축소(상류 0.75D) → 절대 CT/FM은 light와 직접비교 불가,
      판정은 스팬 M²cₙ·w_corr·팁 α로.

    python main.py --config configs/0703_dag_edge_fix/testB_slab5_kleine_straight.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light_slab5",
    # 'wake'는 kleine·dag 공유 키(actuator_line.py:2070-2072). method="kleine"에선
    #   wake ∈ {"straight"(Phase-1 influence_matrix), "free"(Phase-2 convected)}.
    #   ∴ 아래는 Phase-1 straight Kleine (= hvab_hover_c10_kleine.py에서 wake 생략과 동일).
    #   free config와의 차이를 '한 토큰'으로 만들려 명시("free"→"straight").
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "straight",          # ← 유일 변경 (free → straight)
                    "target": "inviscid", "smooth": 2},
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=15, polar_source="nasa_overflow", run_tag="testB_slab5_kleine_straight",
)
