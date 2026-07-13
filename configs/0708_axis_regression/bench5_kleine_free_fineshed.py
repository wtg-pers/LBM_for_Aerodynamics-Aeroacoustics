"""bench5 free-wake — shed 간격 2°→0.5° (팁 근접장 이산화 테스트).

진단(ALM_KLEINE_DIAG)이 확정: free-wake 팁 deficit이 straight의 ~1/10~1/12로 약함.
가설 (A): 원인이 2° shed 간격이 팁 반경서 4.4 lu(>ε~2 lu)라 근접장 과소해상 →
  더 촘촘히(0.5°) shed하면 free 팁 w가 straight로 수렴할 것.
가설 (B): 근본 헬릭스 curve 기하 → 촘촘히 해도 안 수렴(straight가 정답).

이 런을 ALM_KLEINE_DIAG=1로 돌려 팁 w_free가 straight(+0.016급)로 커지는지 확인:
    ALM_KLEINE_DIAG=1 python main.py \
        --config configs/0708_axis_regression/bench5_kleine_free_fineshed.py 2>&1 | grep KLEINE_DIAG

bench5_kleine_free.py와 **wake_azimuth_deg(2.0→0.5) 하나만** 다름 (n_w 자동 4×↑).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="bench5",
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "rebuild_every": 1,
                    "wake_markers": "all", "target": "inviscid", "smooth": 2,
                    "wake_azimuth_deg": 0.5},          # ← 유일 변경 (2.0 → 0.5)
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform",
    n_rev=3, polar_source="nasa_overflow", run_tag="axisreg_kfree_fineshed",
)
