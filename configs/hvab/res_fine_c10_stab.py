"""격리 테스트 3 — fine + MLG coupling 강건화. 5-level NaN이 deep-interface 보간 탓인지.

진단: τ 스케일은 정확(level-4가 더 점성↑) → 버그는 MLG coupling 쪽 추정.
cubic 보간 overshoot / overlap_width=2 부족이 깊은 5번째 interface에서 폭주를 낼 수 있음.
이 변형: overlap_width 2→4, interpolation cubic→linear (둘 다 강건한 쪽).
  → 살면: 원인 = deep coupling(보간/overlap). 그럼 overlap만↑ vs linear만 따로 좁힘.
  → 그래도 죽으면: coupling도 아님 → region 생성/level-4 인덱싱 더 깊게 봐야.
(fine과 동일 ~70GB → DGX. step5+ 넘기면 판정.)
    python main.py --config configs/hvab/res_fine_c10_stab.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="fine",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    run_tag="iso_fine_stab",
)
config["mlg"]["overlap_width"] = 4          # 2→4 (보간 stencil 여유)
config["mlg"]["interpolation"] = "linear"   # cubic→linear (overshoot 제거)
config["time"]["max_steps"] = 30
