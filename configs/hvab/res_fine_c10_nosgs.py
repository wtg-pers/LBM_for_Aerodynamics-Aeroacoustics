"""격리 테스트 2 — fine + **SGS OFF**. fine NaN이 dyn_smag(난류모델) 탓인지 판정.

fine은 use_sgs=True(dyn_smag). 5번째 레벨/고해상 로터 근처(고 strain)에서 dynamic
Smagorinsky가 비물리 ν(음수/폭주)를 낼 수 있음(cf. Luca 2024: 2D Smag 과소산·kernel 의존).
  → nosgs로 step5를 넘기면: 원인 = **dyn_smag**(SGS 모델/계수).
  → 그래도 죽으면: SGS 아님 → 5-level coupling 또는 force 스케일.
(fine과 동일 ~70GB → DGX. 마찬가지로 몇 스텝만 보면 됨.)
    python main.py --config configs/hvab/res_fine_c10_nosgs.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="fine", use_sgs=False,
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    run_tag="iso_fine_nosgs",
)
config["time"]["max_steps"] = 30          # 격리: step5+ 넘기면 SGS 용의
