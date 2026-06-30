"""HVAB hover c10 — 표준 Prandtl (R_tip_eff = R), 보정 없음. 25-rev.

case 2의 Prandtl confound 분리용. 기존 baseline `hvab_hover_c10.py`(=case 2)는
`prandtl_loss=True`(bool)→ legacy `R_tip_eff = R−ε_tip`(외측 ε밴드 hard-zero, 공격적
tuned crutch). 이 파일은 `eps_offset=False`로 **표준 R_tip_eff=R**(BEMT/교과서 일치,
ε-decoupled) → "Prandtl"과 "R−ε hard-zero"를 분리. baseline보다 덜 공격적이라 C_T↑ 예상.

보정(eps_correction)·taper 없음, pure gaussian. 다른 케이스와 동일 light·M0.65.
    python main.py --config configs/hvab/hvab_hover_c10_prtipR.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    prandtl_loss={"enabled": True, "eps_offset": False},   # 표준 R_tip_eff = R
    sampling={"mode": "gaussian"},
    n_rev=25, run_tag="prtipR",
)
