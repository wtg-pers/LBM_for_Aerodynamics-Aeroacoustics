"""HVAB hover c10 — Kleine 2022 비반복 smearing 보정 (Phase 1) A/B용.

pure ALM(prandtl OFF) + eps_correction method="kleine". ABC-A(gaussian) / Dağ와
동일 격자(light, 4-level)에서 dag vs kleine 팁 φ·α·CT 비교.
주의(정직): Phase 1 Kleine는 팁에서 Dağ와 동급(같은 직선-wake 커널). 차이는 inboard 분포
+ 비반복 안정성. 드라마틱한 팁 개선은 Phase 2(free-wake) 예정.
실행 후 fallback 빈도(불안정 솔브→Dağ) 관찰 — 잦으면 relax 옵션 검토.
    python main.py --config configs/hvab/hvab_hover_c10_kleine.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    eps_correction={"enabled": True, "method": "kleine"},   # Phase 1
    run_tag="kleine",
)
