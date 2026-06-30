"""HVAB hover c10 — Kleine 2022 Phase 2 (free-vortex wake) A/B용.

method="kleine" + wake="free": convected free-vortex wake가 팁와류 helix/수축을
포착 → 직선 Phase1(=Dağ)보다 큰 팁 deficit 기대. pure ALM(prandtl OFF), light 격자.
A/B: hvab_hover_c10_kleine.py(straight=Phase1) vs 이 파일(free=Phase2) vs Dağ vs baseline.
성능: per-step free-wake 영향행렬 재빌드(~100ms/step CPU) + wake점 이류 샘플.
실행 후 fallback 빈도·팁 φ/α 회복 관찰. (synthetic smoke는 수축 없어 free≈straight였음 —
실제 LBM 수축에서 차이 확인.)
    python main.py --config configs/hvab/hvab_hover_c10_kleine_free.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "n_w": 50},
    run_tag="kleine_free",
)
