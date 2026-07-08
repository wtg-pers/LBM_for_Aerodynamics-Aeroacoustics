"""HVAB hover c10 — 비등방 Gaussian A/B: ISOTROPIC 대조군 (순수 ALM).

① 비등방 Gaussian(Natelson 2026 Eq.7 / Churchfield 2017) 검증 A/B 쌍의 대조군.
  - 순수 ALM: prandtl_loss OFF, eps_correction OFF, tip_sweep OFF
    → 짝 파일과 유일한 차이가 Gaussian 등방↔비등방 뿐.
  - 등방 = 현행 baseline(구형 ε). 짝: hvab_hover_c10_aniso.py.
  - 판정: 팁 M²Cn(compare_M2Cn) — EXP tip 0.146 대비 과예측(~0.39) 완화 여부.
  light preset(D32, ~12.5GB), NASA OVERFLOW 덱(기존 pure-ALM 벤치와 동일 앵커).
  patch_notes/alm_natelson_aniso_gaussian/.
    python main.py --config configs/hvab/hvab_hover_c10_aniso_iso.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    polar_source="nasa_overflow", n_rev=25, run_tag="aniso_iso",
    # anisotropic=None → 등방 Gaussian (대조군)
)
