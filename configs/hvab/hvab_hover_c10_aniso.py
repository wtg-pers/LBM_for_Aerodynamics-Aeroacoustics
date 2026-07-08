"""HVAB hover c10 — 비등방 Gaussian A/B: ANISOTROPIC (Natelson 2026 Eq.7 / Churchfield 2017).

짝: hvab_hover_c10_aniso_iso.py (등방 대조군). 순수 ALM, Gaussian force projection만 비등방.

폭 배율 (등방 ε_iso=max(chord/4,2Δx) 대비):
  c = 1.0  — chord:  ε_c=ε_iso≈0.25c = Martínez-Tossas 2017 최적(등방과 동일).
  t = 0.5  — thickness: 물리 0.25·두께는 sub-grid(팁 8%익형→0.08·ε_c)라 해상 불가 →
             최소 해상 ε_t≈2Δx, 팁서 ε_iso≈3.9셀 대비 ≈0.5배(해상도 정합값).
  r = 1.0  — radial(span): 요소폭·overlap 위해 chord 스케일 유지.
근거: Martínez-Tossas et al. 2017(등방 최적 0.25c) + Churchfield 2017(비등방 elliptic).
목적: 팁 force 분포를 실제 익형(넓은 chord·얇은 thickness)에 근접 → 팁 M²Cn 과예측 완화 확인.
  light preset(D32), NASA OVERFLOW 덱. GPU 비등방 커널 초회 실사용(로컬 로직검증 완료).
    python main.py --config configs/hvab/hvab_hover_c10_aniso.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    polar_source="nasa_overflow", n_rev=25, run_tag="aniso",
    anisotropic={"c": 1.0, "t": 0.5, "r": 1.0},
)
