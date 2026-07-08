"""HVAB hover c10 — ①+② 결합: 비등방 Gaussian + 링 샘플링 (Natelson 완전체).

2×2 A/B 매트릭스의 (비등방 G, 링 S) 케이스. 나머지 3:
  aniso_iso(등방G,점S)=baseline / aniso(비등방G,점S) / ring(등방G,링S).
네 config는 순수 ALM(prandtl/eps_corr/sweep OFF)·light·NASA덱·25rev로 완전 동일하고
Gaussian(등방↔비등방)과 샘플러(점↔링) 두 knob만 다름 → 깨끗한 2×2.

  ① 비등방 force projection: 폭 배율 c=1.0, t=0.5, r=1.0 (Martínez-Tossas+Churchfield).
  ② 링 인플로우 샘플링: 단면평면 원 20센서 trilinear 평균, R_ring=1.0·ε (Natelson/ROAM).
목적: force 분포(익형 형상 근접)와 인플로우(자기유도 상쇄)를 동시에 개선했을 때
  팁 M²Cn 과예측(등방·점 ~0.39 vs EXP 0.146)이 얼마나 회복되는지 확인.
  patch_notes/alm_natelson_aniso_gaussian/ + alm_natelson_ring_sampling/.
    python main.py --config configs/hvab/hvab_hover_c10_aniso_ring.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False,
    anisotropic={"c": 1.0, "t": 0.5, "r": 1.0},                  # ①
    sampling={"mode": "ring", "ring_n": 20, "ring_r_factor": 1.0},  # ②
    polar_source="nasa_overflow", n_rev=25, run_tag="aniso_ring",
)
