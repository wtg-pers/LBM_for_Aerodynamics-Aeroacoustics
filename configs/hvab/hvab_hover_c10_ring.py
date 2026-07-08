"""HVAB hover c10 — ② 링 인플로우 샘플링 A/B (Natelson 2026 / ROAM·Helios).

대조군: hvab_hover_c10_aniso_iso.py (순수 ALM, Gaussian ±3ε 점 샘플). 이 파일은
샘플러만 링으로 교체 → 유일 차이가 인플로우 샘플링 방식.

링 샘플러: 각 스테이션 주위 원 위 N=20 센서에서 trilinear 샘플 후 평균.
  - 평면 = 익형 단면 평면(ê_c·ê_t span; ê_r에 수직) — Natelson Fig.2.
  - 반경 R_ring = 1.0·ε (논문 미제시 → smearing 폭 스케일 추천값).
  목적: 마커 자기 body-force 결손(중심)을 링 평균으로 상쇄 → 블레이드가 보는 유입에
  근접. 팁 자기유도 오염 완화가 팁 하중에 주는 영향 확인.
  light preset(D32), NASA 덱, 순수 ALM(prandtl/eps_corr/sweep OFF), 25rev.
  patch_notes/alm_natelson_ring_sampling/.
    python main.py --config configs/hvab/hvab_hover_c10_ring.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False,
    sampling={"mode": "ring", "ring_n": 20, "ring_r_factor": 1.0},
    polar_source="nasa_overflow", n_rev=25, run_tag="ring",
)
