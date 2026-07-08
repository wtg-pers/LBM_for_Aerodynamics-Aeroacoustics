"""HVAB hover c10, M_tip=0.65 — marker distribution A/B: COSINE (both ends).

순수 ALM(보정 OFF) + NASA 덱 + light + 25rev 로 Task 2 pureALM_nasa와 동일 셋업,
**marker_distribution만 변경**(uniform → cosine, 양끝 조밀). 마커 배치가 팁/루트
하중·CT·FM에 주는 영향을 격리 비교. cosine_side ∈ {"both","tip","root"}.
    python main.py --config configs/hvab/hvab_hover_c10_markerdist_cosine.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="cosine", cosine_side="both",
    n_rev=25, polar_source="nasa_overflow", run_tag="markerdist_cosine",
)
