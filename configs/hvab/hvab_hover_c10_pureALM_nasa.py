"""HVAB hover c10 — pure ALM, **NASA OVERFLOW C81 덱**, 25-rev.

LUT(폴라 덱) 판별용. 기존 pureALM(`hvab_hover_c10_narrow.py`, NeuralFoil 덱,
폴더 ..._light_pureALM)와 **동일 물리(prandtl·보정 OFF)**, 덱만 NASA OVERFLOW로 교체.
별도 폴더(..._light_pureALM_nasa)라 기존 결과 안 덮음.

목적: Prandtl·smearing 보정이 없어 **덱 교체 효과가 CT/팁하중에 직접** 드러남 →
NeuralFoil(CL_max 과대) → NASA로 바꿔 과대예측이 줄면 LUT 기여 확인.
    python main.py --config configs/hvab/hvab_hover_c10_pureALM_nasa.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False, sampling={"mode": "gaussian"},
    polar_source="nasa_overflow", n_rev=25, run_tag="pureALM_nasa",
)
