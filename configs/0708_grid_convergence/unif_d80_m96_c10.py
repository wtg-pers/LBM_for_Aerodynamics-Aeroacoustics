"""HVAB hover c10 — UNIFORM D=80, 마커 2배(N=96). 마커간격 격리 crossover 2/2.

목적: D-sweep 팁 flatten이 격자 해상도가 아니라 '격자단위 마커 간격 g_LU ∝ D/N'
  효과인지 격리 검증(설계 전문: unif_d40_m24_c10.py 참조). 마커수로 D 증가를 상쇄:
      D80, N=96  →  g_LU ∝ 80/96 = 40/48  = D40 baseline(N=48)과 동일
  ε=max(chord/4, 2Δx)는 N과 무관 → 마커 간격만 순수 격리. 가설이 맞으면:
      [이 케이스 D80·N96]  ==  [unif_d40_c10.py  baseline]
      [unif_d80_c10.py baseline]  ==  [unif_d40_m24_c10.py  D40·N24]

셀=80·80³=40.96M(~16.8GB, 24GB OK, 64-bit 인덱싱 필요 [[project_int32_kernel_ceiling]]),
STEPS_REV≈2513. 격자/도메인/물리는 unif_d80_c10.py와 동일, marker/blade만 48→96.
15rev≈37.7k step — 이 스윕에서 가장 무거움. 상세 배경: unif_d20_c10.py 참조.

    python main.py --config configs/0708_grid_convergence/unif_d80_m96_c10.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="unif80",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform", n_radial=96,
    n_rev=15, polar_source="nasa_overflow", run_tag="unif80_m96",
)
