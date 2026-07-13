"""HVAB hover c10 — UNIFORM D=40, 마커 절반(N=24). 마커간격 격리 crossover 1/2.

목적: D-sweep에서 D80일 때 팁 M²cₙ이 flatten된 것이, 격자 해상도 자체가 아니라
  '격자단위(lattice unit) ALM 마커 간격 g_LU ∝ D/N'의 효과인지 격리 검증.
  마커수 N_RADIAL은 D와 무관한 고정값(48)이라 D40→D80은 dx만 절반 → g_LU 2배.
  이를 마커수로 상쇄:
      D40, N=24  →  g_LU ∝ 40/24 = 80/48  = D80 baseline(N=48)과 동일
      D80, N=96  →  g_LU ∝ 80/96 = 40/48  = D40 baseline(N=48)과 동일
  ε=max(chord/4, 2Δx)는 chord·dx에만 의존(대부분 2Δx=2lu floor)해 N과 무관 →
  마커 간격만 순수 격리됨. 가설이 맞으면:
      [이 케이스 D40·N24]  ==  [unif_d80_c10.py  baseline]
      [unif_d40_c10.py baseline]  ==  [unif_d80_m96_c10.py  D80·N96]

셀=80·40³=5.12M(~2.1GB), STEPS_REV≈1257. 격자/도메인/물리는 unif_d40_c10.py와
동일, marker/blade만 48→24. 상세 배경: unif_d20_c10.py 참조.

    python main.py --config configs/0708_grid_convergence/unif_d40_m24_c10.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hvab"))
from _hvab_hover_base import build_config
config = build_config(
    collective_deg=10.0, mtip=0.65, preset="unif40",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    marker_distribution="uniform", n_radial=24,
    n_rev=15, polar_source="nasa_overflow", run_tag="unif40_m24",
)
