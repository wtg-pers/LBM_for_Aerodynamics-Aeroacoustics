"""HVAB hover c10 — ★Watanabe 해상도 레시피 (fine preset, 5-level), NASA OVERFLOW 덱.

목적: 팁 유도결손을 "보정"이 아니라 "격자 해상도"로 직접 해결하는지 판정.
  - fine preset = D40 / 5-level(4 fine) → 팁 chord 15.7 cell, dx_fine≈5.3mm.
  - 이 해상도에서 ε=max(c/4,2Δx)=**c/4=0.25c**(ε/Δx≈3.93)로 자동 floor 해제
    = Martínez-Tossas 최적 = Watanabe ε=2.5Δx 대역 → 팁와류 직접 해상.
  - Watanabe는 tip-loss/smearing 보정을 **전혀 쓰지 않음** → 여기서도
    eps_correction=None, prandtl_loss=False (순수 ALM). 해상도만으로 팁 φ가
    운동량값으로 회복되면 "보정 불요" 가설 확인. [[reference_hvab_cfd_benchmarks]]

규모: ~202M cells / ~83GB → **DGX(≥80GB) 전용.** 24GB/40GB 불가.
폴더: result_hvab_hover_c10.0_M650_mlg5_D40_fine_watanabe_nasa (기존과 충돌 없음).
참고: SGS는 안정성 위해 base 기본(dyn_smag) 유지. 엄밀 Watanabe(implicit-LES)는
  use_sgs=False지만 과거 fine_nosgs가 step1 NaN 이력 → 먼저 SGS-on으로 계산가능성
  판정 후 필요시 nosgs로 전환. nesting은 sep≥0.15D 안정화본(fine preset).
    python main.py --config configs/hvab/hvab_hover_c10_fine_watanabe_nasa.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="fine",
    eps_correction=None,            # 보정 없음 — 해상도가 직접 팁와류 해상
    prandtl_loss=False,             # Watanabe: tip-loss 함수 미사용
    sampling={"mode": "gaussian"},  # baseline과 동일 (±3ε)
    polar_source="nasa_overflow", n_rev=25, run_tag="watanabe_nasa",
)
