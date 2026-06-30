"""HVAB hover 10° — 속도 샘플러 A/B 실험: **B-ii = anisotropic Gaussian**. 25-rev.

기존 미테스트였던 b2(반경 smoothing) 격리. 등방 Gaussian(±3ε 구)의 **반경방향 폭만**
ε_r = factor·ε 로 좁히고(perpendicular 폭은 ε 유지) off-disk 반경 혼입을 줄임.
   - vs sampA_gauss(등방) → 반경 narrowing의 순효과
   - vs sampB3_mask(디스크 밖 완전배제)/sampB1_point(필터無) → 중간 강도
합성 smoke에선 aniso(0.5)=부분 회복(point/mask는 완전회복). 실제 LBM 미검증분.

조건은 다른 sampler 케이스와 동일(pure ALM=prandtl·eps OFF, preset=light, c10) —
샘플러 mode만 다름. eps_r_factor=0.5 (반경폭 절반).
※ 다른 sampler 런(sampA/B1/B3)이 18rev면 비교는 수렴 tail(--avg-revs 3)로.
    python main.py --config configs/hvab/hvab_hover_c10_sampB2_aniso.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    eps_correction=None, prandtl_loss=False,            # pure ALM (샘플러만 격리)
    sampling={"mode": "aniso", "eps_r_factor": 0.5},    # B-ii — 반경폭 ε_r=0.5ε
    n_rev=25, run_tag="sampB2_aniso",
)
