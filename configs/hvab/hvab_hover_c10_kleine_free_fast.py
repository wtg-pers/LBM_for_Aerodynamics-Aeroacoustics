"""HVAB hover c10 — Kleine 2022 Phase 2 (free-wake), 성능 패치판 (A+B+C).

`hvab_hover_c10_kleine_free.py`와 물리적으로 동일한 케이스. 차이는 free-wake
계산 속도뿐 → 기존 free 결과와 직접 A/B 비교용 (팁 φ/α·CT·FM이 일치하면
근사가 안전, per-step 시간은 단축).

패치 내용 (patch_notes/kleine_freewake_perf/):
  A) free-wake 영향행렬 재빌드를 `rebuild_every` 스텝마다만 수행 (wake는 천천히
     수축 → 사이엔 캐시 재사용). 기본 1=매 스텝(기존과 bit-identical). 여기선 10.
  B) wake 이류 trilinear 샘플을 모든 ring 일괄 1회 호출(+GPU D2H 1회)로 압축
     — bit-identical, 항상 적용(코드 경로).
  C) dΓ/dr 그래디언트 행렬 캐시 — bit-identical, 항상 적용.

rebuild_every만 조절: 10이 보수적(~1% rev). 더 공격적으로 25/50도 가능 —
재빌드 비용이 그만큼 더 분산되나 근사 오차 증가 → 기존 free와 편차 확인 후 결정.
    python main.py --config configs/hvab/hvab_hover_c10_kleine_free_fast.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    eps_correction={"enabled": True, "method": "kleine",
                    "wake": "free", "n_w": 50,
                    "rebuild_every": 10},          # A: 10스텝마다 재빌드
    run_tag="kleine_free_fast",
)
