"""HVAB hover c10 — ε tip-taper (Stage B, 기하적 보정) A/B용.

prandtl OFF + epsilon_mode="tip_taper". Dağ/Kleine(해석적 smearing de-induction)와는
다른 메커니즘 — 팁에서 smearing 커널 폭 ε 자체를 기하적으로 좁혀(2Δx 쪽으로)
over-projection을 줄임. 보정군(epscorr/kleine*)과 같은 pure-ALM·light·gaussian 조건.

★주의(중요): HVAB light는 팁 chord ~6.3 cell이라 baseline 팁 ε이 이미 2Δx **floor**에
걸려 있음(blade.py:371 `max(factor·2Δx, 2Δx)`) → factor=1.0 taper는 **팁에서 거의 no-op**.
유효 구간은 등현부(r/R≈0.7~0.95, chord/4≈2.6→2.0 cell)뿐이라 효과가 작을 것.
taper를 본격적으로 보려면 fine preset(팁 chord↑ → ε headroom 확보)이 필요.
→ light에선 Dağ/Kleine가 주 레버이고, 이 케이스는 "기하적 taper가 light에선 약하다"를
확인하는 음성 대조군 성격.
    python main.py --config configs/hvab/hvab_hover_c10_taper.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _hvab_hover_base import build_config

config = build_config(
    collective_deg=10.0, mtip=0.65, preset="light",
    prandtl_loss=False, sampling={"mode": "gaussian"},
    n_rev=25, run_tag="taper",
)
# ε tip-taper (Stage B). build_config가 노출 안 하므로 actuator_line dict에 직접 주입
# (CT의 ct_hover_t08_m088_prtipR_taper.py와 동일 패턴; factory가 epsilon_mode를 읽음).
config["actuator_line"]["epsilon_mode"] = "tip_taper"
config["actuator_line"]["epsilon_tip_factor"] = 1.0     # 팁 ε 목표 = max(1.0·2Δx, 2Δx)
config["actuator_line"]["epsilon_taper_start"] = 0.7     # r/R 0.7부터 선형 taper
