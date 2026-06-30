"""
Caradonna-Tung hover — θ=8°, M_tip=0.877. 마커수(n_radial) 수렴 테스트.
표준 Prandtl(R_tip_eff=R), no taper, **n_radial=60** (baseline prtipR는 40).

가설(1) 검증: 마커가 cell-center라 최외측이 0.99R(n40) → 더 촘촘하면 R에 근접해
팁-스트립 이산화 과하중↓ + 스팬 순환구배 해상↑. 격자(dx/extent) 불변이라 **값쌈**
(LBM 비용 동일). C_T가 prtipR(0.00639)보다 내려가 수렴하면 마커수 기여, 거의 같으면 기각.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config
config = build_config(collective_deg=8.0, mtip=0.877)
config["actuator_line"]["prandtl_loss"] = {"enabled": True, "eps_offset": False}
config["actuator_line"]["rotor"]["grid"]["n_radial"] = 60
_base = config["output"]["output_dir"].rsplit("/vtk",1)[0] + "_prtipR_n60"
config["output"]["output_dir"] = _base + "/vtk"
config["output"]["checkpoint_dir"] = _base + "/checkpoints"
config["output"]["csv_dir"] = _base + "/csv"
