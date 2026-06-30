"""
Caradonna-Tung hover — θ=8°, M_tip=0.877, EXTENDED fine regions + **Prandtl OFF**.

가장 깨끗한 wake-extent 진단. dx는 'light'와 동일(D=32)이고 finest MLG 박스만 팁
와류를 따라 확장(하류 0.25D→1.0D, 측면 ±0.6D→±0.7D) → **extent를 dx와 분리**.
Prandtl을 끄는 이유: Prandtl이 팁손실을 −31%p나 가려버리므로, extent의 순수 효과를
보려면 꺼야 한다. 비교 기준 = `result_ct_t08.0_M877_mlg4_D32_light_noprandtl`
(이미 확보: C_T=0.00700, +47.9%).

판정:
  - extended_noprandtl C_T가 0.00700보다 **내려가고** 팁 φ가 회복되면
        → 한계는 fine-region EXTENT (격자 확장으로 고칠 수 있음).
  - 거의 안 변하면
        → 한계는 ALM 팁 모델 자체 (→ Stage C / smeared-vortex 과확산 본질).

주의(경미): light_noprandtl 기준은 SGS ON, 이 런은 SGS OFF. 사전 격자수렴 연구상
SGS 효과 ±0.2%로 무시 가능(extent 효과가 그보다 훨씬 큼). 둘 다 SGS로 맞추고 싶으면
아래 use_sgs=True로.

~80M cells, ~33 GB → DGX Spark (128 GB) 필요.

    python main.py --config configs/caradonna_tung/ct_hover_t08_m088_extended_noprandtl.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ct_hover_base import build_config

config = build_config(collective_deg=8.0, mtip=0.877, preset="extended",
                      use_sgs=False)

config["actuator_line"]["prandtl_loss"] = False        # ← extent 순수효과 노출

_base = config["output"]["output_dir"].rsplit("/vtk", 1)[0] + "_noprandtl"
config["output"]["output_dir"] = _base + "/vtk"
config["output"]["checkpoint_dir"] = _base + "/checkpoints"
config["output"]["csv_dir"] = _base + "/csv"
