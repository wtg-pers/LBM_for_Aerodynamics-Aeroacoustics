"""Regression check: marker distribution survives the FULL path
(Rotor.from_config -> to_lattice_units), which is what the HVAB fine-level ALM
actually walks. Run from anywhere; paths are derived from this file's location.

    python patch_notes/alm_marker_distribution/test_full_path_distribution.py
    -> expect "ALL PASS"

Earlier Step-2 verification only exercised Rotor.from_config and MISSED that
Blade.to_lattice_units() regenerated markers with the DEFAULT uniform
distribution, silently discarding cosine/endpoint.
"""
import os, sys, importlib.util
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from src.actuator.rotor import Rotor

CFG = os.path.join(REPO, "configs", "0630_markers_test")
CASES = {"md_uniform_gauss": "uniform", "md_cosine_gauss": "cosine",
         "md_endpoint_gauss": "endpoint"}
ok = True
for cf, label in CASES.items():
    spec = importlib.util.spec_from_file_location("c", os.path.join(CFG, cf + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rc = m.config["actuator_line"]["rotor"]; rc["grid"]["dx"] = 0.1  # dummy; placement-independent
    rot = Rotor.from_config(rc).to_lattice_units(length_scale=1.0, time_scale=1.0)
    bl = rot.blades[0]; rR = bl.marker_r / bl.r_tip; d = np.diff(rR)
    uni = abs(d[0] - d[len(d) // 2]) < 1e-4 and abs(d[-1] - d[0]) < 1e-4
    if label == "uniform":
        checks = [("equal-spacing", uni), ("cell-centred first~0.2597", abs(rR[0] - 0.2597) < 1e-3)]
    elif label == "cosine":
        checks = [("non-uniform (clustered)", not uni), ("tip clustered last>0.999", rR[-1] > 0.999)]
    else:  # endpoint
        checks = [("root-cut first~0.2519", abs(rR[0] - 0.2519) < 2e-3),
                  ("tip endpoint last~1.0", abs(rR[-1] - 1.0) < 1e-4)]
    res = all(v for _, v in checks); ok = ok and res
    print(f"[{'PASS' if res else 'FAIL'}] {cf:16} first={rR[0]:.4f} last={rR[-1]:.4f}  "
          + " ".join(f"{n}={v}" for n, v in checks))
print("\nALL PASS" if ok else "\n**SOME FAILED — blade.py not synced / bug present**")
sys.exit(0 if ok else 1)
