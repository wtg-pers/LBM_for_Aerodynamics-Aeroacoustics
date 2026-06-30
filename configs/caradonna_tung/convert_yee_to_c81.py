"""
Convert Prof. Yee's tab-delimited NACA0012 table (Cl, Cd; Mach-indexed, with
DIFFERENT Mach/AoA grids per coefficient) into a C81 deck readable by
c81_loader.C81PolarSet.  No interpolation: each table keeps its own grid
(the C81 format and our parser both support per-table grids).

    python configs/caradonna_tung/convert_yee_to_c81.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "aeromechanics_workshop", "CT_rotor_datas",
                   "table_NACA0012_Prof_Yee.dat")
OUT = os.path.join(ROOT, "data", "airfoils", "NACA0012_Yee.C81")


def parse_yee(path):
    lines = open(path).read().splitlines()
    blocks, i = {}, 0
    while i < len(lines):
        head = lines[i].split("\t")
        if head and head[0] in ("Cl", "Cd", "Cm"):
            mach = [float(x) for x in lines[i + 1].split("\t") if x.strip()]
            aoa, data, j = [], [], i + 2
            while j < len(lines):
                row = lines[j].split("\t")
                if not row or not row[0].strip():
                    break
                try:
                    a = float(row[0])
                except ValueError:
                    break
                aoa.append(a)
                data.append([float(x) for x in row[1:1 + len(mach)]])
                j += 1
            blocks[head[0]] = (np.array(mach), np.array(aoa), np.array(data))
            i = j
        else:
            i += 1
    return blocks


def write_c81_tables(path, name, tables):
    """tables = [(mach, aoa, data[aoa,mach]) for CL, CD, CM]."""
    counts = []
    for m, a, _ in tables:
        counts += [len(m), len(a)]
    assert all(c < 100 for c in counts), "C81 header fields are 2-digit"
    hdr = "%-28s" % (name + ",")[:28] + "".join("%02d" % c for c in counts)
    lines = [hdr]
    for m, a, d in tables:
        lines.append(" ".join("%.6g" % x for x in m))
        for i in range(len(a)):
            lines.append(" ".join(["%.6g" % a[i]]
                                  + ["%.6g" % v for v in d[i, :]]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    y = parse_yee(SRC)
    cl, cd = y["Cl"], y["Cd"]
    cm = (np.array([0.0, 1.0]), np.array([-180.0, 180.0]), np.zeros((2, 2)))
    write_c81_tables(OUT, "NACA0012", [cl, cd, cm])
    print("wrote:", os.path.relpath(OUT, ROOT))

    from src.actuator.c81_loader import C81PolarSet
    p = C81PolarSet.from_file(OUT)
    print(p.info())
    sl = (p.get_CL(2.0, 0.0) - p.get_CL(-2.0, 0.0)) / 4.0
    print("  M=0: CLa=%.4f/deg, CLmax=%.3f, CDmin=%.5f"
          % (sl, p.get_CL(np.linspace(0, 18, 181), 0.0).max(),
             p.get_CD(np.linspace(-6, 6, 61), 0.0).min()))
    print("  CL(6,0.4)=%.4f CD(6,0.4)=%.5f  (vs raw Yee 0.684 / 0.0110)"
          % (p.get_CL(6.0, 0.4), p.get_CD(6.0, 0.4)))
