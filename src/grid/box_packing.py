"""Box packing — MLG fine region 자동 배치 (mlg_blocks ④).

WHY
---
Fine region 은 지금까지 config 에서 손으로 놓였다 (`build_mlg_4level`,
`GRID_PRESETS`, testrig 리터럴). 손 배치가 어기는 것들은 늘 같다:

  * 좌표가 부모 격자 눈금(2^-(k-1) L0 lu)에 스냅되지 않는다 — setup 이
    round 로 **침묵 스냅**한다 (octo8 로터 박스가 실제로 그랬다).
  * 자식 밴드가 부모 region 을 침범한다 — `validate_block_tree` Rule A 는
    기본 warn 이라 흘려보내기 쉽다.
  * 형제 disjoint / 간극 2·ow 를 빌드까지 가야 안다.
  * "왜 이 자리인가"가 기록되지 않는다.

이 모듈은 씨앗(로터 swept-disk / 몸체 bbox / 명시 박스)에서 합법 배치를
생성하고, 중간 레벨을 합성하고, **setup 과 동일한 기하 기계**
(`OverlapRegion`/`GridBlock`/`validate_block_tree(nesting="error")`)로
검증한 뒤, config `mlg` dict 와 근거 대장(왜/여유)을 함께 낸다. 배치가
불가능하면 어느 제약이 어느 면에서 몇 lu 모자란지 말하고 실패한다
(`alm_placement.select_level` 의 어투).

octree 와의 관계: `patch_notes/mlg_blocks/02` §6 — 자동 배치라는 octree 의
실용 이득을 현 블록 트리 위에서 값싸게 취하는 것이 이 모듈이다.

SCOPE (1단계)
-------------
* 기하·검증은 전부 CPU numpy. GPU/STL 마스크 절단 예보는 후속
  (body_coupling_band_report 를 CPU 마스크로 미리 돌리는 것).
* MPI 실현성(balance_cuts_tree) 리포트는 후속.
* 도메인 flush 는 `flush_faces` 로 명시 허용된 면에만 — fine 레벨 도메인
  BC(patch_notes/mlg_blocks/03)가 그 면을 물리 경계로 만든다.

Author: LBM Development Team / Date: 2026-08
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.grid.block_tree import GridBlock, validate_block_tree
from src.grid.overlap_manager import IndexBox, OverlapRegion

_AXES = ("x", "y", "z")
_FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


# =============================================================================
# Float boxes (L0 lu, inclusive node coords) — the working representation
# =============================================================================
@dataclass(eq=False)   # eq=False: 필드에 ndarray — 기본 __eq__ 는 ambiguous
class FloatBox:
    """Axis-aligned box in L0 lattice units, inclusive node coordinates."""
    lo: np.ndarray
    hi: np.ndarray
    name: str = ""
    level: int = 0
    origin: str = ""          # 근거: 어떤 씨앗/규칙에서 왔는가

    def union(self, other: "FloatBox") -> "FloatBox":
        return FloatBox(np.minimum(self.lo, other.lo),
                        np.maximum(self.hi, other.hi),
                        name=self.name or other.name, level=self.level,
                        origin=f"union({self.origin} + {other.origin})")

    def expanded(self, w: float) -> "FloatBox":
        return FloatBox(self.lo - w, self.hi + w, self.name, self.level,
                        self.origin)

    def intersects(self, other: "FloatBox") -> bool:
        return bool(np.all(self.lo <= other.hi) and np.all(other.lo <= self.hi))

    def gap_to(self, other: "FloatBox") -> float:
        """Separation along the best-separating axis (negative = overlap)."""
        return float(max(max(other.lo[d] - self.hi[d],
                             self.lo[d] - other.hi[d]) for d in range(3)))


def snap_to_level(box: FloatBox, level: int) -> FloatBox:
    """Snap to the PARENT grid (spacing 2^-(level-1) L0 lu), never shrinking.

    setup quantizes region coords with round() against the parent spacing —
    silently. The generator snaps lo down / hi up so the requested volume is
    always covered, and emits values setup's round() maps to themselves.
    """
    g = 2.0 ** -(level - 1)
    lo = np.floor(box.lo / g + 1e-9) * g
    hi = np.ceil(box.hi / g - 1e-9) * g
    return FloatBox(lo, hi, box.name, level, box.origin)


# =============================================================================
# Seeds
# =============================================================================
def rotor_seed(name: str, level: int, hub, radius_lu: float,
               axis=(0.0, 0.0, 1.0),
               pad_d: Tuple[float, float, float] = (0.125, 0.25, 0.6875),
               ) -> Dict:
    """Swept-disk AABB + (up, down, lat) padding in D units.

    pad_d 기본값은 octo8 로터 박스 비율(단일 프로펠러 검증 산):
    up=디스크 위 0.125 D, down=아래 0.25 D, lat=측면 0.6875 D.
    임의 축 방향: per-axis reach = R·sqrt(1-a_d^2) + 패딩의 축/측면 분해
    (alm_placement.rotor_extent 의 swept-disk 산술과 동일한 기하).

    'up' 은 **+axis 방향**이다. 기하 축을 줄 것 — 회전 방향(handedness)은
    rpm 부호의 몫이므로 여기 axis 에 실을 이유가 없다 (octo8 이라면
    rotation_axis=[0,0,-1] 이 아니라 (0,0,1) 을 넘겨 up=+z 로 읽히게).
    """
    return {"kind": "rotor", "name": name, "level": int(level),
            "hub": tuple(float(v) for v in hub),
            "radius_lu": float(radius_lu),
            "axis": tuple(float(v) for v in axis),
            "pad_d": tuple(float(v) for v in pad_d)}


def bbox_seed(name: str, level: int, lo, hi, pad_lu: float = 0.0) -> Dict:
    """Explicit bbox (e.g. STL body from stl_bbox_l0lu) + isotropic pad."""
    return {"kind": "bbox", "name": name, "level": int(level),
            "lo": tuple(float(v) for v in lo),
            "hi": tuple(float(v) for v in hi), "pad_lu": float(pad_lu)}


def box_seed(name: str, level: int, box: Dict) -> Dict:
    """Explicit box dict (x_min..z_max, L0 lu) — 손 배치의 이주 경로."""
    return {"kind": "box", "name": name, "level": int(level),
            "box": {k: float(v) for k, v in box.items()
                    if k in ("x_min", "x_max", "y_min", "y_max",
                             "z_min", "z_max")}}


def _seed_to_floatbox(seed: Dict) -> FloatBox:
    kind = seed["kind"]
    if kind == "box":
        b = seed["box"]
        return FloatBox(np.array([b["x_min"], b["y_min"], b["z_min"]]),
                        np.array([b["x_max"], b["y_max"], b["z_max"]]),
                        seed["name"], seed["level"], origin="explicit box")
    if kind == "bbox":
        p = seed["pad_lu"]
        return FloatBox(np.asarray(seed["lo"], dtype=float) - p,
                        np.asarray(seed["hi"], dtype=float) + p,
                        seed["name"], seed["level"],
                        origin=f"bbox + pad {p:g} lu")
    if kind == "rotor":
        hub = np.asarray(seed["hub"], dtype=float)
        R = seed["radius_lu"]
        D = 2.0 * R
        a = np.asarray(seed["axis"], dtype=float)
        n = np.linalg.norm(a)
        a = a / n if n > 0 else np.array([0.0, 0.0, 1.0])
        up, dn, lat = seed["pad_d"]
        inplane = np.sqrt(np.maximum(0.0, 1.0 - a * a))
        reach = R * inplane                      # swept-disk per-axis
        # 'up' = +axis 쪽 (docstring). 좌표축 성분 부호로 상/하를 분해:
        # a_d > 0 이면 +d 면이 up 쪽, a_d < 0 이면 -d 면이 up 쪽.
        pad_plus = lat * D * inplane + np.where(a >= 0, up, dn) * D * np.abs(a)
        pad_minus = lat * D * inplane + np.where(a >= 0, dn, up) * D * np.abs(a)
        return FloatBox(hub - reach - pad_minus, hub + reach + pad_plus,
                        seed["name"], seed["level"],
                        origin=f"rotor D={D:g} pad_d={seed['pad_d']}")
    raise ValueError(f"unknown seed kind '{kind}'")


# =============================================================================
# Packing
# =============================================================================
class PackingError(ValueError):
    """배치 불가 — 메시지에 제약·면·부족량(lu)을 담는다."""


@dataclass
class PackResult:
    mlg: Dict
    ledger: List[Dict]
    warnings: List[str]
    root: GridBlock
    lines: List[str] = field(default_factory=list)

    def report(self) -> str:
        return "\n".join(self.lines)


def pack(seeds: Sequence[Dict], domain_shape: Tuple[int, int, int],
         overlap_width: int = 2,
         flush_faces: Sequence[str] = (),
         merge_siblings: bool = True,
         interpolation: str = "cubic",
         filter_level: int = 1) -> PackResult:
    """씨앗 → 합법 MLG 배치 + 근거 대장.

    Args:
        seeds: rotor_seed / bbox_seed / box_seed 의 리스트. 레벨은 씨앗이
            명시한다. 중간 레벨에 씨앗이 없으면 자식 요구(자식 박스 + 자식
            밴드)로 합성한다.
        domain_shape: L0 노드 수 (Nx, Ny, Nz).
        overlap_width: C2F/F2C 밴드 폭 (부모 셀 단위).
        flush_faces: 도메인 flush 를 허용할 면 이름들 ('z_min' 등). fine
            레벨 도메인 BC(mlg_blocks/03)가 그 면을 물리 경계로 만들어야
            하며, 허용 없는 면에서 도메인이 모자라면 PackingError.
        merge_siblings: 같은 레벨에서 밴드 포함 겹침이 생기면 합집합으로
            병합(True) 또는 PackingError(False).

    Returns:
        PackResult — .mlg 는 config["mlg"] 에 그대로 대입 가능.
    """
    if not seeds:
        raise PackingError("no seeds")
    dom_hi = np.asarray(domain_shape, dtype=float) - 1.0
    ow = int(overlap_width)
    flush_faces = tuple(flush_faces)
    num_levels = max(int(s["level"]) for s in seeds) + 1
    if num_levels < 2:
        raise PackingError("finest seed level must be >= 1")
    warnings: List[str] = []
    lines: List[str] = [f"[box_packing] {len(seeds)} seeds, "
                        f"{num_levels} levels, ow={ow}, "
                        f"flush_faces={list(flush_faces)}"]

    # ── 1. seeds → per-level float boxes (snapped) ───────────────
    by_level: Dict[int, List[FloatBox]] = {k: [] for k in range(1, num_levels)}
    for s in seeds:
        k = int(s["level"])
        if not 1 <= k < num_levels:
            raise PackingError(f"seed '{s['name']}': level {k} out of range")
        by_level[k].append(snap_to_level(_seed_to_floatbox(s), k))

    # ── 2. coarse synthesis (fine → coarse): 자식 요구를 부모에 합성 ──
    # 자식(level k+1) 밴드 폭 = ow × (level-k spacing) = ow·2^-k [L0 lu].
    # Rule A(에러 수준): 자식 박스 + 자식 밴드 ⊆ 부모 박스.
    for k in range(num_levels - 2, 0, -1):
        band_lu = ow * (2.0 ** -k)
        for child in by_level[k + 1]:
            need = child.expanded(band_lu)
            need.name, need.level = f"need({child.name})", k
            need.origin = f"child '{child.name}' + band {band_lu:g} lu"
            host = None
            for pb in by_level[k]:
                if pb.intersects(child):
                    host = pb
                    break
            if host is None:
                by_level[k].append(snap_to_level(need, k))
            else:
                merged = snap_to_level(host.union(need), k)
                merged.name, merged.origin = host.name, (
                    f"{host.origin}; grown for {child.name}")
                by_level[k][by_level[k].index(host)] = merged

    # ── 3. domain clamp / flush 처리 ─────────────────────────────
    #   레벨 1 의 부모는 도메인. box + 자기 밴드(ow·2^-(k-1))가 도메인을
    #   넘으면: flush 허용 면 → 그 면을 도메인 면에 정확히 붙임(밴드 0),
    #   아니면 PackingError("short by").
    for k in range(1, num_levels):
        # 레벨 1 만 도메인과 직접 커플링 밴드를 주고받는다. k>=2 의 부모
        # 요구는 (2)가 이미 흡수했고, 그 부모의 도메인 적합성은 자기
        # 차례에 검사된다. 다만 박스 자체가 도메인을 벗어나는 건 어느
        # 레벨이든 여기서 flush 허용 여부로 판정한다.
        own_band = ow * (2.0 ** -(k - 1)) if k == 1 else 0.0
        for i, b in enumerate(by_level[k]):
            lo, hi = b.lo.copy(), b.hi.copy()
            for d, ax in enumerate(_AXES):
                fmin, fmax = f"{ax}_min", f"{ax}_max"
                if lo[d] < 0.0 or (own_band > 0.0 and lo[d] - own_band < 0.0):
                    if fmin in flush_faces:
                        lo[d] = 0.0
                    elif lo[d] < 0.0:
                        raise PackingError(
                            f"block '{b.name}' (L{k}) face {fmin}: needs "
                            f"{-lo[d]:g} lu beyond the domain — allow "
                            f"flush_faces=('{fmin}',) or pull the region in")
                    else:
                        raise PackingError(
                            f"block '{b.name}' (L{k}) face {fmin}: band "
                            f"short by {own_band - lo[d]:g} lu to the domain "
                            f"face — move the region in by that much, or "
                            f"allow flush ('{fmin}')")
                if hi[d] > dom_hi[d] or (own_band > 0.0
                                         and hi[d] + own_band > dom_hi[d]):
                    if fmax in flush_faces:
                        hi[d] = dom_hi[d]
                    elif hi[d] > dom_hi[d]:
                        raise PackingError(
                            f"block '{b.name}' (L{k}) face {fmax}: needs "
                            f"{hi[d] - dom_hi[d]:g} lu beyond the domain — "
                            f"allow flush_faces=('{fmax}',) or pull the "
                            f"region in")
                    else:
                        raise PackingError(
                            f"block '{b.name}' (L{k}) face {fmax}: band "
                            f"short by {hi[d] + own_band - dom_hi[d]:g} lu "
                            f"to the domain face — move the region in, or "
                            f"allow flush ('{fmax}')")
            by_level[k][i] = FloatBox(lo, hi, b.name, k, b.origin)

    # ── 4. sibling 처리 (밴드 포함 겹침 → 병합 or 에러, 간극 경고) ──
    for k in range(1, num_levels):
        band_lu = ow * (2.0 ** -(k - 1))
        changed = True
        while changed:
            changed = False
            boxes = by_level[k]
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    gap = a.gap_to(b)
                    if gap < 2.0 * band_lu:
                        if a.expanded(band_lu).intersects(b.expanded(band_lu)):
                            if not merge_siblings:
                                raise PackingError(
                                    f"L{k} blocks '{a.name}' and '{b.name}' "
                                    f"overlap band-inclusive (gap {gap:g} lu "
                                    f"< 2 bands {2 * band_lu:g}) and "
                                    f"merge_siblings=False")
                            m = snap_to_level(a.union(b), k)
                            m.name = f"{a.name}+{b.name}"
                            warnings.append(
                                f"L{k}: merged '{a.name}'+'{b.name}' "
                                f"(band-inclusive overlap, gap {gap:g} lu)")
                            by_level[k] = ([m] + [x for t, x in
                                                  enumerate(boxes)
                                                  if t not in (i, j)])
                            changed = True
                            break
                        warnings.append(
                            f"L{k}: '{a.name}'~'{b.name}' gap {gap:g} lu < "
                            f"2*band {2 * band_lu:g} — coarse solution has "
                            f"little room to relax")
                if changed:
                    break
        # 병합이 부모 요구를 키웠을 수 있으므로, 병합 발생 시 상위 재합성은
        # (2) 단계의 union 논리와 동일하게 부모에 반영해야 한다. 부모는
        # 자식보다 나중(작은 k)에 이 루프를 돌므로 여기서 즉시 반영한다.
        if k > 1:
            for child in by_level[k]:
                need = child.expanded(ow * (2.0 ** -(k - 1)))
                for i, pb in enumerate(by_level[k - 1]):
                    if pb.intersects(child):
                        grown = snap_to_level(pb.union(
                            FloatBox(need.lo, need.hi, pb.name, k - 1)),
                            k - 1)
                        grown.name, grown.origin = pb.name, pb.origin
                        by_level[k - 1][i] = grown
                        break

    # ── 5. setup 과 동일한 기계로 기하 트리 구성 + 검증 ─────────────
    root = GridBlock(level=0, index=0, uid=0, name="L0",
                     shape=tuple(int(n) for n in domain_shape),
                     origin=(0.0, 0.0, 0.0), spacing=1.0)
    prev_level = [root]
    uid = 1
    for k in range(1, num_levels):
        cur: List[GridBlock] = []
        for idx, b in enumerate(by_level[k]):
            host = None
            for pb in prev_level:
                plo = np.asarray(pb.origin, dtype=float)
                phi = plo + (np.asarray(pb.shape) - 1) * pb.spacing
                if np.all(b.lo >= plo - 1e-9) and np.all(b.hi <= phi + 1e-9):
                    host = pb
                    break
            if host is None:
                raise PackingError(
                    f"block '{b.name}' (L{k}) fits inside no L{k - 1} block "
                    f"— synthesis bug or conflicting explicit seeds")
            po = np.asarray(host.origin, dtype=float)
            ps = float(host.spacing)
            loc = np.rint((b.lo - po) / ps).astype(int)
            hic = np.rint((b.hi - po) / ps).astype(int)
            region = OverlapRegion(
                level_coarse=k - 1,
                coarse_shape=tuple(int(n) for n in host.shape),
                fine_region=IndexBox(loc[0], hic[0], loc[1], hic[1],
                                     loc[2], hic[2]),
                overlap_width=ow)
            fdc = region.fine_domain_coarse
            blk = GridBlock(
                level=k, index=idx, uid=uid, name=b.name or f"L{k}b{idx}",
                shape=tuple(region.fine_shape),
                origin=(po[0] + fdc.x_start * ps, po[1] + fdc.y_start * ps,
                        po[2] + fdc.z_start * ps),
                spacing=0.5 ** k, region=region, parent=host)
            host.children.append(blk)
            cur.append(blk)
            uid += 1
        prev_level = cur

    warnings += validate_block_tree(root, overlap_width=ow, nesting="error")

    # ── 6. 대장(ledger) + 리포트 ─────────────────────────────────
    ledger: List[Dict] = []

    def _walk(blk: GridBlock):
        for c in blk.children:
            r = c.region
            fr = r.fine_region
            cells = int(np.prod(r.fine_shape))
            entry = {
                "name": c.name, "level": c.level,
                "fine_shape": tuple(r.fine_shape), "cells": cells,
                "updates_per_coarse_step": cells * (2 ** c.level),
                "flush": {f: bool(v)
                          for f, v in (r.flush_faces or {}).items() if v},
                "region_parent_lu":
                    (fr.x_start, fr.x_end, fr.y_start, fr.y_end,
                     fr.z_start, fr.z_end),
            }
            ledger.append(entry)
            lines.append(
                f"  L{c.level} '{c.name}': shape={entry['fine_shape']} "
                f"{cells:,} cells"
                + (f" flush={sorted(entry['flush'])}" if entry['flush']
                   else ""))
            _walk(c)

    _walk(root)
    total_updates = int(np.prod(domain_shape)) + sum(
        e["updates_per_coarse_step"] for e in ledger)
    lines.append(f"  total updates/coarse step ~ {total_updates:,}")
    for w in warnings:
        lines.append(f"  [warn] {w}")

    # ── 7. mlg dict ──────────────────────────────────────────────
    levels_cfg: List[Dict] = [{}]
    for k in range(1, num_levels):
        regions = []
        for c in (bb for bb in _iter(root) if bb.level == k):
            po = np.asarray(c.parent.origin, dtype=float)
            ps = float(c.parent.spacing)
            fr = c.region.fine_region
            # 순수 파이썬 float 로 — config 는 repr 로 저장될 수 있고
            # np.float64 는 그 경로를 오염시킨다.
            regions.append({
                "name": c.name,
                "x_min": float(po[0] + fr.x_start * ps),
                "x_max": float(po[0] + fr.x_end * ps),
                "y_min": float(po[1] + fr.y_start * ps),
                "y_max": float(po[1] + fr.y_end * ps),
                "z_min": float(po[2] + fr.z_start * ps),
                "z_max": float(po[2] + fr.z_end * ps),
            })
        levels_cfg.append({"regions": regions})
    mlg = {"enabled": True, "num_levels": num_levels,
           "overlap_width": ow, "interpolation": interpolation,
           "filter_level": filter_level, "levels": levels_cfg}
    return PackResult(mlg=mlg, ledger=ledger, warnings=warnings,
                      root=root, lines=lines)


def _iter(root: GridBlock):
    from src.grid.block_tree import iter_blocks
    return iter_blocks(root)
