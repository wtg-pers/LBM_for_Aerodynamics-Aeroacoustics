# ★버그 수정 — to_lattice_units가 marker 분포를 소실 (2026-07-02)

## 증상
클러스터에서 cosine/endpoint config로 재실행해도 결과가 **여전히 uniform 마커**(첫 실행·재실행 모두).
config는 정상(`marker_distribution=cosine/endpoint`), `Rotor.from_config`도 정상(cosine 0.2523 생성).

## 근본 원인
`Blade.to_lattice_units()` (blade.py:872)가 물리→lattice 단위 변환 시 마커를 **재생성하는데
`generate_markers(n_radial=...)`에 distribution을 안 넘겨** 기본값 uniform으로 되돌림:
```python
new_blade.generate_markers(n_radial=self.n_markers)   # distribution 누락 → uniform
```
**HVAB는 fine-level(L3) ALM** → `from_config`(cosine 생성) 후 **`to_lattice_units`가 uniform으로 덮어씀**.
→ 실제 런은 항상 uniform.

## 검증 공백 (자기반성)
Step 2 e2e 검증이 **`from_config`만 테스트하고 `to_lattice_units` 경로를 안 거쳤음**. 그래서 통과했지만
실제 fine-level ALM 경로에서 실패. **전체 경로 미검증이 원인.**

## 수정
1. `generate_markers`: `self._marker_distribution`/`self._cosine_side` 저장.
2. `__init__`: 기본값 "uniform"/"both".
3. `to_lattice_units:872`: `generate_markers(..., distribution=self._marker_distribution, cosine_side=self._cosine_side)`.

## 검증 (`test_full_path_distribution.py`, 전체 경로 from_config→to_lattice_units)
- uniform: first 0.2597/last 0.9922 (변화없음, byte-identical) ✓
- cosine: first 0.2523/last 0.9996 (클러스터 유지) ✓
- endpoint: first 0.2519(root-cut)/last 1.0000(tip) ✓
- **ALL PASS.**

## 조치 (사용자)
**`src/actuator/blade.py` 재동기화 후 cosine/endpoint 재실행**(2번째). uniform+gauss/point은 영향 없음(byte-identical) → 재실행 불요.
