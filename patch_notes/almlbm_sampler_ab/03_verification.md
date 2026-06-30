# 03 — 검증 (smoke)

**스크립트:** `to_claude/sampler_ab_smoke.py` (CPU, 유동장 합성, ~즉시)
실행: `PYTHONPATH=<repo> python to_claude/sampler_ab_smoke.py`

## 결과 (전부 PASS)

### A. import
편집한 `interpolation.py` / `actuator_line.py` 구문·임포트 정상.

### B1. point 정확성
선형장 `u = a + b·x + c·y + d·z`에서 trilinear 해석값과 `max|err| = 7.1e-15`
(부동소수 한계) → `_sample_trilinear` 정확.

### B2. 메커니즘 (호버 모사 step-downwash, u_z=−1 디스크 내부 / 0 외부)
팁 마커(cyl≈R−0.31ε)에서 4모드 u_z (진짜값 −1.0):

| 모드 | u_z | 회복률 |
|------|-----|--------|
| gaussian (A) | **−0.6082** | **61%** (←39% off-disk 희석) |
| point (B-i) | −1.0000 | 100% |
| aniso (B-ii, factor 0.5) | −0.6851 | 69% (부분) |
| mask_disk (B-iii) | −1.0000 | 100% |

→ **gaussian이 팁 다운워시의 ~39%를 잃는 것을 직접 확인**(off-disk 진단과 정합).
point·mask_disk 완전 회복, aniso 부분 회복. 메커니즘 입증.
(주: factor 0.5 aniso는 회복이 약함 → 실제 튜닝 시 더 작은 eps_r_factor 검토.)

### C. end-to-end model.step() 4모드
`Rotor.from_simple`(axis=z, R=30 lu) + dummy polar로 4모드 실행:
- 전부 무오류·유한 force grid, `sum|F| > 0` (BEM 작동).
- `Σ|F_gaussian − F_mask_disk| = 46.2` (>0) → dispatch가 실제로 다른 force 생성.

## 미수행 (의도적)
- **실제 HVAB A/B/C 런**: 무거운 production → 사용자 클러스터 실행
  ([[feedback_simulation_execution]]). 본 패치는 config+CPU smoke까지.
- GPU 경로는 xp-generic 코드라 cupy로 동일 동작 예상(별도 GPU smoke는 클러스터에서).

## 다음
실제 HVAB에서 `sampling.mode`를 gaussian/mask_disk/point로 A/B/C 런 →
팁 φ·α·u_n 스팬분포·C_T·FM 비교 → b1(off-disk)/b2(반경smooth) 정량 분해.
production config 작성은 후속(`06_production_config.md` 예정).
