# Step 3 — endpoint Γ=0 closure (옵션 3): 구현 + ★물리 검증서 결함 발견 (2026-06-30)

## 구현
- `_eps_endpoint_closure` 플래그 + config `eps_correction.endpoint_closure`(기본 False).
- `_viscous_core_correction`: closure ON 시 root-cut·tip(첫/끝 셀 EDGE = `r[0]−Δr₀/2`, `r[-1]+Δr_{N-1}/2`)에
  **Γ=0 가상 trailed 노드** 추가 → 확장격자서 dΓ/dr + 사다리꼴 weight로 kernel 합, 실제 마커서만 평가.

## 검증 (`test_step3_closure.py`, 물리·exact)
| | 결과 |
|---|---|
| P1 closure OFF byte-identical | ✅ (기존 Dağ 불변) |
| P2 net trailed 순환 →0 | ✅ −1.7e-16 (closure 물리적; open=Γ_last−Γ_first) |
| P3 가상노드=root_cut/tip 정확 | ✅ (셀중심 분포서 edge=끝점) |
| **P4 팁 w_corr 부호** | ⚠️ **open +0.012 → closed −0.093 (부호 역전 = 업워시)** |

## ★결론: 옵션 3는 깨끗한 fix가 아님 (물리 검증이 잡음)
w_corr는 "추가 다운워시(양수=de-load)"인데 closure가 팁서 **음수(업워시)**로 뒤집음 → 팁 과부하 **악화** 방향.
원인:
1. **base/smeared wake 불일치(이중계산)**: Dağ deficit는 trailed vorticity가 *마커에만*(smeared projection과 일치)
   존재 가정. ideal 쪽에만 Γ=0 tip shed 추가 → `ideal−smeared` 균형 붕괴 → 비물리 부호.
2. **Γ→0을 dr/2서 강제 = 인공·격자의존 급경사**: 가상 팁노드 grad=−2Γ_last/dr ≈ 내부 17배.
   **dr↓(클러스터)일수록 spurious↑** → 수렴 안 함.

→ 코드는 opt-in·OFF 기본(byte-identical)로 보존하되, **Case 2(closure 비교)는 비물리로 비권장.**
정합 대안 = **옵션 2(endpoint 분포)**: force·보정 둘 다 실제 팁 마커(유한 Γ) 사용 → 일관.

## 권고
- Case 2 보류(또는 "음의 결과"로만 기록). 마커/팁 개선은 **옵션 2(endpoint) + cosine 클러스터(Dağ의 dΓ/dr 정밀화)**로.
- 옵션 3를 살리려면 smeared 쪽도 동일 closure를 줘 deficit 일관성 확보 필요(별도 설계, 후순위).
