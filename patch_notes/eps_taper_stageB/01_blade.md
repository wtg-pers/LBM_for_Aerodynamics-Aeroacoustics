# 1단계 — `src/actuator/blade.py`

**내용:** `Blade`에 ε 테이퍼 본체를 추가. 3개 편집 모두 `epsilon_mode`로 분기되며,
`"default"` 경로는 원본과 byte-identical.

> 참고: `blade.py`에는 이미 이번 작업과 무관한 미커밋 변경(`project_force_to_global`의
> `thrust_axis` 인자)이 있었음. 그건 이 단계와 **무관**하며, 아래 3개 hunk만 이 단계 작업임.

## 편집 1.1 — `__init__` 기본값 (marker 배열 뒤, ~192행)

```python
# Gaussian projection-width (ε) taper controls.
# "default" reproduces ε = max(chord/4, 2·Δx) exactly; "tip_taper"
# narrows ε toward the tip (Diaz 2023 §2.1.4) to reduce tip-vortex
# over-smearing. Set by Rotor.from_config; see set_lattice_spacing().
self.epsilon_mode: str = "default"      # "default" | "tip_taper"
self.epsilon_tip_factor: float = 1.0    # tip ε target = max(factor·2·Δx, 2·Δx)
self.epsilon_taper_start: float = 0.7   # r/R where taper begins
```
**이유:** 모든 `Blade`(`to_lattice_units`에서 새로 만들어지는 것 포함)가 안전한 기본값으로
이 컨트롤들을 갖게 됨 → 손대지 않은 config는 이전과 완전히 동일하게 동작.

## 편집 1.2 — `set_lattice_spacing` 분기 (~327행)

원래의 `np.maximum(chord/4, 2·dx)`를 `eps_base`로 분리한 뒤 분기:

```python
eps_base = np.maximum(self.marker_chord / 4.0, 2.0 * dx)

if self.epsilon_mode == "tip_taper":
    r_norm = self.marker_r / self.r_tip
    taper_start = self.epsilon_taper_start
    t = np.clip((r_norm - taper_start) / (1.0 - taper_start), 0.0, 1.0)
    eps_tip = max(self.epsilon_tip_factor * 2.0 * dx, 2.0 * dx)   # ≥ floor
    self.marker_epsilon = (1.0 - t) * eps_base + t * eps_tip
else:
    self.marker_epsilon = eps_base
```
**이유:** per-marker 기준 ε을 `taper_start` 이후로 더 좁은 팁 값으로 선형 블렌딩
(Diaz 2023 §2.1.4). `eps_tip`은 `2·dx`로 하한이 걸려 ε이 LBM 투영 하한 아래로 절대
내려가지 않음. **`"default"`는 정확히 `eps_base`를 반환** → bit-identical.

**단위 주의:** 런타임에는 `Rotor.to_lattice_units`에서 `dx=1.0`(lattice units)로 호출됨 →
하한 `2·dx = 2.0 lu`, chord는 이미 lu. 물리 dx 재곱셈 없음.

## 편집 1.3 — `to_lattice_units`가 컨트롤을 복사 (~797행)

```python
new_blade = Blade(new_sections)

new_blade.epsilon_mode = self.epsilon_mode
new_blade.epsilon_tip_factor = self.epsilon_tip_factor
new_blade.epsilon_taper_start = self.epsilon_taper_start
```
**이유:** `to_lattice_units`는 **새로운** `Blade()`를 만들고, 그 `__init__`이 컨트롤을
기본값으로 리셋한다. 따라서 `Rotor.to_lattice_units`가
`blade_lu.set_lattice_spacing(dx=1.0)` (rotor.py:1104, 권위 있는 lattice-unit ε 계산)을
호출하기 **전에** 반드시 복사되어야 한다. 이 복사가 없으면 실제 런에서 테이퍼가 조용히
no-op이 된다.

## 검증

- `python -m py_compile src/actuator/blade.py` → **OK**.
- 논리: `epsilon_mode="default"`이면 메서드가 `eps_base`를 그대로 반환(새 분기 진입 안 함)
  → 기준 bit-identical 보장. 전체 수치 게이트는 5단계에서.
