# 3단계 — `src/actuator/actuator_line.py` (팩토리 주입)

**내용:** `actuator_line` config 레벨의 3개 테이퍼 키를 `Rotor.from_config`까지 전달.
single-rotor / multi-rotor 팩토리 양쪽 모두.

## 편집 3a — `create_actuator_line_from_config` (`grid.dx` 주입 직후)

```python
# ε-taper controls are specified at the actuator_line level; forward them
# into rotor_cfg so Rotor.from_config sees them (a per-rotor override placed
# directly under 'rotor' takes precedence).
for _eps_key in ('epsilon_mode', 'epsilon_tip_factor', 'epsilon_taper_start'):
    if _eps_key in config and _eps_key not in rotor_cfg:
        rotor_cfg[_eps_key] = config[_eps_key]
```
**이유:** 기존 `grid.dx` 주입을 그대로 따름. 키는 선택적 — 없으면 루프가 no-op이고
`Rotor.from_config`가 기본값으로 폴백(bit-identical). `not in rotor_cfg` 가드 덕분에
`rotor` 바로 아래 둔 값이 우선(per-rotor override).

## 편집 3b — `create_multi_rotor_from_config`

`shared_defaults`:
```python
'epsilon_mode': config.get('epsilon_mode', 'default'),
'epsilon_tip_factor': config.get('epsilon_tip_factor', 1.0),
'epsilon_taper_start': config.get('epsilon_taper_start', 0.7),
```

`single_config` (`gaussian_cutoff`과 동일하게 shared 폴백 + per-rotor override):
```python
'epsilon_mode': rotor_entry.get('epsilon_mode', shared_defaults['epsilon_mode']),
'epsilon_tip_factor': rotor_entry.get('epsilon_tip_factor', shared_defaults['epsilon_tip_factor']),
'epsilon_taper_start': rotor_entry.get('epsilon_taper_start', shared_defaults['epsilon_taper_start']),
```
**이유:** multi-rotor 덱은 함대 전체 테이퍼(top-level)를 두고 로터별로 override 가능.
`single_config`은 이후 `create_actuator_line_from_config`로 넘어가므로, 3a가 키를 각 로터의
`rotor_cfg`로 전달.

## 커버리지 참고

Fine-level(MLG) ALM은 **이 동일 팩토리**를 `setup.py` 경유로 재사용 → L0와 fine-level 로터
모두 자동으로 테이퍼 적용, 별도 편집 불필요.

## config 사용 예시

```python
actuator_line = {
    "rotor": { ... },
    "gaussian_cutoff": 3.0,
    "prandtl_loss": True,
    # ε taper (opt-in):
    "epsilon_mode": "tip_taper",
    "epsilon_tip_factor": 1.0,
    "epsilon_taper_start": 0.7,
}
```

## 검증

- `python -m py_compile src/actuator/actuator_line.py` → **OK**.
- 전체 config→팁 ε 흐름은 5단계에서 확인.
