# 2단계 — `src/actuator/rotor.py` (`Rotor.from_config`)

**내용:** rotor config에서 3개 테이퍼 키를 읽어, 방금 만들어진 물리 `Blade`에 할당.
`Blade.from_config(...)` 직후, `set_lattice_spacing(dx=dx)` 직전에 위치.

```python
blade = Blade.from_config(blade_cfg)

# ε-projection-width taper controls (default → bit-identical baseline ε).
# Assigned before set_lattice_spacing so both the physical pass here and
# the later lattice-unit pass (Rotor.to_lattice_units) honor the mode.
blade.epsilon_mode = config.get('epsilon_mode', 'default')
blade.epsilon_tip_factor = config.get('epsilon_tip_factor', 1.0)
blade.epsilon_taper_start = config.get('epsilon_taper_start', 0.7)
```

**이유:**
- 물리 blade가 config로부터 태어나는 유일한 지점. 여기서 컨트롤을 설정하면 바로 아래의
  **물리** `set_lattice_spacing(dx_phys)`와 — 결정적으로 — 나중의 **lattice-unit**
  `set_lattice_spacing(dx=1.0)` (`Blade.to_lattice_units` → 1.3단계 복사 경유) 둘 다
  요청한 모드를 보게 된다.
- 기본값은 `Blade.__init__`과 동일(`"default"`, `1.0`, `0.7`) → 키를 생략한 config는 불변.

**키의 출처:** 여기서 `config`는 `create_actuator_line_from_config`가 `Rotor.from_config`로
넘기는 `rotor_cfg` 딕셔너리. 3단계에서 이 딕셔너리에 3개 키를 주입한다(`grid.dx` 주입과
동일 방식)므로, 사용자는 `actuator_line` 레벨에서 한 번만 지정하면 된다.

## 검증

- `python -m py_compile src/actuator/rotor.py` → **OK**.
- 키 흐름 end-to-end는 5단계(스모크 런이 팁 ε을 도출/출력)에서 확인.
