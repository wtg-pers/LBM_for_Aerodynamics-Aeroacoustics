# 02 — actuator_line.py (필드 · dispatch · config)

**파일:** `src/actuator/actuator_line.py`

## 변경 4곳

1. **import** (§ import 블록): `sample_velocity_alt` 추가.

2. **ActuatorLineModel.__init__** (eps_corr 필드 직후):
   ```python
   self._sampling_mode: str = "gaussian"      # gaussian|point|aniso|mask_disk
   self._sampling_eps_r_factor: float = 0.5   # B-ii: ε_r = factor·ε
   ```

3. **step()** (속도 샘플 단계) — dispatch:
   ```python
   if self._sampling_mode == "gaussian":
       u_markers = interpolate_velocity_batch_gpu(...)        # §6, bit-identical
   else:
       u_markers = sample_velocity_alt(
           self._sampling_mode, u_field, positions, epsilon_all,
           xp=xp, n_cut=self.n_cut,
           hub=np.asarray(self.rotor.hub_center, float),
           axis=np.asarray(self.rotor.rotation_axis, float),
           radius=float(self.rotor.radius),
           eps_r_factor=self._sampling_eps_r_factor)
   ```
   → 단일·다중 로터 step 모두 ActuatorLineModel.step을 거치므로 양쪽 커버.

4. **create_actuator_line_from_config** (eps_correction 블록 직후):
   ```python
   samp = config.get('sampling', None)
   if isinstance(samp, dict):
       model._sampling_mode = samp.get('mode', 'gaussian')
       model._sampling_eps_r_factor = samp.get('eps_r_factor', 0.5)
   elif isinstance(samp, str):
       model._sampling_mode = samp
   ```

5. **create_multi_rotor_from_config**: `shared_defaults`·`single_config`에 `'sampling'`
   전파(per-rotor override 허용). 기본 None → "gaussian".

## 기하 출처 (lu, ALM 레벨)
`rotor.hub_center` (lu), `rotor.rotation_axis` (단위벡터), `rotor.radius` (lu).
fine-level ALM은 to_lattice_units로 이미 lu이므로 추가 변환 불필요.

## 절대 제약 확인
`sampling` 미지정 또는 "gaussian" → §6 경로 그대로 → 기존 config 전부 bit-identical.
