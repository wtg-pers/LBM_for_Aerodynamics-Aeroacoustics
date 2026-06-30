# 6단계 — Production A/B config (클러스터 런)

**내용:** `configs/caradonna_tung/ct_hover_t08_m088_taper.py` — 기존 baseline
`ct_hover_t08_m088.py`(C_T ≈ 0.00553을 낸 덱)의 Stage B(테이퍼 ON) 짝.

baseline 대비 차이는 단 하나:
```python
config["actuator_line"]["epsilon_mode"] = "tip_taper"
config["actuator_line"]["epsilon_tip_factor"] = 1.0
config["actuator_line"]["epsilon_taper_start"] = 0.7
# + 별도 출력 폴더  result_ct_t08.0_M877_mlg4_D32_light_taper/
```
격자/preset/SGS/Prandtl 전부 baseline과 동일 → 깔끔한 A/B. Prandtl은 baseline에 맞춰 ON
유지 (의도된 팁-ε ↔ R_tip_eff 커플링 주의; Prandtl 격리 연구는 `prandtl_loss=False`).

**검증 (빌드만, 런 없음):**
- 테이퍼 키 존재; `prandtl_loss=True`; device `gpu`; D32, mlg 4-level; `max_steps=18090`.
- 출력 폴더 `..._light_taper`는 baseline `..._light`와 **분리** → baseline 덮어쓰지 않음.
- baseline config는 `epsilon_mode` 없음 → `default`로 해석 (bit-identical 경로).

**D=16 스모크에선 안 보이고 여기선 테이퍼가 작동하는 이유:** D32 `light`에서는 fine-level
`chord_lu ≈ 21.3` → `chord/4 ≈ 5.3 > floor 2.0`이라 ε이 floor에 안 걸림 → 테이퍼가 외측
스팬을 눈에 띄게 좁힘 (`05_verification.md`에서 수학 검증). D=16 스모크는 `chord/4 = 0.667 <
2.0`이라 ε이 균일하게 floor (테이퍼 no-op).

## 클러스터 런 + 즉시 sanity check

```
python main.py --config configs/caradonna_tung/ct_hover_t08_m088_taper.py
```
**런 시작 수 초 내**(setup 시점, timestep 전 기록됨)에 테이퍼 작동 여부 확인:
`result_ct_t08.0_M877_mlg4_D32_light_taper/csv/blade_diagnostics/blade_geometry.csv`
→ `epsilon_lu` 컬럼이 r/R 0.7 안쪽은 ~일정, 팁으로 갈수록 **감소**(~2.0 쪽)해야 함.
균일하면 테이퍼 미작동 — 거기서 멈추고 점검.

## A/B 비교 (런 종료 후) — 커맨드

### ⭐ 가장 쉬운 방법: 단일 스크립트 (복붙 불필요, 한 줄 실행)
아래 (1)(2)(3)을 한 번에 수행. heredoc 복붙 문제 없음.
```bash
python src/utilities/compare_taper_ab.py        # 기본 폴더명(M0.877 light) 사용
# 또는 폴더/옵션 지정:
python src/utilities/compare_taper_ab.py \
    --A result_ct_t08.0_M877_mlg4_D32_light \
    --B result_ct_t08.0_M877_mlg4_D32_light_taper \
    --mtip 0.877 --avg-revs 3 --exp-ct 0.00473
```
출력: 콘솔에 C_T(A/B/exp) + 팁 요약표, `aeromechanics_workshop/temp_results/`에
`AB_taper_compare.png`(eps_lu·phi·M2CL·F_n A/B 오버레이) + `spanwise_{A,B}_*.csv`.
(plot 라벨은 ASCII — 클러스터 한글 폰트 없어도 안전. 콘솔은 한글.)
실데이터로 end-to-end 검증 완료(2026-06-22).

---

아래는 위 스크립트가 내부적으로 하는 일을 직접 돌리고 싶을 때의 수동 단계.

폴더 변수:
```bash
A=result_ct_t08.0_M877_mlg4_D32_light          # baseline
B=result_ct_t08.0_M877_mlg4_D32_light_taper    # taper
```

### (1) spanwise 분포 — `spanwise_post.py`  (주의: `--result`는 플래그, 위치인자 아님)
```bash
python -m src.utilities.spanwise_post --result $A --mtip 0.877 --avg-revs 3
python -m src.utilities.spanwise_post --result $B --mtip 0.877 --avg-revs 3
# → aeromechanics_workshop/temp_results/spanwise_<폴더명>.csv (+ 플롯)
#   컬럼: r_R, alpha, phi, Re, CL, CD, mach, M2CL, F_n, F_theta, eps_lu
```

### (2) C_T 비교 (rotor_performance.csv, 마지막 3 rev 평균) vs 실험 0.00473
```bash
python - <<'PY'
import pandas as pd
for tag,d in [("A base","result_ct_t08.0_M877_mlg4_D32_light"),
              ("B taper","result_ct_t08.0_M877_mlg4_D32_light_taper")]:
    df = pd.read_csv(f"{d}/csv/rotor_performance.csv")
    w  = df[df.revolutions >= df.revolutions.max()-3]   # 마지막 3 rev
    T  = w.thrust_lu.mean()
    CT = T / (w.rho_ref.mean()*w.area_lu.mean()*w.tip_speed_lu.mean()**2)  # rotorcraft
    print(f"{tag:8s}  C_T={CT:.5f}   (T_lu={T:.4f})")
print("exp                C_T=0.00473")
PY
```

### (3) A vs B 오버레이 (eps_lu / phi / M2CL vs r/R) — (1) 산출물 사용
```bash
python - <<'PY'
import pandas as pd, matplotlib.pyplot as plt
base="aeromechanics_workshop/temp_results"
A=pd.read_csv(f"{base}/spanwise_result_ct_t08.0_M877_mlg4_D32_light.csv")
B=pd.read_csv(f"{base}/spanwise_result_ct_t08.0_M877_mlg4_D32_light_taper.csv")
fig,ax=plt.subplots(1,3,figsize=(15,4))
for col,a in zip(["eps_lu","phi","M2CL"],ax):
    a.plot(A.r_R,A[col],"-o",ms=3,label="A base")
    a.plot(B.r_R,B[col],"-s",ms=3,label="B taper")
    a.set(xlabel="r/R",title=col); a.grid(alpha=.3); a.legend()
fig.tight_layout(); fig.savefig(f"{base}/AB_taper_compare.png",dpi=130)
print("saved", f"{base}/AB_taper_compare.png")
PY
```

**기대 신호:** B에서 `eps_lu` 팁 감소(2.0 쪽) → 팁 `phi` 상승(φ→0 결손 회복) →
팁 `M2CL` roll-off → `C_T` 감소(0.00553 → 0.00473 쪽).

이후 결정: B 충분, 또는 Stage C(filtered lifting-line, JFM-2019 — hover 타당성 caveat) 진행.
HART2 FM calibration 재평가.

## 선택적 민감도 (한 점으로 불충분하면)

`epsilon_tip_factor`는 2·dx에서만 floor(그 아래 불가)이므로, 스윕할 노브는
`epsilon_taper_start`(예: 0.6 / 0.7 / 0.8) — 시작이 빠를수록 외측 스팬을 더 공격적으로 좁힘.
이 config를 다른 start + 폴더 suffix로 복제하면 됨.
