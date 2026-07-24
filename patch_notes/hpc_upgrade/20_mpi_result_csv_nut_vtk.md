# 20 — MPI 런 결과폴더 CSV 복구 + nu_t 필드 VTK (2026-07-21)

## 문제 (사용자 지적)
1. **rotor_performance.csv / blade_diagnostics/*.csv 헤더만 기록** — setup이
   헤더를 쓰지만 main_mpi 락스텝 루프는 OutputManager.process()를 안 거쳐
   행이 영원히 안 붙음. 0718 β 매트릭스·0721 factorial 결과폴더가 전부
   헤더-온리였던 원인 (그때마다 마커 VTP 재적분으로 우회).
   (--csv 덴스 시계열은 클러스터 런치 디렉토리에만 남음.)
2. **필드 VTK에 nu_t 부재** — MLGVTKWriter는 level.nu_t를 이미 지원하나
   MPI 브리지 _LevelView가 rho/u만 gather → dyn_smag 런에서도 미기록.
   과소산 분해 트랙(SGS 기여 정량)에 필요한 필드.

## 수정
- `src/solver/output_manager.py`: 행 기록을 모듈 함수
  `log_rotor_performance_row(model, path, step)` /
  `log_blade_diagnostics_rows(model, dir, step)`로 추출(단일 소스),
  OutputManager 메서드는 위임. blade 행에 eps 6열 추가(EPS_FLOORS 연계).
- `main_mpi.py`: rank 0 on_log(--log-every 주기)에서 두 함수 호출 —
  단일-GPU 루프와 동일 스키마/주기로 결과폴더 CSV 채움.
- `src/parallel/output.py`: dyn_smag 레벨의 `L.nut`(flat N) gather 추가
  (tag 600+2k), _LevelView에 nu_t 전달 → VTI에 nu_t 배열 기록.
- `src/io/marker_vtk_writer.py` + `get_blade_diagnostics()`: 마커에
  eps_lu + 스프레딩(eps_c/t/r) + 샘플링(eps_samp_c/t/r) 기록.
  iso 경로는 3축 모두 eps_lu로 fallback(스키마 모드-불변).

## 검증 (로컬 3090, main_mpi 단일 프로세스)
- bench5_fact_s4(aniso both+dyn_smag) 40步: rotor_performance 5행,
  blade_diag 26열(eps_t=2.0 floor 확인, eps_r=2.0), VTI arrays =
  density/velocity/**nu_t**, VTP에 eps 7배열. CT 안정.
- bench5_purealm_m3(iso) 16步: trio=eps_lu fallback ✓, 회귀 무.

## ⚠배포 주의
main_mpi.py는 src/ 밖(레포 루트) — **src/만 복사 배포하면 main_mpi.py
수정이 누락**됨(과거 동일 사고 선례). 이번 변경은 main_mpi.py + src/ 4개
파일 동시 배포 필수. 현재 돌고 있는 l4wake A/B는 구코드라 여전히
헤더-온리 → 기존 VTP 재적분 경로로 분석(이미 검증된 워크플로).
