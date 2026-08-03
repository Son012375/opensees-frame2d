# 해석시스템 갭 백로그 (적대적 감사 2026-06-24)

> **✅ 전 12갭 구현·검증 완료 (2026-06-24).** Phase A→B→C 순차 구현 + 페이즈별 적대적
> 리뷰 워크플로우(98 agents) 통과. 회귀: `tests/test_analysis_gaps.py` (44 케이스),
> 전체 스위트 825 pass, 벤치마크 case6 V2 무영향(office/II → 저장창고·증폭계수 0). 아래
> 원본 핸드오프 명세(A1~A5, B1~B3, C1~C3)는 전부 구현됨. 리뷰에서 발견·수정된 추가 결함은
> `memory/analysis_gaps_impl_2026-06.md` 참조.
>
> 구현 위치는 대부분 `mcp-server/core/design_check.py` (검토함수) +
> `result_interpreter.py`(finding R13~R25) + `narrative_interpreter.build_facts`(facts) +
> `visualization_calc_report.renderSec10`(§10) + `frame_3d.py`(modal/disp 극값).
>
> ── 원본 명세 (핸드오프) ──
> 26-agent 워크플로우 감사에서 confirmed된 **미해결 갭 12건**. 버그 8건(B1~B6 + #1)은
> 이미 수정·검증 완료(`tests/test_analysis_bugfixes.py`, `memory/analysis_audit_2026-06.md`).
>
> 각 갭은 '안전' 판정을 뒤집을 수 있으므로, 구현 시 `run_design_check` overall_status +
> `result_interpreter._classify_severity` + `visualization_calc_report.renderSec10` 판정 게이트에
> 반드시 연결하고, `narrative_interpreter.build_facts`에 facts로 노출해야 §10이 서술한다.

## Phase A — 빠른 보강 (판정 불변, 즉시 가치)

### A1. 유효지진중량 W가 창고 25% 활하중 누락 (docstring 모순)
- 파일: `mcp-server/core/load_generator.py` `_calculate_story_weights` (docstring "활하중은 창고 25%만 포함"이라 했으나 본문은 `w = DL*area`만), 동일 결함 `_estimate_story_weights`(frame_3d.py 모달 질량).
- Fix: 창고류(storage/storage_light/storage_heavy/factory) 용도에 `+0.25*LL*area`. 비창고는 0(현 단순화 유지). 모달질량도 동일 적용해 일관. 또는 미구현 시 docstring 정정 + 창고시 경고.

### A2. 지배 하중조합 §10/facts 미표기
- 파일: `mcp-server/core/design_check.py`(summary에 drift_governing_combo/member_governing_combo 추가) → `narrative_interpreter.build_facts`(max_drift_ratio/max_interaction_ratio 옆에 지배조합 문자열), `visualization_calc_report.py` §9 footer.
- 데이터 이미 존재: `drift_check["critical"]["combo"]`, member의 `governing_combo`(B3 수정으로 이제 실제 H1 지배조합). 집계·노출만.

### A3. V2 모달 누적참여 ≥90% 충분성 플래그 부재
- 파일: `mcp-server/core/frame_3d.py` `_run_eigen_analysis_v2` 반환 dict에 `cumulative_participation: {x_pct, y_pct, sufficient_90pct}` 추가(V1과 동형). num_modes=min(3*n,15) 고층 미달 가능 → 경고/모드수 증대.

### A4. V2 모달 비틀림모드 미탐지 (rz_pct 하드코딩 0, ROTN-Z 미분류)
- 파일: `mcp-server/core/frame_3d.py` `_run_eigen_analysis_v2`. 현재 direction='X'/'Y'/'XY'만, rz_pct=0 고정 → `result_interpreter._interpret_modal`의 torsion_risk(first_dir=='ROTN-Z')와 R10/R11이 V2에서 영영 안뜸(false R12 가능).
- Fix: 마스터에 회전질량(I_eff 이미 계산됨) 배정 + rz 참여 계산 + 'TRAN-X'/'TRAN-Y'/'ROTN-Z' 라벨(V1과 통일). 또는 `_interpret_modal`을 라벨 무관(rz_pct 우세로 비틀림 탐지)하게.

### A5. (=⑪) 데이터 envelope 성분별 worst 아님
- 파일: `design_check.py` `_governing_member_forces`. B3로 H1·전단은 조합독립 최댓값이나, demand(Pu/Mux/Muy)는 H1지배 단일조합 스냅샷. 축력·강축·약축 최대가 다른 조합일 때 성분별 worst로 보강(보수성).

## Phase B — 사용성·적용성 (과대주장 해소, 판정 강화)

### B1. 보 수직처짐/사용성 검토 부재 (L/360·L/240)
- 현 `run_design_check`는 (1)지진 층간변위 (2)부재강도 H1만. 보 중력처짐(KDS 41 31 사용성: 활하중 L/360, 전체 L/240) 전무. `simple_beam.py`/`continuous_beam.py`엔 있으나 3D 경로엔 없음.
- **§10 "사용성 만족"이 drift만으로 단정 = 과대주장.** Fix: `check_beam_deflections` 신설 — 중력조합서 보 중앙 상대처짐(지점침하 차감) 산정(보를 중간절점 메싱하거나 형상함수 후처리해 mid-span sag 포착) → `summary.max_deflection_ratio` → overall_status + _classify_severity + §10 판정 게이트 + facts. 미수행 시 §10에 "처짐 미검토" 명시.

### B2. 내진설계범주(SDC) + 횡력저항시스템 적용성/높이제한
- 지진리포트에 SDS/SD1/R/Ω0/Cd/seismic_system은 있으나 SDC 판정·시스템 적격성·높이제한 없음. SDS=0.46g(SDC C~D)에서 OMF(R=3.5) 적용가능 여부 미검토 → 부재·drift OK면 §10이 '적합' 결론.
- Fix: (1) (SDS,SD1,Ie)→SDC 매핑(밴드데이터 `data/kds_output/05_seismic_dry_run.json`의 seismic_design_category_Sds). (2) SDC별 허용시스템+높이제한 테이블(SEISMIC_SYSTEM_MAP 키 기준). (3) `run_design_check`에 system_check → 부적격시 critical_issues + overall_status NG.

### B3. 인장부재를 압축내력 φPn로 검토
- 파일: `design_check.py` `_interaction_H1`/axial_ratio가 항상 `abs(Pu)/phiPn`(압축 Fcr). 순인장(인발기둥·가새)은 AISC H1.2로 인장내력 Pc=φt·Fy·Ag(>압축) 사용해야. 보수적이나 오류.
- Fix: envelope에 부호있는 N 보존 → 부호 분기(인장이면 Pc=φ·Fy·A/1000, H1.2형식). 순단면파단(D2-b) 미검토는 가정에 명기.

## Phase C — 전역안정 (고급, 판정 뒤집을 수 있음)

### C1. 전도(overturning)·활동(sliding) 안정
- V·층별 Fx·hx로 전도모멘트 M_ot=Σ(Fx·hx), 저항 M_r=0.9·W·(평면폭/2) 산정가능하나 어디에도 없음(grep overturning/sliding/전도/활동 → 0).
- Fix: `run_design_check`에 global-stability: 방향별 M_r/M_ot ≥ 1.5(또는 양압조합) + 활동(ΣFx ≤ μN). overall_status + _classify_severity 연결. (참고: visualization_3d.py의 overturning_check는 평형확인일 뿐 안정평가 아님 → 개명 권장.)

### C2. 비틀림 비정형 정량 + 변위증폭 Ax
- 현재 modal 1차 ROTN-Z일 때 정성 flag만(result_interpreter). KDS 41 17: δmax/δavg>1.2 비정형, >1.4 극한; Ax=clamp((δmax/(1.2·δavg))²,1,3). `check_story_drifts`는 inelastic=Cd·δe/Ie만(Ax 누락).
- Fix: (a) frame_3d 층변위서 다이어프램 절점 corner δmax/평균 δavg 추적. (b) ratio·비정형 분류. (c) Ax 적용해 inelastic_drift=Ax·Cd·δe/Ie. (d) 등가정적에 우발편심 0.05L 옵션.

### C3. P-Delta 안정계수 θ
- KDS 41 17 θ=(Px·Δ)/(Vx·hsx·Cd), θ≤0.1 무시, θ>θmax=min(0.5/(β·Cd),0.25) 불안정. 어디에도 없음. 3D는 Corotational로 2차효과 반영하나 θ 정량판정·보고 없음.
- Fix: `design_check`에 stability_check: Px(층상부 누적중량), Vx(Fx 누적), Δ(비탄성 층간변위), hsx → θ. 0.1<θ≤θmax면 1/(1-θ) 증폭, θ>θmax면 NG. result_interpreter finding 코드 추가.

## 미해결 잔여(낮음)
- ⑩ = A3(V2 충분성). ⑫ = A2(지배조합 보고). (Phase A에 흡수)

## 구현 공통 체크리스트
1. 각 검토 → `run_design_check` 반환 dict + `summary` + `overall_status` 반영.
2. `result_interpreter._classify_severity`의 max_ratio/severity에 새 비율 포함, finding 코드 추가.
3. `narrative_interpreter.build_facts`에 facts(숫자 문자열 사전포맷) 추가 → §10 서술. anti-hallucination 규칙 준수(allowlist 자기씨앗).
4. `visualization_calc_report.renderSec10` 판정/박스에 연결.
5. regression: `tests/test_analysis_bugfixes.py` 패턴으로 케이스 추가. 벤치마크(case6 V2/importance II) 영향 확인.
6. 적대적 리뷰(워크플로우)로 각 페이즈 검증.
