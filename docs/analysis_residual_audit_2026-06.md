# 해석시스템 잔여 이슈 감사 (2026-06-24, 갭12 구현 이후)

> 적대적 감사 워크플로우(6축 / 73 agents)로 **이미 수정한 8버그 + 12갭을 제외하고** 추가로
> 발견·검증한 미해결 이슈 **49건**. 새 세션에서 이어받기 위한 핸드오프.
> 우선순위: **Tier 1(안전-비보수, 최우선) → Tier 2(코드기준 누락검토) → Tier 3(견고성/모델링) → Tier 4(보수적/문서/테스트)**.
> "안전-비보수(safety-unconservative)" = 실제로 위험할 수 있는데 '안전'으로 표시될 수 있는 것 → 최우선.
> 직전 작업: `memory/analysis_gaps_impl_2026-06.md`, `tests/test_analysis_gaps.py`.

## Tier 1 — CRITICAL (안전-비보수 / 핵심 누락) ⚠️

1. **K=1.0 유효좌굴길이 — 모멘트골조에서 비보수** (`design_check.py:616, 734`)
   - 비가새 모멘트골조(sway)는 K≈1.2~2.0+인데 K=1.0 고정 → 세장비 과소 → φPn 과대 → 기둥이 실제보다 안전하게 표시.
   - 가정문구 "K=1.0 보수적, 비가새 골조는 K>1 필요"가 **거꾸로 서술됨**(K=1이 비보수인데 보수라 주장).
   - Fix: 골조형식별 K 산정(AISC A2) 또는 모멘트골조 최소 K≥1.2 적용. 문구 정정.

2. **보 횡-비틀림좌굴(LTB) 미검토** (`design_check.py:130-138 _bending_capacity`)
   - Mn=Fy·Zx로 항상 완전소성+완전횡지지 가정. 중간횡지지 없으면 LTB 지배 → 용량 10~30% 과대.
   - Fix: AISC F2 — Lb/Lp/Lr/Cb로 Mn=min(Mp, Mcr). member_info에 비지지길이 Lb 추가(없으면 Lb=부재길이 보수가정).

3. **`_infer_member_story`가 실제 층정보 무시 → P-Delta 증폭이 엉뚱한 부재에 적용** (`design_check.py:525-543`)
   - `member_id % len(stories)`로 층 추정. 그러나 `frame_3d.py:676`이 **이미 `member_info["story"]`(실제 층)을 채움**.
   - C3 P-Delta 부재증폭(`pdelta_amp_by_story[story]`)·NG 부재 층보고가 잘못된 층에 매핑될 수 있음.
   - **Quick win(저위험):** `_infer_member_story`를 `minfo.get("story") or (기존 폴백)`로 교체. (단 story=None 엣지 처리 — Tier3 참고)

4. **솔버 실패가 0(영) 결과로 '안전' 보고** (`frame_3d.py:~2559 정적, ~1649 고유치`)
   - 비수렴/특이행렬 시 빈 dict 또는 0 변위·0 부재력이 그대로 흘러 design_check가 '안전'으로 판정 가능.
   - Fix: 케이스 결과에 `error`/solver_meta 플래그를 싣고, run_design_check가 error면 NG/not_checked로 게이트. (수렴실패를 침묵시키지 말 것.)

5. **RSA(응답스펙트럼) 경로가 부재력 미생성 → RSA 조합 부재강도 미검토** (`response_spectrum_analysis.py`)
   - RSA는 절점 변위/층변위만 산출, 요소력(N/V/M) 미추출. `rsa_result_to_case_data`가 force를 0으로 하드코딩.
   - RSA 조합이 combo_names에 추가되지만 member_forces 없음 → 부재검토 누락(또는 ELF force 오용). 모듈이 사실상 미통합.
   - Fix: 모드별 `ops.eleResponse(eid,'localForce')` → SRSS/CQC 조합 → member_forces 구성. 또는 RSA UI 노출 시 "부재검토 미수행" 명시.

## Tier 2 — HIGH (KDS/AISC 의무검토 누락)

6. **세장비 한계 미검토** (`design_check.py:101-127`) — 압축 KL/r≤200, 인장 L/r≤300 미확인 → 초과부재가 통과.
7. **국부좌굴/단면 콤팩트 분류 누락** (`design_check.py:130-148`, `section_3d.py`) — b/2tf·h/tw 미분류, 항상 compact 가정 → 비콤팩트/세장 단면 용량 과대.
8. **우발 비틀림(±5% 편심) 미적용** (`load_generator.py generate_seismic_loads`) — KDS 41 17 등가정적 의무. C2 비틀림은 해석 실변위만 사용(우발편심 별도).
9. **직교지진 100%+30% 조합 미적용** (`load_generator.py:~793`) — SDC C/D 의무(EQX_100+EQY_30 등).
10. **풍하중 사용성 변위(H/400) 미검토** (`design_check.py`) — 지진 층간변위만 있음. `check_wind_drifts` 신설.
11. **모달 모드수 상한 min(3n,15) → 고층 <90%** (`frame_3d.py:~1646, ~2865`) — A3로 경고만, 자동증대 없음. 미달 시 num_modes 상향 재시도.
12. **수직 비정형(연층·중량·강도 불연속) 자동탐지 부재** (`building_model.py`/`frame_3d.py`) — KDS 41 17 표5.3-2. 층강성 K_i 비교로 탐지.
13. **요약 서술이 일부 'not_checked'인데 '전부 통과'로 단정** (`result_interpreter.py:_generate_summary`/`_classify_severity`) — 처짐/시스템/안정성 미검토 시 명시. 모든 NG 하위검토가 severity에 반영되는지 점검.
14. **json.dumps NaN/Inf 디폴트 핸들러 없음** (`visualization_calc_report.py:151`, `visualization_3d.py`) — 방금 고친 튜플키처럼 500 유발 가능. `default=`/sanitize 또는 NaN/Inf 클램프.
15. **인장 부재 순단면파단(D2-b)·블록전단** (B3 후속) — 총단면항복만. 연결부 검토 권고는 R21로 표기됨(정량검토는 미구현).

## Tier 3 — MEDIUM (견고성 / 모델링 근사)

- `member_info["story"]`가 None 가능(`frame_3d.py:667`, node_to_story 누락 시) → 검증/폴백 필요(3번과 연계).
- 영길이 요소 필터가 V2만 있고 V1 없음(`frame_3d.py:2523 vs 600-643`) → 비정형 좌표병합 시 특이행렬 위험.
- 강막(rigidDiaphragm) 절점 <3개 시 rank 결손 경고 없음(`frame_3d.py:515-538`).
- 부재 릴리즈: elem_type↔torsion_dof 불일치 시 특이행렬 사전경고 없음; 부분릴리즈(반강접) 미지원(`frame_3d.py:604-625`).
- P-Delta 부재 2차모멘트는 1/(1−θ) 선형근사 — 매우 세장한 기둥/θ>0.2에서 Cm(AISC) 고려 권고(`design_check.py:1173-1250`).
- 적설 불균형/표류 미모델(`load_generator.py:680-738`); 풍속 V0 폴백 사용 시 경고 없음.
- Cs_min의 IE 적용 해석 모호(`load_generator.py:444`) — 조항 명시 + 민감도.
- 유연/반강성 다이어프램 미처리(rigid 가정만).
- 기둥 비지지길이를 층고로 가정(횡지지 입력 없음).

## Tier 4 — LOW (보수적 / 명확성 / 테스트)

- 활하중 면적저감(KDS 41 12 §3) 미적용 — 보수적이나 데이터(03_live_reduction)는 존재, 옵션화 가능.
- 보 약축휨 용량(phiMny) H1에 포함 — 보수적/문서 명확화(`design_check.py:638-643`).
- kN/m↔N/mm 단위변환 주석 보강(`frame_3d.py:747,867,926`).
- 단층건물 story_nodes_map 완전성 검증.
- 비틀림 미평가 서술: drift·torsion 둘 다 None일 때 facts에 'torsion_performed' 플래그.
- C-phase 수치(FS·θ·δmax/δavg) anti-hallucination allowlist 통합 테스트 추가.

## 공통 권장 절차 (직전 작업과 동일)
1. 각 검토 → `run_design_check` 반환 + `summary` + `overall_status` 반영.
2. `result_interpreter`(finding R-code) + `narrative_interpreter.build_facts`(facts, allowlist 자기씨앗) + `visualization_calc_report.renderSec10`(§10) 연결.
3. **design_check에 넣는 dict 키/값은 JSON-safe**(튜플키·set·numpy·NaN/Inf 금지) — `json.dumps(dc)` 회귀 테스트 필수.
4. regression `tests/test_analysis_gaps.py` 패턴 + 벤치마크(case6 V2) 영향 확인 + 페이즈별 적대적 리뷰 워크플로우.
5. 안전-비보수 항목은 **실해석 케이스로 NG가 실제로 떠야** 검증 완료(예: 세장 기둥/비지지 보/θ>0.1).
