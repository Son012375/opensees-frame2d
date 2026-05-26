# Chat Tool Matrix

> Phase 0 산출물 — 챗봇이 무엇을 할 수 있어야 하는지 범위를 잠그는 문서.
> 이번 단계(A→D)에서 구현되는 도구와, 차기 phase로 미루는 기능을 한 표에 담아 범위 팽창을 막는다.
> 연관 plan: `~/.claude/plans/c-users-youm-claude-plans-v2-editor-fede-cosmic-hare.md`

## Phase 정의

| Phase | 목표 | LLM이 할 수 있는 것 | 차단 |
|-------|------|---------------------|------|
| **A** | Read-only inspect | 분석 요약/부재 조회 답변 | 모델 mutation 일체 X |
| **B** | Recommendations | 추천 목록·평가·해설 트리거 | candidate_id 생성/점수 변경 X |
| **C** | Edit (Read-First) | 단면 변경 **미리보기**까지만 | Apply는 사용자 클릭 필수, LLM whitelist 제외 |
| **D** | KDS RAG | 조항 인용 | `validate_code_reference()` 미통과 chunk 차단 |
| **Later** | NL/IFC/run/export/view | (이번 범위 밖, 매트릭스에만 등재) | — |

## Tool Matrix

| Feature | User phrase 예시 | Phase | Chat tool | 기존 UI / API | Data source | Mutates model? | Safety |
|---------|------------------|-------|-----------|---------------|-------------|----------------|--------|
| 부재 선택 inspect | "이 부재 안전해?" / "K3-2 ratio 보여줘" | **A** | `inspect_selection` | 3D viewer 클릭 → Property panel | `cache.member_info_by_elem_id` + `cache.member_ratios_by_elem_id` | No | ui_context의 `selected_element_ids` (node 제외 필터) 우선, 모호하면 LLM이 사용자에게 재확인 |
| 분석 요약 | "현재 분석 결과 요약" / "NG 부재 몇 개?" | **A** | `get_analysis_summary` | DC 탭 / Modal 탭 | `cache.analysis_summary` + `cache.modal_summary` + `cache.envelope` | No | summary는 deterministic dict 그대로 반환, LLM은 paraphrase만 |
| 추천 목록 | "추천 후보 뭐 있어?" | **B** | `list_recommendations` | 추천 탭 (`switchResultTab('recommend')`) | `cache.candidates_by_id` 읽기 | No | candidate_id 신규 생성 불가 |
| 추천 평가 | "위험한 부재 추천 평가해" | **B** | `evaluate_recommendations` | Evaluate 버튼 → `/api/v2/recommendations/evaluate` | `services.recommendation_jobs._eval_executor` (Phase B 진입 전 `submit_eval`/`poll_eval` public wrapper로 추출 예정) | No (재해석은 trigger, 모델은 그대로) | `_eval_executor` 단일 워커 직렬화, polling 5초/60초 cap, 진행률을 `status` 이벤트로 stream |
| 추천 설명 | "왜 그 후보가 추천됐어?" | **B** | `explain_recommendation` | 카드 클릭 modal | `core.recommendation.explainer.explain_candidate` | No | KDS 인용은 evidence만, `validate_code_reference()` 통과 chunk만 |
| 단면 변경 미리보기 | "3층 기둥 H-400으로 키워줘" | **C** | `preview_section_change` | `/api/v2/recommendations/preview-apply` + diff modal | `apply_candidate_to_model` (deepcopy) | **Preview only** (저장 X) | virtual candidate를 `chat_session.virtual_candidates_by_id[session][preview_id]`에 등록, `EditorV2ChatBridge.openDiffPreview()`로 modal 주입 |
| Apply 확인 | (LLM이 직접 호출 불가) | **C** | `confirm_apply_preview` | 사용자가 modal "Apply" 클릭 | — | **사용자 확인 후 Yes** | `tool_registry` whitelist에서 제외, 클라이언트가 별도 `POST /api/v2/chat/apply-preview/{preview_id}`로만 트리거 |
| KDS 조항 인용 | "층간변위 기준이 뭐야?" | **D** | `query_kds` | (없음, 신규) | `core.kds_rag.factory.get_default_kds_retriever()` | No | quote 400자 cap, `validate_code_reference()` 통과 chunk만, Noop fallback이면 "RAG 미사용" 배지 |
| NL → config | "10층 오피스 서울 강남" | **Later (K)** | `nl_to_building_config` | NL 탭 → `/api/claude/parse-building` | `core.nl_resolver.resolve_building_config` | No | — |
| IFC parse | "이 IFC 파일 열어" | **Later (K)** | `parse_ifc_file` | IFC 탭 → `/api/ifc/parse` | `core.ifc_parser` | No (config만) | 파일 업로드는 클라이언트에서 |
| 해석 실행 | "지금 모델 해석 돌려" | **Later (K)** | `run_analysis` | Run 버튼 → `/api/v2/analyze` | OpenSees | **No (모델은 그대로지만 cache 갱신)** | Phase C와 동일 risk class, 별도 phase에서 안전성 재검토 후 도입 |
| Export | "Excel 결과 내보내" | **Later (K)** | `export_result` | DXF/XLSX 버튼 → `/api/v2/export/*` | jobs_db | No | 단순 trigger |
| View / filter / selection | "NG 부재만 보여줘" | **Later (K)** | `view_filter` (가칭) | 클라이언트 only (`showDCColors` 등) | — | No (시각 표시만) | bridge에 별도 `setViewFilter()` 추가 필요 |

## 운영 원칙

1. **신규 도구 추가 절차** — 이 매트릭스에 행을 먼저 추가하고, Phase 컬럼을 정한 뒤에 코드 작성. Phase가 "Later"면 plan의 "향후 확장" 섹션과 연결.
2. **도구 이름 = Python 함수명** — `tools/{inspect,recommend,edit,kds}.py`의 함수와 1:1 매칭.
3. **`tool_registry`에 등록되지 않으면 LLM이 호출 불가** — env `CHAT_TOOLS_ENABLED`로 phase별 토글, whitelist 위반 호출은 orchestrator가 거부.
4. **Mutation? 컬럼이 "Yes"인 도구는 존재하지 않아야 한다** — 모든 변경은 preview → bridge → 사용자 확인 경로로만.
5. **확장은 작게** — 새 phase 시작 전 plan + 이 매트릭스 동시 갱신, 단일 PR 원칙.

## Phase 진입 기준

- **A 시작**: chat_router + orchestrator + OllamaProvider + Bridge.getContext() 통합 동작
- **B 시작**: `services.recommendation_jobs.submit_eval/poll_eval` public wrapper 추출 완료
- **C 시작**: `chat_session.virtual_candidates_by_id` 라이브 사용 (Phase 0에서는 skeleton만), `EditorV2ChatBridge.openDiffPreview` Phase C 본구현
- **D 시작**: `VOYAGE_API_KEY` + `KDS_RAG_INDEX_PATH` env 설정 시 자동 활성, 미설정이면 Noop 응답으로 graceful degrade
