# Apply MVP — 최소 적용 파이프라인

DB(SSOT) + ContractInterpreter(결정론) + OpenSees 3D의 end-to-end 동작을 검증하는 MVP.

## 실행

```bash
python -m apply_mvp.runner --input data/apply_mvp_input.json
# 상세 로그:
python -m apply_mvp.runner --input data/apply_mvp_input.json -v
```

## 파이프라인 흐름

```
JSON 입력 → LoadCaseBuilder(DB 조회) → ContractInterpreter(combo 해석)
         → OpenSees 3D(하중 적용 + 선형해석) → 결과/로그 출력
```

## 파일 구조

| 파일 | 역할 |
|------|------|
| `schemas.py` | 입력/출력 데이터클래스 |
| `load_case_builder.py` | DB에서 LL 조회, 면하중→선하중 변환, load_cases 생성 |
| `opensees_3d_mvp.py` | 최소 3D frame (2층 1bay×1bay) 생성/해석 |
| `runner.py` | CLI 진입점, 전체 파이프라인 오케스트레이션 |

## Supabase 연동

- `SUPABASE_URL` + `SUPABASE_KEY` 환경변수가 있으면 실제 DB 조회
- 없으면 `data/kds_output/03_normalized.json` (로컬 DB) + `docs/specs/examples/combo_apply_example.json` (예시 combo) fallback

## 제약사항 (MVP)

- IFC / Analysis Graph 미사용 — 임시 JSON 입력
- 지붕활하중(RLL) = 0 (지붕 미모델링)
- 3D 모델: 1bay×1bay, 단면/재료 고정 (H-300×300 기둥, H-400×200 보, SS275)
- Diaphragm 생략
- 선형 정적 해석 1스텝

## 확장 계획

1. 실제 Supabase 연결로 live_load + load_combo 동시 조회
2. 다층/다bay 프레임 확장
3. IFC → 자동 geometry 생성
4. P-Delta / 비선형 해석
5. HTML 리포트 생성 (기존 visualization.py 연동)

## 테스트

```bash
pytest tests/test_apply_mvp_pipeline.py -v
```
