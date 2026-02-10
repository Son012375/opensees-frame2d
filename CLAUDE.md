# OpenSees-MCP Claude 가이드

> **프로젝트 상세 문서:** `.claude/PROJECT_CONTEXT.md`

## Quick Start

새 세션에서 컨텍스트 복원:
```
/read .claude/PROJECT_CONTEXT.md
```

## 현재 상태 요약

| 해석 유형 | 상태 |
|-----------|------|
| Simple Beam | ✅ Ready |
| Continuous Beam | ✅ Ready |
| Frame 2D | ✅ Ready |
| Frame 3D | 🚧 Planned |

## 핵심 파일

| 파일 | 설명 |
|------|------|
| `mcp-server/core/frame_2d.py` | 2D 프레임 해석 엔진 |
| `mcp-server/core/visualization.py` | HTML 리포트 생성 (~3500줄) |
| `mcp-server/core/sign_convention.py` | 부호규약 변환 |
| `webapp/backend/app/main_simple.py` | FastAPI 앱 |

## 부호규약

- **V > 0:** 좌측면 상향 (↑)
- **M > 0:** Sagging (하부 인장)
- **변환:** `V_textbook = -V_opensees`, `M_textbook = -M_opensees`

## 실행

```bash
cd webapp/backend
python -m uvicorn app.main_simple:app --port 8001
```

## 진행 예정 (장기)

1. 3D Frame 해석
2. 부재 릴리즈 (힌지)
3. P-Delta 해석
