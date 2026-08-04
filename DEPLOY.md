# 배포 가이드

> ## ⚠️ 불특정 다수에게 링크를 보낸다면: `DEMO_MODE=1`
>
> 공개 배포에서 잠가야 하는 것이 여럿인데, 하나라도 빠뜨리면 그게 곧 노출입니다.
> 그래서 스위치를 하나로 모았습니다. **`DEMO_MODE=1` 하나만 켜면** 아래가 전부 적용됩니다.
>
> | | `DEMO_MODE` 미설정 (기본) | `DEMO_MODE=1` |
> |---|---|---|
> | `/docs` · `/openapi.json` · `/redoc` | 공개 | **404** |
> | 챗봇 라우터 (`/api/v2/chat/*`) | 마운트됨 | **미마운트** |
> | `/editor-v2` · `/editor-lab` (개발용) | 200 | **404** |
> | `/api/v2/recommendations/{evaluate,explain}` | 공개 | **404** |
> | `/api/building/analyze` (레거시 V1) | 공개 | **404** |
> | `DEMO_MAX_MEMBERS` | 0 (무제한) | **250** |
> | `LLM_CALLS_PER_IP_PER_HOUR` / `_PER_DAY` | 0 (무제한) | **5 / 100** |
> | `KDS_CACHE_MODE` | `auto` (DB 우선) | **`prefer`** (로컬 스냅샷 우선) |
> | 에디터 10층 프리셋 | 있음 | **숨김** (250부재 한도에 반드시 걸림) |
>
> 개별 환경변수를 명시하면 그쪽이 항상 우선합니다. **플래그를 끄면 이전과 동일하게 동작**하는 것이
> 이 설계의 수용 기준이라, 로컬 개발에는 아무 영향이 없습니다.
>
> `render.yaml`에는 이미 들어 있습니다. Azure/Docker로 직접 띄운다면 아래 `--env-vars`에 추가하세요.
>
> **Supabase는 `DEMO_MODE=1`에서 선택사항입니다.** KDS 참조표(하중·지진·단면 3,808행)의 로컬
> 스냅샷이 `data/kds_cache/`에 들어 있고 `KDS_CACHE_MODE=prefer`가 그걸 먼저 읽습니다.
> 그래서 DB가 잠자거나 잠겨 있어도 해석이 돌고, 오히려 조회당 왕복이 사라져 더 빠릅니다
> (3층 프리셋 9.5초 → 6.3초). 스냅샷은 DB에서 그대로 덤프한 것이라 값이 다르지 않습니다
> (`scripts/snapshot_kds_tables.py`). 요약하면 **공개 배포에서 Supabase 자격증명을 빼도 됩니다.**
>
> ### `DEMO_AUTH_TOKEN` (챗봇을 켠 배포에서만 해당)
>
> `DEMO_MODE=1`이면 챗봇 라우터 자체가 마운트되지 않으므로 아래 항목은 해당 없습니다.
> **챗봇을 의도적으로 켠 배포**에서만 읽으세요.
>
> `DEMO_AUTH_TOKEN`을 설정하지 않으면 KDS 챗봇 **감사 로그 엔드포인트**
> `GET /api/v2/chat/audit/{analysis_id}`가 운영자 모드로 열립니다 — 누구나 analysis_id만으로
> 다른 사용자의 설계 검토 메타데이터(부재 ratio, 위반 항목)와, `include_quotes=true` 시
> 인용 quote까지 조회할 수 있습니다. 설정 시에는 (a) 토큰 보유자 또는 (b) 해당 레코드를 작성한
> 본인 세션(`session_id`)만 조회 가능합니다.
> 감사 로그 파일(`data/chat_audit/`)은 quote·ratio·analysis_id를 포함하는 민감 산출물이며
> `.gitignore`로 커밋이 차단됩니다. 저장 위치는 `CHAT_AUDIT_LOG_PATH`로 재지정 가능.

## 사전 준비

### 1. Azure for Students 계정 생성
1. https://azure.microsoft.com/ko-kr/free/students/ 접속
2. 학교 이메일(.ac.kr)로 인증 → **$100 크레딧 + 12개월 무료** (신용카드 불필요)

### 2. Azure CLI 설치
```bash
# Windows (PowerShell 관리자)
winget install Microsoft.AzureCLI

# 설치 확인
az --version
```

### 3. Azure 로그인
```bash
az login
```

---

## 배포 (한 번만 실행)

### Step 1: 리소스 그룹 생성
```bash
az group create --name opensees-demo-rg --location koreacentral
```

### Step 2: Container Registry (ACR) 생성
```bash
# ACR 이름은 전 세계 고유해야 함 (영소문자+숫자만)
az acr create \
  --resource-group opensees-demo-rg \
  --name openseesdemo \
  --sku Basic \
  --admin-enabled true
```

### Step 3: Docker 이미지 빌드 & 푸시
```bash
# 프로젝트 루트에서 실행
cd d:/son/opensees-MCP

# ACR에 로그인
az acr login --name openseesdemo

# 이미지 빌드 & 푸시 (ACR에서 직접 빌드 - 로컬 Docker 불필요!)
az acr build \
  --registry openseesdemo \
  --image opensees-app:latest \
  --file Dockerfile \
  .
```

> **참고**: `az acr build`는 Azure 클라우드에서 빌드하므로 로컬에 Docker Desktop이 없어도 됩니다.

### Step 4: Container Apps 환경 생성
```bash
az containerapp env create \
  --name opensees-demo-env \
  --resource-group opensees-demo-rg \
  --location koreacentral
```

### Step 5: Container App 배포
```bash
# ACR 비밀번호 가져오기
ACR_PASSWORD=$(az acr credential show --name openseesdemo --query "passwords[0].value" -o tsv)

az containerapp create \
  --name opensees-app \
  --resource-group opensees-demo-rg \
  --environment opensees-demo-env \
  --image openseesdemo.azurecr.io/opensees-app:latest \
  --registry-server openseesdemo.azurecr.io \
  --registry-username openseesdemo \
  --registry-password "$ACR_PASSWORD" \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 0 \
  --max-replicas 2 \
  --env-vars \
    DEMO_MODE="1" \
    SUPABASE_URL="your-supabase-url" \
    SUPABASE_KEY="your-supabase-key" \
    ANTHROPIC_API_KEY="your-anthropic-key" \
    MCP_SERVER_PATH="/app/mcp-server"
```

### Step 6: URL 확인
```bash
az containerapp show \
  --name opensees-app \
  --resource-group opensees-demo-rg \
  --query "properties.configuration.ingress.fqdn" \
  -o tsv
```

결과: `opensees-app.<random>.koreacentral.azurecontainerapps.io`

접속: `https://opensees-app.<random>.koreacentral.azurecontainerapps.io/?token=your-secret-token-here`

---

## 업데이트 (코드 수정 후)

```bash
# 1. 이미지 다시 빌드 & 푸시
az acr build \
  --registry openseesdemo \
  --image opensees-app:latest \
  --file Dockerfile \
  .

# 2. 컨테이너 업데이트 (새 이미지 반영)
az containerapp update \
  --name opensees-app \
  --resource-group opensees-demo-rg \
  --image openseesdemo.azurecr.io/opensees-app:latest
```

> 이 두 명령어면 끝입니다. 1~2분이면 반영됩니다.

---

## 환경변수 변경

```bash
# 토큰 변경
az containerapp update \
  --name opensees-app \
  --resource-group opensees-demo-rg \
  --set-env-vars DEMO_AUTH_TOKEN="new-token"

# 여러 변수 한번에
az containerapp update \
  --name opensees-app \
  --resource-group opensees-demo-rg \
  --set-env-vars \
    DEMO_AUTH_TOKEN="new-token" \
    ANTHROPIC_API_KEY="new-key"
```

---

## 비용 관리

### 현재 설정 예상 비용
| 리소스 | 월 비용 (예상) |
|--------|---------------|
| Container Apps (idle 시 0대) | ~$0~5 |
| ACR Basic | ~$5 |
| **합계** | **~$5~10/월** |

### 비용 절약 팁
- `--min-replicas 0`: 접속 없으면 컨테이너 0대 → 과금 $0
- 첫 요청 시 cold start ~10~30초 (컨테이너 시작 시간)
- `--min-replicas 1`로 변경하면 cold start 없지만 상시 과금

### 크레딧 확인
```bash
# Azure Portal에서 확인
# https://portal.azure.com → Cost Management → Credits
```

---

## 삭제 (정리)

```bash
# 리소스 그룹 통째로 삭제 (모든 리소스 제거)
az group delete --name opensees-demo-rg --yes --no-wait
```

---

## 로컬 테스트 (Docker)

```bash
# 로컬에서 Docker로 테스트 (Docker Desktop 필요)
docker compose up --build

# 접속: http://localhost:8000
# 인증 테스트: http://localhost:8000/?token=test123
# (DEMO_AUTH_TOKEN=test123 설정 시)
```

---

## 트러블슈팅

### 로그 확인
```bash
az containerapp logs show \
  --name opensees-app \
  --resource-group opensees-demo-rg \
  --follow
```

### 컨테이너 재시작
```bash
az containerapp revision restart \
  --name opensees-app \
  --resource-group opensees-demo-rg
```

### Cold start가 너무 느릴 때
```bash
# 최소 1대 항상 유지 (상시 과금 발생)
az containerapp update \
  --name opensees-app \
  --resource-group opensees-demo-rg \
  --min-replicas 1
```
