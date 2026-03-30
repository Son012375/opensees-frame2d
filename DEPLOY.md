# Azure Container Apps 배포 가이드

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
    DEMO_AUTH_TOKEN="your-secret-token-here" \
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
