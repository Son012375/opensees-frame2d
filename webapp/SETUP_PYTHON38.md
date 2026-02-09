# Python 3.8 환경 설정 가이드

OpenSeesPy Windows 버전은 **Python 3.8**만 지원합니다.

## 방법 1: Python 3.8 직접 설치 (권장)

1. **Python 3.8 다운로드**
   - https://www.python.org/downloads/release/python-3810/
   - Windows installer (64-bit) 다운로드 및 설치
   - 설치 시 "Add Python to PATH" 체크하지 않음 (기존 환경과 충돌 방지)

2. **새 가상환경 생성**
   ```batch
   # Python 3.8이 설치된 경로 (예: C:\Python38)
   C:\Python38\python.exe -m venv d:\son\opensees-MCP\venv38

   # 가상환경 활성화
   d:\son\opensees-MCP\venv38\Scripts\activate

   # 패키지 설치
   pip install openseespy fastapi uvicorn[standard] pydantic jinja2 python-multipart supabase plotly numpy
   ```

3. **웹앱 실행**
   ```batch
   cd d:\son\opensees-MCP\webapp
   d:\son\opensees-MCP\venv38\Scripts\activate
   run_simple.bat
   ```

## 방법 2: Anaconda/Miniconda 사용

1. **Miniconda 설치**
   - https://docs.conda.io/en/latest/miniconda.html

2. **환경 생성**
   ```batch
   conda create -n opensees38 python=3.8
   conda activate opensees38
   pip install openseespy fastapi uvicorn[standard] pydantic jinja2 python-multipart supabase plotly numpy
   ```

## 방법 3: WSL2 사용 (Linux 환경)

WSL2에서는 Python 3.9+ 에서도 OpenSeesPy가 작동합니다.

```bash
# WSL2 Ubuntu에서
sudo apt update
sudo apt install python3-pip
pip3 install openseespy fastapi uvicorn pydantic jinja2 python-multipart plotly numpy

cd /mnt/d/son/opensees-MCP/webapp
python3 -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8000
```

---

## LLM 관련 참고

현재 웹앱은 **LLM(Claude API)을 사용하지 않습니다**.
- 사용자 입력 → OpenSees 직접 해석 → HTML 리포트 생성
- Claude API 연동은 별도 개발 필요 (자연어 → 구조 입력 변환)
