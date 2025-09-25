# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

무조건 코드 이외의 모든 답변은 한국어로 해주세요.

## 프로젝트 개요
AI 기반 주간업무 보고서 자동 생성 시스템. Slack, Notion, OneDrive, Outlook의 4개 데이터 소스에서 활동 데이터를 수집하여 시간순 통합 타임라인을 생성하고, AI로 보고서를 자동 생성합니다.

## 핵심 아키텍처

### 백엔드 시스템
- **메인 API 서버**: `user_timeline_api.py` (FastAPI, 포트 8000)
- **데이터 모델**: `model/data.py` (Pydantic 스키마)
- **AI 보고서 생성**: `fetch_reports.py` (LangChain + OpenAI)
- **벡터 임베딩**: `vector_embedding.py` (pgvector 기반 유사도 검색)

### 데이터베이스 (PostgreSQL + pgvector)
- **컨테이너**: pgvector/pgvector:pg16 (포트 5433→5432)
- **4개 데이터 테이블**: slack, notion, onedrive, outlook
- **벡터 검색**: 각 테이블에 embedding 컬럼 (768차원) + ivfflat 인덱스
- **스키마 파일**: `backup_v9.sql` (전체), `table.sql` (벡터 컬럼 추가)

### 프론트엔드
- **Vue.js 대시보드**: `fastapi-project/front/` (Vite + Vue 3)
- **Streamlit 앱**: `fastapi-project/streamlit_app.py`

## 개발 명령어

### 데이터베이스 실행
```bash
# Docker로 PostgreSQL 실행
docker-compose up -d

# 스키마 적용 (필요시)
psql -U postgres -d weekly_report_db -f backup_v9.sql
```

### 백엔드 서버 실행
```bash
# 가상환경 활성화 (Windows)
weekly-report-generator\Scripts\activate

# FastAPI 서버 실행 (포트 8000)
python user_timeline_api.py

# Streamlit 앱 실행
cd fastapi-project
python streamlit_app.py
```

### 프론트엔드 개발
```bash
cd fastapi-project/front
npm install
npm run dev
```

## API 아키텍처

### 핵심 엔드포인트
- `GET /api/user-timeline/{user_id}` - 사용자별 통합 타임라인 (4개 소스 통합)
- `POST /api/generate-report` - AI 기반 주간 보고서 생성
- `GET /api/activities/{source}/{task_id}` - 소스별 활동 데이터 조회

### 데이터 통합 플로우
1. **데이터 수집**: 4개 소스(Slack, Notion, OneDrive, Outlook)에서 사용자 활동 조회
2. **시간순 정렬**: timestamp 기준 통합 타임라인 생성
3. **벡터 검색**: pgvector 기반 유사 활동 검색
4. **AI 생성**: LangChain + OpenAI로 주간 보고서 자동 생성

### 데이터베이스 연결
- **로컬 DB**: `postgresql+asyncpg://postgres:1234@localhost:5432/weekly_report_db`
- **Docker DB**: myuser/mypassword@localhost:5433/mydatabase