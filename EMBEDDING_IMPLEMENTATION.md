# 📊 임베딩 기능 구현 문서

> **구현 날짜:** 2025-09-26
> **구현자:** 조성호
> **목적:** 보고서 자동 생성 후 임베딩 벡터 생성 및 DB 저장

## 🎯 구현 개요

### 핵심 기능
- `/api/generate-summary` API에서 보고서 생성 후 **자동 임베딩 생성 및 DB 저장**
- **임베딩 모델:** `jhgan/ko-sbert-nli` (한국어 특화, 768차원)
- **기존 보고서 생성 로직 유지**, 임베딩 저장은 추가 기능으로 구현

### 파이프라인
1. 더미 보고서 생성 (OpenAI API 또는 더미 텍스트)
2. 한국어 임베딩 모델로 768차원 벡터 생성
3. PostgreSQL vector 타입으로 DB 저장

---

## 🏗️ 아키텍처 변경사항

### 1. DB 스키마 수정
```sql
-- AS-IS (1536차원)
report_embedded public.vector(1536)

-- TO-BE (768차원)
report_embedded public.vector(768)
```

**변경 이유:** `jhgan/ko-sbert-nli` 모델이 768차원 출력

### 2. 새로 추가된 코드 구조

#### 📦 새 의존성
```python
from sentence_transformers import SentenceTransformer
import numpy as np
```

#### 🔧 ReportEmbeddingService 클래스
```python
class ReportEmbeddingService:
    """보고서 임베딩 전용 서비스"""
    - create_embedding(): 텍스트 → 768차원 벡터
    - create_vector_string(): 벡터 → PostgreSQL vector 형식
```

#### 💾 store_report_with_embedding() 함수
```python
async def store_report_with_embedding():
    """보고서 + 임베딩 통합 저장 함수"""
    - 임베딩 생성
    - DB에 보고서와 임베딩 동시 저장
    - 에러 핸들링 포함
```

#### 🔄 generate_summary API 수정
기존 로직 + 임베딩 저장 기능 추가

---

## 🔑 OpenAI API 키 문제 해결

### 문제 상황
- 개발 환경에서 OpenAI API 키 없이 테스트 필요
- 기존 `manager_chain.ainvoke()` 호출 시 401 에러 발생

### 해결 방식
```python
# OpenAI API 키 검증 로직
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("OPENAI_A") or len(OPENAI_API_KEY) < 20:
    # 더미 보고서 사용
    manager_summary = "[더미 보고서] Task ... 샘플 텍스트..."
else:
    # 실제 OpenAI API 호출
    manager_summary = await manager_chain.ainvoke({"team_reports": dummy_reports})
```

**장점:**
- API 키 없이도 임베딩 기능 테스트 가능
- 실제 운영 시에는 OpenAI API 정상 작동
- 개발/테스트 환경 분리

---

## 🔧 DB 시퀀스 충돌 해결

### 문제 원인
```sql
-- sample_data.sql 기존 설정 (문제)
SELECT pg_catalog.setval('public.report_id_seq', 1, false);
```
- 기존 샘플 데이터: id=1~4
- 새 INSERT 시도: id=1 (중복 키 에러)

### 해결 방법
```sql
-- sample_data.sql 수정 (해결)
SELECT pg_catalog.setval('public.report_id_seq', (SELECT COALESCE(MAX(id), 0) FROM public.report), true);
```
- 자동으로 기존 데이터의 최대 ID + 1부터 시작
- 어떤 샘플 데이터든 호환

---

## 🛠️ 설치 및 테스트 가이드

### 1. 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# 주요 추가 패키지
- sentence-transformers>=2.2.2
- huggingface_hub>=0.16.0
```

### 2. DB 초기화
```bash
# PostgreSQL 로컬 DB 설정
psql -U postgres -h localhost -p 5432 -c "DROP DATABASE IF EXISTS weekly_report_db;"
psql -U postgres -h localhost -p 5432 -c "CREATE DATABASE weekly_report_db;"
psql -U postgres -h localhost -p 5432 -d weekly_report_db -f sample_data.sql
```

### 3. API 테스트
```bash
# 서버 실행
python user_timeline_api.py

# Swagger UI 접속
http://localhost:8001/docs

# /api/generate-summary 테스트
{
  "task_id": 1,
  "start_date": "2025-09-20",
  "end_date": "2025-09-25"
}
```

### 4. 결과 확인
```sql
-- 임베딩 저장 확인
psql -U postgres -h localhost -p 5432 -d weekly_report_db -c "SELECT id, task_id, writer, vector_dims(report_embedded) as dimension FROM public.report ORDER BY id DESC LIMIT 3;"

-- 상세 확인
SELECT id, task_id, timestamp, writer,
       LEFT(report, 100) as report_preview,
       vector_dims(report_embedded) as embedding_dimension,
       report_embedded IS NOT NULL as has_embedding
FROM public.report
ORDER BY id DESC LIMIT 5;
```

---

## 🎯 협업자를 위한 정보

### 주요 함수별 역할
- `ReportEmbeddingService.__init__()`: 모델 초기화 (최초 실행 시 다운로드)
- `create_embedding()`: 텍스트 → 768차원 numpy array → Python list
- `create_vector_string()`: Python list → PostgreSQL vector 형식 문자열
- `store_report_with_embedding()`: 전체 파이프라인 통합 실행

### 에러 해결
1. **ImportError (huggingface_hub)**: `pip install --upgrade huggingface_hub`
2. **시퀀스 중복 키**: sample_data.sql 재실행 또는 시퀀스 수동 설정
3. **vector 타입 에러**: `vector_dims()` 함수 사용 (array_length 대신)

### 향후 개선 방향
- [ ] **실제 프로덕션 환경에서 OpenAI API 키 설정**
  - ⚠️ **중요:** API 키 설정 후 `user_timeline_api.py` 504-511라인의 더미 텍스트 로직 제거 필요
  ```python
  # 제거해야 할 코드 (504-511라인)
  if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("OPENAI_A") or len(OPENAI_API_KEY) < 20:
      manager_summary = f"[더미 보고서] Task {request.task_id}에 대한..."
      print("🔄 OpenAI API 키가 없어서 더미 보고서를 사용합니다.")
  else:
      manager_summary = await manager_chain.ainvoke({"team_reports": dummy_reports})

  # 정리 후 코드
  manager_summary = await manager_chain.ainvoke({"team_reports": dummy_reports})
  ```
- [ ] 임베딩 벡터 유사도 검색 API 추가
- [ ] 배치 처리로 다중 보고서 임베딩 최적화
- [ ] 임베딩 캐싱 시스템 도입

---

## 📋 체크리스트

- [x] 768차원 임베딩 생성 성공
- [x] PostgreSQL vector 타입 저장 성공
- [x] API 키 없이 테스트 환경 구축
- [x] DB 시퀀스 충돌 해결
- [x] 기존 보고서 생성 로직 보존
- [x] 에러 핸들링 및 로깅 구현

**✅ 임베딩 기능 구현 완료!**