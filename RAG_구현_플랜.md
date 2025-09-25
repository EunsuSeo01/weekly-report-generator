# RAG 기반 관리자 질의응답 시스템 구현 완료

## 📋 프로젝트 개요
기존 주간 보고서 시스템에 **관리자용 RAG(Retrieval-Augmented Generation) 질의응답 기능**을 추가하여, 관리자가 자연어로 질의하면 과거 보고서들을 검색하여 종합적인 답변을 제공하는 시스템

## 🎯 핵심 기능
**관리자 사용 시나리오:**
```
관리자: "OO프로젝트 DB 성능 이슈 어떻게 처리됐는지 알려줘"
↓
시스템: 질의 임베딩 → 전체 보고서 유사도 검색 → Top-5 추출 → LLM 종합 답변
↓
결과: "과거 3건의 유사 사례에서 다음과 같이 해결했습니다..."
```

## 🔧 기술 설정
- **임베딩 모델**: `all-MiniLM-L6-v2` (384차원, 가볍고 빠름)
- **벡터DB**: PostgreSQL + pgvector (기존 인프라 활용)
- **유사도 임계값**: 0.6 (관리자용, 포괄적 검색)
- **검색 개수**: Top-5 (다양한 사례 참조)
- **검색 범위**: 전체 report 테이블 (task_id 제한 없음)
- **응답 형태**: LLM이 과거 사례를 종합한 구체적 답변

## ✅ 구현 완료 상태

### Phase 1: DB 스키마 확장 ✅
```sql
-- add_embedding_column.sql
ALTER TABLE public.report ADD COLUMN embedding vector(384);
CREATE INDEX report_embedding_idx ON public.report
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 5);
```

### Phase 2: 관리자용 RAG API 구현 ✅
```python
# 새로운 API 엔드포인트
@app.post("/api/admin-query", response_model=ReportResponse)
async def admin_rag_query(request: AdminQueryRequest, session: AsyncSession):
    """관리자의 자유 텍스트 질의에 대한 RAG 기반 답변"""
```

### Phase 3: 핵심 함수들 구현 ✅
1. **`create_embedding(text: str)`** - all-MiniLM-L6-v2 모델로 384차원 벡터 생성
2. **`search_similar_reports(query_text: str, session, top_k=5)`** - 전체 report 테이블 유사도 검색
3. **`insert_report()` 확장** - 보고서 저장 시 자동 임베딩 처리
4. **관리자용 프롬프트** - 유사 사례 종합하여 구체적 답변 생성

## 📊 데이터 플로우

### 직원용 보고서 생성 (기존 기능 + 임베딩 추가)
```
POST /reports/weekly
→ 플랫폼 데이터 수집 → task별 그룹핑 → LLM 보고서 생성
→ DB 저장 + 자동 임베딩 처리 (관리자 검색용)
```

### 관리자 RAG 질의응답 🆕
```
POST /api/admin-query {"query": "OO프로젝트 트러블슈팅 사례"}
↓
1. 질의 텍스트 임베딩 (all-MiniLM-L6-v2)
2. 전체 report 테이블 유사도 검색 (cosine similarity ≥ 0.6)
3. Top-5 유사 보고서 추출 (담당자, 날짜, 내용 포함)
4. 관리자용 프롬프트로 LLM 종합 답변 생성
↓
결과: 과거 사례 기반 구체적이고 실용적인 답변
```

## 💻 코드 변경점 상세

### 1. user_timeline_api.py 추가된 함수들

#### 새로운 임포트
```python
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
```

#### 임베딩 모델 초기화
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

#### 새로운 데이터 모델
```python
class AdminQueryRequest(BaseModel):
    query: str  # "OO프로젝트 DB 성능 이슈 어떻게 처리됐는지 알려줘"
```

#### 핵심 함수들
```python
# 1. 임베딩 생성
async def create_embedding(text: str) -> List[float]:
    return embeddings.embed_query(text)

# 2. 유사 보고서 검색
async def search_similar_reports(query_text: str, session, top_k=5):
    # 전체 report 테이블 대상 벡터 검색 (cosine similarity ≥ 0.6)

# 3. 관리자용 RAG API
@app.post("/api/admin-query", response_model=ReportResponse)
async def admin_rag_query(request: AdminQueryRequest, session):
    # 질의 → 검색 → LLM 답변 생성

# 4. insert_report 함수 확장
# 보고서 저장 시 자동 임베딩 생성 및 저장
```

### 2. 관리자용 프롬프트 템플릿
```python
manager_rag_prompt = PromptTemplate.from_template("""
# 역할: 조직의 업무 히스토리를 잘 아는 관리자 어시스턴트
# 지시: 관리자 질문에 대해 과거 보고서들을 참고하여 구체적 답변
# 답변 지침:
- 구체적인 날짜, 담당자, 처리 과정 포함
- 유사한 사례가 여러 개 있다면 패턴 분석
- 정보가 부족하면 "추가 정보가 필요합니다" 명시
""")
```

## 🚀 테스트 가이드

### Step 1: 환경 설정
```bash
# 1. PostgreSQL 실행 확인
# 2. .env 파일 설정
DATABASE_URL="postgresql+asyncpg://postgres:1234@localhost:5432/weekly_report_db"
OPENAI_API_KEY="sk-your-actual-openai-api-key"

# 3. Python 의존성 설치
pip install langchain-huggingface sentence-transformers
```

### Step 2: DB 스키마 적용
```bash
# sample_data.sql로 기본 DB 구성 (이미 완료된 경우 스킵)
psql -U postgres -d weekly_report_db -f sample_data.sql

# 임베딩 컬럼 추가
psql -U postgres -d weekly_report_db -f add_embedding_column.sql
```

### Step 3: API 서버 실행
```bash
python user_timeline_api.py
# 서버 시작: http://localhost:8001
```

### Step 4: 기능 테스트
```bash
# 1. 서버 상태 확인
curl http://localhost:8001/health

# 2. 더미 보고서 생성 (임베딩 자동 생성됨)
curl -X POST http://localhost:8001/reports/weekly \
-H "Content-Type: application/json" \
-d '{
  "platform_ids": {"slack": [1,2], "notion": [1]},
  "start": "2025-09-22",
  "end": "2025-09-28",
  "writer": "테스터",
  "email": "test@skax.co.kr"
}'

# 3. 관리자 RAG 질의 테스트
curl -X POST http://localhost:8001/api/admin-query \
-H "Content-Type: application/json" \
-d '{"query": "SK하이닉스 프로젝트에서 발생한 문제들과 해결 과정을 알려줘"}'
```

### Step 5: 예상 결과
- 🟢 **성공**: 과거 보고서 기반으로 구체적이고 상세한 답변
- 🔵 **부분 성공**: 일부 유사 사례 발견하여 제한적 답변
- 🟡 **정보 부족**: "관련된 과거 보고서를 찾을 수 없습니다"

## ✨ 실용적 효과
1. **관리자 업무 효율성**: 과거 사례 즉시 검색으로 의사결정 속도 향상
2. **조직 지식 활용**: 축적된 보고서가 검색 가능한 지식베이스로 활용
3. **패턴 발견**: 반복되는 문제와 해결책의 패턴 자동 분석
4. **기존 시스템 호환**: 직원용 기능은 그대로 유지, 관리자만 추가 혜택

## 🔒 보안 및 성능
- **접근 제어**: 관리자 전용 API (필요시 인증 추가 가능)
- **검색 성능**: pgvector ivfflat 인덱스로 대용량 데이터에서도 빠른 검색
- **데이터 품질**: 임베딩 품질은 보고서 내용 품질에 비례