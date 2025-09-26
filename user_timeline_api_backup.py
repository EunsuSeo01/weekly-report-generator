from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from typing import List, Dict, Any
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional

# ====================================
# 환경 변수 로드
# ====================================
dotenv.load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ====================================
# RAG용 임베딩 모델 초기화 (관리자용)
# ====================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
print(f"'{EMBEDDING_MODEL}' 임베딩 모델을 로드합니다...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL 환경 변수가 설정되지 않았습니다.")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# ====================================
# FastAPI app
# ====================================
app = FastAPI(title="User Timeline + Weekly Report Service", version="1.0")

# ====================================
# DB 연결
# ====================================
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession)

async def get_db_session():
    async with async_session() as session:
        yield session

# ====================================
# 데이터 모델
# ====================================
class TimelineActivity(BaseModel):
    source: str
    timestamp: str
    content: str
    metadata: Dict[str, Any]

class UserTimelineResponse(BaseModel):
    user_id: str
    start_date: str
    end_date: str
    activities: List[TimelineActivity]
    summary: Dict[str, Any]

class ReportRequest(BaseModel):
    task_id: int
    start_date: str
    end_date: str

class AdminQueryRequest(BaseModel):
    query: str  # "OO프로젝트 DB 성능 이슈 어떻게 처리됐는지 알려줘"

class ReportResponse(BaseModel):
    summary: str

class ReportIn(BaseModel):
    platform_ids: Dict[str, List[int]]
    start: str
    end: str
    writer: str
    email: str

# ====================================
# LLM
# ====================================
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=OPENAI_API_KEY)
output_parser = StrOutputParser()

# 관리자 RAG 질의응답 프롬프트
manager_rag_prompt = PromptTemplate.from_template("""
# 역할
당신은 조직의 업무 히스토리를 잘 아는 관리자 어시스턴트입니다.

# 지시
관리자의 질문에 대해 아래 제공된 과거 보고서들을 참고하여 정확하고 구체적인 답변을 해주세요.

# 관리자 질문
{user_query}

# 참고할 과거 보고서들
{similar_reports}

# 답변 지침
- 제공된 보고서의 내용만을 기반으로 답변하세요
- 구체적인 날짜, 담당자, 처리 과정을 포함하세요
- 유사한 사례가 여러 개 있다면 패턴을 분석해서 알려주세요
- 정보가 부족하면 "추가 정보가 필요합니다"라고 명시하세요

# 답변:
""")

# 기존 관리자 요약 보고서 프롬프트 (기존 기능 유지)
manager_prompt = PromptTemplate.from_template("""
# 역할
당신은 팀의 성과를 한눈에 파악해야 하는 유능한 팀장입니다.

# 지시
아래에 제공되는 task별 팀원들의 주간 보고서 내용을 바탕으로, 팀 전체의 관점에서 **핵심 성과, 발견된 문제점, 그리고 다음 주 공통 목표**를 요약하여 관리자용 보고서를 작성해 주세요.

# 팀원별 보고 내용
{team_reports}

# 관리자용 요약 보고서:
""")

manager_chain = manager_prompt | llm | output_parser
manager_rag_chain = manager_rag_prompt | llm | output_parser

# 주간 업무 보고서 프롬프트
REPORT_TEMPLATE = """
## 작성 지침
- 반드시 아래 제공된 context와 task_description만을 근거로 작성하세요.
- 제공되지 않은 사실은 추측하거나 임의로 작성하지 마세요.
- context가 부족하면 '자료 없음'이라고 명시하세요.

## 1) 주간 요약
이번 주 task {task_id} ({task_description}) 관련 진행 상황과 핵심 논의를 요약하세요.
{context}

## 2) 사람별 주요 산출물
다음은 참여자의 주요 산출물입니다:
{member_list}

## 3) 협업 내역
Slack/Notion/Outlook/OneDrive 기록을 바탕으로 어떤 사람들이 어떤 방식으로 협업했는지 구체적으로 정리하세요.

## 4) 리스크/이슈
대화와 회의록에서 드러난 문제점, 잠재 리스크, 해결 필요 사항을 정리하세요.

## 5) 차주 계획
다음 주에 진행해야 할 후속 작업과 개선점을 제시하세요.

(기간: {start} ~ {end})
"""

report_prompt = PromptTemplate(
    template=REPORT_TEMPLATE,
    input_variables=["context", "task_id", "task_description", "member_list", "start", "end"],
)

# ====================================
# 유틸 함수
# ====================================
async def get_task_description(task_id: int, session: AsyncSession) -> str:
    query = text("SELECT description FROM public.task WHERE id = :task_id")
    result = await session.execute(query, {"task_id": task_id})
    row = result.fetchone()
    return row[0] if row else "(설명 없음)"

async def create_embedding(text: str) -> List[float]:
    """텍스트를 768차원 벡터로 임베딩"""
    try:
        vector = embeddings.embed_query(text)
        return vector
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return None

async def search_similar_reports(query_text: str, session: AsyncSession, top_k: int = 5) -> str:
    """관리자 질의에 대한 유사 보고서 검색 (전체 report 테이블 대상)"""
    try:
        # 쿼리 임베딩 생성
        query_embedding = await create_embedding(query_text)
        if not query_embedding:
            return "임베딩 생성에 실패했습니다."

        # 벡터를 문자열로 변환 (pgvector 형식)
        vector_str = str(query_embedding)

        # 전체 report 테이블에서 유사도 검색
        query = text("""
            SELECT content, task_id, writer, "timestamp"::text as timestamp,
                   cosine_similarity(embedding, :query_vector) as similarity
            FROM public.report
            WHERE embedding IS NOT NULL
            AND cosine_similarity(embedding, :query_vector) >= 0.6
            ORDER BY cosine_similarity(embedding, :query_vector) DESC
            LIMIT :top_k
        """)

        result = await session.execute(query, {
            "query_vector": vector_str,
            "top_k": top_k
        })

        similar_reports = []
        for row in result.fetchall():
            row_dict = dict(row._mapping)
            similar_reports.append(
                f"[Task {row_dict['task_id']} | {row_dict['writer']} | {row_dict['timestamp'][:10]} | 유사도: {row_dict['similarity']:.2f}]\n"
                f"{row_dict['content'][:800]}...\n"
            )

        if similar_reports:
            return "\n" + "="*80 + "\n".join(similar_reports)
        else:
            return "관련된 과거 보고서를 찾을 수 없습니다."

    except Exception as e:
        print(f"유사 보고서 검색 오류: {e}")
        return "검색 중 오류가 발생했습니다."

async def insert_report(task_id: int, writer: str, email: str, content: str, session: AsyncSession):
    now = datetime.utcnow()

    # 보고서 저장
    query = text("""
        INSERT INTO public.report (task_id, "timestamp", writer, email, content)
        VALUES (:task_id, :timestamp, :writer, :email, :content)
        RETURNING id
    """)
    result = await session.execute(query, {
        "task_id": task_id,
        "timestamp": now,
        "writer": writer,
        "email": email,
        "content": content
    })
    report_id = result.fetchone()[0]

    # 🚀 NEW: 임베딩 생성 및 저장 (관리자 검색용)
    embedding_vector = await create_embedding(content)
    if embedding_vector:
        update_query = text("""
            UPDATE public.report
            SET embedding = :embedding
            WHERE id = :report_id
        """)
        await session.execute(update_query, {
            "embedding": str(embedding_vector),
            "report_id": report_id
        })
        print(f"📝 보고서 {report_id} 임베딩 처리 완료")

    await session.commit()

async def generate_report_for_task(task_id: int, platform_data: List[Dict[str, Any]], start_ts: str, end_ts: str, session: AsyncSession) -> str:
    docs = [Document(page_content=d.get("content", "")) for d in platform_data]
    actors = {d.get("actor") for d in platform_data if d.get("actor")}
    actor_list = "- " + "\n- ".join(actors) if actors else "- (none)"
    task_description = await get_task_description(task_id, session)

    context = "\n".join([doc.page_content for doc in docs])
    chain = report_prompt | llm | output_parser
    body = await chain.ainvoke({
        "context": context,
        "task_id": task_id,
        "task_description": task_description,
        "member_list": actor_list,
        "start": start_ts,
        "end": end_ts,
    })

    return f"# 업무 {task_id}: {task_description} 주간 보고서\n\n{body}"

# ====================================
# API 엔드포인트
# ====================================
@app.get("/api/user-timeline/{user_id}", response_model=UserTimelineResponse)
async def get_user_timeline(
    user_id: str,
    start_date: str,
    end_date: str,
    session: AsyncSession = Depends(get_db_session)
):
    # 문자열 → date 변환
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()

    query = text("""
        SELECT id, content, sender, receiver, task_id, "timestamp"::text as timestamp
        FROM public.slack
        WHERE (sender = :user_id OR receiver = :user_id)
          AND DATE("timestamp") BETWEEN :start_date AND :end_date
        ORDER BY "timestamp" DESC
    """)
    result = await session.execute(
        query,
        {"user_id": user_id, "start_date": start_date_obj, "end_date": end_date_obj}
    )

    activities = []
    for row in result.fetchall():
        row_dict = dict(row._mapping)
        activities.append(TimelineActivity(
            source="slack",
            timestamp=row_dict["timestamp"],
            content=row_dict["content"],
            metadata={
                "sender": row_dict["sender"],
                "receiver": row_dict["receiver"],
                "task_id": row_dict["task_id"],
                "slack_id": row_dict["id"]
            }
        ))

    return UserTimelineResponse(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        activities=activities,
        summary={"total_count": len(activities), "slack_count": len(activities)}
    )

@app.get("/api/user-summary/{user_id}")
async def get_user_summary(
    user_id: str,
    start_date: str,
    end_date: str,
    session: AsyncSession = Depends(get_db_session)
):
    # 문자열 → date 변환
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()

    query = text("""
        SELECT COUNT(*) FROM public.slack
        WHERE (sender = :user_id OR receiver = :user_id)
          AND DATE("timestamp") BETWEEN :start_date AND :end_date
    """)
    result = await session.execute(
        query,
        {"user_id": user_id, "start_date": start_date_obj, "end_date": end_date_obj}
    )
    count = result.scalar()
    return {"user_id": user_id, "total_count": count}


@app.get("/api/users")
async def get_available_users(session: AsyncSession = Depends(get_db_session)):
    query = text("SELECT DISTINCT name FROM public.employee ORDER BY name")
    result = await session.execute(query)
    users = [row[0] for row in result.fetchall()]
    return {"users": users, "count": len(users)}

@app.get("/api/db-health")
async def db_health(session: AsyncSession = Depends(get_db_session)):
    try:
        result = await session.execute(text("SELECT 1"))
        return {"database_status": "connected", "result": result.scalar()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 🚀 NEW: 관리자 RAG 기반 질의응답 API
@app.post("/api/admin-query", response_model=ReportResponse)
async def admin_rag_query(request: AdminQueryRequest, session: AsyncSession = Depends(get_db_session)):
    """관리자의 자유 텍스트 질의에 대한 RAG 기반 답변"""
    try:
        # 유사 보고서 검색
        similar_reports = await search_similar_reports(request.query, session)

        # RAG 기반 답변 생성
        rag_response = await manager_rag_chain.ainvoke({
            "user_query": request.query,
            "similar_reports": similar_reports
        })

        return ReportResponse(summary=rag_response)
    except Exception as e:
        return ReportResponse(summary=f"오류가 발생했습니다: {str(e)}")

# 기존 관리자 요약 API (기존 기능 유지)
@app.post("/api/generate-summary", response_model=ReportResponse)
async def generate_summary(request: ReportRequest):
    dummy_reports = f"Task {request.task_id} 보고서 (기간 {request.start_date}~{request.end_date})"
    manager_summary = await manager_chain.ainvoke({"team_reports": dummy_reports})
    return ReportResponse(summary=manager_summary)

@app.post("/reports/weekly")
async def make_weekly_report(p: ReportIn, session: AsyncSession = Depends(get_db_session)):
    reports = []

    # 1. 모든 플랫폼 데이터 수집
    all_platform_data = []
    for platform, ids in p.platform_ids.items():
        if not ids:
            continue

        query = None
        if platform == "slack":
            query = text("SELECT id, content, sender AS actor, receiver, task_id, \"timestamp\"::text as ts FROM public.slack WHERE id = ANY(:ids)")
        elif platform == "notion":
            query = text("SELECT id, content, NULL as actor, task_id, \"timestamp\"::text as ts FROM public.notion WHERE id = ANY(:ids)")
        elif platform == "outlook":
            query = text("SELECT id, content, sender AS actor, receiver, task_id, \"timestamp\"::text as ts FROM public.outlook WHERE id = ANY(:ids)")
        elif platform == "onedrive":
            query = text("SELECT id, content, writer AS actor, task_id, \"timestamp\"::text as ts FROM public.onedrive WHERE id = ANY(:ids)")

        if query is not None:
            result = await session.execute(query, {"ids": ids})
            rows = [dict(r._mapping) for r in result.fetchall()]
            all_platform_data.extend(rows)

    # 2. task_id별로 그룹핑
    grouped = {}
    for d in all_platform_data:
        task_id = d.get("task_id")
        if not task_id:
            continue
        grouped.setdefault(task_id, []).append(d)

    # 3. 보고서 생성
    for task_id, items in grouped.items():
        task_id_int = int(task_id)   # <-- 여기서 변환
        report_md = await generate_report_for_task(task_id_int, items, p.start, p.end, session)
        await insert_report(task_id_int, p.writer, p.email, report_md, session)
        reports.append({"task_id": task_id_int, "report": report_md})

    return {
        "platform_ids": p.platform_ids,
        "range": {"start": p.start, "end": p.end},
        "reports": reports
    }


# ====================================
# 실행
# ====================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
