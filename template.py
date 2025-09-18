# pip install -r pjt-requirements.txt
# pip install langchain-core langchain-community langchain-openai faiss-cpu pypdf python-multipart
# pjt-main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
import build_vector_db
from build_vector_db import build_vector_db
from datetime import date
from dotenv import load_dotenv
load_dotenv()

# OpenAI API 키 확인
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

# 환경변수 설정 또는 .env로 관리 권장
os.environ["OPENAI_API_KEY"] = openai_api_key

app = FastAPI(
    title="논문 업로드 및 벡터화 API",
    description="PDF 파일을 업로드하면 텍스트를 추출하고 FAISS 벡터 DB로 저장합니다.",
    version="1.0.0",
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.schema import Document

# ✅ 요청 스키마
class PromptRequest(BaseModel):
    user_request: str

# ✅ 응답 스키마
class AnswerResponse(BaseModel):
    report: str

sample_docs = [
    # 1. 슬랙 대화 내용 예시
    Document(
        page_content="로그인 API 성능 개선 작업 완료했습니다. 기존 500ms에서 150ms로 응답 시간 단축했고, 오늘 오후에 배포 예정입니다.",
        metadata={"source": "Slack #backend-dev 채널"}
    ),
    # 2. 노션 회의록 예시
    Document(
        page_content="주간 기획 회의록입니다. 신규 '스마트 리포트' 기능의 MVP 범위를 확정했습니다. 주요 기능은 자동 데이터 연동과 템플릿 기반 리포트 생성입니다. UI/UX 디자인은 다음 주까지 초안을 공유하기로 했습니다.",
        metadata={"source": "Notion - 2025년 9월 3주차 기획 회의"}
    ),
    # 3. 또 다른 슬랙 대화 예시
    Document(
        page_content="CS팀에서 전달된 '데이터 다운로드 오류' 버그 재현 및 원인 파악 완료. 핫픽스 준비 중이며, 내일 오전 중으로 해결 가능할 것 같습니다.",
        metadata={"source": "Slack #cs-support 채널"}
    )
]

@app.post("/generate-report", response_model=AnswerResponse, summary="프롬프트 기반 질문 처리", tags=["질의 응답"])
async def generate_report(request: PromptRequest):
    if request is None:
        raise HTTPException(status_code=400, detail="인풋이 없습니다.")
    
    try:
        context = "\n\n".join([doc.page_content for doc in sample_docs])
        global VECTORSTORE
        VECTORSTORE = build_vector_db(docs=sample_docs)
        if VECTORSTORE is None:
            raise HTTPException(status_code=500, detail="벡터 DB가 준비되지 않았습니다.")
        
        # 관련 문서 검색
        related_docs = VECTORSTORE.similarity_search("이번 주에 진행한 업무 보고", k=3) # 요청과 유사한 top3를 가지고 와서 related_docs에 저장

        print(VECTORSTORE)
        print(related_docs)
        
        # 프롬프트 생성
        full_prompt = f"""
            # [역할 부여]
            당신은 유능한 PM(Project Manager)이자, 여러 업무 기록을 종합하여 핵심을 파악하는 성과 분석 전문가입니다.

            # [작업 지시]
            주어지는 '[이번 주 업무 기록]'만을 바탕으로, 동료와 상급자에게 공유할 주간업무 보고서 초안을 작성해 주세요.

            # [주간업무 보고서 예시 제공]
            # 주간 업무보고서 (예시)

            보고자: OOO  
            부서: AI 서비스 개발팀  
            기간: 2025년 9월 8일 ~ 2025년 9월 12일

            ---

            ## 1. 금주 주요 업무 진행 사항
            - **AI 모델 개발**
            - 고객 질의 데이터셋 기반 Intent Classification 모델 초기 버전 학습 완료
            - F1 Score 0.82 달성 (목표치 0.85 대비 96% 수준)
            - 에러 케이스 분석을 통해 오분류 유형(동의어 처리, 맥락 부족) 확인

            - **데이터 파이프라인**
            - Slack/Notion API 연동 완료 → 내부 문서 자동 수집 파이프라인 구축
            - 수집 데이터 총 5,200건, 중복 제거 및 클리닝 작업 진행

            - **서비스 연동**
            - 내부 챗봇 MVP 환경에서 검색+응답 모듈 통합 테스트 수행
            - 주요 시나리오(FAQ 검색, 정책 문서 검색)에서 정상 응답률 87% 확인

            ---

            ## 2. 협업 및 커뮤니케이션
            - **기획팀**: MVP 사용자 시나리오 검증 및 요구사항 피드백 수령
            - **인프라팀**: Kubernetes 환경에서 모델 배포 컨테이너화 작업 협업
            - **QA팀**: 데이터 라벨링 가이드라인 조율 및 품질 점검 회의 참여

            ---

            ## 3. 차주 계획 (2025년 9월 15일 ~ 9월 19일)
            - 모델 성능 개선 (F1 Score 0.85 이상 달성 목표)
            - 데이터 증강 및 동의어 사전 구축
            - Hard Negative Mining 기법 적용
            - Redis 기반 캐싱 레이어 적용하여 API 호출 효율화
            - 챗봇 UX 개선 → 응답 속도 1.5초 이하 목표
            - 내부 데모 준비 (9월 19일 금요일 예정)

            ---

            ## 4. 기타
            - 팀 내 기술 세미나(9/17): "RAG 기반 기업 문서 검색 시스템 사례" 발표 예정
            - 신규 인턴 온보딩 지원 및 데이터 클리닝 교육 진행

            ---

            # [이번 주 업무 기록]
            \n\n{context}\n\n
        """

        # LLM 호출
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        response = llm.invoke([HumanMessage(content=full_prompt)]).content
        
        return AnswerResponse(report=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"응답 생성 중 오류 발생: {str(e)}")
