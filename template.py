# pip install -r pjt-requirements.txt
# pip install langchain-core langchain-community langchain-openai faiss-cpu pypdf python-multipart
# pjt-main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
import tempfile

# 환경변수 설정 또는 .env로 관리 권장
os.environ["OPENAI_API_KEY"] = ""

app = FastAPI(
    title="논문 업로드 및 벡터화 API",
    description="PDF 파일을 업로드하면 텍스트를 추출하고 FAISS 벡터 DB로 저장합니다.",
    version="1.0.0",
)

# 전역 벡터스토어 저장소 (데모용, 실제에선 외부 저장소 권장)
VECTORSTORE = None

def load_paper(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

# 논문 업로드 API 엔드포인트 설정
@app.post("/upload-paper", summary="논문 업로드 및 벡터화", tags=["PDF 처리"])
async def upload_pdf(file: UploadFile = File(...)):
    # 파일 유효성 검사
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 허용됩니다.")
    
    try:
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 1단계: 텍스트 추출
        paper_text = load_paper(tmp_path)

        # 2단계: 문서 분할
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([paper_text])

        # 3단계: 벡터 저장 (전역 변수에 저장)
        global VECTORSTORE
        VECTORSTORE = FAISS.from_documents(docs, OpenAIEmbeddings()) # 임베딩 처리

        return JSONResponse(content={
            "message": f"총 {len(docs)}개의 청크가 벡터로 저장되었습니다."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")


from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ✅ 요청 스키마
class PromptRequest(BaseModel):
    prompt: str
    top_k: int = 3  # 관련 문서 수, 기본값 3

# ✅ 응답 스키마
class AnswerResponse(BaseModel):
    prompt: str
    response: str

@app.post("/ask", response_model=AnswerResponse, summary="프롬프트 기반 질문 처리", tags=["질의 응답"])
async def ask_question(request: PromptRequest):
    global VECTORSTORE
    if VECTORSTORE is None:
        raise HTTPException(status_code=400, detail="벡터스토어가 초기화되지 않았습니다. 먼저 PDF를 업로드하세요.")
    
    try:
        # 관련 문서 검색
        related_docs = VECTORSTORE.similarity_search(request.prompt, k=request.top_k) # 요청과 유사한 top3를 가지고 와서 related_docs에 저장
        context = "\n\n".join([doc.page_content for doc in related_docs])
        
        # 프롬프트 생성 (할루시네이션을 방지하는 문구가 들어있음)
        full_prompt = f"""너는 논문 기반 연구 보조 AI이다. 논문에서 발췌한 내용에 대해서만 답변하고 마무리하라. 논문에서 발췌한 내용에서 답변이 없으면 벡터 DB에서 답변을 찾을 수 없다고 하라. 다음은 논문에서 발췌한 내용이다:\n\n{context}\n\n이제 아래 복합 질문에 단계적으로 답변하라:\n{request.prompt}"""

        # LLM 호출
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        response = llm.invoke([HumanMessage(content=full_prompt)]).content
        
        return AnswerResponse(prompt=request.prompt, response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"응답 생성 중 오류 발생: {str(e)}")
