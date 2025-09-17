from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def build_vector_db(docs: list[Document]) -> Optional[FAISS]:
    """문서 리스트를 기반으로 FAISS 벡터 DB를 생성합니다."""

    if not docs:
        print("검색할 문서가 없습니다. API 설정 및 권한을 확인해주세요.")
        return None
    
    print(f"\n총 {len(docs)}개의 문서를 기반으로 DB 생성을 시작합니다.")

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        print(f"문서를 총 {len(split_docs)}개의 조각(chunk)으로 분할했습니다.")

        # 한국어 문장의 의미를 훨씬 잘 이해하는 모델입니다.
        EMBEDDING_MODEL = "jhgan/ko-sbert-nli"
        print(f"'{EMBEDDING_MODEL}' 임베딩 모델을 로드합니다...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
        )

        print("문서 조각을 벡터로 변환하고 데이터베이스를 생성합니다...")
        vector_db = FAISS.from_documents(split_docs, embeddings)

        print("✅ 데이터베이스 생성이 완료되었습니다.")
        return vector_db

    except Exception as e:
        print(f"벡터 DB 생성 중 오류 발생: {e}")
        return None
