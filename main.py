import os
from langchain_community.document_loaders import (
    SlackDirectoryLoader,
    NotionDirectoryLoader,
    OneDriveLoader,
    OutlookMessageLoader, # GoogleDriveLoader 대신 추가
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Slack 설정
SLACK_BOT_TOKEN = "[YOUR_SLACK_BOT_TOKEN]"
SLACK_CHANNEL_IDS = ["[YOUR_CHANNEL_ID_1]"]

# 2. Notion 설정
NOTION_INTEGRATION_TOKEN = "[YOUR_NOTION_INTEGRATION_TOKEN]"
NOTION_PAGE_IDS = ["[YOUR_PAGE_ID_1]"]

# 3. Microsoft 365 설정 (OneDrive & Outlook 공통)
# Azure Portal에서 발급받은 ID를 입력하세요.
MS_CLIENT_ID = "[YOUR_MS_APP_CLIENT_ID]"
MS_TENANT_ID = "[YOUR_MS_APP_TENANT_ID]"

ONEDRIVE_FOLDER_PATH = "[YOUR_ONEDRIVE_FOLDER_PATH]" # 예: "/drive/root:/Documents/Work"
OUTLOOK_FOLDER_NAME = "[YOUR_OUTLOOK_FOLDER_NAME]" # 예: "Inbox"

# 4. 임베딩 모델 (이전과 동일)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# --- 코드 본문 ---

def load_all_documents():
    """각 소스에서 문서를 로드하고 하나의 리스트로 합칩니다."""
    all_docs = []
    
    # 1. Slack 로더 (이전과 동일)
    try:
        if SLACK_BOT_TOKEN.startswith("[YOUR_"):
            print("⚠️ Slack 토큰이 설정되지 않았습니다. Slack 로딩을 건너뜁니다.")
        else:
            print("🔄 Slack에서 문서를 가져옵니다...")
            loader = SlackDirectoryLoader(token=SLACK_BOT_TOKEN, channel_ids=SLACK_CHANNEL_IDS)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"✅ Slack에서 {len(docs)}개의 문서를 가져왔습니다.")
    except Exception as e:
        print(f"❌ Slack 로딩 중 오류 발생: {e}")

    # 2. Notion 로더 (이전과 동일)
    try:
        if NOTION_INTEGRATION_TOKEN.startswith("[YOUR_"):
            print("⚠️ Notion 토큰이 설정되지 않았습니다. Notion 로딩을 건너뜁니다.")
        else:
            print("🔄 Notion에서 문서를 가져옵니다...")
            loader = NotionDirectoryLoader(integration_token=NOTION_INTEGRATION_TOKEN, page_ids=NOTION_PAGE_IDS)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"✅ Notion에서 {len(docs)}개의 문서를 가져왔습니다.")
    except Exception as e:
        print(f"❌ Notion 로딩 중 오류 발생: {e}")

    # 3. OneDrive 로더 (Google Drive 대체)
    try:
        if MS_CLIENT_ID.startswith("[YOUR_"):
            print("⚠️ Microsoft Client ID가 설정되지 않았습니다. OneDrive 로딩을 건너뜁니다.")
        else:
            print("🔄 OneDrive에서 문서를 가져옵니다...")
            loader = OneDriveLoader(
                client_id=MS_CLIENT_ID,
                tenant_id=MS_TENANT_ID,
                folder_path=ONEDRIVE_FOLDER_PATH,
                # 최초 실행 시 브라우저를 통해 MS 계정 인증이 필요합니다.
            )
            docs = loader.load()
            all_docs.extend(docs)
            print(f"✅ OneDrive에서 {len(docs)}개의 문서를 가져왔습니다.")
    except Exception as e:
        print(f"❌ OneDrive 로딩 중 오류 발생: {e}")

    # 4. Outlook 로더 (신규 추가)
    try:
        if MS_CLIENT_ID.startswith("[YOUR_"):
            print("⚠️ Microsoft Client ID가 설정되지 않았습니다. Outlook 로딩을 건너뜁니다.")
        else:
            print("🔄 Outlook에서 메일을 가져옵니다...")
            loader = OutlookMessageLoader(
                client_id=MS_CLIENT_ID,
                tenant_id=MS_TENANT_ID,
                folder_name=OUTLOOK_FOLDER_NAME,
                # OneDrive와 동일한 인증을 사용합니다.
            )
            docs = loader.load()
            all_docs.extend(docs)
            print(f"✅ Outlook에서 {len(docs)}개의 메일을 가져왔습니다.")
    except Exception as e:
        print(f"❌ Outlook 로딩 중 오류 발생: {e}")
        
    return all_docs

# --- 아래 부분은 이전 코드와 완전히 동일합니다 ---

def setup_knowledge_base(docs):
    """문서 리스트를 기반으로 벡터 DB를 생성합니다."""
    if not docs:
        print("검색할 문서가 없습니다. API 설정 및 권한을 확인해주세요.")
        return None
    print(f"\n총 {len(docs)}개의 문서를 기반으로 DB 생성을 시작합니다.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    print(f"문서를 총 {len(split_docs)}개의 조각(chunk)으로 분할했습니다.")
    print(f"'{EMBEDDING_MODEL}' 임베딩 모델을 로드합니다...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
    )
    print("문서 조각을 벡터로 변환하고 데이터베이스를 생성합니다...")
    vector_db = FAISS.from_documents(split_docs, embeddings)
    print("✅ 데이터베이스 생성이 완료되었습니다.")
    return vector_db

def search_documents(db, query, k=3):
    """벡터 DB에서 질문과 유사한 문서를 검색합니다."""
    if db is None:
        print("데이터베이스가 준비되지 않았습니다.")
        return
    print(f"\n[질문] {query}")
    print("-" * 30)
    results = db.similarity_search_with_score(query, k=k)
    if not results:
        print("관련된 문서를 찾을 수 없습니다.")
        return
    print(f"[답변] 가장 관련성 높은 문서 {len(results)}개를 찾았습니다.\n")
    for i, (doc, score) in enumerate(results):
        source_path = doc.metadata.get('source', '출처 불명')
        content = doc.page_content
        print(f"문서 #{i+1} (유사도: {score:.4f}):")
        print(f"  - 출처: {source_path}")
        print(f"  - 내용: \"{content.strip()}\"")
        print("-" * 20)

if __name__ == "__main__":
    all_documents = load_all_documents()
    db = setup_knowledge_base(all_documents)
    if db:
        search_documents(db, "프로젝트 관련해서 지난 주에 공유된 메일 찾아줘")
        search_documents(db, "온보딩 관련 자료는 어디에 있어?")