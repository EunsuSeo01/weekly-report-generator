"""
user_timeline_api_test.py 파일로 테스트가 안돼서 만든 파일

가벼운 RAG 테스트 스크립트
단계별로 임베딩 모델 로드 -> 검색 테스트

"""

import asyncio
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from typing import List, Optional
import dotenv

# 환경 변수 로드
dotenv.load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL 환경 변수가 설정되지 않았습니다.")

# DB 연결
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession)

# 전역 변수로 임베딩 모델 저장
embeddings = None

async def test_light_embedding():
    """가벼운 임베딩 모델 테스트"""
    global embeddings

    print("\n🔵 Step 1: 가벼운 임베딩 모델 로드 테스트")

    try:
        # 가장 가벼운 모델부터 시도
        print("📦 all-MiniLM-L6-v2 모델 로드 시도... (384차원)")

        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # 384차원 모델
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 간단한 임베딩 테스트
        test_text = "테스트 문장입니다."
        test_vector = embeddings.embed_query(test_text)

        print(f"✅ 임베딩 모델 로드 성공!")
        print(f"📊 테스트 임베딩 차원: {len(test_vector)}")
        print(f"📄 테스트 벡터 샘플: {test_vector[:5]}...")

        return True

    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        print("💡 sentence-transformers 설치 필요: pip install sentence-transformers")
        return False

async def create_embedding(text: str) -> Optional[List[float]]:
    """텍스트를 임베딩으로 변환"""
    global embeddings

    if embeddings is None:
        print("❌ 임베딩 모델이 로드되지 않았습니다.")
        return None

    try:
        vector = embeddings.embed_query(text)
        return vector
    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {e}")
        return None

async def test_sample_embeddings():
    """샘플 텍스트들로 임베딩 테스트"""
    print("\n🔵 Step 2: 샘플 임베딩 생성 테스트")

    sample_texts = [
        "로그인 API 성능 개선 작업을 완료했습니다.",
        "데이터베이스 쿼리 최적화를 진행했습니다.",
        "사용자 인터페이스 버그를 수정했습니다.",
    ]

    for i, text in enumerate(sample_texts, 1):
        print(f"📝 샘플 {i}: {text}")

        vector = await create_embedding(text)
        if vector:
            print(f"✅ 임베딩 성공 (차원: {len(vector)})")
        else:
            print(f"❌ 임베딩 실패")
        print()

async def test_basic_similarity():
    """기본적인 유사도 계산 테스트"""
    print("\n🔵 Step 3: 유사도 계산 테스트")

    try:
        import numpy as np

        text1 = "API 성능 개선"
        text2 = "성능 최적화 작업"
        text3 = "사용자 인터페이스 디자인"

        vec1 = await create_embedding(text1)
        vec2 = await create_embedding(text2)
        vec3 = await create_embedding(text3)

        if vec1 and vec2 and vec3:
            # 코사인 유사도 계산
            def cosine_similarity(v1, v2):
                v1_arr = np.array(v1)
                v2_arr = np.array(v2)
                return np.dot(v1_arr, v2_arr) / (np.linalg.norm(v1_arr) * np.linalg.norm(v2_arr))

            sim_1_2 = cosine_similarity(vec1, vec2)
            sim_1_3 = cosine_similarity(vec1, vec3)

            print(f"📊 '{text1}' vs '{text2}': {sim_1_2:.3f}")
            print(f"📊 '{text1}' vs '{text3}': {sim_1_3:.3f}")
            print(f"✅ 유사도 계산 성공! (높을수록 유사함)")

    except ImportError:
        print("💡 numpy 설치 필요: pip install numpy")
    except Exception as e:
        print(f"❌ 유사도 계산 실패: {e}")

async def test_db_embedding_insert():
    """DB에 임베딩 저장 테스트"""
    print("\n🔵 Step 4: DB 임베딩 저장 테스트")

    async with async_session() as session:
        try:
            # 테스트용 더미 보고서 생성
            test_content = "테스트 보고서입니다. API 성능 개선 작업을 완료했습니다."
            test_embedding = await create_embedding(test_content)

            if test_embedding:
                # 보고서 저장
                insert_query = text("""
                    INSERT INTO public.report (task_id, "timestamp", writer, email, content, embedding)
                    VALUES (:task_id, NOW(), :writer, :email, :content, :embedding)
                    RETURNING id
                """)

                result = await session.execute(insert_query, {
                    "task_id": 999,  # 테스트용 task_id
                    "writer": "테스터",
                    "email": "test@example.com",
                    "content": test_content,
                    "embedding": str(test_embedding)  # pgvector 형식
                })

                report_id = result.fetchone()[0]
                await session.commit()

                print(f"✅ 테스트 보고서 저장 성공! ID: {report_id}")
                print(f"📄 내용: {test_content}")
                print(f"📊 임베딩 차원: {len(test_embedding)}")

        except Exception as e:
            print(f"❌ DB 임베딩 저장 실패: {e}")
            await session.rollback()

async def test_similarity_search():
    """유사도 기반 검색 테스트"""
    print("\n🔵 Step 5: 유사도 검색 테스트")

    async with async_session() as session:
        try:
            # 검색 쿼리
            search_query = "성능 최적화 작업"
            query_embedding = await create_embedding(search_query)

            if not query_embedding:
                print("❌ 검색 쿼리 임베딩 실패")
                return

            # 유사도 검색 (pgvector 사용)
            search_sql = text("""
                SELECT id, writer, content,
                       cosine_similarity(embedding, :query_vector) as similarity
                FROM public.report
                WHERE embedding IS NOT NULL
                ORDER BY cosine_similarity(embedding, :query_vector) DESC
                LIMIT 5
            """)

            result = await session.execute(search_sql, {
                "query_vector": str(query_embedding)
            })

            print(f"🔍 검색 쿼리: '{search_query}'")
            print("📋 검색 결과:")

            found_results = False
            for row in result.fetchall():
                found_results = True
                row_dict = dict(row._mapping)
                print(f"  ID {row_dict['id']} | {row_dict['writer']} | 유사도: {row_dict['similarity']:.3f}")
                print(f"  내용: {row_dict['content'][:100]}...")
                print()

            if not found_results:
                print("❌ 임베딩이 있는 보고서가 없습니다.")
                print("💡 먼저 add_embedding_column.sql을 실행하고 보고서를 생성해주세요.")

        except Exception as e:
            print(f"❌ 유사도 검색 실패: {e}")

async def main():
    """전체 테스트 실행"""
    print("🚀 가벼운 RAG 테스트 시작")
    print("=" * 60)

    # Step별 테스트 실행
    if await test_light_embedding():
        await test_sample_embeddings()
        await test_basic_similarity()
        await test_db_embedding_insert()
        await test_similarity_search()
    else:
        print("💡 임베딩 모델 로드 실패. 다음 명령어로 패키지를 설치해주세요:")
        print("   pip install sentence-transformers numpy")

    print("=" * 60)
    print("✅ 가벼운 RAG 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main())