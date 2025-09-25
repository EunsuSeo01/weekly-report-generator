"""
user_timeline_api_test.py 파일로 테스트가 안돼서 만든 파일

DB 연결 테스트 전용 스크립트
임베딩 모델 없이 DB 기본 기능만 테스트
"""

import asyncio
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import dotenv

# 환경 변수 로드
dotenv.load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

print(f"📊 DATABASE_URL: {DATABASE_URL}")

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL 환경 변수가 설정되지 않았습니다.")

# DB 연결
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession)

async def test_basic_db():
    """기본 DB 연결 테스트"""
    print("\n🔵 Step 1: 기본 DB 연결 테스트")

    try:
        async with async_session() as session:
            result = await session.execute(text("SELECT 1 as test"))
            value = result.scalar()
            print(f"✅ DB 연결 성공! 테스트 값: {value}")
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return False

    return True

async def test_tables():
    """테이블 존재 확인"""
    print("\n🔵 Step 2: 테이블 존재 확인")

    tables_to_check = ["employee", "task", "report", "slack", "notion", "onedrive", "outlook"]

    async with async_session() as session:
        for table in tables_to_check:
            try:
                result = await session.execute(text(f"SELECT COUNT(*) FROM public.{table}"))
                count = result.scalar()
                print(f"✅ {table} 테이블: {count}개 레코드")
            except Exception as e:
                print(f"❌ {table} 테이블 오류: {e}")

async def test_report_table():
    """report 테이블 상세 확인"""
    print("\n🔵 Step 3: report 테이블 상세 확인")

    async with async_session() as session:
        try:
            # 컬럼 구조 확인
            result = await session.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'report' AND table_schema = 'public'
                ORDER BY ordinal_position
            """))

            print("📋 report 테이블 컬럼 구조:")
            for row in result.fetchall():
                print(f"  - {row[0]}: {row[1]}")

            # 샘플 데이터 확인
            result = await session.execute(text("""
                SELECT id, task_id, writer,
                       SUBSTRING(content, 1, 100) as content_preview,
                       embedding IS NOT NULL as has_embedding
                FROM public.report
                ORDER BY id
                LIMIT 5
            """))

            print("\n📄 report 테이블 샘플 데이터:")
            for row in result.fetchall():
                row_dict = dict(row._mapping)
                print(f"  ID {row_dict['id']}: Task {row_dict['task_id']} | {row_dict['writer']} | Embedding: {row_dict['has_embedding']}")
                print(f"    내용: {row_dict['content_preview']}...")

        except Exception as e:
            print(f"❌ report 테이블 확인 실패: {e}")

async def test_embedding_column():
    """embedding 컬럼 존재 확인"""
    print("\n🔵 Step 4: embedding 컬럼 존재 확인")

    async with async_session() as session:
        try:
            result = await session.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'report'
                AND table_schema = 'public'
                AND column_name = 'embedding'
            """))

            if result.fetchone():
                print("✅ embedding 컬럼이 존재합니다")

                # embedding 데이터 확인
                result = await session.execute(text("""
                    SELECT COUNT(*) as total,
                           COUNT(embedding) as with_embedding
                    FROM public.report
                """))
                row = result.fetchone()
                print(f"📊 전체 보고서: {row[0]}개, 임베딩 있음: {row[1]}개")
            else:
                print("❌ embedding 컬럼이 없습니다. add_embedding_column.sql을 실행해주세요.")

        except Exception as e:
            print(f"❌ embedding 컬럼 확인 실패: {e}")

async def main():
    """전체 테스트 실행"""
    print("🚀 DB 연결 테스트 시작")
    print("=" * 50)

    # Step별 테스트 실행
    if await test_basic_db():
        await test_tables()
        await test_report_table()
        await test_embedding_column()

    print("=" * 50)
    print("✅ DB 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main())