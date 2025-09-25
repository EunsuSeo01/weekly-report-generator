--
-- RAG 기능을 위한 report 테이블 스키마 확장
-- 실행 방법: psql -U postgres -d weekly_report_db -f add_embedding_column.sql

-- 1. report 테이블에 embedding 컬럼 추가 (384차원)
ALTER TABLE public.report
ADD COLUMN IF NOT EXISTS embedding vector(384);

-- 2. 벡터 유사도 검색을 위한 인덱스 생성
CREATE INDEX IF NOT EXISTS report_embedding_idx
ON public.report
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 5);

-- 3. 확인용 쿼리
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'report'
AND table_schema = 'public';

-- 4. 인덱스 확인
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'report'
AND schemaname = 'public';