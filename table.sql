ALTER TABLE public.task
    DROP COLUMN task_id;
-- 텍스트 컬럼
ALTER TABLE public.task
    ADD COLUMN description text;

-- 임베딩 컬럼 (pgvector 1536차원 예시)
ALTER TABLE public.task
    ADD COLUMN embedding vector(1536);

ALTER TABLE public.employee
    ADD COLUMN email VARCHAR(100) UNIQUE NOT NULL,
    ADD COLUMN password VARCHAR(255) NOT NULL;

-- Notion 테이블에 embedding 추가
ALTER TABLE public.notion
    ADD COLUMN embedding vector(768);

-- OneDrive 테이블에 embedding 추가
ALTER TABLE public.onedrive
    ADD COLUMN embedding vector(768);

-- Outlook 테이블에 embedding 추가
ALTER TABLE public.outlook
    ADD COLUMN embedding vector(768);

-- Slack 테이블에 embedding 추가
ALTER TABLE public.slack
    ADD COLUMN embedding vector(768);

-- Notion
CREATE INDEX notion_embedding_idx
    ON public.notion
        USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 5);

-- OneDrive
CREATE INDEX onedrive_embedding_idx
    ON public.onedrive
        USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 5);

-- Outlook
CREATE INDEX outlook_embedding_idx
    ON public.outlook
        USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 5);

-- Slack
CREATE INDEX slack_embedding_idx
    ON public.slack
        USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 5);

DROP TABLE IF EXISTS public.description CASCADE;
