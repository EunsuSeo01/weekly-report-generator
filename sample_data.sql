--
-- PostgreSQL database dump
--

\restrict 1abCLlIIK2cJV8kuve9W1GBvhQ1CTbWD8mlvLHOmTdspP4UaoDf84nbwm02lWJx

-- Dumped from database version 17.6 (Homebrew)
-- Dumped by pg_dump version 17.6 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: project; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA project;


ALTER SCHEMA project OWNER TO postgres;

--
-- Name: pgaudit; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgaudit WITH SCHEMA public;


--
-- Name: EXTENSION pgaudit; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pgaudit IS 'provides auditing functionality';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: cosine_similarity(public.vector, public.vector); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cosine_similarity(v1 public.vector, v2 public.vector) RETURNS double precision
    LANGUAGE plpgsql IMMUTABLE
    AS $$
BEGIN
  -- pgvector의 <=> 는 코사인 거리(cosine distance)를 반환하므로
  -- 1 - 거리 = 코사인 유사도
  RETURN 1 - (v1 <=> v2);
END;
$$;


ALTER FUNCTION public.cosine_similarity(v1 public.vector, v2 public.vector) OWNER TO postgres;

--
-- Name: get_exec_time(text); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.get_exec_time(query text) RETURNS double precision
    LANGUAGE plpgsql
    AS $$
DECLARE
    plan JSON;
    exec_time FLOAT;
BEGIN
    EXECUTE format('EXPLAIN (ANALYZE, FORMAT JSON) %s', query) INTO plan;
    exec_time := (plan->0->>'Execution Time')::FLOAT;
    RETURN exec_time;
END;
$$;


ALTER FUNCTION public.get_exec_time(query text) OWNER TO postgres;

--
-- Name: normalize_vector(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.normalize_vector() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
  norm FLOAT;
  zero_vec VECTOR(384);
BEGIN
  -- 384차원 영벡터 생성
  zero_vec := array_fill(0.0::float, ARRAY[384])::vector;

  -- L2 Norm 계산: 벡터와 영벡터 간의 L2 거리
  norm := NEW.embedding_vector <-> zero_vec;

  -- norm이 양수일 때만 정규화
  IF norm > 0 THEN
    NEW.embedding_vector := NEW.embedding_vector / norm;
  END IF;

  RETURN NEW;
END;
$$;


ALTER FUNCTION public.normalize_vector() OWNER TO postgres;

--
-- Name: random_vector(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.random_vector() RETURNS public.vector
    LANGUAGE sql
    AS $$
    SELECT array_agg(random())::vector(384)
    FROM generate_series(1,384);
$$;


ALTER FUNCTION public.random_vector() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: administer; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.administer (
    id integer NOT NULL,
    name character varying(50) NOT NULL,
    email character varying(100) NOT NULL,
    password character varying(255) NOT NULL,
    job_grade character varying(50) NOT NULL,
    task_involved integer[]
);


ALTER TABLE public.administer OWNER TO postgres;

--
-- Name: administer_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.administer_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.administer_id_seq OWNER TO postgres;

--
-- Name: administer_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.administer_id_seq OWNED BY public.administer.id;


--
-- Name: employee; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.employee (
    id integer NOT NULL,
    name character varying(50) NOT NULL,
    email character varying(100) NOT NULL,
    password character varying(255) NOT NULL,
    job_grade character varying(50)
);


ALTER TABLE public.employee OWNER TO postgres;

--
-- Name: employee_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.employee_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.employee_id_seq OWNER TO postgres;

--
-- Name: employee_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.employee_id_seq OWNED BY public.employee.id;


--
-- Name: notion; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.notion (
    id integer NOT NULL,
    task_id integer,
    "timestamp" timestamp without time zone NOT NULL,
    participant_id character varying(50),
    content text,
    embedding public.vector(768)
);


ALTER TABLE public.notion OWNER TO postgres;

--
-- Name: notion_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.notion_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.notion_id_seq OWNER TO postgres;

--
-- Name: notion_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.notion_id_seq OWNED BY public.notion.id;


--
-- Name: onedrive; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.onedrive (
    id integer NOT NULL,
    task_id integer,
    "timestamp" timestamp without time zone NOT NULL,
    writer character varying(50),
    content text,
    embedding public.vector(768)
);


ALTER TABLE public.onedrive OWNER TO postgres;

--
-- Name: onedrive_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.onedrive_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.onedrive_id_seq OWNER TO postgres;

--
-- Name: onedrive_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.onedrive_id_seq OWNED BY public.onedrive.id;


--
-- Name: outlook; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.outlook (
    id integer NOT NULL,
    receiver character varying(50),
    sender character varying(50) NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    task_id integer,
    content text,
    embedding public.vector(768)
);


ALTER TABLE public.outlook OWNER TO postgres;

--
-- Name: outlook_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.outlook_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.outlook_id_seq OWNER TO postgres;

--
-- Name: outlook_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.outlook_id_seq OWNED BY public.outlook.id;


--
-- Name: participant; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.participant (
    id integer NOT NULL,
    notion_id integer,
    p1 character varying(50),
    p2 character varying(50),
    p3 character varying(50),
    p4 character varying(50),
    p5 character varying(50),
    p6 character varying(50)
);


ALTER TABLE public.participant OWNER TO postgres;

--
-- Name: participant_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.participant_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.participant_id_seq OWNER TO postgres;

--
-- Name: participant_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.participant_id_seq OWNED BY public.participant.id;


--
-- Name: report; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.report (
    id integer NOT NULL,
    task_id integer,
    "timestamp" timestamp without time zone NOT NULL,
    writer character varying(50),
    email character varying(100),
    report text,
    report_embedded public.vector(1536)
);


ALTER TABLE public.report OWNER TO postgres;

--
-- Name: report_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.report_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.report_id_seq OWNER TO postgres;

--
-- Name: report_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.report_id_seq OWNED BY public.report.id;


--
-- Name: slack; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.slack (
    id integer NOT NULL,
    receiver character varying(50),
    sender character varying(50) NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    task_id integer,
    content text,
    embedding public.vector(768)
);


ALTER TABLE public.slack OWNER TO postgres;

--
-- Name: slack_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.slack_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.slack_id_seq OWNER TO postgres;

--
-- Name: slack_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.slack_id_seq OWNED BY public.slack.id;


--
-- Name: task; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.task (
    id integer NOT NULL,
    task_uuid character varying(50),
    description text
);


ALTER TABLE public.task OWNER TO postgres;

--
-- Name: task_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.task_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.task_id_seq OWNER TO postgres;

--
-- Name: task_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.task_id_seq OWNED BY public.task.id;


--
-- Name: administer id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.administer ALTER COLUMN id SET DEFAULT nextval('public.administer_id_seq'::regclass);


--
-- Name: employee id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employee ALTER COLUMN id SET DEFAULT nextval('public.employee_id_seq'::regclass);


--
-- Name: notion id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.notion ALTER COLUMN id SET DEFAULT nextval('public.notion_id_seq'::regclass);


--
-- Name: onedrive id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.onedrive ALTER COLUMN id SET DEFAULT nextval('public.onedrive_id_seq'::regclass);


--
-- Name: outlook id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.outlook ALTER COLUMN id SET DEFAULT nextval('public.outlook_id_seq'::regclass);


--
-- Name: participant id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.participant ALTER COLUMN id SET DEFAULT nextval('public.participant_id_seq'::regclass);


--
-- Name: report id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report ALTER COLUMN id SET DEFAULT nextval('public.report_id_seq'::regclass);


--
-- Name: slack id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.slack ALTER COLUMN id SET DEFAULT nextval('public.slack_id_seq'::regclass);


--
-- Name: task id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.task ALTER COLUMN id SET DEFAULT nextval('public.task_id_seq'::regclass);


--
-- Data for Name: administer; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.administer (id, name, email, password, job_grade, task_involved) FROM stdin;
1	김민준	kimminjun@skax.co.kr	1111	관리자	{1,2}
2	박서연	parkseoyeon@skax.co.kr	2222	관리자	{1,2}
3	이수진	leesujin@skax.co.kr	3333	관리자	{1,2}
4	최준영	choijunyoung@skax.co.kr	4444	관리자	{1,2}
\.


--
-- Data for Name: employee; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.employee (id, name, email, password, job_grade) FROM stdin;
1	서은수	eunsuseo@skax.co.kr	1111	매니저
2	윤소현	shyoun@skax.co.kr	2222	매니저
3	박현규	bakhg@skax.co.kr	3333	매니저
4	정도현	dohyunj@skax.co.kr	4444	매니저
5	박범준	parkbj@skax.co.kr	5555	매니저
6	조성호	choseongho@skax.co.kr	6666	매니저
\.


--
-- Data for Name: notion; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.notion (id, task_id, "timestamp", participant_id, content, embedding) FROM stdin;
1	1	2025-09-23 10:00:00	1	이번 주 온라인 쇼핑몰 시스템 구축 관련 회의 내용 정리/n참석자: 서은수(PM), 윤소현(백엔드 개발)/n/n1. 결제 모듈 안정화 테스트 결과/n - 지난주 발견된 오류 3건 중 2건 수정 완료/n - 결제 승인 속도 평균 1.2초 → 0.8초 개선됨/n - 카드사별 연동 체크 진행, 아직 일부 테스트 남음/n/n2. 상품 검색 속도 개선 작업 진행 상황/n - DB 인덱스 최적화 시도 → 검색 속도 약 20% 빨라짐/n - 캐시 서버 적용 방안 논의 중, 다음 주 PoC 예정/n/n3. 이슈 사항/n - 특정 시간대(오후 8~9시)에 서버 응답 지연 발생/n - 원인: 동시 접속자 수 증가, 서버 자원 부족 추정/n - 단기 대책: 서버 모니터링 강화 및 로그 수집/n/n4. 차주 계획/n - 결제 모듈 최종 안정화 테스트 완료 목표/n - 검색 속도 개선 PoC 결과 리뷰 후 적용 범위 확정/n/n추가 논의사항: SK 회사에서 클라우드 인프라 확장 검토 필요성 제기/n	\N
2	2	2025-09-23 10:00:00	2	이번 주 병원 예약·진료 시스템 통합 회의 노트입니다./n금주 목표는 크게 두 가지입니다. 첫째는 모바일 예약 화면 UI를 개선해서 환자들이 더 쉽게 예약할 수 있도록 만드는 것이고, 둘째는 검사 결과 자동 연동 기능을 본격적으로 개발에 착수하는 것입니다./nUI 개선 작업에서는 버튼 크기와 배치를 바꿔서 시력이 안 좋은 환자도 쉽게 클릭할 수 있게 했습니다. 또 예약 확인 화면에서 바로 진료 시간표를 보여줄 수 있도록 설계를 바꿨습니다./n검사 결과 자동 연동 기능은 환자의 검사 데이터를 병원 서버에서 불러와 진료 기록과 연결하는 작업입니다. 이번 주는 데이터베이스 구조를 검토하고 샘플 데이터를 연동하는 데 집중했습니다./n이슈 사항으로는 일부 환자 데이터가 중복 저장되는 문제가 있었습니다. 예를 들어 같은 환자가 여러 번 예약을 시도하면 데이터베이스에 두 번 저장되는 문제가 발생했습니다./n해결 방안은 예약 ID를 고유값으로 설정하는 것이며, 추후 전화번호와 함께 묶어서 중복을 방지하는 로직을 추가할 계획입니다./n회의 결론은 병원 시스템은 환자의 편의성과 안정성이 가장 중요하다는 점이며, 앞으로도 환자 경험을 개선하는 방향으로 개발을 이어가기로 했습니다./n	\N
\.


--
-- Data for Name: onedrive; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.onedrive (id, task_id, "timestamp", writer, content, embedding) FROM stdin;
1	1	2025-09-24 17:15:00	윤소현	결제 모듈 안정화 테스트 결과 보고서/n/n1. 테스트 환경/n - 서버: SK 회사 클라우드 서버 3대/n - DB: PostgreSQL 17/n/n2. 테스트 항목/n - 결제 승인 속도 측정/n - 오류 코드 발생 여부/n - 카드사 연동 호환성/n/n3. 주요 결과/n - 평균 응답 속도: 0.8초 (이전 대비 30% 개선)/n - 오류 발생: 10건 → 2건/n - 카드사 연동 정상: 5개 중 4개 성공, 1개 보완 필요/n/n4. 추가 확인사항/n - 특정 시간대 서버 과부하 문제 지속 발생/n - DB 인덱스 최적화 필요성 확인/n/n5. 결론/n이번 주 테스트에서는 결제 모듈 안정화가 대부분 달성되었으나, 서버 응답 지연은 여전히 해결 과제임/n	\N
2	2	2025-09-24 16:00:00	박범준	환자 예약 데이터/n예약 ID, 환자 이름, 예약 날짜, 예약 시간, 진료과, 담당의사/n1001, 김민수, 2025-09-25, 09:00, 내과, 박지훈/n1002, 이서연, 2025-09-25, 10:00, 외과, 김태현/n/n중복 저장 발생 사례/n예약 ID 없이 두 번 들어온 환자 데이터: 환자명 김민수, 예약일 2025-09-25, 예약시간 09:00/n해결 방안: 예약 ID를 고유 키 값으로 설정하고, 중복 방지 로직 추가/n/nUI 개선 내역/n- 버튼 크기 확대 적용/n- 예약 확인 화면에서 진료 시간표 바로 확인 기능 추가/n/n검사 결과 자동 연동 기능 준비/n- 데이터베이스 샘플 테이블 생성 완료/n- 검사 결과와 진료 기록 매칭 테스트 시작/n	\N
\.


--
-- Data for Name: outlook; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.outlook (id, receiver, sender, "timestamp", task_id, content, embedding) FROM stdin;
1	윤소현	서은수	2025-09-23 08:30:00	1	안녕하세요/n이번 주 온라인 쇼핑몰 결제 모듈 점검 회의를 9월 24일 오후 2시에 진행하려 합니다./n장소: SK 회사 2층 회의실/n안건: 결제 모듈 안정화 진행 상황, 서버 응답 지연 문제 논의/n참석자: 서은수, 윤소현/n감사합니다/n	\N
2	서은수	윤소현	2025-09-24 17:20:00	1	안녕하세요/n이번 주 테스트 결과 정리 문서(payment_module_test_report.docx)를 OneDrive에 업로드했습니다./n경로: /project/online_shop/payment/n확인 후 의견 부탁드립니다/n감사합니다/n	\N
3	박범준	박현규	2025-09-23 14:00:00	2	안녕하세요/n이번 주 병원 예약·진료 시스템 통합 관련 회의는 9월 25일(목) 오후 3시에 진행합니다./n장소: SK헬스케어 회의실/n온라인 접속 링크도 전달드립니다./n안건: 모바일 예약 UI 개선, 검사 결과 자동 연동 기능 개발/n감사합니다/n	\N
4	박현규	박범준	2025-09-24 16:30:00	2	안녕하세요/n이번 주 작업 중간 결과를 정리한 파일을 OneDrive에 업로드했습니다./n파일명: hospital_reservation_integration.xlsx/n예약 UI 개선 내역과 DB 구조 변경 사항이 포함돼 있습니다./n확인 부탁드립니다/n	\N
\.


--
-- Data for Name: participant; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.participant (id, notion_id, p1, p2, p3, p4, p5, p6) FROM stdin;
1	1	서은수	윤소현	\N	\N	\N	\N
2	2	박현규	박범준	\N	\N	\N	\N
\.


--
-- Data for Name: report; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.report (id, task_id, "timestamp", writer, email, report, report_embedded) FROM stdin;
1	1	2025-09-24 14:30:00	서은수	eunsuseo@skax.co.kr	\N	\N
2	1	2025-09-24 11:00:00	윤소현	shyoun@skax.co.kr	\N	\N
3	2	2025-09-24 17:45:00	박현규	bakhg@skax.co.kr	\N	\N
4	2	2025-09-24 10:25:00	박범준	parkbj@skax.co.kr	\N	\N
\.


--
-- Data for Name: slack; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.slack (id, receiver, sender, "timestamp", task_id, content, embedding) FROM stdin;
1	윤소현	서은수	2025-09-22 09:10:00	1	이번 주 결제 모듈 안정화 테스트 잘 진행되고 있어? SK 회사 보고에 들어갈 거라 중요해/n	\N
2	서은수	윤소현	2025-09-22 09:12:00	1	네, 온라인 쇼핑몰 결제 모듈 속도가 좀 더 빨라졌어요. 평균 0.8초 정도까지 줄였습니다/n	\N
3	윤소현	서은수	2025-09-22 09:15:00	1	좋네! 근데 어제 저녁에 서버 응답 지연 있었다던데 확인했어?/n	\N
4	서은수	윤소현	2025-09-22 09:17:00	1	네 확인했는데 동시 접속자가 몰려서 그런 것 같아요. 서버 로그 보니 CPU 95%까지 올라갔습니다/n	\N
5	윤소현	서은수	2025-09-22 09:20:00	1	검색 속도 개선도 같이 보고해줘. 이번 주 보고서는 임원진도 본대/n	\N
6	서은수	윤소현	2025-09-22 11:05:00	1	DB 인덱스 조정했더니 상품 검색 속도가 약 20% 개선됐습니다/n	\N
7	윤소현	서은수	2025-09-22 11:10:00	1	좋네. 근데 캐시 서버 적용은 언제쯤 가능할까?/n	\N
8	서은수	윤소현	2025-09-22 11:13:00	1	PoC는 다음 주쯤 시작 가능할 것 같아요. SK 회사 클라우드 자원 할당만 확정되면요/n	\N
9	윤소현	서은수	2025-09-22 13:00:00	1	점심 먹었어? 오늘 카페테리아 붐볐다더라/n	\N
10	서은수	윤소현	2025-09-22 13:02:00	1	네, 간단히 먹었어요. 온라인 쇼핑몰 검색 기능 생각하다 늦게 갔더니 자리 없더라고요/n	\N
11	윤소현	서은수	2025-09-22 15:45:00	1	오후 8시쯤 서버 지연 현상 또 나올까봐 걱정된다/n	\N
12	서은수	윤소현	2025-09-22 15:47:00	1	오늘은 모니터링을 강화해둘게요. 서버 자원 부족이 원인일 가능성이 크니까요/n	\N
13	윤소현	서은수	2025-09-22 16:10:00	1	SK 회사 회의실 예약했어? 내일 임원 보고 준비도 해야 해/n	\N
14	서은수	윤소현	2025-09-22 16:15:00	1	네, 오후 3시 회의실 확보했습니다. 보고 자료는 제가 정리해둘게요/n	\N
15	윤소현	서은수	2025-09-23 09:00:00	1	어제 저녁 서버 상태 어땠어?/n	\N
16	서은수	윤소현	2025-09-23 09:05:00	1	다행히 큰 지연은 없었습니다. 다만 CPU가 여전히 높아서 개선이 필요합니다/n	\N
17	윤소현	서은수	2025-09-23 10:20:00	1	결제 모듈 안정화 테스트 로그 업로드했습니다. OneDrive 확인해보세요/n	\N
18	서은수	윤소현	2025-09-23 10:25:00	1	좋아. 결과 요약해서 보고용으로 정리 부탁해/n	\N
19	윤소현	서은수	2025-09-23 12:30:00	1	오늘 점심 메뉴 추천 좀 해주세요. 머리 아파요 ㅋㅋ/n	\N
20	서은수	윤소현	2025-09-23 12:32:00	1	온라인 쇼핑몰에서 할인하던 김밥세트 어때? ㅋㅋ/n	\N
21	윤소현	서은수	2025-09-23 15:00:00	1	검색 속도 개선 내용도 보고서에 넣어야지/n	\N
22	서은수	윤소현	2025-09-23 15:05:00	1	네, DB 최적화 결과랑 캐시 서버 적용 계획 같이 정리하겠습니다/n	\N
23	윤소현	서은수	2025-09-24 09:10:00	1	결제 모듈 카드사 연동 테스트 중인데 한 군데서 오류 발생했습니다/n	\N
24	서은수	윤소현	2025-09-24 09:15:00	1	어느 카드사야? SK 회사랑 협의 필요할 수도 있겠다/n	\N
25	윤소현	서은수	2025-09-24 09:18:00	1	H카드 쪽입니다. 에러 로그 정리 중이에요/n	\N
26	서은수	윤소현	2025-09-24 14:00:00	1	회의 준비됐어? 임원 보고 자료 같이 확인하자/n	\N
27	윤소현	서은수	2025-09-24 14:05:00	1	네, 온라인 쇼핑몰 결제 모듈 개선 사항과 서버 응답 지연 이슈 정리했습니다/n	\N
28	서은수	윤소현	2025-09-25 09:30:00	1	어제 회의 피드백 괜찮았어/n	\N
29	윤소현	서은수	2025-09-25 09:35:00	1	네, 결제 모듈 속도 개선은 긍정적으로 봤습니다. 다만 서버 지연은 조속히 해결해야 한다고 했습니다/n	\N
30	서은수	윤소현	2025-09-26 10:00:00	1	이번 주 보고서 최종 버전 마무리했어?/n	\N
31	박범준	박현규	2025-09-22 09:05:00	2	어제 모바일 예약 화면 버튼 크기 조정했는데, SK헬스케어 테스트 환경에서 반응 괜찮았어?/n	\N
32	박현규	박범준	2025-09-22 09:07:00	2	응 괜찮아 보였어. 다만 글자 크기는 조금 더 크게 해야 할 듯./n	\N
33	박범준	박현규	2025-09-22 09:10:00	2	알겠어. 예약 UI 개선 목표가 ‘누구나 쉽게’니까 글자도 맞춰야지./n	\N
34	박현규	박범준	2025-09-22 09:20:00	2	DB 로그 보니까 예약 데이터 중복 저장되는 케이스가 또 발생했어./n특히 같은 환자가 여러 번 시도할 때./n	\N
35	박범준	박현규	2025-09-22 09:23:00	2	헐, 그거 큰 문제네. 예약 ID 고유값 설정 제대로 안 먹힌 건가?/n	\N
36	박현규	박범준	2025-09-22 09:30:00	2	아직 샘플 데이터에는 중복 방지 로직이 없어서 그래. 이번 주에 수정할게./n	\N
37	박범준	박현규	2025-09-22 10:00:00	2	점심 뭐 먹을래? 병원 근처 식당 가자. 어제부터 UI 코드 붙잡느라 머리 아프네./n	\N
38	박현규	박범준	2025-09-22 10:05:00	2	좋아, 삼계탕 어때? SK헬스케어 회의실 근처에 맛집 있잖아./n	\N
39	박범준	박현규	2025-09-22 14:10:00	2	검사 결과 자동 연동 기능 DB 샘플 테이블 만들었어. 오후에 확인해줄래?/n	\N
40	박현규	박범준	2025-09-22 14:15:00	2	오케이. 연동 잘 되면 UI 쪽에도 바로 연결해볼게./n	\N
41	박범준	박현규	2025-09-23 09:00:00	2	어제 병원 예약 UI에 캘린더 추가했어. 이제 날짜 선택이 훨씬 보기 좋을 거야./n	\N
42	박현규	박범준	2025-09-23 09:05:00	2	좋네. 그럼 DB에서 예약 시간과 매칭 잘 되는지만 확인하면 돼./n	\N
43	박범준	박현규	2025-09-23 11:00:00	2	근데 중복 예약 문제 아직 완전히 해결 안 됐어. 새로운 테스트 케이스에서 또 떴어./n	\N
44	박현규	박범준	2025-09-23 11:05:00	2	그럼 예약 ID 외에 전화번호도 같이 체크하도록 바꿀까?/n	\N
45	박범준	박현규	2025-09-23 13:00:00	2	좋은 생각이야. 고유값을 두 개 이상 묶으면 중복 줄일 수 있지./n	\N
46	박현규	박범준	2025-09-23 15:20:00	2	SK헬스케어 쪽에서 이번 주 목요일에 회의 일정 잡았어. 오후 3시래./n	\N
47	박범준	박현규	2025-09-23 15:25:00	2	오케이. UI 개선 현황이랑 DB 중복 이슈 해결 방안 같이 보고하자./n	\N
48	박현규	박범준	2025-09-23 17:00:00	2	야, 오늘 야근해야 할지도 몰라. 캘린더랑 검사 결과 화면 같이 붙이는 중인데 쉽지 않네./n	\N
49	박범준	박현규	2025-09-23 17:05:00	2	나도 DB 정리 때문에 남아야 할 듯. 우리 둘 다 병원 예약 시스템 붙잡고 있네./n	\N
50	박현규	박범준	2025-09-24 09:30:00	2	어제 테스트 버전 올렸어. SK헬스케어 내부 계정으로 로그인해봐./n	\N
51	박범준	박현규	2025-09-24 09:40:00	2	봤어. UI 반응 속도 괜찮아. 다만 검사 결과 불러오는 게 조금 느려./n	\N
52	박현규	박범준	2025-09-24 10:00:00	2	DB 인덱스 추가하면 빨라질까?/n	\N
53	박범준	박현규	2025-09-24 10:05:00	2	응, 인덱스 처리하면 조회 속도 개선될 거야. 이번 주 안에 반영할게./n	\N
54	박현규	박범준	2025-09-24 13:20:00	2	OneDrive에 hospital_reservation_integration.xlsx 올렸어. 확인해줘./n	\N
55	박범준	박현규	2025-09-24 13:25:00	2	봤어. 정리 잘했네. UI 스크린샷도 넣으면 좋을 듯./n	\N
56	박현규	박범준	2025-09-25 09:00:00	2	오늘 회의 준비됐어? SK헬스케어 담당자 앞에서 발표해야 해./n	\N
57	박범준	박현규	2025-09-25 09:10:00	2	응. UI 개선 자료랑 DB 구조 수정안 다 챙겼어./n	\N
58	박현규	박범준	2025-09-25 15:00:00	2	회의 시작했어. 지금 예약 화면 시연 중./n	\N
59	박범준	박현규	2025-09-25 15:30:00	2	담당자가 검사 결과 자동 연동 부분 관심 많네. 속도 빨리 개선해야겠다./n	\N
60	박현규	박범준	2025-09-26 11:00:00	2	이번 주 마무리로 중복 예약 방지 로직 다시 점검하자. 병원 예약 시스템 핵심이잖아./n	\N
\.


--
-- Data for Name: task; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.task (id, task_uuid, description) FROM stdin;
1	t001	온라인 쇼핑몰 시스템 구축
2	t002	병원 예약·진료 시스템 통합
\.


--
-- Name: administer_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.administer_id_seq', 1, false);


--
-- Name: employee_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.employee_id_seq', 1, false);


--
-- Name: notion_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.notion_id_seq', 1, false);


--
-- Name: onedrive_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.onedrive_id_seq', 1, false);


--
-- Name: outlook_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.outlook_id_seq', 1, false);


--
-- Name: participant_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.participant_id_seq', 1, false);


--
-- Name: report_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.report_id_seq', 1, false);


--
-- Name: slack_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.slack_id_seq', 1, false);


--
-- Name: task_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.task_id_seq', 1, false);


--
-- Name: administer administer_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.administer
    ADD CONSTRAINT administer_email_key UNIQUE (email);


--
-- Name: administer administer_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.administer
    ADD CONSTRAINT administer_pkey PRIMARY KEY (id);


--
-- Name: employee employee_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employee
    ADD CONSTRAINT employee_email_key UNIQUE (email);


--
-- Name: employee employee_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employee
    ADD CONSTRAINT employee_pkey PRIMARY KEY (id);


--
-- Name: notion notion_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.notion
    ADD CONSTRAINT notion_pkey PRIMARY KEY (id);


--
-- Name: onedrive onedrive_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.onedrive
    ADD CONSTRAINT onedrive_pkey PRIMARY KEY (id);


--
-- Name: outlook outlook_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.outlook
    ADD CONSTRAINT outlook_pkey PRIMARY KEY (id);


--
-- Name: participant participant_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.participant
    ADD CONSTRAINT participant_pkey PRIMARY KEY (id);


--
-- Name: report report_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report
    ADD CONSTRAINT report_pkey PRIMARY KEY (id);


--
-- Name: slack slack_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.slack
    ADD CONSTRAINT slack_pkey PRIMARY KEY (id);


--
-- Name: task task_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.task
    ADD CONSTRAINT task_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

\unrestrict 1abCLlIIK2cJV8kuve9W1GBvhQ1CTbWD8mlvLHOmTdspP4UaoDf84nbwm02lWJx

