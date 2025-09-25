--
-- PostgreSQL database dump
--

-- Dumped from database version 14.17 (Homebrew)
-- Dumped by pg_dump version 14.17 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

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

ALTER TABLE public.employee ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.employee_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: notion; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.notion (
    id integer NOT NULL,
    task_id character varying(50),
    "timestamp" timestamp without time zone NOT NULL,
    participant_id character varying(50),
    content text,
    embedding public.vector(768)
);


ALTER TABLE public.notion OWNER TO postgres;

--
-- Name: onedrive; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.onedrive (
    id integer NOT NULL,
    task_id character varying(50),
    "timestamp" timestamp without time zone NOT NULL,
    writer character varying(50),
    content text,
    embedding public.vector(768)
);


ALTER TABLE public.onedrive OWNER TO postgres;

--
-- Name: outlook; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.outlook (
    id integer NOT NULL,
    receiver character varying(50),
    sender character varying(50) NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    task_id character varying(50),
    content text,
    embedding public.vector(768)
);


ALTER TABLE public.outlook OWNER TO postgres;

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
-- Name: report; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.report (
    id integer NOT NULL,
    task_id integer,
    "timestamp" timestamp without time zone NOT NULL,
    writer character varying(50),
    email character varying(100),
    content text
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


ALTER TABLE public.report_id_seq OWNER TO postgres;

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
    task_id character varying(50),
    content text,
    embedding public.vector(768)
);


ALTER TABLE public.slack OWNER TO postgres;

--
-- Name: task; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.task (
    id integer NOT NULL,
    description text,
    embedding public.vector(1536),
    task_uuid character varying(50)
);


ALTER TABLE public.task OWNER TO postgres;

--
-- Name: task_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.task ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.task_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: report id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report ALTER COLUMN id SET DEFAULT nextval('public.report_id_seq'::regclass);


--
-- Data for Name: employee; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.employee (id, name, email, password, job_grade) FROM stdin;
1	서은수	eunsuseo@skax.co.kr	1111	\N
2	윤소현	shyoun@skax.co.kr	2222	\N
3	박현규	bakhg@skax.co.kr	3333	\N
4	정도현	dohyunj@skax.co.kr	4444	\N
5	박범준	parkbj@skax.co.kr	5555	\N
6	조성호	choseongho@skax.co.kr	6666	\N
7	테스트	test@skax.co.kr	1234	\N
9	김민수	example@skax.kr	$2b$12$I6bVjxzGltdOncJho/vxQOgZZQbbRpcRthc3xispNG3klYv2bQxUu	\N
\.


--
-- Data for Name: notion; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.notion (id, task_id, "timestamp", participant_id, content, embedding) FROM stdin;
1	1	2025-09-22 10:00:00	1	회의명: SK하이닉스 스마트 팹 예지보전 주간 킥오프(Week of 0922)/n참석자: 서은수(PM), 윤소현(Data Eng), 박현규(AI Dev)/n목적: 센서 데이터 수집 파이프라인 구축 진행 상황 점검 및 이상 탐지 모델 PoC 범위 확정./n/n[안건1] 데이터 수집 파이프라인 구조 확정/n- 소스: 설비 PLC 로그(OPC-UA), 공정 이력 MES, 장비 이벤트 로그(벤더 포맷)/n- 수집: Edge Collector → Kafka(Topic: skhynix.smartfab.sensors) → Stream Processor(Flink) → 시계열DB(TimescaleDB)/n- 보조: 원본 장기보관용 Object Storage(S3-호환) 30일 핫/180일 콜드 정책/n- 보안: 사내망 전용 VPC + SK AX Bastion, 전송 구간 TLS, PII 없음 확인/n리스크: 장비 로그 포맷 비표준(벤더 별 CSV/JSON/바이너리 혼재) → 전처리 표준화 레이어 필요/n/n[안건2] 이상 탐지 모델 PoC 범위/n- 1차 대상: 노광/식각 2개 라인, 주요 센서 12종(챔버온도, 진공압, 스핀속도, 가스유량 등)/n- 레이블: 설비 알람/정지 이벤트를 약한 라벨로 활용(주석: 유지보수 이력은 2주 후 연동)/n- 기법: 시계열 Reconstruction(Conv-VAE) + 변동성 기반 Thresholding, 비교군으로 STL 분해 + MAD 기준치/n- 평가: 알람 검출 시점(선행시간), 과검출율, 주당 알람 건수 제한 정책(운영 경보 피로도)/n/n[안건3] 운영 대시보드/n- 지표: 라인별 설비가동률(OEE), 알람 Top-5, 이상Score 분포, 센서별 Drift 트렌드/n- 기술: Grafana + TimescaleDB + Alertmanager 연동(서비스: Slack, Outlook)/n/n[액션아이템]/n- 윤소현: OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결(9/24)/n- 박현규: Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인리포트(9/25)/n- 서은수: 벤더 로그 사양 수합, 표준 변환 스키마 확정(9/26)/n/n[의존성 및 이슈]/n- 장비 로그 포맷 미통일로 ETL 지연 가능성 큼 → 공통 스키마(v1) 우선 도입 후 점진 확장/n- SK하이닉스 내부 보안 정책에 따른 네트워크 포트/방화벽 승인 절차 필요(담당자 전달받음)/n/n[추가 논의]/n- Kafka 파티션 수는 초기 6, 일일 1.5배 증가 트래픽 가정 시 12까지 확장 계획/n- TimescaleDB 하이퍼테이블 청크 1일, 보조 인덱스는 line, tool_id, sensor 조합으로 구성/n- 변환 스키마 v1 필수/옵션 필드 목록 확정 및 샘플 데이터셋 1차 라벨 점검 일정 확정/n/n[성능 목표]/n- 쓰기 TPS 2배 개선 달성, 대시보드 렌더링 지연 1.5초 이내 유지/n- 알람 과검출률 15% 이하, 선행시간 +5분 이상 확보 목표/n/n[결론]/n- 이번 주 목표는 '수집 파이프라인 최소 동작'과 '이상 탐지 PoC 첫 결과' 확보./n- 다음 주 목표는 라인 확대와 알람 정책 튜닝 초안 수립./n/n회사 언급: SK하이닉스, SK hynix, SK AX/n키워드: 스마트 팹, 예지보전, OPC-UA, Kafka, TimescaleDB, Conv-VAE, STL+MAD, 대시보드, PoC, 가동률(OEE)	\N
2	2	2025-09-23 16:00:00	2	회의명: 현대자동차 디지털 트윈 주간 킥오프(울산 공장)/n참석자: 서은수(PM, SK AX), 박범준(시스템 아키텍트, SK AX), 조성호(IoT/ML, SK AX), 현대자동차 제조혁신팀 2명/n아젠다:/n1) 디지털 트윈 모델 최신 상태 공유(공정 흐름 완성)/n2) IoT 센서 연동 테스트 범위/일정/리스크 점검/n3) 실시간 데이터 동기화 지연(latency) 이슈 현황/대응책/n4) OneDrive 산출물 관리 전략 및 파일명 규칙(hyundai_digitaltwin_factory_design.pdf 포함)/n주요 논의:/n- 서은수: 지난주까지 공정 흐름 디지털 트윈 모델링을 완료했으며, 이번 주에는 울산 2라인의 설비 태그를 OPC-UA로 연결하는 IoT 테스트를 진행한다고 설명./n- 박범준: 시뮬레이션 엔진에서 라인 병목 구간 예측, Kafka 파티션 증설 제안./n- 조성호: IoT 게이트웨이에서 구형 PLC 표준화 문제, 야간(22~24시) 동기화 지연(latency) 로그 공유./n- 현대자동차 제조혁신팀: KPI(라인 가동률, 생산량, 불량률, 에너지 사용량) 대시보드 반영 요청./n결정사항:/n1) IoT 연동: 울산 2라인부터 센서 48종, 태그 1,200개 연결./n2) 데이터 파이프라인: Kafka 파티션 6→12 증설, 백업 스케줄 현대자동차 IT와 조정./n3) 모델 정합성: 매일 09시 편차 리포트 자동 생성./n4) 산출물 관리: OneDrive 루트 ‘hyundai_digitwin/design’에 ‘hyundai_digitaltwin_factory_design.pdf’ 관리./n리스크: 실시간 지연, 구형 PLC 태그 문제, 네트워크 부하./n대응: 백업 조정, 태그 변환 룰, Kafka/스토리지 성능 상향./n액션아이템:/n- 서은수: 일정 조정안 커뮤니케이션/n- 박범준: KPI 대시보드 초안 배포/n- 조성호: OPC-UA 변환 룰 적용 및 센서 노이즈 필터 튜닝/n다음 회의: 2025-09-26 16:00 온라인 진행./n비고: 보안등급 ‘내부’ 유지.	\N
3	3	2025-09-22 10:00:00	3	회의명: 신한은행 오픈뱅킹 ERP 연동 프로젝트 주간 점검/n참석자: 서은수(PM, SK AX), 정도현(백엔드 개발자, SK AX), 신한은행 IT 담당자 1명/n아젠다:/n1) 계좌조회 API 테스트 환경 점검/n2) ERP 매핑 로직 검토/n3) 실거래 트래픽 시뮬레이션 결과 공유/n4) 보안 가이드 준수 현황 확인/n주요 논의:/n- 서은수: ERP와 신한은행 API 간 매핑 현황을 검토, 일부 JSON 필드 변환 규칙 재정의 필요성 제기./n- 정도현: 이체 API 호출 시 토큰 만료 오류 발생 사례 보고, 재발 방지를 위해 재시도 로직 설계 중이라고 설명./n- 신한은행 IT: 실거래 트래픽 시뮬레이션 중 응답 지연이 발생한 로그를 제공, 병목 원인 분석을 지원하겠다고 답변./n결정사항:/n1) ERP 매핑 규칙에 계좌 상태 필드 추가./n2) API 호출 재시도 로직 3회 적용 후 실패시 알람 전송./n3) 보안 키 관리 AWS Secret Manager로 일원화./n리스크:/n- 실거래 트래픽 응답 지연 지속시 SLA 준수 위협./n- 배포 일정 차질 우려./n대응:/n- 병목 구간 비동기 처리 모듈 도입 검토./n- CloudWatch 기반 알람 세분화./n액션아이템:/n- 서은수: 보안 가이드 정리 후 팀 공유/n- 정도현: 재시도 로직 적용 및 테스트 리포트 제출/n다음 회의: 2025-09-29 10:00 온라인 진행./n비고: 보안등급 ‘내부’ 유지.	\N
4	4	2025-09-22 10:00:00	4	회의명: KakaoBrain LLM 데이터 파이프라인 주간 킥오프(Week of 0922)/n참석자: 윤소현(Data Eng), 박현규(AI Dev)/n목적: 한국어 LLM 학습용 대규모 데이터 파이프라인 진행 상황 점검./n/n[안건1] 데이터 수집 모듈 설계 및 크롤링 테스트/n[안건2] 텍스트 정제 규칙 적용 및 샘플셋 구축/n[이슈] 스토리지 I/O 병목 발생./n/n액션아이템:/n- 윤소현: 크롤러 rate-limit, 중복 제거 캐시 레이어 PoC/n- 박현규: 문장 분리/정제 룰셋 커밋, profanity/PII 마스킹 구축/n/n회사 언급: SK AX, Kakao Brain/n키워드: LLM, 데이터 파이프라인, 크롤러, 정제, I/O 병목	\N
\.


--
-- Data for Name: onedrive; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.onedrive (id, task_id, "timestamp", writer, content, embedding) FROM stdin;
1	1	2025-09-25 18:00:00	윤소현	문서명: skhynix_smartfab_predictive_maintenance_report.docx/n/n1. 개요/n이 문서는 SK하이닉스 스마트 팹 예지보전 PoC에서 데이터 수집 파이프라인의 최소 동작 구성과 로그 표준화 방안을 기술한다./n목표는 생산 라인의 설비 이상을 조기에 감지하여 다운타임을 최소화하는 것으로, 초기 범위는 노광 및 식각 라인에 한정한다./n회사 표기: SK하이닉스, SK hynix, SK AX./n/n2. 아키텍처/nEdge 수집기(OPC-UA 커넥터)가 설비 PLC에서 센서 데이터를 폴링하여 Kafka 토픽(skhynix.smartfab.sensors)으로 송신한다./n스트림 프로세서(Flink)는 벤더별 이기종 로그를 공통 스키마 v1로 변환하고, TimescaleDB에 적재한다./n원본은 S3 호환 Object Storage에 핫30일/콜드180일 정책으로 보관한다./n보안은 사내망 전용 VPC, 전송 TLS, 접근 IAM Role 기반으로 구성하며, PII 데이터는 수집 대상에 포함되지 않는다./n/n3. 데이터 스키마 v1/n필수 필드: ts(epoch_ms), line(text), tool_id(text), sensor(text), value(float), vendor(text), unit(text)./n옵션 필드: status(text), qc_flag(text), batch_id(text)./n스케일 맵: 벤더A/B/C 별 센서 단위와 배율을 표준화 테이블에 등록한다./n/n4. 전처리 규칙/n결측 30초 이내는 선형보간, 그 이상은 마스킹 처리한다./n스파이크 제거는 이동평균 대비 5*IQR 초과 구간을 후보로 표기한다./n타임스탬프 드리프트는 NTP 기준 200ms 초과 시 보정한다./n/n5. 이상 탐지 PoC 요약/n모델1(Conv-VAE Reconstruction): Reconstruction 에러 분포 기반 임계값을 탐색한다./n모델2(STL+MAD): 주기/추세/잔차 분해 후 잔차의 이상 구간을 검출한다./n선행시간, 과검출율, 주당 알람 수를 핵심 지표로 삼고, 운영 알람 피로도를 낮추기 위해 상한 정책을 적용한다./n/n6. 대시보드/nGrafana 기반으로 라인별 OEE, 이상Score 분포, 알람 Top-5, 센서 Drift 트렌드를 시각화한다./nAlertmanager를 통해 Slack/Outlook으로 알림을 전송하며, 알림 임계치를 주간 단위로 튜닝한다./n/n7. 성능 및 병목/nKafka→DB 경로에서 쓰기 TPS가 초기 구성 대비 2배 개선되었으나, 일부 쿼리 인덱스 최적화가 필요하다./nObject Storage는 멀티파트 업로드와 수명주기 정책으로 비용을 최적화한다./n/n8. 리스크 및 대응/n가장 큰 리스크는 장비 로그 포맷의 비표준성으로, 변환 스키마의 점진 확장을 전제로 한다./n네트워크/방화벽 정책 승인 절차는 SK하이닉스 보안 기준에 맞춰 진행한다./n/n9. 결론/n본 구성은 스마트 팹 예지보전의 최소 요건을 충족하며, 다음 단계는 라인 확대와 모델 임계치 자동 튜닝이다./n본 문서는 SK AX 내부 표준에 따라 작성되었다.	\N
2	2	2025-09-24 14:20:00	박범준	문서명: hyundai_digitaltwin_factory_design.pdf/n개요: 현대자동차 울산 공장의 디지털 트윈 설계 문서./n구성: 공정 흐름 모델링, IoT 센서 연동, Kafka 스트리밍, KPI 대시보드./n세부: 센서 48종, 태그 1,200개 OPC-UA 연동, 지연 측정 포인트, KPI 4종./n리스크: latency 급증, 구형 PLC, 네트워크 부하./n대응: Kafka 파티션 6→12, 스토리지 IOPS 상향, 변환 룰 표준화./n보안: 접근제어, 로그 180일 보관./n결론: 현대자동차 디지털 트윈의 실시간성과 안정성 달성을 목표로 하며 SK AX가 운영 최적화./n	\N
3	3	2025-09-26 11:00:00	정도현	문서명: shinhan_openbanking_erp_integration.sql/nCREATE TABLE erp_shinhan_account_mapping (/n    id SERIAL PRIMARY KEY,/n    erp_account_id VARCHAR(50),/n    shinhan_account_no VARCHAR(50),/n    balance NUMERIC,/n    status VARCHAR(20),/n    last_updated TIMESTAMP DEFAULT now()/n);/n/n-- 신한은행 계좌조회 API 응답을 ERP 테이블과 매핑하기 위한 SQL 스크립트/n-- 테스트 환경에서 정상 동작 확인 완료/n-- 운영 배포 시 보안 로직 삽입 예정	\N
4	4	2025-09-26 09:25:00	윤소현	문서명: kakaobrain_llm_data_pipeline.py/n/n1. 개요/nSK AX와 Kakao Brain이 협업하여 한국어 LLM 학습 데이터 파이프라인을 구축./n2. 기능/n- 크롤러 모듈: robots.txt 준수, rate-limit 적용/n- 정제 모듈: 문장 분리, 광고/내비게이션 제거, PII 마스킹/n- 저장: SSD 버퍼링 후 오브젝트 스토리지 업로드/n3. 리스크/n스토리지 I/O 병목 발생./n/n회사명: SK AX, Kakao Brain/n키워드: LLM, 데이터 수집, 파이프라인, SSD 버퍼링	\N
\.


--
-- Data for Name: outlook; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.outlook (id, receiver, sender, "timestamp", task_id, content, embedding) FROM stdin;
1	윤소현	서은수	2025-09-24 08:30:00	1	제목: [일정확인] SK하이닉스 스마트 팹 예지보전 데모 리허설 안내(9/26)/n안녕하세요, 서은수입니다./n9/26(금) 11:00 내부 리허설, 15:00 SK하이닉스 현업 데모 예정입니다./n장소: 온라인(Teams)/n아젠다: 1) 수집 파이프라인 현황 2) 이상 탐지 모델 PoC 결과 3) 다음 주 확대 계획/n참석: 서은수, 윤소현, 박현규/n감사합니다./n-SK AX PM 서은수-	\N
2	서은수	윤소현	2025-09-25 18:05:00	1	제목: [업로드 완료] skhynix_smartfab_predictive_maintenance_report.docx/n안녕하세요, 윤소현입니다./n요청하신 수집 파이프라인 다이어그램과 로그 표준화 초안이 포함된 문서를 OneDrive에 업로드했습니다./n파일명: skhynix_smartfab_predictive_maintenance_report.docx/n설명: Kafka 토픽 구조, TimescaleDB 스키마, S3 라이프사이클 정책, OPC-UA 커넥터 설정 포함./n확인 부탁드립니다./n-SK AX 데이터 엔지니어 윤소현-	\N
3	조성호	서은수	2025-09-23 15:22:00	2	제목: [회의 일정] 9/26(금) 16:00 현대자동차 디지털 트윈 주간 점검/n온라인 회의, 안건: IoT 센서 테스트, latency 로그 리뷰, KPI 반영./n- 서은수(Sk AX PM)	\N
4	박범준	조성호	2025-09-24 14:25:30	2	제목: [업로드 안내] OneDrive에 hyundai_digitaltwin_factory_design.pdf 업로드 완료/n변경 사항: OPC-UA 변환 룰 추가, KPI 정의 업데이트./n- SK AX 조성호	\N
5	정도현	서은수	2025-09-22 08:30:00	3	제목: [회의 일정] 신한은행 오픈뱅킹 ERP 연동 프로젝트 주간 회의/n일시: 2025-09-22 10:00/n안건: 계좌조회 API 연동, ERP 매핑 로직, 응답 지연 이슈	\N
6	서은수	정도현	2025-09-26 11:10:00	3	제목: [업로드 안내] OneDrive에 shinhan_openbanking_erp_integration.sql 업로드 완료/nERP 매핑 테이블 포함, 최종 확인 부탁드립니다.	\N
7	윤소현	박현규	2025-09-23 09:00:00	4	제목: [회의 일정 공유] Kakao Brain x SK AX 주간 싱크/n안건: 수집 현황, 정제 룰셋, I/O 병목 공유/n장소: 온라인(Teams)/n참석: 윤소현, 박현규	\N
8	윤소현	박현규	2025-09-26 09:30:00	4	제목: [업로드 완료] kakaobrain_llm_data_pipeline.py/nOneDrive에 파이프라인 코드 초안 업로드 완료/nAirflow DAG, 정제 함수, SSD 버퍼링 업로드 포함/n확인 부탁드립니다.	\N
\.


--
-- Data for Name: participant; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.participant (id, notion_id, p1, p2, p3, p4, p5, p6) FROM stdin;
1	1	서은수	윤소현	박현규	\N	\N	\N
2	2	서은수	박범준	조성호	\N	\N	\N
3	3	서은수	정도현	\N	\N	\N	\N
4	4	윤소현	박현규	\N	\N	\N	\N
\.


--
-- Data for Name: report; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.report (id, task_id, "timestamp", writer, email, content) FROM stdin;
21	1	2025-09-24 05:28:43.580749	박현규	bakhg@skax.co.kr	# 업무 1: SK하이닉스 스마트 팹 예지보전 플랫폼 주간 보고서\n\n## 1) 주간 요약\n이번 주 SK하이닉스 스마트 팹 예지보전 플랫폼의 주요 진행 상황은 데이터 수집 파이프라인 구축과 이상 탐지 모델 PoC 범위 확정입니다. 데이터 수집 파이프라인은 설비 PLC 로그, 공정 이력 MES, 장비 이벤트 로그를 소스로 하여 Edge Collector, Kafka, Flink, TimescaleDB를 통해 수집됩니다. 보조 저장소로는 Object Storage가 사용되며, 보안은 사내망 전용 VPC와 TLS로 강화되었습니다. 이상 탐지 모델 PoC는 노광/식각 2개 라인에서 시작하며, Conv-VAE와 STL+MAD 기법을 사용하여 알람 선행 감지 여부를 평가합니다. 운영 대시보드는 Grafana와 TimescaleDB를 사용하여 구성됩니다.\n\n## 2) 사람별 주요 산출물\n- 박현규: Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인 리포트 작성\n- 윤소현: OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결\n- 서은수: 벤더 로그 사양 수합 및 표준 변환 스키마 확정\n\n## 3) 협업 내역\n서은수, 윤소현, 박현규는 Slack과 Notion을 통해 협업하였습니다. 윤소현은 데이터 수집 파이프라인의 실시간 스트림 연결을 담당하였고, 박현규는 이상 탐지 모델의 학습 스크립트를 작성하였습니다. 서은수는 벤더 로그 사양을 수합하고 표준 변환 스키마를 확정하는 역할을 맡았습니다. 이들은 각자의 역할을 수행하면서도 서로의 진행 상황을 공유하고 조율하였습니다.\n\n## 4) 리스크/이슈\n- 장비 로그 포맷의 비표준화로 인해 ETL 과정에서 지연이 발생할 가능성이 큽니다. 이를 해결하기 위해 공통 스키마(v1)를 우선 도입하고 점진적으로 확장할 계획입니다.\n- SK하이닉스 내부 보안 정책에 따라 네트워크 포트 및 방화벽 승인 절차가 필요합니다.\n\n## 5) 차주 계획\n- 라인 확대와 알람 정책 튜닝 초안을 수립합니다.\n- 공통 스키마(v1)의 필수 및 옵션 필드 목록을 확정하고, 샘플 데이터셋의 1차 라벨 점검을 완료합니다.\n- Kafka 파티션 수를 초기 6에서 트래픽 증가를 가정하여 12까지 확장하는 계획을 실행합니다.\n- TimescaleDB의 하이퍼테이블 청크 및 보조 인덱스 구성을 최적화합니다.
27	1	2025-09-24 05:51:15.335855	박현규	bakhg@skax.co.kr	# 업무 1: SK하이닉스 스마트 팹 예지보전 플랫폼 주간 보고서\n\n## 1) 주간 요약\n이번 주 SK하이닉스 스마트 팹 예지보전 플랫폼의 진행 상황은 다음과 같습니다. 센서 데이터 수집 파이프라인의 구조가 확정되었으며, 노광 및 식각 두 개 라인에서 이상 탐지 모델 PoC를 시작하기로 결정했습니다. 데이터 수집은 Edge Collector에서 Kafka를 통해 TimescaleDB로 이루어지며, 장비 로그의 비표준 포맷 문제로 전처리 표준화 레이어가 필요합니다. 이상 탐지 모델은 Conv-VAE를 사용하여 시계열 데이터를 재구성하고, STL+MAD를 비교군으로 설정하여 성능을 평가합니다. 운영 대시보드는 Grafana와 TimescaleDB를 연동하여 구축됩니다. 이번 주 목표는 수집 파이프라인의 최소 동작과 이상 탐지 PoC의 첫 결과 확보입니다.\n\n## 2) 사람별 주요 산출물\n- 박현규: Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인 리포트 작성\n- 윤소현: OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결\n- 서은수: 벤더 로그 사양 수합 및 표준 변환 스키마 확정\n\n## 3) 협업 내역\n참여자들은 Slack과 Outlook을 통해 주로 협업하였습니다. 윤소현은 OPC-UA 커넥터 PoC를 진행하며 실시간 스트림 연결을 담당하였고, 박현규는 Conv-VAE 모델의 학습 스크립트를 작성하였습니다. 서은수는 벤더 로그 사양을 수합하고 표준 변환 스키마를 확정하는 역할을 맡았습니다. 이들은 주간 킥오프 회의에서 각자의 진행 상황을 공유하고, 필요한 조치를 논의하였습니다.\n\n## 4) 리스크/이슈\n장비 로그의 포맷이 비표준화되어 ETL 과정에서 지연이 발생할 가능성이 큽니다. 이를 해결하기 위해 공통 스키마(v1)를 우선 도입하고 점진적으로 확장할 계획입니다. 또한, SK하이닉스 내부 보안 정책에 따른 네트워크 포트 및 방화벽 승인 절차가 필요하여 관련 담당자와 협의가 필요합니다.\n\n## 5) 차주 계획\n다음 주에는 라인 확대와 알람 정책 튜닝 초안을 수립할 예정입니다. 또한, 장비 로그의 표준화 작업을 지속하여 ETL 지연 문제를 해결하고, 이상 탐지 모델의 성능을 개선하기 위한 추가적인 실험을 진행할 계획입니다.
28	4	2025-09-24 05:51:24.216836	박현규	bakhg@skax.co.kr	# 업무 4: 카카오브레인 LLM 데이터 파이프라인 주간 보고서\n\n## 1) 주간 요약\n이번 주 Kakao Brain과 SK AX의 주간 싱크에서는 카카오브레인 LLM 데이터 파이프라인의 진행 상황에 대해 논의가 이루어졌습니다. 주요 안건으로는 데이터 수집 현황, 정제 룰셋, 그리고 I/O 병목 현상이 다뤄졌습니다. 회의는 온라인으로 Teams 플랫폼을 통해 진행되었으며, 참석자는 윤소현과 박현규였습니다.\n\n## 2) 사람별 주요 산출물\n- 박현규: 데이터 정제 룰셋 초안 작성 및 I/O 병목 현상에 대한 분석 보고서 제출.\n\n## 3) 협업 내역\n- 윤소현과 박현규는 Slack을 통해 데이터 수집 현황에 대한 실시간 업데이트를 주고받았습니다. \n- Notion을 활용하여 데이터 정제 룰셋의 초안을 공동으로 작성하고 피드백을 주고받았습니다.\n- Outlook을 통해 회의 일정을 조율하고 주요 안건을 사전에 공유했습니다.\n- OneDrive를 통해 데이터 분석 보고서 및 관련 자료를 공유하며 협업을 진행했습니다.\n\n## 4) 리스크/이슈\n- I/O 병목 현상이 데이터 처리 속도를 저하시키고 있어 이에 대한 해결책이 필요합니다.\n- 데이터 정제 룰셋이 아직 초안 단계에 머물러 있어, 구체적인 기준과 절차의 확립이 요구됩니다.\n\n## 5) 차주 계획\n- I/O 병목 현상 해결을 위한 기술적 대안 모색 및 테스트 진행.\n- 데이터 정제 룰셋의 세부 기준 확립 및 최종안 작성.\n- 데이터 수집 프로세스의 효율성을 높이기 위한 추가적인 방안 검토.
22	4	2025-09-24 05:28:52.719271	박현규	bakhg@skax.co.kr	# 업무 4: 카카오브레인 LLM 데이터 파이프라인 주간 보고서\n\n## 1) 주간 요약\n이번 주 Kakao Brain x SK AX 주간 싱크에서는 카카오브레인 LLM 데이터 파이프라인의 진행 상황에 대해 논의했습니다. 주요 안건은 데이터 수집 현황, 정제 룰셋, 그리고 I/O 병목 현상에 대한 공유였습니다. 윤소현과 박현규가 참석하여 온라인으로 진행된 회의에서 데이터 수집의 현재 상태와 정제 과정에서의 문제점, 그리고 데이터 처리 속도를 저해하는 I/O 병목 현상에 대해 심도 있는 논의가 이루어졌습니다.\n\n## 2) 사람별 주요 산출물\n- 박현규: 데이터 정제 룰셋 초안 작성 및 I/O 병목 현상 분석 보고서 작성\n\n## 3) 협업 내역\n윤소현과 박현규는 Slack과 Notion을 통해 지속적으로 소통하며 데이터 파이프라인의 문제점을 공유하고 해결 방안을 모색했습니다. 윤소현은 데이터 수집 현황을 정리하여 공유하였고, 박현규는 이를 바탕으로 정제 룰셋을 작성하였습니다. 또한, I/O 병목 현상에 대한 분석 결과를 Notion에 게시하여 팀원들과 공유하였습니다.\n\n## 4) 리스크/이슈\n현재 데이터 파이프라인에서 I/O 병목 현상이 발생하여 데이터 처리 속도가 저하되고 있습니다. 이로 인해 전체 프로젝트 일정에 차질이 생길 가능성이 있습니다. 또한, 데이터 정제 룰셋이 아직 초안 단계에 머물러 있어 추가적인 검토와 수정이 필요합니다.\n\n## 5) 차주 계획\n다음 주에는 I/O 병목 현상을 해결하기 위한 기술적 대안을 모색하고, 데이터 정제 룰셋을 최종 확정하는 작업이 필요합니다. 이를 위해 추가적인 팀 회의를 통해 다양한 해결 방안을 논의하고, 각 팀원이 맡은 부분에 대한 진행 상황을 점검할 예정입니다.
23	1	2025-09-24 05:39:48.459337	박현규	bakhg@skax.co.kr	# 업무 1: SK하이닉스 스마트 팹 예지보전 플랫폼 주간 보고서\n\n죄송합니다. 제공된 context가 없어 요청하신 내용을 작성할 수 없습니다. 필요한 정보를 제공해 주시면 도움을 드리겠습니다.
24	1	2025-09-24 05:48:58.075097	박현규	bakhg@skax.co.kr	# 업무 1: SK하이닉스 스마트 팹 예지보전 플랫폼 주간 보고서\n\n## 1) 주간 요약\n이번 주 SK하이닉스 스마트 팹 예지보전 플랫폼 프로젝트에서는 데이터 수집 파이프라인 구축과 이상 탐지 모델 PoC의 초기 설정이 주요 논의 주제였습니다. 데이터 수집 파이프라인은 설비 PLC 로그, 공정 이력 MES, 장비 이벤트 로그를 소스로 하여 Edge Collector, Kafka, Stream Processor, TimescaleDB로 이어지는 구조로 확정되었습니다. 이상 탐지 모델 PoC는 노광/식각 두 개 라인에서 시작하며, Conv-VAE 기반의 시계열 Reconstruction과 STL+MAD를 비교하여 알람 선행 감지 여부를 평가합니다. 이번 주 목표는 '수집 파이프라인 최소 동작'과 '이상 탐지 PoC 첫 결과' 확보입니다.\n\n## 2) 사람별 주요 산출물\n- **박현규**: Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인 리포트 작성.\n- **윤소현**: OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결.\n- **서은수**: 벤더 로그 사양 수합 및 표준 변환 스키마 확정.\n\n## 3) 협업 내역\n- **Slack**: 팀원들 간의 실시간 커뮤니케이션 도구로 사용되어, 데이터 수집 파이프라인과 이상 탐지 모델 관련 논의가 이루어졌습니다.\n- **Notion**: 프로젝트 진행 상황과 주요 산출물, 액션 아이템 등을 기록하고 공유하는 데 사용되었습니다.\n- **Outlook**: 회의 일정 조율 및 회의록 공유에 활용되었습니다.\n- **OneDrive**: 산출물 및 관련 문서의 저장과 공유를 위한 플랫폼으로 사용되었습니다.\n\n## 4) 리스크/이슈\n- 장비 로그 포맷이 비표준화되어 있어 ETL 과정에서 지연이 발생할 가능성이 큽니다. 이를 해결하기 위해 공통 스키마(v1)를 우선 도입하고 점진적으로 확장할 계획입니다.\n- SK하이닉스 내부 보안 정책에 따라 네트워크 포트 및 방화벽 승인 절차가 필요하며, 이로 인한 일정 지연 가능성이 있습니다.\n\n## 5) 차주 계획\n- 데이터 수집 파이프라인의 라인 확대와 안정화 작업을 진행합니다.\n- 이상 탐지 모델의 알람 정책 튜닝 초안을 수립하고, PoC 결과를 바탕으로 모델 성능을 개선합니다.\n- 장비 로그 포맷 표준화를 위한 전처리 표준화 레이어 개발을 진행합니다.\n- 네트워크 포트 및 방화벽 승인 절차를 완료하여 보안 문제를 해결합니다.
25	1	2025-09-24 05:49:20.256598	박현규	bakhg@skax.co.kr	# 업무 1: SK하이닉스 스마트 팹 예지보전 플랫폼 주간 보고서\n\n## 1) 주간 요약\n이번 주 SK하이닉스 스마트 팹 예지보전 플랫폼 프로젝트에서는 센서 데이터 수집 파이프라인 구축과 이상 탐지 모델 PoC 범위 확정이 주요 논의 사항이었다. 데이터 수집 파이프라인은 설비 PLC 로그, 공정 이력 MES, 장비 이벤트 로그를 소스로 하여 Edge Collector, Kafka, Stream Processor, 시계열DB로 구성되었으며, 보조 저장소로 Object Storage를 사용한다. 이상 탐지 모델 PoC는 노광/식각 2개 라인에서 주요 센서 12종을 대상으로 Conv-VAE와 STL+MAD 기법을 비교하여 평가한다. 운영 대시보드는 Grafana를 활용하여 설비 가동률, 알람, 이상 점수 등을 시각화한다. 주요 목표는 수집 파이프라인의 최소 동작과 이상 탐지 PoC의 첫 결과 확보이다.\n\n## 2) 사람별 주요 산출물\n- **박현규**: Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인 리포트 준비.\n- **윤소현**: OPC-UA 커넥터 PoC로 3개 설비의 실시간 스트림 연결 작업.\n- **서은수**: 벤더 로그 사양 수합 및 표준 변환 스키마 확정.\n\n## 3) 협업 내역\n참여자들은 Slack, Notion, Outlook, OneDrive를 통해 협업했다. 윤소현은 OPC-UA 커넥터 PoC 작업을 진행하며 실시간 스트림 연결을 담당했고, 박현규는 Conv-VAE 학습 스크립트를 작성하여 모델 개발을 지원했다. 서은수는 벤더 로그 사양을 수합하고 표준 변환 스키마를 확정하여 데이터 전처리 표준화를 추진했다. 이들은 주간 킥오프 회의를 통해 진행 상황을 공유하고, 각자의 작업을 조율하며 협업을 이어갔다.\n\n## 4) 리스크/이슈\n- 장비 로그 포맷이 비표준화되어 있어 ETL 과정에서 지연이 발생할 가능성이 크다. 이를 해결하기 위해 공통 스키마(v1)를 우선 도입하고 점진적으로 확장할 계획이다.\n- SK하이닉스 내부 보안 정책에 따른 네트워크 포트 및 방화벽 승인 절차가 필요하며, 담당자에게 전달된 상태이다.\n\n## 5) 차주 계획\n다음 주에는 라인 확대와 알람 정책 튜닝 초안을 수립할 계획이다. 이를 위해 데이터 수집 파이프라인의 안정성을 높이고, 이상 탐지 모델의 성능을 평가하여 개선점을 도출할 예정이다. 또한, 변환 스키마 v1의 필수 및 옵션 필드 목록을 확정하고 샘플 데이터셋의 1차 라벨 점검을 진행할 계획이다.
26	4	2025-09-24 05:49:27.743756	박현규	bakhg@skax.co.kr	# 업무 4: 카카오브레인 LLM 데이터 파이프라인 주간 보고서\n\n## 1) 주간 요약\n이번 주 Kakao Brain과 SK AX 간의 주간 싱크 회의에서는 LLM 데이터 파이프라인의 수집 현황, 정제 룰셋, I/O 병목 현상에 대한 논의가 이루어졌습니다. 윤소현과 박현규가 참석하여 데이터 수집의 진행 상황을 점검하고, 데이터 정제 과정에서의 룰셋 적용 방법을 검토했습니다. 또한, I/O 병목 현상이 발생하는 부분을 공유하고 이를 해결하기 위한 방안을 모색했습니다.\n\n## 2) 사람별 주요 산출물\n- 박현규: 데이터 정제 룰셋 초안 작성 및 I/O 병목 현상 분석 보고서 작성\n\n## 3) 협업 내역\n윤소현과 박현규는 Teams를 통해 온라인 회의를 진행하며 데이터 수집 및 정제에 대한 의견을 교환했습니다. 박현규는 Notion에 데이터 정제 룰셋 초안을 공유하였고, 윤소현은 이에 대한 피드백을 제공했습니다. 또한, Slack을 통해 실시간으로 수집된 데이터를 검토하고, Outlook을 통해 회의 일정을 조율했습니다.\n\n## 4) 리스크/이슈\n현재 데이터 파이프라인에서 I/O 병목 현상이 발생하고 있어 데이터 처리 속도가 저하되고 있습니다. 이로 인해 전체 프로젝트 일정에 차질이 생길 가능성이 있으며, 이를 해결하기 위한 기술적 지원이 필요합니다. 또한, 데이터 정제 룰셋의 세부 항목에 대한 합의가 아직 이루어지지 않아 추가적인 논의가 필요합니다.\n\n## 5) 차주 계획\n다음 주에는 I/O 병목 현상을 해결하기 위한 기술적 방안을 구체화하고, 이를 테스트할 계획입니다. 또한, 데이터 정제 룰셋에 대한 최종 합의를 도출하고, 이를 바탕으로 정제 작업을 시작할 예정입니다. 추가적으로, 데이터 수집의 효율성을 높이기 위한 방안을 모색할 것입니다.
29	1	2025-09-24 06:12:43.241856	박현규	bakhg@skax.co.kr	# 업무 1: SK하이닉스 스마트 팹 예지보전 플랫폼 주간 보고서\n\n## 1) 주간 요약\n이번 주에는 SK하이닉스 스마트 팹 예지보전 플랫폼의 데이터 수집 파이프라인 구축과 이상 탐지 모델 PoC 범위를 확정하는 데 중점을 두었습니다. 데이터 수집 파이프라인은 설비 PLC 로그, 공정 이력 MES, 장비 이벤트 로그를 소스로 하여 Edge Collector, Kafka, Stream Processor, 시계열DB로 구성되었습니다. 이상 탐지 모델 PoC는 노광/식각 2개 라인과 주요 센서 12종을 대상으로 하며, Conv-VAE와 STL+MAD 기법을 사용하여 알람 검출 시점, 과검출율 등을 평가합니다. 운영 대시보드는 Grafana와 TimescaleDB를 연동하여 라인별 설비가동률, 알람 Top-5 등을 모니터링합니다. 이번 주의 목표는 '수집 파이프라인 최소 동작'과 '이상 탐지 PoC 첫 결과' 확보입니다.\n\n## 2) 사람별 주요 산출물\n- 윤소현: OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결\n- 서은수: 벤더 로그 사양 수합 및 표준 변환 스키마 확정\n- 박현규: Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인 리포트 작성\n\n## 3) 협업 내역\n참여자들은 Slack과 Outlook을 통해 주간 킥오프 회의를 진행하며, Notion에 회의록과 진행 상황을 기록했습니다. 윤소현은 OPC-UA 커넥터 PoC를 통해 실시간 스트림 연결을 담당했고, 서은수는 벤더 로그 사양을 수합하여 표준 변환 스키마를 확정했습니다. 박현규는 Conv-VAE 학습 스크립트를 작성하고 베이스라인 리포트를 작성하여 팀과 공유했습니다.\n\n## 4) 리스크/이슈\n장비 로그 포맷의 비표준화로 인해 ETL 과정에서 지연이 발생할 가능성이 큽니다. 이를 해결하기 위해 공통 스키마(v1)를 우선 도입하고 점진적으로 확장할 계획입니다. 또한, SK하이닉스 내부 보안 정책에 따른 네트워크 포트/방화벽 승인 절차가 필요하며, 담당자에게 이를 전달받았습니다.\n\n## 5) 차주 계획\n다음 주에는 라인 확대와 알람 정책 튜닝 초안을 수립할 계획입니다. 이를 위해 데이터 수집 파이프라인의 안정성을 검토하고, 이상 탐지 모델의 성능을 평가하여 개선점을 도출할 예정입니다. 또한, 변환 스키마의 필수/옵션 필드 목록을 확정하고, 샘플 데이터셋의 1차 라벨 점검을 진행할 계획입니다.
30	1	2025-09-24 06:15:24.744	박현규	bakhg@skax.co.kr	# 업무 1: SK하이닉스 스마트 팹 예지보전 플랫폼 주간 보고서\n\n## 1) 주간 요약\n이번 주 SK하이닉스 스마트 팹 예지보전 플랫폼 관련 진행 상황은 다음과 같습니다. 센서 데이터 수집 파이프라인의 구조가 확정되었으며, 노광 및 식각 두 개 라인에서 PoC를 시작하기로 결정했습니다. 주요 센서 12종에 대해 이상 탐지 모델을 적용할 계획이며, Conv-VAE 기반의 시계열 Reconstruction과 STL+MAD를 비교하여 임계값을 설정할 예정입니다. 또한, 운영 대시보드의 지표와 기술 스택이 확정되었습니다. 이번 주의 목표는 수집 파이프라인의 최소 동작과 이상 탐지 PoC의 첫 결과를 확보하는 것이었습니다.\n\n## 2) 사람별 주요 산출물\n- 서은수: 벤더 로그 사양 수합 및 표준 변환 스키마 확정\n- 박현규: Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인 리포트 작성\n- 윤소현: OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결\n\n## 3) 협업 내역\n참여자들은 Slack, Notion, Outlook, OneDrive를 통해 협업을 진행했습니다. 서은수는 벤더 로그 사양을 수합하고 표준 변환 스키마를 확정하는 데 주력했으며, 박현규는 Conv-VAE 모델의 학습 스크립트를 작성하고 성능을 평가하는 베이스라인 리포트를 작성했습니다. 윤소현은 OPC-UA 커넥터를 통해 3개 설비의 실시간 스트림 연결을 담당했습니다. 이들은 주간 킥오프 회의를 통해 진행 상황을 공유하고, 각자의 작업 내용을 조율했습니다.\n\n## 4) 리스크/이슈\n장비 로그 포맷의 비표준화로 인해 ETL 과정에서 지연이 발생할 가능성이 큽니다. 이를 해결하기 위해 공통 스키마(v1)를 우선 도입하고 점진적으로 확장할 계획입니다. 또한, SK하이닉스 내부 보안 정책에 따른 네트워크 포트 및 방화벽 승인 절차가 필요하며, 관련 담당자에게 전달되었습니다.\n\n## 5) 차주 계획\n다음 주에는 라인 확대와 알람 정책 튜닝 초안을 수립할 예정입니다. 이를 위해 데이터 수집 파이프라인의 안정성을 높이고, PoC 결과를 바탕으로 알람 검출의 선행시간과 과검출률을 개선하는 작업이 필요합니다. 또한, 변환 스키마 v1의 필수 및 옵션 필드 목록을 확정하고, 샘플 데이터셋의 1차 라벨 점검을 진행할 계획입니다.
31	1	2025-09-24 06:25:48.411496	박현규	bakhg@skax.co.kr	# 업무 1: SK하이닉스 스마트 팹 예지보전 플랫폼 주간 보고서\n\n## 1) 주간 요약\n\n**Task 1: SK하이닉스 스마트 팹 예지보전 플랫폼 진행 상황**\n\n- **센서 샘플 JSON 요청**: Conv-VAE 입력 시퀀스 길이 조정을 위해 필요.\n- **PoC 시작 라인**: 노광/식각 2개 라인에서 시작, 알람 선행 감지 여부 확인 필요.\n- **모델 임계값 설정**: Reconstruction 에러 기반 임계값 설정 후 STL+MAD 비교 예정.\n\n**회의: SK하이닉스 스마트 팹 예지보전 주간 킥오프 (Week of 0922)**\n- **참석자**: 서은수(PM), 윤소현(Data Eng), 박현규(AI Dev)\n- **목적**: 센서 데이터 수집 파이프라인 구축 진행 상황 점검 및 이상 탐지 모델 PoC 범위 확정.\n\n**안건 및 논의 사항**\n\n1. **데이터 수집 파이프라인 구조 확정**\n   - **소스**: 설비 PLC 로그(OPC-UA), 공정 이력 MES, 장비 이벤트 로그(벤더 포맷)\n   - **수집**: Edge Collector → Kafka → Stream Processor(Flink) → 시계열DB(TimescaleDB)\n   - **보조**: Object Storage(S3-호환) 30일 핫/180일 콜드 정책\n   - **보안**: 사내망 전용 VPC, 전송 구간 TLS, PII 없음 확인\n   - **리스크**: 장비 로그 포맷 비표준으로 전처리 표준화 레이어 필요\n\n2. **이상 탐지 모델 PoC 범위**\n   - **대상**: 노광/식각 2개 라인, 주요 센서 12종\n   - **레이블**: 설비 알람/정지 이벤트를 약한 라벨로 활용\n   - **기법**: 시계열 Reconstruction(Conv-VAE) + 변동성 기반 Thresholding, STL 분해 + MAD 비교\n   - **평가**: 알람 검출 시점, 과검출율, 주당 알람 건수 제한 정책\n\n3. **운영 대시보드**\n   - **지표**: 라인별 설비가동률(OEE), 알람 Top-5, 이상Score 분포, 센서별 Drift 트렌드\n   - **기술**: Grafana + TimescaleDB + Alertmanager 연동\n\n**액션아이템**\n- 윤소현: OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결 (9/24)\n- 박현규: Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인 리포트 (9/25)\n- 서은수: 벤더 로그 사양 수합, 표준 변환 스키마 확정 (9/26)\n\n**의존성 및 이슈**\n- 장비 로그 포맷 미통일로 ETL 지연 가능성\n- SK하이닉스 내부 보안 정책에 따른 네트워크 포트/방화벽 승인 절차 필요\n\n**추가 논의**\n- Kafka 파티션 수 초기 6, 트래픽 증가 시 12까지 확장 계획\n- TimescaleDB 하이퍼테이블 청크 1일, 보조 인덱스 구성\n- 변환 스키마 v1 필수/옵션 필드 목록 확정 및 샘플 데이터셋 라벨 점검 일정 확정\n\n**성능 목표**\n- 쓰기 TPS 2배 개선, 대시보드 렌더링 지연 1.5초 이내 유지\n- 알람 과검출률 15% 이하, 선행시간 +5분 이상 확보\n\n**결론**\n- 이번 주 목표: '수집 파이프라인 최소 동작'과 '이상 탐지 PoC 첫 결과' 확보\n- 다음 주 목표: 라인 확대와 알람 정책 튜닝 초안 수립\n\n## 2) 사람별 주요 산출물\n\n- **박현규**: Conv-VAE 기본 구조 학습 스크립트 및 베이스라인 리포트\n- **서은수**: 벤더 로그 사양 수합 및 표준 변환 스키마 확정\n- **윤소현**: OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결\n\n## 3) 협업 내역\n\n- **Slack/Notion/Outlook/OneDrive 기록 기반 협업 내역 정리**: 각종 협업 도구를 통해 실시간으로 진행 상황 공유 및 피드백 교환.\n\n## 4) 리스크/이슈\n\n- **장비 로그 포맷 미통일**: ETL 지연 가능성, 전처리 표준화 레이어 필요.\n- **보안 승인 절차**: SK하이닉스 내부 보안 정책에 따른 네트워크 포트/방화벽 승인 절차 필요.\n\n## 5) 차주 계획\n\n- **후속 작업 및 개선점**:\n  - 수집 파이프라인 안정화 및 성능 최적화\n  - PoC 결과 분석 및 모델 개선\n  - 알람 정책 튜닝 초안 수립 및 테스트\n\n(기간: 2025-09-22T09:14:22 ~ 2025-09-27T09:14:22)
32	1	2025-09-24 06:34:15.18509	박현규	bakhg@skax.co.kr	# 업무 1: SK하이닉스 스마트 팹 예지보전 플랫폼 주간 보고서\n\n## 1) 주간 요약\n\n**Task 1 (SK하이닉스 스마트 팹 예지보전 플랫폼) 관련 진행 상황 요약:**\n\n- **센서 샘플 JSON 요청:** Conv-VAE 입력 시퀀스 길이 조정을 위해 센서 샘플 JSON 요청.\n- **PoC 시작:** 노광/식각 2개 라인부터 PoC 시작, 알람 선행 감지 여부 확인 필요.\n- **모델 임계값 설정:** Reconstruction 에러 기반 임계값 설정 후 STL+MAD 비교 예정.\n\n**회의명:** SK하이닉스 스마트 팹 예지보전 주간 킥오프(Week of 0922)  \n**참석자:** 서은수(PM), 윤소현(Data Eng), 박현규(AI Dev)  \n**목적:** 센서 데이터 수집 파이프라인 구축 진행 상황 점검 및 이상 탐지 모델 PoC 범위 확정.\n\n**[안건1] 데이터 수집 파이프라인 구조 확정**\n- **소스:** 설비 PLC 로그(OPC-UA), 공정 이력 MES, 장비 이벤트 로그(벤더 포맷)\n- **수집:** Edge Collector → Kafka → Stream Processor(Flink) → 시계열DB(TimescaleDB)\n- **보조:** Object Storage(S3-호환) 30일 핫/180일 콜드 정책\n- **보안:** 사내망 전용 VPC + SK AX Bastion, 전송 구간 TLS, PII 없음 확인\n- **리스크:** 장비 로그 포맷 비표준 → 전처리 표준화 레이어 필요\n\n**[안건2] 이상 탐지 모델 PoC 범위**\n- **1차 대상:** 노광/식각 2개 라인, 주요 센서 12종\n- **레이블:** 설비 알람/정지 이벤트를 약한 라벨로 활용\n- **기법:** 시계열 Reconstruction(Conv-VAE) + 변동성 기반 Thresholding, 비교군으로 STL 분해 + MAD 기준치\n- **평가:** 알람 검출 시점, 과검출율, 주당 알람 건수 제한 정책\n\n**[안건3] 운영 대시보드**\n- **지표:** 라인별 설비가동률(OEE), 알람 Top-5, 이상Score 분포, 센서별 Drift 트렌드\n- **기술:** Grafana + TimescaleDB + Alertmanager 연동\n\n**[액션아이템]**\n- **윤소현:** OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결(9/24)\n- **박현규:** Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인 리포트(9/25)\n- **서은수:** 벤더 로그 사양 수합, 표준 변환 스키마 확정(9/26)\n\n**[의존성 및 이슈]**\n- 장비 로그 포맷 미통일로 ETL 지연 가능성 → 공통 스키마(v1) 우선 도입 후 점진 확장\n- SK하이닉스 내부 보안 정책에 따른 네트워크 포트/방화벽 승인 절차 필요\n\n**[추가 논의]**\n- Kafka 파티션 수 초기 6, 일일 1.5배 증가 트래픽 가정 시 12까지 확장 계획\n- TimescaleDB 하이퍼테이블 청크 1일, 보조 인덱스는 line, tool_id, sensor 조합으로 구성\n- 변환 스키마 v1 필수/옵션 필드 목록 확정 및 샘플 데이터셋 1차 라벨 점검 일정 확정\n\n**[성능 목표]**\n- 쓰기 TPS 2배 개선, 대시보드 렌더링 지연 1.5초 이내 유지\n- 알람 과검출률 15% 이하, 선행시간 +5분 이상 확보 목표\n\n**[결론]**\n- 이번 주 목표는 '수집 파이프라인 최소 동작'과 '이상 탐지 PoC 첫 결과' 확보.\n- 다음 주 목표는 라인 확대와 알람 정책 튜닝 초안 수립.\n\n## 2) 사람별 주요 산출물\n\n- **서은수:** 벤더 로그 사양 수합, 표준 변환 스키마 확정\n- **박현규:** Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인 리포트\n- **윤소현:** OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결\n\n## 3) 협업 내역\n\n- **Slack/Notion/Outlook/OneDrive 기록 기반 협업 내역 정리:** 주간 회의 및 액션 아이템 공유, 진행 상황 업데이트 및 피드백 교환.\n\n## 4) 리스크/이슈\n\n- **문제점:** 장비 로그 포맷 비표준으로 인한 ETL 지연 가능성\n- **리스크:** SK하이닉스 내부 보안 정책에 따른 네트워크 포트/방화벽 승인 절차 필요\n- **해결 필요 사항:** 공통 스키마(v1) 도입 및 점진 확장, 보안 승인 절차 진행\n\n## 5) 차주 계획\n\n- **후속 작업 및 개선점:**\n  - 라인 확대 및 알람 정책 튜닝 초안 수립\n  - 공통 스키마(v1) 도입 및 ETL 프로세스 개선\n  - 보안 승인 절차 완료 및 네트워크 설정 최적화\n\n(기간: 2025-09-22T09:14:22 ~ 2025-09-27T09:14:22)
33	4	2025-09-24 06:34:21.075596	박현규	bakhg@skax.co.kr	# 업무 4: 카카오브레인 LLM 데이터 파이프라인 주간 보고서\n\n## 1) 주간 요약\n이번 주에는 Task 4와 관련하여 카카오브레인과 SK AX 간의 주간 싱크 회의가 진행되었습니다. 회의에서는 데이터 수집 현황, 정제 룰셋, I/O 병목 현상에 대한 논의가 이루어졌습니다. 회의는 온라인으로 Teams를 통해 진행되었으며, 윤소현과 박현규가 참석하였습니다.\n\n## 2) 사람별 주요 산출물\n- 박현규: 회의에서 논의된 내용을 바탕으로 데이터 파이프라인 개선 방안 초안 작성.\n\n## 3) 협업 내역\n이번 주에는 Slack, Notion, Outlook, OneDrive를 활용하여 협업이 이루어졌습니다. 각 플랫폼을 통해 회의 일정 조율, 문서 공유, 피드백 수렴 등의 활동이 진행되었습니다.\n\n## 4) 리스크/이슈\n현재 데이터 파이프라인에서 I/O 병목 현상이 발생하고 있으며, 이를 해결하기 위한 방안이 필요합니다. 또한, 데이터 정제 룰셋에 대한 추가적인 검토가 요구됩니다.\n\n## 5) 차주 계획\n다음 주에는 I/O 병목 현상을 해결하기 위한 구체적인 방안을 마련하고, 데이터 정제 룰셋을 최종 확정할 계획입니다. 이를 위해 추가적인 회의와 검토 작업이 예정되어 있습니다.
34	1	2025-09-24 07:00:00.849372	박현규	bakhg@skax.co.kr	# 업무 1: SK하이닉스 스마트 팹 예지보전 플랫폼 주간 보고서\n\n## 1) 주간 요약\n\n### Task 1: SK하이닉스 스마트 팹 예지보전 플랫폼 진행 상황\n- **센서 샘플 JSON**: Conv-VAE 입력 시퀀스 길이 조정을 위해 샘플 요청.\n- **PoC 시작**: 노광/식각 2개 라인에서 시작, 알람 선행 감지 여부 확인 필수.\n- **모델 기법**: Reconstruction 에러 기반 임계값 설정, STL+MAD 비교 검토.\n- **회의**: SK하이닉스 스마트 팹 예지보전 주간 킥오프(Week of 0922)\n  - **참석자**: 서은수(PM), 윤소현(Data Eng), 박현규(AI Dev)\n  - **목적**: 센서 데이터 수집 파이프라인 구축 진행 상황 점검 및 이상 탐지 모델 PoC 범위 확정.\n\n#### 안건 1: 데이터 수집 파이프라인 구조 확정\n- **소스**: 설비 PLC 로그(OPC-UA), 공정 이력 MES, 장비 이벤트 로그(벤더 포맷)\n- **수집**: Edge Collector → Kafka → Stream Processor(Flink) → TimescaleDB\n- **보조**: Object Storage(S3-호환) 30일 핫/180일 콜드 정책\n- **보안**: 사내망 전용 VPC + SK AX Bastion, 전송 구간 TLS, PII 없음 확인\n- **리스크**: 장비 로그 포맷 비표준으로 전처리 표준화 레이어 필요\n\n#### 안건 2: 이상 탐지 모델 PoC 범위\n- **1차 대상**: 노광/식각 2개 라인, 주요 센서 12종\n- **레이블**: 설비 알람/정지 이벤트를 약한 라벨로 활용\n- **기법**: 시계열 Reconstruction(Conv-VAE) + 변동성 기반 Thresholding, STL 분해 + MAD 기준치\n- **평가**: 알람 검출 시점, 과검출율, 주당 알람 건수 제한 정책\n\n#### 안건 3: 운영 대시보드\n- **지표**: 라인별 설비가동률(OEE), 알람 Top-5, 이상 Score 분포, 센서별 Drift 트렌드\n- **기술**: Grafana + TimescaleDB + Alertmanager 연동\n\n#### 액션아이템\n- **윤소현**: OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결(9/24)\n- **박현규**: Conv-VAE 기본 구조 학습 스크립트 작성 및 베이스라인 리포트(9/25)\n- **서은수**: 벤더 로그 사양 수합, 표준 변환 스키마 확정(9/26)\n\n#### 의존성 및 이슈\n- 장비 로그 포맷 미통일로 ETL 지연 가능성 → 공통 스키마(v1) 우선 도입\n- 네트워크 포트/방화벽 승인 절차 필요\n\n#### 추가 논의\n- Kafka 파티션 수 확장 계획\n- TimescaleDB 하이퍼테이블 청크 및 보조 인덱스 구성\n- 변환 스키마 v1 필수/옵션 필드 목록 및 샘플 데이터셋 라벨 점검 일정 확정\n\n#### 성능 목표\n- 쓰기 TPS 2배 개선, 대시보드 렌더링 지연 1.5초 이내 유지\n- 알람 과검출률 15% 이하, 선행시간 +5분 이상 확보\n\n#### 결론\n- 이번 주 목표: '수집 파이프라인 최소 동작'과 '이상 탐지 PoC 첫 결과' 확보\n- 다음 주 목표: 라인 확대와 알람 정책 튜닝 초안 수립\n\n## 2) 사람별 주요 산출물\n\n- **서은수**: 벤더 로그 사양 수합 및 표준 변환 스키마 확정\n- **박현규**: Conv-VAE 기본 구조 학습 스크립트 및 베이스라인 리포트 작성\n- **윤소현**: OPC-UA 커넥터 PoC로 3개 설비 실시간 스트림 연결\n\n## 3) 협업 내역\n\n- **Slack/Notion/Outlook/OneDrive** 기록 기반으로 협업 내역 정리\n\n## 4) 리스크/이슈\n\n- **문제점**: 장비 로그 포맷 미통일로 인한 ETL 지연 가능성\n- **리스크**: 네트워크 포트/방화벽 승인 절차 필요\n- **해결 필요 사항**: 공통 스키마(v1) 우선 도입 및 점진 확장\n\n## 5) 차주 계획\n\n- **후속 작업**: 라인 확대 및 알람 정책 튜닝 초안 수립\n- **개선점**: 데이터 수집 파이프라인 및 이상 탐지 모델 최적화\n\n(기간: 2025-09-22T09:14:22 ~ 2025-09-27T09:14:22)
\.


--
-- Data for Name: slack; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.slack (id, receiver, sender, "timestamp", task_id, content, embedding) FROM stdin;
54	서은수	조성호	2025-09-23 14:34:10	2	확장 승인합니다. 다만 latency 모니터링 강화해 주세요./n	\N
1	윤소현	서은수	2025-09-22 09:12:03	1	이번 주 SK하이닉스 스마트 팹 예지보전 킥오프 일정 확인했어요/n수집 파이프라인 초안 공유 가능합니다.	\N
2	서은수	윤소현	2025-09-22 09:13:41	1	좋아요, Kafka 토픽 이름 skhynix.smartfab.sensors 맞죠?/nOPC-UA 커넥터 상태도 알려주세요.	\N
3	박현규	윤소현	2025-09-22 09:14:22	1	센서 샘플 JSON 조금 받을 수 있을까요?/nConv-VAE 입력 시퀀스 길이 맞춰보려 합니다.	\N
4	윤소현	박현규	2025-09-22 09:15:10	1	여기 1분치 샘플/n필드: ts,line,tool_id,sensor,value/vendor별 포맷 달라 스키마 v1로 정규화 중.	\N
5	서은수	박현규	2025-09-22 09:18:55	1	PoC는 노광/식각 2개 라인부터 시작/n알람 선행 감지 여부 꼭 확인해주세요.	\N
6	박현규	서은수	2025-09-22 10:27:12	1	Reconstruction 에러 기반 임계값 먼저 잡아보고/nSTL+MAD도 비교로 돌려볼게요.	\N
7	윤소현	서은수	2025-09-22 11:03:02	1	TimescaleDB 파티셔닝 구성했고/nS3 라이프사이클 핫30/콜드180 반영했습니다.	\N
8	박현규	윤소현	2025-09-22 13:22:44	1	샘플에 드롭 값이 있어요/nNaN 처리 규칙 합의 필요합니다.	\N
9	윤소현	박현규	2025-09-22 13:24:18	1	드롭은 30초 이내 선형보간, 초과는 마스킹으로 가시죠/n스키마 문서에 추가하겠습니다.	\N
10	서은수	윤소현	2025-09-22 15:01:57	1	벤더 로그 표준화 이슈 오늘 SK하이닉스와 콜 잡아둘게요/n자료는 Notion에 올렸습니다.	\N
11	박현규	서은수	2025-09-22 16:40:31	1	초기 Conv-VAE 학습 결과 과검출 조금 있어요/n임계치 조정해보겠습니다.	\N
12	윤소현	박현규	2025-09-23 09:02:11	1	Kafka→TimescaleDB 적재 튜닝으로 쓰기 TPS 2배 개선했습니다.	\N
13	박현규	윤소현	2025-09-23 09:04:33	1	굿!/n윈도우 60→90초 실험도 곧 시도해볼게요.	\N
14	서은수	박현규	2025-09-23 10:18:05	1	PoC 리포트에 ‘알람 피로도’ 섹션 추가 부탁/n운영팀이 중요하대요.	\N
15	박현규	서은수	2025-09-23 11:37:22	1	주당 알람 상한 정책/임계값 ROC 비교 넣겠습니다/nSK AX 템플릿 적용할게요.	\N
16	윤소현	서은수	2025-09-23 14:53:50	1	OPC-UA 연결 3대 설비 스트림 정상/n라인 태그 및 변환 완료했습니다.	\N
17	서은수	윤소현	2025-09-23 15:07:11	1	좋아요!/n수집 파이프라인 다이어그램 OneDrive에 버전 올려주세요.	\N
18	박현규	윤소현	2025-09-23 16:40:09	1	NaN 처리 후 재학습에서 과검출 12% 감소/n선행시간 +6분 개선입니다.	\N
19	서은수	박현규	2025-09-24 09:10:26	1	내일 현업 미팅 전 데모 대시보드 스샷 3장 부탁해요/n알람Top5, 이상Score, OEE.	\N
20	박현규	서은수	2025-09-24 09:12:41	1	준비 가능해요/nSK하이닉스 라벨 넣은 화면 제공하겠습니다.	\N
21	윤소현	서은수	2025-09-24 11:31:55	1	Object Storage 라이프사이클 설정 반영 완료/n감사 추적용 쿼리 정리.	\N
22	박현규	윤소현	2025-09-24 13:05:07	1	벤더C 로그 스케일 다릅니다/n정규화 테이블 룰 추가할까요?	\N
23	윤소현	박현규	2025-09-24 13:06:20	1	추가하죠/n벤더A/B/C 단위/배율 맵 표 정의해서 변환에 반영.	\N
24	서은수	윤소현	2025-09-24 17:42:38	1	내부 보고 상단 한 줄 설명 필요/n‘스마트 팹 예지보전’ 개요요.	\N
25	윤소현	서은수	2025-09-24 17:44:03	1	“SK하이닉스 생산라인 설비 이상을 사전 감지해 다운타임 최소화하는 예지보전 PoC”로 하겠습니다.	\N
26	박현규	서은수	2025-09-25 09:21:57	1	새 모델에서 False Positive 감소/n임계값 0.82 근처가 균형 좋아요.	\N
27	서은수	박현규	2025-09-25 11:00:22	1	좋습니다/n데모 때 임계값 슬라이더 조정 가능한 UI도 언급합시다.	\N
28	윤소현	박현규	2025-09-25 14:33:40	1	TimescaleDB 인덱스 재구성으로 조회 40% 개선/n대시보드 렌더 지연 완화.	\N
29	서은수	윤소현	2025-09-26 09:08:12	1	오늘 11시 리허설/n샘플 알람 3건만 준비 부탁합니다.	\N
30	박현규	서은수	2025-09-26 12:27:59	1	리허설 OK/n15시 실제 데모 자료 갱신했고 회사명 표기(SK하이닉스, SK AX) 확인했습니다.	\N
31	박범준	서은수	2025-09-22 09:05:12	2	현대자동차 디지털 트윈 대시보드 첫 배포 일정 오늘 17시로 잡을까요?/n	\N
32	서은수	박범준	2025-09-22 09:06:40	2	좋습니다. 시뮬레이션 엔진 편차 리포트 점검 후 SK AX 명의로 공유하겠습니다./n	\N
33	조성호	서은수	2025-09-22 09:08:02	2	울산 2라인 OPC-UA 태그 최신본 OneDrive에 올려둘게요./n	\N
34	박범준	조성호	2025-09-22 10:31:11	2	시뮬 예측 vs 실제 생산량 편차가 오전 9시 타임에서 6%까지 발생. 센서 결측 있었나요?/n	\N
35	조성호	박범준	2025-09-22 10:32:27	2	압력 센서 2개에서 스파이크 있었습니다. 필터 파라미터 튜닝 중이에요./n	\N
36	서은수	전체	2025-09-22 11:50:42	2	현대자동차 회의 오후 4시 진행 예정. latency 로그 공유 부탁드려요./n	\N
37	조성호	서은수	2025-09-22 13:15:23	2	latency 로그 보니 야간 시간대 동기화 지연이 3초까지 치솟습니다./n	\N
38	박범준	조성호	2025-09-22 13:16:45	2	3초면 KPI 오차가 커집니다. Kafka 파티션 증설을 서은수님께 제안드리죠./n	\N
39	서은수	박범준	2025-09-22 13:18:01	2	네, SK AX 인프라팀과 협의하겠습니다. 우선 증설안 준비해 주세요./n	\N
40	조성호	박범준	2025-09-22 14:40:11	2	울산 공장 태그 수는 1,200개 이상입니다. 표준화 룰 다시 정리해야겠어요./n	\N
41	박범준	조성호	2025-09-22 15:02:22	2	표준화 룰 문서 Notion에 올리겠습니다. 현대자동차 제조혁신팀도 참조 가능하도록 할게요./n	\N
42	조성호	서은수	2025-09-22 16:05:33	2	대시보드 KPI 지표 중 ‘에너지 사용량’이 아직 반영 안 됐습니다./n	\N
43	서은수	조성호	2025-09-22 16:07:10	2	추가 반영 부탁드립니다. 현대자동차 측에서도 강하게 요청했어요./n	\N
44	박범준	서은수	2025-09-22 17:22:41	2	대시보드 초안 배포 완료했습니다. OEE와 불량률 지표 확인해 주세요./n	\N
45	서은수	박범준	2025-09-22 17:25:00	2	확인했습니다. 이번 배포안에 ‘현대자동차 울산 공장’ 라벨도 추가합시다./n	\N
46	조성호	박범준	2025-09-23 09:11:54	2	OPC-UA 게이트웨이 재기동했습니다. 센서 신호 안정화 확인 부탁드립니다./n	\N
47	박범준	조성호	2025-09-23 09:12:58	2	안정화 확인했습니다. latency는 여전히 1.2초 수준입니다./n	\N
48	서은수	전체	2025-09-23 09:14:30	2	latency 1초 이내로 줄여야 합니다. 대응책 제시 부탁드립니다./n	\N
49	조성호	서은수	2025-09-23 11:43:18	2	IoT 게이트웨이에 버퍼링 레벨 조정해서 latency 0.8초까지 줄였습니다./n	\N
50	박범준	서은수	2025-09-23 11:45:22	2	좋습니다. Kafka 파티션 증설안도 곧 제출할게요./n	\N
51	서은수	박범준	2025-09-23 13:01:45	2	현대자동차와 공유할 PoC 리포트에 KPI 그래프 캡처 넣어주세요./n	\N
52	박범준	서은수	2025-09-23 13:02:59	2	넣겠습니다. KPI는 OEE, 생산량, 불량률, 에너지 사용량 포함./n	\N
53	조성호	서은수	2025-09-23 14:33:00	2	울산 2라인에서 태그 추가 요청 들어왔습니다. 50개 확장 필요./n	\N
55	박범준	조성호	2025-09-23 15:15:42	2	latency 모니터링 대시보드 추가했습니다. 지연 구간 색상 표시./n	\N
56	조성호	박범준	2025-09-24 09:20:33	2	구형 PLC 태그에서 단위 불일치 발견했습니다. 변환 룰 테이블에 추가하겠습니다./n	\N
57	박범준	조성호	2025-09-24 09:22:44	2	좋습니다. 표준화 테이블 최신본을 OneDrive에 올려주세요./n	\N
58	조성호	서은수	2025-09-24 10:55:00	2	latency 개선 후 KPI 편차가 3% 이하로 줄었습니다./n	\N
59	서은수	박범준	2025-09-25 11:33:23	2	내일 현대자동차 회의 리허설 준비합시다. 발표 자료 초안 부탁해요./n	\N
60	박범준	서은수	2025-09-25 11:34:30	2	발표 자료 초안 완성했습니다. SK AX와 현대자동차 로고 넣어두었습니다./n	\N
61	서은수	정도현	2025-09-22 09:12:00	3	신한은행 API 테스트 서버 열렸는지 확인했어?/n	\N
62	정도현	서은수	2025-09-22 09:13:00	3	네, 정상 동작합니다. ERP 쪽 호출까지는 문제없어요./n	\N
63	서은수	정도현	2025-09-22 09:20:00	3	어제 시뮬레이션에서 응답 지연 있었잖아. 원인 추적했어?/n	\N
64	정도현	서은수	2025-09-22 09:21:00	3	ERP 게이트웨이 큐에 메시지 쌓여 있더라구요. 비동기 처리로 개선 중이에요./n	\N
65	서은수	정도현	2025-09-22 10:05:00	3	ERP 매핑 로직 최종 확정된 거지?/n	\N
66	정도현	서은수	2025-09-22 10:07:00	3	네, 신한은행 계좌조회 JSON과 ERP 테이블 매핑 완료했습니다./n	\N
67	서은수	정도현	2025-09-22 10:30:00	3	오늘 오후엔 이체 API 연동 테스트하자./n	\N
68	정도현	서은수	2025-09-22 10:31:00	3	좋습니다. 인증 토큰 만료만 주의하면 될 듯합니다./n	\N
69	서은수	정도현	2025-09-22 11:45:00	3	보안 규정 검토했어?/n	\N
70	정도현	서은수	2025-09-22 11:46:00	3	아직이에요. 금융보안원 가이드 오늘 밤에 확인하고 공유할게요./n	\N
71	서은수	정도현	2025-09-23 09:10:00	3	트래픽 시뮬 돌렸는데 신한은행 API 응답이 2초 이상 걸리네./n	\N
72	정도현	서은수	2025-09-23 09:12:00	3	ERP 서버랑 신한은행 서버 간 네트워크 병목 같아요. AWS 리전 확인해볼게요./n	\N
73	서은수	정도현	2025-09-23 09:15:00	3	로드밸런서 로그도 한번 체크해줘./n	\N
74	정도현	서은수	2025-09-23 09:16:00	3	넵, SLA 문서도 확인할게요./n	\N
75	서은수	정도현	2025-09-23 15:20:00	3	ERP 매핑 관련해서 신한은행 담당자랑 통화했는데 계좌 상태 필드 추가 필요하대./n	\N
76	정도현	서은수	2025-09-23 15:21:00	3	DB 스키마 업데이트 하겠습니다./n	\N
77	서은수	정도현	2025-09-24 09:00:00	3	보안 키 관리는 어떻게 하고 있어?/n	\N
78	정도현	서은수	2025-09-24 09:02:00	3	AWS Secret Manager 사용 중이에요./n	\N
79	서은수	정도현	2025-09-24 10:20:00	3	모니터링 알람도 추가해놔야겠네./n	\N
80	정도현	서은수	2025-09-24 10:21:00	3	CloudWatch 알람 세팅할게요./n	\N
81	서은수	정도현	2025-09-25 14:10:00	3	어제 테스트에서 계좌조회 100건 중 2건 실패했어./n	\N
82	정도현	서은수	2025-09-25 14:11:00	3	신한은행 API 응답 지연 원인으로 보여요. 재시도 로직 넣겠습니다./n	\N
83	서은수	정도현	2025-09-25 14:12:00	3	3회까지 재시도 적용해줘./n	\N
84	서은수	정도현	2025-09-26 09:00:00	3	오늘 최종 리포트 정리하자./n	\N
85	정도현	서은수	2025-09-26 09:01:00	3	네, 로그 분석 마무리하고 올리겠습니다./n	\N
86	서은수	정도현	2025-09-26 11:00:00	3	OneDrive에 SQL 스크립트 올려놨으니 확인해봐./n	\N
87	정도현	서은수	2025-09-26 11:01:00	3	봤습니다! 매핑 테이블 포함돼 있네요./n	\N
88	서은수	정도현	2025-09-26 11:05:00	3	좋아, 오늘 회의에서 공유하자./n	\N
89	정도현	서은수	2025-09-26 15:00:00	3	신한은행 SLA 문서 읽어보니 트래픽 제한이 있네요. 조정해야 할 듯합니다./n	\N
90	서은수	정도현	2025-09-26 15:05:00	3	그럼 우리 ERP 쪽 배치 로직도 수정합시다./n	\N
91	윤소현	박현규	2025-09-22 09:12:05	4	이번 주 Kakao Brain LLM 파이프라인 시작!/n크롤러 rate-limit 테스트 먼저 돌려볼게.	\N
92	박현규	윤소현	2025-09-22 09:13:47	4	좋아!/n정제 룰셋 베이스라인도 같이 시작함./n광고 패턴 제거 목록 정리 중.	\N
93	윤소현	박현규	2025-09-22 10:21:10	4	robots.txt 예외 케이스 발견./n크롤링 스킵 리스트에 추가했어.	\N
94	박현규	윤소현	2025-09-22 11:05:33	4	문장 분리기 테스트 중인데 약어 때문에 경계가 깨져./n룰 하나 더 추가할게.	\N
95	윤소현	박현규	2025-09-22 13:40:58	4	배치 크롤링 1회차 완료!/n원시 텍스트 12GB 확보했어./n스토리지 I/O 병목도 확인됨.	\N
96	박현규	윤소현	2025-09-22 14:12:11	4	12GB? 꽤 많네./nSSD 버퍼링 없으면 계속 병목 날 듯.	\N
97	윤소현	박현규	2025-09-22 15:20:44	4	SSD 캐시 붙여서 테스트 중./n쓰기 TPS 2배 개선됨.	\N
98	박현규	윤소현	2025-09-22 15:25:37	4	굿!/n정제 모듈이랑 같이 연결해봐야겠다.	\N
99	윤소현	박현규	2025-09-22 16:10:23	4	중복 문서 제거 로직 추가했어./n해시 기반 필터링.	\N
100	박현규	윤소현	2025-09-22 17:32:12	4	PII 마스킹 룰 정리 중./n전화번호, 주민번호 패턴 우선 적용.	\N
101	윤소현	박현규	2025-09-23 09:05:41	4	오늘 오전은 크롤러 병렬화 확인할게./n멀티프로세싱 풀 적용.	\N
102	박현규	윤소현	2025-09-23 09:22:14	4	정제 로그 보니까 광고 문구 필터 잘 작동함./n‘구매하기’ 패턴 거의 제거.	\N
103	윤소현	박현규	2025-09-23 10:12:56	4	좋네!/n이제 profanity 필터링 추가하자.	\N
104	박현규	윤소현	2025-09-23 11:01:37	4	욕설 리스트 베이스라인 커밋했어./n정제 속도 15% 개선됨.	\N
105	윤소현	박현규	2025-09-23 14:23:18	4	샘플셋 100MB 정제 완료./n텍스트 길이 분포 확인했어.	\N
106	박현규	윤소현	2025-09-23 14:45:32	4	길이 10자 이하 문장은 대부분 제거하는 게 좋아보여./n분석 결과 공유할게.	\N
107	윤소현	박현규	2025-09-23 15:55:44	4	스토리지 전송 지연 다시 확인./nI/O 병목 여전함.	\N
108	박현규	윤소현	2025-09-23 16:02:10	4	멀티파트 업로드 옵션 켜봐./n예전에 SK AX 클라우드에서 비슷한 문제 있었음.	\N
109	윤소현	박현규	2025-09-24 09:11:58	4	멀티파트 적용 후 TPS 1.8배 개선./n아직 100%는 아님.	\N
110	박현규	윤소현	2025-09-24 09:30:14	4	충분히 쓸만해./n이번 주 데모에선 이 정도 성능이면 돼.	\N
111	윤소현	박현규	2025-09-24 11:10:07	4	대시보드에 크롤링 속도 그래프 추가했어./n어제랑 비교 그래프도.	\N
112	박현규	윤소현	2025-09-24 11:15:42	4	좋아./n데모용 스샷 3장 뽑아둘게.	\N
113	윤소현	박현규	2025-09-24 13:02:55	4	벤더 로그 포맷 차이 많음./n공통 스키마 v1 적용 중.	\N
114	박현규	윤소현	2025-09-24 13:17:28	4	스키마 매핑표 작성 중./nKakao Brain 팀에도 공유해야겠어.	\N
115	윤소현	박현규	2025-09-25 09:12:39	4	샘플셋 1GB 정제 완료./n정제 규칙 정상 작동.	\N
116	박현규	윤소현	2025-09-25 09:22:01	4	1GB면 충분히 데모 가능하겠다./nLLM 입력 토큰 분포 확인할게.	\N
117	윤소현	박현규	2025-09-25 11:18:54	4	데이터 파이프라인 다이어그램 업데이트./nOneDrive에 올려둘게.	\N
118	박현규	윤소현	2025-09-25 11:21:46	4	확인할게./n그래프 노드별 처리율도 같이 표시됐어?	\N
119	윤소현	박현규	2025-09-26 09:08:13	4	오늘 11시 리허설 예정./n데모용 샘플 알람 준비 완료.	\N
120	박현규	윤소현	2025-09-26 17:20:45	4	이번 주 마감!/nI/O 병목 완화, 정제 베이스라인, 샘플셋 준비까지 완료.	\N
\.


--
-- Data for Name: task; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.task (id, description, embedding, task_uuid) FROM stdin;
1	SK하이닉스 스마트 팹 예지보전 플랫폼	\N	t001
2	현대자동차 디지털 트윈 기반 스마트팩토리	\N	t002
3	신한은행 오픈뱅킹 ERP 연동	\N	t003
4	카카오브레인 LLM 데이터 파이프라인	\N	t004
\.


--
-- Name: employee_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.employee_id_seq', 9, true);


--
-- Name: report_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.report_id_seq', 34, true);


--
-- Name: task_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.task_id_seq', 4, true);


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
-- Name: notion_embedding_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX notion_embedding_idx ON public.notion USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='5');


--
-- Name: onedrive_embedding_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX onedrive_embedding_idx ON public.onedrive USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='5');


--
-- Name: outlook_embedding_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX outlook_embedding_idx ON public.outlook USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='5');


--
-- Name: slack_embedding_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX slack_embedding_idx ON public.slack USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='5');


--
-- PostgreSQL database dump complete
--

