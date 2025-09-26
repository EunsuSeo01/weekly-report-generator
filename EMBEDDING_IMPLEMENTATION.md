# 📊 임베딩 기능 구현 문서

> **구현 날짜:** 2025-09-26
> **목적:** 주간 보고서 자동 생성 후 임베딩 벡터 생성 및 DB 저장

## 🎯 구현 개요

### 핵심 기능
- **올바른 위치**: `/reports/weekly` API에서 보고서 생성 후 **자동 임베딩 생성 및 DB 저장**
- **임베딩 모델**: `jhgan/ko-sbert-nli` (768차원)
- **기존 함수 보존**: 새로운 함수 추가로 기존 로직 유지

### 파이프라인
1. 플랫폼 데이터 수집 및 보고서 생성 (실제 LLM 또는 더미)
2. 한국어 임베딩 모델로 768차원 벡터 생성
3. PostgreSQL vector 타입으로 DB 저장

---

## 🏗️ 아키텍처 변경사항

### 1. API 위치 수정
**변경 전**: `/api/generate-summary`에 임베딩 기능 잘못 구현
**변경 후**: `/reports/weekly`에 올바르게 이동

### 2. 새로 추가된 함수들

#### 🔧 generate_report_with_fallback()
```python
# OpenAI API 키 상태에 따른 자동 분기
- API 키 있음: 실제 LLM 보고서 생성 (기존 함수 호출)
- API 키 없음: 더미 보고서 생성 (개발/테스트용)
```

#### 💾 store_report_embedding_only()
```python
# 이미 저장된 보고서에 임베딩만 추가
- 임베딩 생성 (jhgan/ko-sbert-nli, 768차원)
- PostgreSQL vector 타입으로 UPDATE
```

#### 🔄 insert_report() 수정
```python
# 기존: 저장만
# 수정: 저장 + report_id 반환 (임베딩 저장 시 사용)
```

### 3. 더미 보고서 존재 이유
**목적**: OpenAI API 키 없는 개발/테스트 환경 대응
**⚠️ 중요**: 프로덕션 환경에서는 반드시 제거 필요
**형식**: 실제 보고서와 동일한 구조로 생성하여 의미있는 임베딩 생성

---

## 🔄 새로운 흐름

### `/reports/weekly` API 처리 순서
```python
1. 플랫폼 데이터 수집 (Slack, Notion, Outlook, OneDrive)
2. task_id별 그룹핑
3. generate_report_with_fallback() → 보고서 생성
4. insert_report() → DB 저장 (report_id 반환)
5. store_report_embedding_only() → 임베딩 저장
6. return 결과
```

### API 키 자동 분기
- **API 키 있음**: 실제 OpenAI LLM 보고서 생성
- **API 키 없음**: 더미 보고서 생성 (임시, 제거 예정)

---

## 📋 협업자를 위한 체크리스트

### ✅ 완료된 작업
- [x] 임베딩 기능을 `/reports/weekly`로 이동
- [x] 기존 함수들 보존하며 새 함수 추가
- [x] 768차원 임베딩 생성 및 저장
- [x] API 키 없는 환경에서 더미 보고서 생성

### ⚠️ 프로덕션 배포 시 필수 작업
- [ ] **OpenAI API 키 설정**
- [ ] **더미 보고서 로직 제거** (generate_report_with_fallback 함수 내)
- [ ] 임베딩 기능 정상 작동 확인

### 🔍 테스트 방법
```bash
# 1. /reports/weekly API 테스트
curl -X POST "http://localhost:8001/reports/weekly" \
  -H "Content-Type: application/json" \
  -d '{"platform_ids": {"slack": [1,2]}, "start": "2025-09-22", "end": "2025-09-26", "writer": "테스트", "email": "test@skax.co.kr"}'

# 2. 임베딩 저장 확인
psql -U postgres -h localhost -p 5432 -d weekly_report_db \
  -c "SELECT id, vector_dims(report_embedded) FROM report ORDER BY id DESC LIMIT 3;"
```

---

## 🔧 기술적 세부사항

### DB 스키마
```sql
-- report 테이블
report_embedded public.vector(768)  -- 768차원 임베딩 벡터
```

### 의존성
```python
# 새로 추가된 패키지
sentence-transformers>=2.2.2
huggingface_hub>=0.16.0
```

### 함수별 역할
- `ReportEmbeddingService`: 임베딩 생성 전용 서비스
- `generate_report_with_fallback()`: API 키 상태별 보고서 생성
- `store_report_embedding_only()`: 임베딩만 저장하는 함수
- `insert_report()`: 기존 함수, report_id 반환하도록 수정

---

## 🚨 다음 개발자를 위한 중요 사항

### 1. 프로덕션 배포 전 필수 작업
```python
# generate_report_with_fallback() 함수에서 제거해야 할 부분
if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("OPENAI_A") and len(OPENAI_API_KEY) >= 20:
    return await generate_report_for_task(task_id, platform_data, start_ts, end_ts, session)
else:
    # 이 부분 전체 제거하고 위의 return만 남기기
    print(f"🔄 OpenAI API 키가 없어서 더미 보고서를 생성합니다...")
    # ... 더미 보고서 생성 로직 ...
```

### 2. 코드 위치 및 역할
- **임베딩 기능**: `/reports/weekly` API에 구현됨
- **기존 함수**: 모두 보존, 새 함수만 추가
- **더미 보고서**: 개발용, 프로덕션에서 제거 필요

### 3. 테스트 확인 방법
```sql
-- 임베딩이 제대로 저장되었는지 확인
SELECT id, task_id, vector_dims(report_embedded) as dimension
FROM public.report
WHERE report_embedded IS NOT NULL
ORDER BY id DESC LIMIT 5;
```

**✅ 임베딩 기능 `/reports/weekly`에 올바르게 구현 완료**