import streamlit as st
import requests
from datetime import date, timedelta

# 페이지 설정
st.set_page_config(
    page_title="주간 업무 보고서 생성기",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }

    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 500;
    }

    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }

    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }

    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }

    .report-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        min-height: fit-content;
        height: auto;
        overflow: visible;
    }

    .report-content {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        min-height: fit-content;
        height: auto;
        overflow: visible;
        line-height: 1.6;
        font-size: 14px;
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    .stTextArea > div > div > textarea {
        background-color: #ffffff;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        font-size: 16px;
    }

    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown('<h1 class="main-header">📊 주간 업무 보고서 생성기</h1>', unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.markdown("### ⚙️ 설정")

    # FastAPI 서버 URL
    api_url = st.text_input(
        "FastAPI 서버 URL",
        value="http://localhost:3306",
        help="FastAPI 서버가 실행되고 있는 주소를 입력하세요"
    )

    st.markdown("---")

    # 현재 날짜 정보
    today = date.today()
    st.markdown(f"**📅 오늘 날짜:** {today.strftime('%Y년 %m월 %d일')}")

    # 이번 주 정보
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    st.markdown(f"**📆 이번 주:** {week_start.strftime('%m/%d')} ~ {week_end.strftime('%m/%d')}")

    st.markdown("---")

    # 도움말
    with st.expander("❓ 사용법"):
        st.markdown("""
        1. **보고서 생성**: 버튼을 클릭하여 AI가 보고서를 생성하게 하세요
        2. **결과 확인**: 생성된 보고서를 검토하고 필요시 수정하세요

        **💡 팁:**
        - 이번 주 업무 데이터를 기반으로 자동으로 보고서가 생성됩니다
        """)

# 기능 선택 탭
tab1, tab2 = st.tabs(["📊 이번주 주간 보고서", "🔍 최근 2주 키워드 열람"])

with tab1:
    st.markdown('<h2 class="sub-header">📊 이번주 주간 보고서 생성</h2>', unsafe_allow_html=True)

    # 서버 상태 확인
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        server_status = "🟢 연결됨"
        status_color = "success-box"
    except:
        server_status = "🔴 연결 실패"
        status_color = "error-box"

    st.markdown(f'<div class="{status_color}"><strong>서버 상태:</strong> {server_status}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="info-box">📋 <strong>자동 생성 기능</strong><br>이번 주 업무 데이터를 기반으로 자동으로 주간 보고서를 생성합니다.</div>', unsafe_allow_html=True)

    with col2:
        # 보고서 생성 버튼
        if st.button("📝 이번주 주간 보고서 생성", key="generate_weekly_report", use_container_width=True):
            with st.spinner("🔄 AI가 이번주 업무 데이터를 분석하여 보고서를 생성하고 있습니다..."):
                try:
                    # FastAPI 요청 (user_request 없이)
                    response = requests.post(
                        f"{api_url}/generate-report",
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        report = result.get("report", "")

                        # 세션 상태에 보고서 저장
                        st.session_state.generated_report = report
                        st.success("✅ 주간 보고서가 성공적으로 생성되었습니다!")

                    else:
                        st.error(f"❌ 오류 발생: {response.status_code} - {response.text}")

                except requests.exceptions.Timeout:
                    st.error("⏰ 요청 시간이 초과되었습니다. 다시 시도해주세요.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
                except Exception as e:
                    st.error(f"❌ 예상치 못한 오류가 발생했습니다: {str(e)}")

with tab2:
    st.markdown('<h2 class="sub-header">🔍 최근 2주 키워드 열람</h2>', unsafe_allow_html=True)

    # 키워드 입력 영역
    st.markdown('<div class="info-box">🔍 <strong>키워드 검색 기능</strong><br>최근 2주간의 업무 데이터에서 특정 키워드를 포함한 내용을 검색할 수 있습니다.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        # 키워드 입력
        keywords = st.text_input(
            "검색할 키워드를 입력하세요 (최대 3개, 쉼표로 구분):",
            placeholder="예시: API, 개발, 회의",
            help="키워드는 쉼표(,)로 구분하여 최대 3개까지 입력 가능합니다"
        )

    with col2:
        # 검색 버튼
        search_disabled = not keywords.strip()
        if st.button("🔍 키워드 검색", key="search_keywords", use_container_width=True, disabled=search_disabled):
            # 키워드 파싱
            keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]

            if len(keyword_list) > 3:
                st.error("⚠️ 키워드는 최대 3개까지만 입력 가능합니다!")
            else:
                with st.spinner("🔍 최근 2주 데이터에서 키워드를 검색하고 있습니다..."):
                    # TODO: 키워드 검색 API 호출 (나중에 구현)
                    st.markdown("""
                    <div class="warning-box">
                        <strong>🚧 키워드 검색 기능은 현재 개발 중입니다. 곧 제공될 예정입니다!</strong>
                    </div>
                    """, unsafe_allow_html=True)

                    # 임시 결과 표시 (개발용)
                    st.markdown("**🔍 검색된 키워드:**")
                    for i, keyword in enumerate(keyword_list, 1):
                        st.markdown(f"{i}. `{keyword}`")

    # 키워드 검색 결과 영역 (향후 구현)
    if "keyword_search_results" in st.session_state:
        st.markdown("---")
        st.markdown("**📋 검색 결과:**")
        # TODO: 검색 결과 표시 로직

# 생성된 보고서 표시
if "generated_report" in st.session_state:
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📄 생성된 보고서</h2>', unsafe_allow_html=True)

    # 보고서 내용을 동적 크기 조절 박스에 표시
    st.markdown(f'<div class="report-content">{st.session_state.generated_report}</div>', unsafe_allow_html=True)

    # 액션 버튼들
    col_action1, col_action2, col_action3 = st.columns(3)

    with col_action1:
        if st.button("📋 복사용 텍스트 보기", key="show_copy_text"):
            # 복사 가능한 텍스트 박스로 표시
            st.text_area(
                "📋 아래 텍스트를 선택하여 복사하세요:",
                value=st.session_state.generated_report,
                height=150,
                key="copy_text_area"
            )

    with col_action2:
        # 다운로드 버튼
        st.download_button(
            label="💾 텍스트 파일로 다운로드",
            data=st.session_state.generated_report,
            file_name=f"weekly_report_{today.strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

    with col_action3:
        if st.button("🔄 새로 생성", key="regenerate"):
            if "generated_report" in st.session_state:
                del st.session_state.generated_report
            st.rerun()

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6c757d; font-size: 14px; margin-top: 2rem;">
        📊 주간 업무 보고서 생성기 | Powered by FastAPI + Streamlit + OpenAI
    </div>
    """,
    unsafe_allow_html=True
)