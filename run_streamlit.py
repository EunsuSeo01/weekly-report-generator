#!/usr/bin/env python3
"""
Streamlit 앱 실행 스크립트
사용법: python run_streamlit.py
"""

import subprocess
import sys
import os

def run_streamlit():
    """Streamlit 앱을 실행합니다."""
    print("🚀 주간 업무 보고서 생성기를 시작합니다...")
    print("📱 브라우저에서 http://localhost:8501 로 접속하세요")
    print("⚠️  종료하려면 Ctrl+C를 누르세요")
    print("-" * 50)

    try:
        # Streamlit 앱 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "streamlit_app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n✅ 애플리케이션이 종료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 현재 디렉토리가 올바른지 확인
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py 파일을 찾을 수 없습니다.")
        print("올바른 디렉토리에서 실행해주세요.")
        sys.exit(1)

    run_streamlit()