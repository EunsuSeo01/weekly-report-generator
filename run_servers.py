#!/usr/bin/env python3
"""
FastAPI 서버와 Streamlit 앱을 동시에 실행하는 스크립트
사용법: python run_servers.py
"""

import subprocess
import sys
import os
import time
import signal
from threading import Thread

class ServerRunner:
    def __init__(self):
        self.fastapi_process = None
        self.streamlit_process = None

    def run_fastapi(self):
        """FastAPI 서버를 실행합니다."""
        print("🔧 FastAPI 서버를 시작합니다...")
        try:
            self.fastapi_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "template:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ])
        except Exception as e:
            print(f"❌ FastAPI 서버 시작 오류: {e}")

    def run_streamlit(self):
        """Streamlit 앱을 실행합니다."""
        print("🖥️  Streamlit 앱을 시작합니다...")
        # FastAPI 서버가 시작될 시간을 기다림
        time.sleep(3)

        try:
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run",
                "streamlit_app.py",
                "--server.port=8501",
                "--server.address=localhost",
                "--browser.gatherUsageStats=false"
            ])
        except Exception as e:
            print(f"❌ Streamlit 앱 시작 오류: {e}")

    def stop_servers(self):
        """모든 서버를 종료합니다."""
        print("\n🛑 서버들을 종료합니다...")

        if self.fastapi_process:
            self.fastapi_process.terminate()
            print("✅ FastAPI 서버 종료됨")

        if self.streamlit_process:
            self.streamlit_process.terminate()
            print("✅ Streamlit 앱 종료됨")

    def run(self):
        """두 서버를 모두 실행합니다."""
        print("🚀 주간 업무 보고서 생성기를 시작합니다...")
        print("=" * 60)
        print("📍 FastAPI 서버: http://localhost:8000")
        print("📍 Streamlit 앱: http://localhost:8501")
        print("⚠️  종료하려면 Ctrl+C를 누르세요")
        print("=" * 60)

        try:
            # FastAPI 서버를 별도 스레드에서 실행
            fastapi_thread = Thread(target=self.run_fastapi)
            fastapi_thread.daemon = True
            fastapi_thread.start()

            # Streamlit 앱을 별도 스레드에서 실행
            streamlit_thread = Thread(target=self.run_streamlit)
            streamlit_thread.daemon = True
            streamlit_thread.start()

            # 메인 스레드에서 대기
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            self.stop_servers()
            print("\n✅ 모든 서버가 종료되었습니다.")
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")
            self.stop_servers()

def main():
    # 필요한 파일들이 존재하는지 확인
    required_files = ["template.py", "streamlit_app.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ 다음 파일들을 찾을 수 없습니다: {', '.join(missing_files)}")
        print("올바른 디렉토리에서 실행해주세요.")
        sys.exit(1)

    # 서버 실행
    runner = ServerRunner()

    # 시그널 핸들러 등록
    def signal_handler(signum, frame):
        runner.stop_servers()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    runner.run()

if __name__ == "__main__":
    main()