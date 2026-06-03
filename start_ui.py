"""
一键启动 OPS-SAT 新前端 (FastAPI + React)

用法:
    python start_ui.py          # 同时启动后端 + 前端
    python start_ui.py --api    # 仅启动 FastAPI 后端
    python start_ui.py --web    # 仅启动 React 前端
"""
import subprocess
import sys
import os
import signal
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

processes = []


def cleanup(signum=None, frame=None):
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def start_api():
    """启动 FastAPI 后端 (port 8000)"""
    print("[*] Starting FastAPI backend on :8000")
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--reload", "--port", "8000", "--host", "127.0.0.1"],
        cwd=PROJECT_ROOT,
    )
    processes.append(p)
    return p


def start_web():
    """启动 React 前端 (port 5180)"""
    print("[*] Starting React frontend on :5180")
    p = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        shell=True,
    )
    processes.append(p)
    return p


def main():
    parser = argparse.ArgumentParser(description="OPS-SAT UI Launcher")
    parser.add_argument("--api", action="store_true", help="仅启动 FastAPI")
    parser.add_argument("--web", action="store_true", help="仅启动 React")
    args = parser.parse_args()

    start_both = not args.api and not args.web

    print("=" * 50)
    print("  OPS-SAT Telemetry Diagnostics - Launcher")
    print("=" * 50)

    if start_both or args.api:
        start_api()
    if start_both or args.web:
        start_web()

    if start_both:
        print("\n[OK] System started")
        print("  Backend API:  http://localhost:8000/docs")
        print("  Frontend:     http://localhost:5180")
        print("  Legacy UI:    streamlit run src/ui/app.py")
        print("\n  Press Ctrl+C to stop all services\n")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
