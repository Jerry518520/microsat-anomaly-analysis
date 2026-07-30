"""
一键启动 OPS-SAT 遥测异常诊断系统 (FastAPI + React)

用法:
    python scripts/start_ui.py              # 同时启动后端 + 前端 + 自动打开浏览器
    python scripts/start_ui.py --api        # 仅启动 FastAPI 后端
    python scripts/start_ui.py --web        # 仅启动 React 前端
    python scripts/start_ui.py --no-open    # 不自动打开浏览器
"""
import subprocess
import sys
import os
import signal
import argparse
import socket
import time
import webbrowser
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")

API_PORT = 8000
WEB_PORT = 5180
API_URL = f"http://localhost:{API_PORT}"
WEB_URL = f"http://localhost:{WEB_PORT}"

# 保存子进程的 PID，用于清理
child_pids = []


# ── 工具函数 ──────────────────────────────────────────────

def log(tag, msg):
    icons = {"OK": "[OK]", "ERR": "[!!]", "WAIT": "[..]", "INFO": "[>>]"}
    print(f"  {icons.get(tag, '   ')} {msg}")


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(port, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def kill_port(port):
    """强制杀掉占用指定端口的进程"""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                subprocess.run(["taskkill", "/F", "/PID", pid],
                               capture_output=True, timeout=5)
    except Exception:
        pass


def kill_pid_tree(pid):
    """杀掉进程树（包括子进程）"""
    # taskkill /T 杀掉进程树
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                   capture_output=True, timeout=10)


def cleanup(signum=None, frame=None):
    print("\n")
    log("INFO", "正在停止所有服务...")
    for pid in child_pids:
        try:
            kill_pid_tree(pid)
        except Exception:
            pass
    # 双保险：强制杀掉端口
    kill_port(API_PORT)
    kill_port(WEB_PORT)
    log("OK", "所有服务已停止")
    print()
    sys.exit(0)


def install_signal_handlers():
    """注册信号处理 + Windows atexit"""
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    # Windows: 控制台窗口关闭时也会触发
    try:
        import atexit
        atexit.register(cleanup)
    except Exception:
        pass


# ── 前置检查 ──────────────────────────────────────────────

def check_python():
    if not os.path.exists(VENV_PYTHON):
        log("ERR", f"未找到虚拟环境: {VENV_PYTHON}")
        log("INFO", "请先执行: python -m venv .venv")
        return False
    log("OK", "Python 虚拟环境就绪")
    return True


def check_node():
    try:
        r = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        log("OK", f"Node.js {r.stdout.strip()}")
        return True
    except FileNotFoundError:
        log("ERR", "未找到 Node.js，请先安装 https://nodejs.org/")
        return False
    except Exception:
        log("ERR", "Node.js 检查失败")
        return False


def check_node_modules():
    pkg = os.path.join(FRONTEND_DIR, "node_modules", ".package-lock.json")
    if not os.path.exists(pkg):
        log("WAIT", "首次运行，正在安装前端依赖 (npm install)...")
        r = subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, shell=True, timeout=120)
        if r.returncode != 0:
            log("ERR", "npm install 失败")
            return False
        log("OK", "前端依赖安装完成")
    else:
        log("OK", "前端依赖已就绪")
    return True


def check_api_deps():
    code = "import fastapi, uvicorn, pandas"
    r = subprocess.run([VENV_PYTHON, "-c", code], capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        log("WAIT", "正在安装后端依赖...")
        subprocess.run([VENV_PYTHON, "-m", "pip", "install", "fastapi", "uvicorn", "pandas"],
                       capture_output=True, timeout=60)
    log("OK", "后端依赖已就绪")
    return True


def check_port(port, name):
    if port_in_use(port):
        log("WAIT", f"端口 {port} 被占用，正在清理...")
        kill_port(port)
        time.sleep(1)
        if port_in_use(port):
            log("ERR", f"端口 {port} 无法释放，请手动关闭占用进程")
            return False
        log("OK", f"端口 {port} 已释放")
    return True


# ── 启动服务 ──────────────────────────────────────────────

def start_api():
    log("INFO", f"启动 FastAPI 后端 -> {API_URL}")

    log_file = os.path.join(PROJECT_ROOT, ".backend.log")
    lf = open(log_file, "w", encoding="utf-8")

    p = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "src.api.main:app",
         "--port", str(API_PORT), "--host", "127.0.0.1"],
        cwd=PROJECT_ROOT,
        stdout=lf,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    child_pids.append(p.pid)

    if wait_for_port(API_PORT):
        log("OK", f"后端已就绪  {API_URL}/docs")
        return True

    # 启动失败，读取日志
    p.terminate()
    lf.close()
    log("ERR", "后端启动失败")
    try:
        with open(log_file, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            for line in lines[-6:]:
                print(f"       {line}")
    except Exception:
        pass
    return False


def start_web():
    log("INFO", f"启动 React 前端 -> {WEB_URL}")

    log_file = os.path.join(PROJECT_ROOT, ".frontend.log")
    lf = open(log_file, "w", encoding="utf-8")

    p = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        shell=True,
        stdout=lf,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    child_pids.append(p.pid)

    if wait_for_port(WEB_PORT):
        log("OK", f"前端已就绪  {WEB_URL}")
        return True

    # 启动失败
    p.terminate()
    lf.close()
    log("ERR", "前端启动失败")
    try:
        with open(log_file, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            for line in lines[-6:]:
                print(f"       {line}")
    except Exception:
        pass
    return False


def open_browser(url, delay=2):
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


# ── 健康检查 ──────────────────────────────────────────────

def health_check():
    """启动后验证所有 API 端点"""
    import urllib.request
    import json

    checks = [
        ("/api/health", "health"),
        ("/api/dashboard/metrics", "metrics"),
        ("/api/dashboard/channels", "channels"),
        ("/api/dashboard/alerts", "alerts"),
    ]

    all_ok = True
    for path, name in checks:
        try:
            req = urllib.request.urlopen(f"{API_URL}{path}", timeout=10)
            json.loads(req.read().decode("utf-8"))
            log("OK", f"  {name}")
        except Exception as e:
            log("ERR", f"  {name}: {e}")
            all_ok = False

    return all_ok


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OPS-SAT 遥测异常诊断系统 - 一键启动")
    parser.add_argument("--api", action="store_true", help="仅启动 FastAPI 后端")
    parser.add_argument("--web", action="store_true", help="仅启动 React 前端")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    install_signal_handlers()
    start_both = not args.api and not args.web

    print()
    print("  +================================================+")
    print("  |  OPS-SAT Telemetry Diagnostics - Launcher      |")
    print("  +================================================+")
    print()

    # ── 前置检查 ──
    print("  [ Environment Check ]")

    if not check_python():
        sys.exit(1)

    if start_both or args.web:
        if not check_node():
            sys.exit(1)
        if not check_node_modules():
            sys.exit(1)

    if start_both or args.api:
        check_api_deps()
        if not check_port(API_PORT, "FastAPI"):
            sys.exit(1)

    if start_both or args.web:
        if not check_port(WEB_PORT, "Vite"):
            sys.exit(1)

    print()

    # ── 启动服务 ──
    print("  [ Starting Services ]")

    api_ok = True
    web_ok = True

    if start_both or args.api:
        api_ok = start_api()

    if start_both or args.web:
        web_ok = start_web()

    if not api_ok or not web_ok:
        print()
        log("ERR", "部分服务启动失败，请检查上方错误信息")
        cleanup()

    # ── API 健康检查 ──
    print()
    print("  [ Health Check ]")
    if not health_check():
        log("ERR", "部分 API 检查失败")
    print()

    # ── 打印摘要 ──
    print("  +================================================+")
    print("  |  All services started                          |")
    print("  +================================================+")
    print(f"  |  Backend API   {API_URL:<32s}|")
    print(f"  |  API Docs      {API_URL}/docs{' ' * (32 - len(API_URL) - 6)}|")
    print(f"  |  Frontend      {WEB_URL:<32s}|")
    print("  +------------------------------------------------+")
    print("  |  Press Ctrl+C to stop all services             |")
    print("  +================================================+")
    print()

    # ── 自动打开浏览器 ──
    if not args.no_open and web_ok:
        open_browser(WEB_URL)
        log("INFO", "正在打开浏览器...")

    # ── 保持运行 ──
    try:
        while True:
            # 定期检查子进程是否还活着
            for pid in list(child_pids):
                try:
                    os.kill(pid, 0)
                except OSError:
                    log("ERR", f"子进程 {pid} 已退出，正在重启...")
                    child_pids.remove(pid)
                    # 可以在这里加自动重启逻辑
            time.sleep(3)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
