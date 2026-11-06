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

# 自带 Node.js，不依赖 PATH。
# 实测：项目内 node-runtime 只带 node.exe（v22.16.0），不含 npm
#（node_modules 下只有 pnpm）；而 PATH 里的 npm 可能来自别的 Node 安装
#（本机实测是 WorkBuddy 托管的 v22.22.2）。从 Explorer 双击 .bat 打开的
# cmd 不一定有那套 PATH，裸 `npm run dev` 会直接失败。
# 因此前端一律用「node.exe + vite.js」直接拉起，绕开 npm。
NODE_EXE = os.path.join(PROJECT_ROOT, "node-runtime", "node.exe")
VITE_JS = os.path.join(FRONTEND_DIR, "node_modules", "vite", "bin", "vite.js")

API_PORT = 8000
WEB_PORT = 5180
API_URL = f"http://localhost:{API_PORT}"
WEB_URL = f"http://localhost:{WEB_PORT}"

# 保存子进程的 PID，用于清理
child_processes = []


# ── 工具函数 ──────────────────────────────────────────────

def log(tag, msg):
    icons = {"OK": "[OK]", "ERR": "[!!]", "WAIT": "[..]", "INFO": "[>>]"}
    print(f"  {icons.get(tag, '   ')} {msg}")


def child_env(name, value):
    """子进程环境副本：名字未显式设置时用给定默认值，其余全部继承。

    **为什么后端默认关鉴权**：``.env`` 被 ``.gitignore:51`` 忽略，队友
    clone 下来 ``.env`` 里没有 ``API_AUTH_KEYS``，而 ``src/api/security.py``
    的语义是「未配置即拒绝全部」—— 于是除 ``/api/health`` 外所有接口
    返回 503，前端满屏报错。启动器是本地开发入口，这里兜底关掉；
    生产部署由 K8s/Docker 显式注入变量开启，不依赖本文件。
    """
    env = os.environ.copy()
    env.setdefault(name, value)
    return env


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


def kill_pid_tree(pid):
    """杀掉进程树（包括子进程）"""
    # taskkill /T 杀掉进程树
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                   capture_output=True, timeout=10)


def cleanup(signum=None, frame=None):
    for process in list(child_processes):
        if process.poll() is None:
            try:
                kill_pid_tree(process.pid)
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                log("ERR", f"服务 {process.pid} 停止失败")
    child_processes.clear()
    if signum is not None:
        raise SystemExit(128 + signum)


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
    """校验自带 Node.js。

    不再用裸 ``node`` —— 双击启动器时 PATH 里可能没有 Node。
    """
    if not os.path.exists(NODE_EXE):
        log("ERR", f"未找到 Node.js: {NODE_EXE}")
        return False
    try:
        # encoding/errors 必须显式给：默认按 locale 解码，中文 Windows 下
        # Node 输出一个非 GBK 字节就会抛 UnicodeDecodeError（实测发生过）。
        r = subprocess.run(
            [NODE_EXE, "--version"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15,
        )
        log("OK", f"Node.js {r.stdout.strip()}")
        return True
    except Exception as e:
        log("ERR", f"Node.js 检查失败: {e}")
        return False


def check_node_modules():
    if not os.path.isfile(VITE_JS):
        log("ERR", "前端依赖缺失。请先在 frontend 目录安装依赖。")
        return False
    log("OK", "前端依赖已就绪")
    return True


def check_api_deps():
    code = """
from dotenv import load_dotenv
load_dotenv('.env')
import fastapi, uvicorn, pandas, numpy, scipy, sklearn, faiss, torch
import sentence_transformers, langchain_core, langchain_community, langchain_text_splitters
import pypdf, yaml, joblib, requests, sqlite3
from src.detection.frozen import load_bundle
from src.rag.embedding import resolve_embedding_model_path
from src.utils.paths import read_faiss_index_status
from src.api.result_data import confirmed_results
config = yaml.safe_load(open('configs/rag_config.yaml', encoding='utf-8'))
resolve_embedding_model_path(config['embedding'])
load_bundle()
confirmed_results()
status, _, reason = read_faiss_index_status()
if status != 'ok':
    raise RuntimeError(reason)
"""
    try:
        result = subprocess.run([VENV_PYTHON, "-c", code], cwd=PROJECT_ROOT,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log("ERR", f"后端检查失败。{exc}")
        return False
    if result.returncode != 0:
        log("ERR", "后端依赖或冻结资源检查失败。请按部署说明准备依赖、数据、模型和索引。")
        log("ERR", result.stderr[-1200:])
        return False
    log("OK", "后端依赖、数据、模型和索引已就绪")
    return True


def check_port(port, name):
    if port_in_use(port):
        log("ERR", f"{name} 端口 {port} 已被占用。启动已停止。")
        return False
    return True


# ── 启动服务 ──────────────────────────────────────────────

def start_api():
    log("INFO", f"启动 FastAPI 后端 -> {API_URL}")

    if child_env("API_AUTH_DISABLED", "true").get("API_AUTH_DISABLED", "").lower() in {"true", "1", "yes"}:
        log("INFO", "本地开发模式已关闭 API 鉴权。服务监听 127.0.0.1。")
    log_file = os.path.join(PROJECT_ROOT, ".backend.log")
    lf = open(log_file, "w", encoding="utf-8")

    p = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "src.api.main:app",
         "--port", str(API_PORT), "--host", "127.0.0.1"],
        cwd=PROJECT_ROOT,
        stdout=lf,
        stderr=subprocess.STDOUT,
        env=child_env("API_AUTH_DISABLED", "true"),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    child_processes.append(p)

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
        # 直接 node + vite.js，不用 npm run dev：项目自带 node-runtime 无 npm，
        # 且 PATH 里的 npm 可能来自另一套 Node 安装（双击启动时不在 PATH）。
        [NODE_EXE, VITE_JS, "--port", str(WEB_PORT), "--strictPort"],
        cwd=FRONTEND_DIR,
        stdout=lf,
        stderr=subprocess.STDOUT,
        env=child_env("BACKEND_URL", API_URL),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    child_processes.append(p)

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
            from dotenv import dotenv_values
            keys = os.getenv("API_AUTH_KEYS") or dotenv_values(os.path.join(PROJECT_ROOT, ".env")).get("API_AUTH_KEYS", "")
            headers = {"X-API-Key": keys.split(",")[0].strip()} if keys else {}
            request = urllib.request.Request(f"{API_URL}{path}", headers=headers)
            req = urllib.request.urlopen(request, timeout=10)
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
        if not check_api_deps():
            sys.exit(1)
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
        sys.exit(1)

    # ── API 健康检查 ──
    print()
    print("  [ Health Check ]")
    if (start_both or args.api) and not health_check():
        log("ERR", "部分 API 检查失败")
        cleanup()
        sys.exit(1)
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
            for process in list(child_processes):
                if process.poll() is not None:
                    log("ERR", f"服务 {process.pid} 已退出")
                    cleanup()
                    sys.exit(1)
            time.sleep(3)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
