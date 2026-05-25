"""
一键启动多智能体诊断平台

启动 FastAPI 后端 (port 8000) 和 Flask 前端 (port 5000)
Usage: python run.py
"""

import subprocess
import sys
import time
import signal
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")

processes = []


def start_backend():
    print("[run] 启动 FastAPI 后端 (port 8000)...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=BACKEND_DIR,
        env=env,
    )
    processes.append(("FastAPI", p))
    print("[run] FastAPI 后端 PID:", p.pid)


def start_frontend():
    print("[run] 启动 Flask 前端 (port 5000)...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=FRONTEND_DIR,
        env=env,
    )
    processes.append(("Flask", p))
    print("[run] Flask 前端 PID:", p.pid)


def shutdown(signum=None, frame=None):
    print("\n[run] 正在关闭所有服务...")
    for name, p in processes:
        if p.poll() is None:
            print(f"[run] 终止 {name} (PID {p.pid})...")
            p.terminate()
    for name, p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"[run] 强制终止 {name}...")
            p.kill()
    print("[run] 所有服务已关闭.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    start_backend()
    time.sleep(2)
    start_frontend()

    print("\n[run] ========================================")
    print("[run]   FastAPI 后端:  http://127.0.0.1:8000")
    print("[run]   Flask  前端:  http://127.0.0.1:5000")
    print("[run]   按 Ctrl+C 关闭所有服务")
    print("[run] ========================================\n")

    try:
        while True:
            for name, p in processes:
                if p.poll() is not None:
                    print(f"[run] {name} 意外退出 (code {p.returncode}), 正在关闭...")
                    shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()
