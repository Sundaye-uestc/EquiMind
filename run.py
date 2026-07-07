"""
一键启动多智能体诊断平台

启动 FastAPI 后端 (port 8000) 和 Flask 前端 (port 5000)

Usage:
  python run.py                        # 默认启动
  python run.py --skip-vllm-check      # 跳过 vLLM 健康检查（调试用）
  python run.py --backend-only         # 仅启动 FastAPI 后端
  python run.py --frontend-only        # 仅启动 Flask 前端
"""

import subprocess
import sys
import time
import signal
import os
import argparse
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")

# 确保 backend 在 sys.path 中，以便导入 config_handler
sys.path.insert(0, BACKEND_DIR)

processes = []


def check_vllm_health(port: int, timeout: float = 2.0) -> bool:
    """检测本地 vLLM 服务是否在运行。"""
    url = f"http://localhost:{port}/health"
    try:
        req = urllib.request.Request(url)
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


def preflight_check(skip_vllm: bool = False) -> bool:
    """启动前检查。

    当 rag.yaml 中 backend 为 local 时，验证 vLLM 服务是否就绪。
    """
    try:
        from utils.config_handler import rag_conf
    except Exception as e:
        print(f"[run] ⚠️ 无法加载配置: {e}")
        return True  # 配置加载失败不阻断，让后续错误自然暴露

    backend = rag_conf.get("backend", "dashscope")

    if backend != "local":
        print(f"[run] 后端: {backend}（云端模式，无需本地 vLLM）")
        return True

    if skip_vllm:
        print("[run] 后端: local（--skip-vllm-check，跳过 vLLM 检测）")
        return True

    print("[run] 后端: local，检测 vLLM 服务状态...")

    # 从配置读取端口
    chat_base = rag_conf.get("chat_api_base", "http://localhost:8002/v1")
    embed_base = rag_conf.get("embedding_api_base", "http://localhost:8003/v1")

    chat_port = 8002
    embed_port = 8003
    try:
        from urllib.parse import urlparse
        chat_port = urlparse(chat_base).port or 8002
        embed_port = urlparse(embed_base).port or 8003
    except Exception:
        pass

    chat_ok = check_vllm_health(chat_port)
    embed_ok = check_vllm_health(embed_port)

    if chat_ok and embed_ok:
        print(f"[run] ✓ vLLM Chat (:{chat_port}) 就绪")
        print(f"[run] ✓ vLLM Embedding (:{embed_port}) 就绪")
        return True

    # 有服务未就绪
    print("\n[run] ╔══════════════════════════════════════════╗")
    print("[run] ║  ⚠️  vLLM 服务未完全就绪！              ║")
    print("[run] ╠══════════════════════════════════════════╣")
    if not chat_ok:
        print(f"[run] ║  ✗ vLLM Chat (:{chat_port}) — 未响应     ║")
    else:
        print(f"[run] ║  ✓ vLLM Chat (:{chat_port}) — 就绪       ║")
    if not embed_ok:
        print(f"[run] ║  ✗ vLLM Embedding (:{embed_port}) — 未响应 ║")
    else:
        print(f"[run] ║  ✓ vLLM Embedding (:{embed_port}) — 就绪   ║")
    print("[run] ╠══════════════════════════════════════════╣")
    print("[run] ║  请先启动 vLLM 服务：                     ║")
    print("[run] ║    ./scripts/start_all.sh                ║")
    print("[run] ║  或跳过检查：                             ║")
    print("[run] ║    python run.py --skip-vllm-check       ║")
    print("[run] ╚══════════════════════════════════════════╝\n")

    # 询问用户是否继续
    try:
        answer = input("[run] 继续启动 FastAPI + Flask？(vLLM 不可用时会报错) [y/N]: ")
        if answer.lower() not in ("y", "yes"):
            print("[run] 已取消。")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\n[run] 已取消。")
        return False

    return True


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
    parser = argparse.ArgumentParser(description="EquiMind 一键启动")
    parser.add_argument(
        "--skip-vllm-check",
        action="store_true",
        help="跳过 vLLM 健康检查（调试用）",
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="仅启动 FastAPI 后端",
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="仅启动 Flask 前端",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 启动前检查
    if not preflight_check(skip_vllm=args.skip_vllm_check):
        sys.exit(1)

    if not args.frontend_only:
        start_backend()
        time.sleep(2)

    if not args.backend_only:
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
