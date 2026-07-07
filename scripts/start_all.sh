#!/bin/bash
# ============================================================
# EquiMind — 一站式启动脚本
# ============================================================
# 启动顺序：
#   1. vLLM Chat (8002)      — 后台，等待 /health 就绪
#   2. vLLM Embedding (8003) — 后台，等待 /health 就绪
#   3. FastAPI 后端 (8000)
#   4. Flask  前端 (5000)
#
# Usage:
#   ./scripts/start_all.sh              # 全部启动
#   ./scripts/start_all.sh --frontend-only  # 仅前端（vLLM 已运行）
#   ./scripts/start_all.sh --backend-only   # 仅后端
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
LOG_DIR="$PROJECT_DIR/logs"

CONDA_ENV="/data/sqx/conda/equimind"
VLLM_CHAT_URL="http://localhost:8002/health"
VLLM_EMBED_URL="http://localhost:8003/health"

START_FRONTEND=true
START_BACKEND=true
SKIP_VLLM=false

# ---- 参数解析 ----
for arg in "$@"; do
    case $arg in
        --frontend-only)
            SKIP_VLLM=true
            START_BACKEND=false
            ;;
        --backend-only)
            SKIP_VLLM=true
            START_FRONTEND=false
            ;;
        --skip-vllm)
            SKIP_VLLM=true
            ;;
        --help|-h)
            echo "Usage: $0 [--frontend-only|--backend-only|--skip-vllm]"
            exit 0
            ;;
    esac
done

# ---- 清理函数 ----
PIDS=()
cleanup() {
    echo ""
    echo "[start_all] 正在关闭所有服务..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "[start_all] 终止 PID $pid..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    # 等待子进程退出
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "[start_all] 所有服务已关闭."
}
trap cleanup EXIT INT TERM

# ---- conda 激活 ----
activate_conda() {
    source /home/lab/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
}

# ---- 健康检查 ----
wait_for_health() {
    local url=$1
    local name=$2
    local max_wait=${3:-300}
    local waited=0
    echo -n "[start_all] 等待 $name ($url)"
    while [ $waited -lt $max_wait ]; do
        if curl -sf -o /dev/null "$url" 2>/dev/null; then
            echo " ✓ 就绪 (${waited}s)"
            return 0
        fi
        sleep 3
        waited=$((waited + 3))
        echo -n "."
    done
    echo " ✗ 超时 (${max_wait}s)"
    return 1
}

# ---- 创建日志目录 ----
mkdir -p "$LOG_DIR"

# ============================================================
# 1. 启动 vLLM 服务
# ============================================================
if [ "$SKIP_VLLM" = false ]; then
    echo "[start_all] ========================================"
    echo "[start_all] 启动 vLLM 推理服务..."
    echo "[start_all] ========================================"

    activate_conda

    # 启动 vLLM Chat（后台）
    echo "[start_all] 启动 vLLM Chat (端口 8002)..."
    vllm serve /data/sqx/models/huggingface/Qwen3-14B-AWQ \
      --host 0.0.0.0 --port 8002 \
      --tensor-parallel-size 1 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.85 \
      --dtype auto --trust-remote-code \
      > "$LOG_DIR/vllm_chat.log" 2>&1 &
    PIDS+=($!)

    # 启动 vLLM Embedding（后台）
    echo "[start_all] 启动 vLLM Embedding (端口 8003)..."
    vllm serve /data/sqx/models/modelscope/models/BAAI--bge-m3/snapshots/master \
      --host 0.0.0.0 --port 8003 \
      --tensor-parallel-size 1 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.10 \
      --task embed --trust-remote-code \
      > "$LOG_DIR/vllm_embedding.log" 2>&1 &
    PIDS+=($!)

    # 等待两个 vLLM 服务就绪
    echo ""
    wait_for_health "$VLLM_CHAT_URL" "vLLM Chat" 300 || {
        echo "[start_all] ⚠️ vLLM Chat 启动失败，检查日志: $LOG_DIR/vllm_chat.log"
        exit 1
    }
    wait_for_health "$VLLM_EMBED_URL" "vLLM Embedding" 120 || {
        echo "[start_all] ⚠️ vLLM Embedding 启动失败，检查日志: $LOG_DIR/vllm_embedding.log"
        exit 1
    }
    echo ""
fi

# ============================================================
# 2. 启动 FastAPI 后端 (port 8000)
# ============================================================
if [ "$START_BACKEND" = true ]; then
    echo "[start_all] 启动 FastAPI 后端 (端口 8000)..."
    activate_conda
    cd "$BACKEND_DIR"
    python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 &
    PIDS+=($!)
    cd "$PROJECT_DIR"
    sleep 2
    echo "[start_all] FastAPI 后端 PID: ${PIDS[-1]}"
fi

# ============================================================
# 3. 启动 Flask 前端 (port 5000)
# ============================================================
if [ "$START_FRONTEND" = true ]; then
    echo "[start_all] 启动 Flask 前端 (端口 5000)..."
    activate_conda
    cd "$FRONTEND_DIR"
    python app.py &
    PIDS+=($!)
    cd "$PROJECT_DIR"
    sleep 1
    echo "[start_all] Flask 前端 PID: ${PIDS[-1]}"
fi

# ============================================================
# 4. 就绪提示
# ============================================================
echo ""
echo "[start_all] ========================================"
echo "[start_all]   EquiMind 全服务已启动"
echo "[start_all]   FastAPI 后端:  http://localhost:8000"
echo "[start_all]   Flask  前端:  http://localhost:5000"
echo "[start_all]   vLLM Chat:     http://localhost:8002"
echo "[start_all]   vLLM Embed:    http://localhost:8003"
echo "[start_all]   按 Ctrl+C 关闭所有服务"
echo "[start_all] ========================================"
echo ""

# ---- 主循环：监控子进程 ----
while true; do
    all_dead=true
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            all_dead=false
        fi
    done
    if $all_dead; then
        echo "[start_all] 所有子进程已退出"
        exit 0
    fi
    sleep 2
done
