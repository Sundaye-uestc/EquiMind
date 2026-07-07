#!/bin/bash
# ============================================================
# EquiMind — 启动 vLLM Chat 服务 (Qwen3-14B-AWQ)
# ============================================================
# 端口: 8002  |  显存: ~9.4 GB  |  GPU: RTX 4090 ×1
# 参数与 backend/config/vllm.yaml 保持同步
# ============================================================
set -euo pipefail

# ---- conda 环境 ----
source /home/lab/anaconda3/etc/profile.d/conda.sh
conda activate /data/sqx/conda/equimind

# ---- 模型与服务参数 ----
MODEL_PATH="/data/sqx/models/huggingface/Qwen3-14B-AWQ"
HOST="0.0.0.0"
PORT=8002
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.85
TP=1

# ---- 日志 ----
LOG_DIR="/data/sqx/Equimind/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_chat_$(date +%Y%m%d_%H%M%S).log"

echo "[start_vllm_chat] 模型: $MODEL_PATH"
echo "[start_vllm_chat] 端口: $PORT  |  显存利用率: $GPU_MEM_UTIL  |  max_model_len: $MAX_MODEL_LEN"
echo "[start_vllm_chat] 日志: $LOG_FILE"
echo "[start_vllm_chat] 加载 AWQ 模型约需 1-3 分钟，请等待 /health 就绪..."

vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype auto \
  --trust-remote-code \
  2>&1 | tee "$LOG_FILE"
